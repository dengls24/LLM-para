"""
LLM-Para  Multi-Chip Parallelism & Communication Analyzer (Beta)
=================================================================
Models LLM inference distributed across multiple devices with:
  - Tensor Parallelism (TP): split weight matrices across devices
  - Pipeline Parallelism (PP): split layers across pipeline stages
  - Data Parallelism (DP): replicate model, partition batch

Communication primitives modeled:
  - AllReduce: 2(N-1)/N * msg_size (ring algorithm)
  - AllGather: (N-1)/N * msg_size
  - Point-to-Point: msg_size (pipeline stages)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math


@dataclass
class ParallelismConfig:
    num_devices: int = 8
    tp_degree: int = 8
    pp_degree: int = 1
    dp_degree: int = 1
    inter_chip_bw_gbs: float = 900.0
    topology: str = 'ring'


class ParallelismAnalyzer:
    """Analyze multi-chip LLM inference with TP/PP/DP strategies."""

    def __init__(self, model_config: dict, hardware: dict, parallel_config: ParallelismConfig):
        self.cfg = model_config
        self.hw = hardware
        self.par = parallel_config

        # Model params
        self.h = self.cfg.get('hidden_size', 4096)
        self.n_heads = self.cfg.get('num_heads', 32)
        self.n_kv_heads = self.cfg.get('num_key_value_heads', self.n_heads)
        self.n_layers = self.cfg.get('num_layers', 32)
        self.ffn_inter = self.cfg.get('intermediate_size', 4 * self.h)
        self.vocab = self.cfg.get('vocab_size', 32000)
        self.seq_len = self.cfg.get('seq_len', 2048)
        self.batch = self.cfg.get('batch_size', 1)
        self.max_gen = self.cfg.get('max_gen_len', 4096)

        # Quantization
        qc = self.cfg.get('quant_config', {})
        self.w_bits = qc.get('weight_attn', 16)
        self.act_bits = qc.get('activation', 16)
        self.kv_bits = qc.get('kv_cache', 16)

        # Hardware
        self.per_dev_peak = hardware.get('per_device_peak', hardware.get('peak_performance', 67e12))
        self.per_dev_bw = hardware.get('per_device_bw', hardware.get('memory_bandwidth', 3.35e12))
        self.per_dev_cap = hardware.get('per_device_capacity', hardware.get('memory_capacity', 80e9))
        self.inter_bw = self.par.inter_chip_bw_gbs * 1e9  # convert to B/s

    # ── Sharding ──────────────────────────────────────────────────────────────

    def compute_sharding(self) -> dict:
        tp = self.par.tp_degree
        pp = self.par.pp_degree

        # Weight sizes (bytes)
        bytes_per_param = self.w_bits / 8
        # Attention weights per layer: Q, K, V, O projections
        qkv_params = self.h * self.h * 3  # Q + K + V (simplified for GQA below)
        # GQA: K,V have fewer heads
        head_dim = self.h // self.n_heads
        q_params = self.h * self.h
        k_params = self.h * (self.n_kv_heads * head_dim)
        v_params = k_params
        o_params = self.h * self.h
        attn_params_per_layer = q_params + k_params + v_params + o_params

        # FFN weights per layer
        gate = 1 if self.cfg.get('use_gate_ffn', True) else 0
        ffn_params_per_layer = self.h * self.ffn_inter * (2 + gate)

        # Norms per layer
        norm_params_per_layer = self.h * 2  # 2 norms

        params_per_layer = attn_params_per_layer + ffn_params_per_layer + norm_params_per_layer
        total_params = params_per_layer * self.n_layers + self.vocab * self.h * 2  # embed + lm_head

        total_weight_bytes = total_params * bytes_per_param
        layers_per_stage = self.n_layers // max(1, pp)

        # TP sharding: attention heads split across TP, FFN columns split
        weight_per_device = total_weight_bytes / tp / pp

        # KV cache per device: KV heads divided by TP
        kv_heads_per_dev = max(1, self.n_kv_heads // tp)
        kv_per_token = 2 * kv_heads_per_dev * head_dim * (self.kv_bits / 8) * layers_per_stage
        total_kv = kv_per_token * (self.seq_len + self.max_gen) * self.batch

        # Activations per device (per micro-batch)
        act_per_device = self.batch * self.seq_len * self.h * (self.act_bits / 8) * 2  # rough

        return {
            'total_params': total_params,
            'total_weight_bytes': total_weight_bytes,
            'weight_per_device_gb': weight_per_device / 1e9,
            'kv_per_device_gb': total_kv / 1e9,
            'activation_per_device_gb': act_per_device / 1e9,
            'layers_per_stage': layers_per_stage,
            'kv_heads_per_device': kv_heads_per_dev,
            'attn_params_per_layer': attn_params_per_layer,
            'ffn_params_per_layer': ffn_params_per_layer,
            'strategy': f'TP={tp}, PP={pp}, DP={self.par.dp_degree}',
        }

    # ── Communication ────────────────────────────────────────────────────────

    def compute_communication_volume(self) -> dict:
        tp = self.par.tp_degree
        pp = self.par.pp_degree
        act_bytes = self.act_bits / 8

        # TP communication per layer:
        # After O-projection: AllReduce of size (batch * seq * hidden * act_bytes)
        # After FFN-down: AllReduce of same size
        msg_size_tp = self.batch * 1 * self.h * act_bytes  # decode: seq=1

        allreduce_volume = self._allreduce_volume(msg_size_tp)
        allreduce_per_layer = allreduce_volume * 2  # two AllReduces per layer (attn + FFN)
        allreduce_time_per_layer_s = allreduce_per_layer / self.inter_bw if self.inter_bw > 0 else 0

        # PP communication: activation transfer between stages
        pp_msg = self.batch * 1 * self.h * act_bytes  # decode
        pp_per_boundary = pp_msg / self.inter_bw if self.inter_bw > 0 and pp > 1 else 0

        # Prefill communication (larger messages)
        msg_size_prefill = self.batch * self.seq_len * self.h * act_bytes
        ar_prefill_per_layer = self._allreduce_volume(msg_size_prefill) * 2
        ar_prefill_time = ar_prefill_per_layer / self.inter_bw if self.inter_bw > 0 else 0

        total_decode_comm = allreduce_time_per_layer_s * (self.n_layers // max(1, pp))
        total_prefill_comm = ar_prefill_time * (self.n_layers // max(1, pp))

        return {
            'tp_allreduce_msg_bytes': msg_size_tp,
            'tp_allreduce_volume_per_layer': allreduce_per_layer,
            'allreduce_per_layer_ms': allreduce_time_per_layer_s * 1000,
            'total_decode_comm_ms': total_decode_comm * 1000,
            'total_prefill_comm_ms': total_prefill_comm * 1000,
            'pp_activation_bytes': pp_msg if pp > 1 else 0,
            'pp_latency_per_boundary_ms': pp_per_boundary * 1000,
            'inter_chip_bw_gbs': self.par.inter_chip_bw_gbs,
            'tp_degree': tp,
            'pp_degree': pp,
        }

    def _allreduce_volume(self, msg_bytes: float) -> float:
        """Ring AllReduce: 2(N-1)/N * msg_size"""
        n = self.par.tp_degree
        if n <= 1:
            return 0.0
        return 2 * (n - 1) / n * msg_bytes

    # ── Per-device Memory ────────────────────────────────────────────────────

    def compute_per_device_memory(self) -> dict:
        s = self.compute_sharding()
        total = s['weight_per_device_gb'] + s['kv_per_device_gb'] + s['activation_per_device_gb']
        capacity_gb = self.per_dev_cap / 1e9
        return {
            'weights_gb': s['weight_per_device_gb'],
            'kv_cache_gb': s['kv_per_device_gb'],
            'activations_gb': s['activation_per_device_gb'],
            'total_per_device_gb': total,
            'device_capacity_gb': capacity_gb,
            'utilization_pct': min(100, total / capacity_gb * 100) if capacity_gb > 0 else 0,
            'fits': total <= capacity_gb,
        }

    # ── Throughput ───────────────────────────────────────────────────────────

    def compute_throughput(self) -> dict:
        tp = self.par.tp_degree
        pp = self.par.pp_degree
        sharding = self.compute_sharding()
        comm = self.compute_communication_volume()

        # Decode: per-token time
        # Compute time per device per layer (weight loading dominant in decode)
        weight_per_layer = (sharding['attn_params_per_layer'] + sharding['ffn_params_per_layer']) * (self.w_bits / 8) / tp
        compute_time_per_layer = weight_per_layer / self.per_dev_bw  # seconds

        # Communication per layer
        comm_time_per_layer = comm['allreduce_per_layer_ms'] / 1000  # seconds

        # Overlap: max(compute, comm) per layer
        time_per_layer = max(compute_time_per_layer, comm_time_per_layer)
        layers_per_stage = self.n_layers // max(1, pp)

        # Pipeline efficiency
        pp_efficiency = 1.0 if pp <= 1 else (layers_per_stage) / (layers_per_stage + pp - 1)

        decode_time_per_token = time_per_layer * layers_per_stage / pp_efficiency
        tokens_per_sec = 1.0 / decode_time_per_token if decode_time_per_token > 0 else 0

        # Single-device baseline (no parallelism)
        single_weight_per_layer = (sharding['attn_params_per_layer'] + sharding['ffn_params_per_layer']) * (self.w_bits / 8)
        single_time_per_layer = single_weight_per_layer / self.per_dev_bw
        single_total = single_time_per_layer * self.n_layers
        single_tps = 1.0 / single_total if single_total > 0 else 0

        speedup = tokens_per_sec / single_tps if single_tps > 0 else 0
        ideal_speedup = tp * pp  # ideal linear scaling

        # Communication overhead
        compute_total = compute_time_per_layer * layers_per_stage
        comm_total = comm_time_per_layer * layers_per_stage
        overhead_pct = comm_total / (compute_total + comm_total) * 100 if (compute_total + comm_total) > 0 else 0

        # Scaling curve: 1 to num_devices
        scaling = self._scaling_curve(sharding)

        return {
            'decode_tokens_per_sec': tokens_per_sec,
            'single_device_tokens_per_sec': single_tps,
            'speedup': speedup,
            'ideal_speedup': ideal_speedup,
            'parallel_efficiency_pct': speedup / ideal_speedup * 100 if ideal_speedup > 0 else 0,
            'pp_efficiency_pct': pp_efficiency * 100,
            'comm_overhead_pct': overhead_pct,
            'compute_time_per_layer_us': compute_time_per_layer * 1e6,
            'comm_time_per_layer_us': comm_time_per_layer * 1e6,
            'scaling': scaling,
        }

    def _scaling_curve(self, sharding: dict) -> list:
        """Compute scaling from 1 to num_devices."""
        curve = []
        act_bytes = self.act_bits / 8
        single_weight_per_layer = (sharding['attn_params_per_layer'] + sharding['ffn_params_per_layer']) * (self.w_bits / 8)

        for n in range(1, self.par.num_devices + 1):
            if n == 1:
                single_time = single_weight_per_layer / self.per_dev_bw * self.n_layers
                curve.append({'devices': 1, 'tokens_per_sec': 1.0 / single_time if single_time > 0 else 0, 'speedup': 1.0, 'efficiency_pct': 100.0})
                continue

            tp_n = min(n, self.par.tp_degree)
            weight_per_dev = single_weight_per_layer / tp_n
            comp_time = weight_per_dev / self.per_dev_bw

            msg = self.batch * 1 * self.h * act_bytes
            ar_vol = 2 * (tp_n - 1) / tp_n * msg * 2
            comm_time = ar_vol / self.inter_bw if self.inter_bw > 0 else 0

            layer_time = max(comp_time, comm_time)
            total = layer_time * self.n_layers
            tps = 1.0 / total if total > 0 else 0
            base_tps = curve[0]['tokens_per_sec']
            su = tps / base_tps if base_tps > 0 else 0

            curve.append({
                'devices': n,
                'tokens_per_sec': tps,
                'speedup': su,
                'efficiency_pct': su / n * 100 if n > 0 else 0,
            })

        return curve

    # ── Full Analysis ────────────────────────────────────────────────────────

    def run_full_analysis(self) -> dict:
        sharding = self.compute_sharding()
        comm = self.compute_communication_volume()
        memory = self.compute_per_device_memory()
        throughput = self.compute_throughput()

        return {
            'sharding': sharding,
            'communication': comm,
            'per_device_memory': memory,
            'throughput': throughput,
            'summary': {
                'num_devices': self.par.num_devices,
                'tp_degree': self.par.tp_degree,
                'pp_degree': self.par.pp_degree,
                'dp_degree': self.par.dp_degree,
                'strategy': sharding['strategy'],
                'weight_per_device_gb': sharding['weight_per_device_gb'],
                'comm_overhead_pct': throughput['comm_overhead_pct'],
                'decode_tokens_per_sec': throughput['decode_tokens_per_sec'],
                'speedup': throughput['speedup'],
                'parallel_efficiency_pct': throughput['parallel_efficiency_pct'],
            },
        }
