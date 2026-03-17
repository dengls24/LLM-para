"""
LLM-Para Heterogeneous Architecture Analyzer
=============================================
Models LLM inference on multi-tier memory systems:
  - Cambricon-LLM (chiplet-based SRAM + DRAM + Flash)
  - Flash-LLM (NAND-based weight offloading)
  - NAND-PIM (near-storage processing)

Key concepts from Cambricon-LLM (Yu et al. 2024):
  1. Data placement: assign weights/KV/activations to optimal tier
  2. Per-operator effective bandwidth: determined by bottleneck tier
  3. Decode bottleneck: weight loading from appropriate tier
  4. Prefill: compute-bound (batched processing)

Memory access pattern:
  - Prefill: weights loaded once per layer × batch → amortized
  - Decode:  weights loaded every token → bottleneck!
    * Weights in SRAM  → SRAM BW (fast)
    * Weights in DRAM  → DRAM BW (medium)
    * Weights in Flash → Flash BW (slow)

Reference: arxiv.org/abs/2312.03134 (LLMCompass), Yu et al. 2024 (Cambricon-LLM)
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict


@dataclass
class TierPlacement:
    tier: str           # 'SRAM' | 'DRAM' | 'Flash'
    data_type: str      # 'weight' | 'kv_cache' | 'activation'
    size_gb: float
    bandwidth: float    # Byte/s for this tier
    energy_pj_per_byte: float
    latency_ns: float


class HeteroAnalyzer:
    """
    Analyzes LLM inference on a heterogeneous multi-tier memory architecture.

    For each operator, determines:
    1. Which memory tier holds the data
    2. Effective bandwidth (limited by bottleneck tier)
    3. Performance and energy for that tier
    4. Comparison vs. ideal (single flat memory)
    """

    def __init__(self, model_config: dict, hardware: dict):
        self.cfg = model_config
        self.hw  = hardware
        self._tiers = hardware.get('memory_tiers', {})

        if not self._tiers:
            raise ValueError(f"Hardware '{hardware['name']}' is not a heterogeneous architecture. "
                             "Set is_heterogeneous=True and provide memory_tiers.")

    def _tier_bw(self, tier: str) -> float:
        return self._tiers[tier]['bandwidth']

    def _tier_cap(self, tier: str) -> float:
        return self._tiers[tier]['capacity']

    def _tier_energy(self, tier: str) -> float:
        return self._tiers[tier].get('energy_per_byte_pj', 3.0)

    def compute_data_placement(self) -> dict:
        """
        Determine optimal data placement across memory tiers.

        Strategy (greedy, hottest data first):
          1. SRAM: active layer weights + activations (hot, frequent)
          2. DRAM: KV cache + remaining weights (warm)
          3. Flash: overflow weights (cold, infrequent)
        """
        cfg = self.cfg
        h   = cfg['hidden_size']
        n   = cfg['num_heads']
        kv_n = cfg.get('num_key_value_heads', n)
        d   = h // n
        L   = cfg['num_layers']
        s   = cfg['seq_len']
        b   = cfg['batch_size']
        mgl = cfg.get('max_gen_len', 2048)
        q   = cfg['quant_config']
        w_attn = q['weight_attn']
        w_ffn  = q['weight_ffn']
        kv_bit = q['kv_cache']
        a_bit  = q['activation']
        inter  = cfg.get('intermediate_size', 4 * h)
        vocab  = cfg.get('vocab_size', 32000)

        n_exp_tok   = cfg.get('num_experts_per_tok', 1)
        n_exp_total = cfg.get('num_local_experts', 1)

        # Per-layer weight size
        w_qkv_per_layer = (h*h + 2*h*int(h*kv_n/n)) * w_attn / 8
        w_o_per_layer   = h * h * w_attn / 8
        if cfg.get('num_experts_per_tok'):
            w_ffn_per_layer = inter * h * 3 * w_ffn / 8 * n_exp_total
        elif cfg.get('use_gate_ffn'):
            w_ffn_per_layer = inter * h * 3 * w_ffn / 8
        else:
            w_ffn_per_layer = inter * h * 2 * w_ffn / 8
        w_norm_per_layer = h * 2 * a_bit / 8

        w_per_layer_gb  = (w_qkv_per_layer + w_o_per_layer + w_ffn_per_layer + w_norm_per_layer) / 1e9
        w_total_gb      = w_per_layer_gb * L + vocab * h * w_attn / 8 * 2 / 1e9

        kv_bytes_per_token = L * kv_n * d * 2 * (kv_bit / 8)
        kv_max_gb = kv_bytes_per_token * (s + mgl) * b / 1e9

        act_peak_gb = max(b * n * s * s, b * s * inter) * (a_bit / 8) / 1e9

        # Available capacities
        sram_cap  = self._tier_cap('SRAM')  / 1e9  # GB
        dram_cap  = self._tier_cap('DRAM')  / 1e9
        flash_cap = self._tier_cap('Flash') / 1e9

        # Greedy placement
        sram_used = 0.0
        dram_used = 0.0
        flash_used = 0.0
        placement = []

        # 1. Activations → SRAM (hot, must be fast)
        act_sram = min(act_peak_gb, sram_cap - sram_used)
        sram_used += act_sram
        placement.append(TierPlacement('SRAM', 'activation', act_sram,
                                        self._tier_bw('SRAM'), self._tier_energy('SRAM'), 1))

        # 2. How many layers' weights fit in SRAM?
        layers_sram = int((sram_cap - sram_used) / w_per_layer_gb) if w_per_layer_gb > 0 else 0
        layers_sram = min(layers_sram, L)
        w_sram_gb   = layers_sram * w_per_layer_gb
        sram_used  += w_sram_gb
        if w_sram_gb > 0:
            placement.append(TierPlacement('SRAM', 'weight', w_sram_gb,
                                            self._tier_bw('SRAM'), self._tier_energy('SRAM'), 1))

        # 3. KV cache → DRAM (warm, moderate access)
        kv_dram = min(kv_max_gb, dram_cap - dram_used)
        dram_used += kv_dram
        placement.append(TierPlacement('DRAM', 'kv_cache', kv_dram,
                                        self._tier_bw('DRAM'), self._tier_energy('DRAM'), 100))
        kv_overflow = max(0.0, kv_max_gb - kv_dram)

        # 4. Remaining weights → DRAM
        w_remaining = w_total_gb - w_sram_gb
        w_dram_gb   = min(w_remaining, dram_cap - dram_used)
        dram_used  += w_dram_gb
        if w_dram_gb > 0:
            placement.append(TierPlacement('DRAM', 'weight', w_dram_gb,
                                            self._tier_bw('DRAM'), self._tier_energy('DRAM'), 100))

        # 5. Overflow → Flash
        w_flash_gb  = max(0.0, w_remaining - w_dram_gb)
        kv_flash_gb = kv_overflow
        if w_flash_gb > 0:
            placement.append(TierPlacement('Flash', 'weight', w_flash_gb,
                                            self._tier_bw('Flash'), self._tier_energy('Flash'), 100000))
        if kv_flash_gb > 0:
            placement.append(TierPlacement('Flash', 'kv_cache', kv_flash_gb,
                                            self._tier_bw('Flash'), self._tier_energy('Flash'), 100000))

        # Determine primary bottleneck for decode (weight loading tier)
        if w_sram_gb >= w_total_gb:
            decode_bottleneck = 'SRAM'
        elif w_sram_gb + w_dram_gb >= w_total_gb:
            decode_bottleneck = 'DRAM'
        else:
            decode_bottleneck = 'Flash'

        layers_in_dram  = min(L - layers_sram, int(w_dram_gb / w_per_layer_gb)) if w_per_layer_gb > 0 else 0
        layers_in_flash = max(0, L - layers_sram - layers_in_dram)

        return {
            'placement': [asdict(p) for p in placement],
            'summary': {
                'total_weights_gb':      w_total_gb,
                'weights_in_sram_gb':    w_sram_gb,
                'weights_in_dram_gb':    w_dram_gb,
                'weights_in_flash_gb':   w_flash_gb,
                'kv_cache_gb':           kv_max_gb,
                'activations_gb':        act_peak_gb,
                'layers_in_sram':        layers_sram,
                'layers_in_dram':        layers_in_dram,
                'layers_in_flash':       layers_in_flash,
                'decode_bottleneck':     decode_bottleneck,
                'decode_bw':             self._tier_bw(decode_bottleneck),
                'sram_utilization_pct':  sram_used / sram_cap * 100 if sram_cap > 0 else 0,
                'dram_utilization_pct':  dram_used / dram_cap * 100 if dram_cap > 0 else 0,
                'flash_spills':          w_flash_gb > 0 or kv_flash_gb > 0,
            }
        }

    def analyze_per_operator(self, ops_results: List[dict], placement_summary: dict) -> List[dict]:
        """
        For each operator, compute effective performance on the heterogeneous system.
        Bottleneck is determined by which tier holds the relevant data.
        """
        peak_perf = self.hw['peak_performance']
        ps        = placement_summary

        decode_bw    = ps['decode_bw']
        sram_bw      = self._tier_bw('SRAM')
        dram_bw      = self._tier_bw('DRAM')
        flash_bw     = self._tier_bw('Flash')
        bottleneck   = ps['decode_bottleneck']

        # Effective bandwidth per phase for weight-loading ops
        phase_weight_bw = {
            'Prefill': dram_bw,   # batch load → amortized, use DRAM BW
            'Decode':  decode_bw, # per-token weight load → bottleneck
            'Output':  dram_bw,
        }

        hetero_ops = []
        for r in ops_results:
            density     = r['density']
            phase       = r['phase']
            flops       = r['flops_total']
            total_bytes = r['total_bytes_total']
            weight_bytes = r.get('weight_bytes', 0) * r.get('num_layers', 1)

            # Determine effective bandwidth
            is_weight_dominated = weight_bytes > total_bytes * 0.4 if total_bytes > 0 else False
            if is_weight_dominated and phase == 'Decode':
                eff_bw = phase_weight_bw['Decode']
                data_tier = bottleneck
            elif r['category'] == 'Attention' and 'kv_cache' in r.get('note', '').lower():
                eff_bw = dram_bw  # KV cache in DRAM
                data_tier = 'DRAM'
            else:
                eff_bw = sram_bw  # activations/hot data in SRAM
                data_tier = 'SRAM'

            # Roofline with effective BW
            ridge_eff   = peak_perf / eff_bw
            attain_perf = min(density * eff_bw, peak_perf) if density > 0 else 0.0
            time_s      = flops / attain_perf if attain_perf > 0 else 0.0

            # Compare with ideal flat-memory (original bandwidth)
            orig_bw   = self.hw['memory_bandwidth']
            ideal_perf = min(density * orig_bw, peak_perf) if density > 0 else 0.0
            slowdown   = ideal_perf / attain_perf if attain_perf > 0 else 1.0

            # Energy per byte for this tier
            tier_energy_pj = self._tiers[data_tier].get('energy_per_byte_pj', 3.0)
            energy_j = total_bytes * tier_energy_pj / 1e12 if total_bytes > 0 else 0.0

            hetero_ops.append({
                'operation':         r['operation'],
                'phase':             r['phase'],
                'category':          r['category'],
                'density':           density,
                'data_tier':         data_tier,
                'effective_bw_gbs':  eff_bw / 1e9,
                'attain_perf_gflops': attain_perf / 1e9,
                'ideal_perf_gflops': ideal_perf / 1e9,
                'slowdown_vs_ideal': round(slowdown, 2),
                'time_s':            time_s,
                'energy_j':          energy_j,
                'ridge_point_eff':   ridge_eff,
                'bound':             'Memory' if density < ridge_eff else 'Compute',
                'flops_total':       flops,
                'total_bytes_total': total_bytes,
            })

        return hetero_ops

    def compute_decode_throughput(self, placement_summary: dict) -> dict:
        """
        Compute decode throughput (tokens/s) limited by weight loading bandwidth.
        This is the key metric for interactive inference.

        tokens/s = weight_loading_bw / bytes_per_token_decode
        """
        cfg    = self.cfg
        h      = cfg['hidden_size']
        n      = cfg['num_heads']
        kv_n   = cfg.get('num_key_value_heads', n)
        L      = cfg['num_layers']
        q      = cfg['quant_config']
        w_attn = q['weight_attn']
        w_ffn  = q['weight_ffn']
        inter  = cfg.get('intermediate_size', 4 * h)
        n_exp_tok   = cfg.get('num_experts_per_tok', 1)
        n_exp_total = cfg.get('num_local_experts', 1)
        b      = cfg['batch_size']

        # Weight bytes to load per token per layer
        kv_h   = int(h * kv_n / n)
        w_qkv  = (h*h + 2*h*kv_h) * w_attn / 8
        w_o    = h * h * w_attn / 8
        if cfg.get('num_experts_per_tok'):
            w_ffn_l = inter * h * 3 * w_ffn / 8 * n_exp_tok  # only active experts
        elif cfg.get('use_gate_ffn'):
            w_ffn_l = inter * h * 3 * w_ffn / 8
        else:
            w_ffn_l = inter * h * 2 * w_ffn / 8

        bytes_per_token_per_layer = (w_qkv + w_o + w_ffn_l) * b
        bytes_per_token_total     = bytes_per_token_per_layer * L

        decode_bw = placement_summary['decode_bw']  # Byte/s
        bottleneck = placement_summary['decode_bottleneck']

        tokens_per_sec = decode_bw / bytes_per_token_total if bytes_per_token_total > 0 else 0.0

        # Compare with each tier
        results = {}
        for tier_name, tier in self._tiers.items():
            tps = tier['bandwidth'] / bytes_per_token_total if bytes_per_token_total > 0 else 0.0
            results[f'tokens_per_sec_{tier_name}'] = tps

        # Latency to first token (prefill) — compute bound at peak
        prefill_flops = self._estimate_prefill_flops()
        ttft_s = prefill_flops / self.hw['peak_performance'] if self.hw['peak_performance'] > 0 else 0.0

        return {
            'decode_tokens_per_sec':      tokens_per_sec,
            'decode_bottleneck_tier':     bottleneck,
            'decode_bottleneck_bw_gbs':   decode_bw / 1e9,
            'bytes_per_token':            bytes_per_token_total,
            'ttft_ms':                    ttft_s * 1000,
            **results,
        }

    def _estimate_prefill_flops(self) -> float:
        cfg  = self.cfg
        h, n, L, b = cfg['hidden_size'], cfg['num_heads'], cfg['num_layers'], cfg['batch_size']
        s    = cfg['seq_len']
        inter = cfg.get('intermediate_size', 4 * h)
        kv_n = cfg.get('num_key_value_heads', n)
        kv_h = int(h * kv_n / n)
        n_ep = cfg.get('num_experts_per_tok', 1)
        use_gate = cfg.get('use_gate_ffn', False)
        ffn_mul = 3 if use_gate else 2
        per_layer = (2*b*s*h*h + 2*b*s*h*kv_h*2 + 4*b*n*s*s*h//n +
                     2*b*s*h*inter*ffn_mul*n_ep)
        return per_layer * L

    def run_full_analysis(self) -> dict:
        """Run complete heterogeneous architecture analysis."""
        from analyzer import LLMAnalyzer
        analyzer = LLMAnalyzer(self.cfg)
        ops_results = analyzer.analyze()

        placement = self.compute_data_placement()
        hetero_ops = self.analyze_per_operator(ops_results, placement['summary'])
        throughput = self.compute_decode_throughput(placement['summary'])

        # Aggregate stats
        total_time = sum(op['time_s'] for op in hetero_ops if op['time_s'] > 0)
        total_energy = sum(op['energy_j'] for op in hetero_ops)
        avg_slowdown = (sum(op['slowdown_vs_ideal'] for op in hetero_ops)
                        / max(1, len(hetero_ops)))

        prefill_ops = [op for op in hetero_ops if op['phase'] == 'Prefill']
        decode_ops  = [op for op in hetero_ops if op['phase'] == 'Decode']

        tier_dist = {}
        for op in hetero_ops:
            t = op['data_tier']
            tier_dist[t] = tier_dist.get(t, 0) + 1

        return {
            'placement':    placement,
            'hetero_ops':   hetero_ops,
            'throughput':   throughput,
            'summary': {
                'total_time_s':         total_time,
                'total_energy_j':       total_energy,
                'avg_slowdown':         round(avg_slowdown, 2),
                'tier_distribution':    tier_dist,
                'decode_tps':           throughput['decode_tokens_per_sec'],
                'bottleneck_tier':      placement['summary']['decode_bottleneck'],
                'flash_spills':         placement['summary']['flash_spills'],
                'prefill_ops_count':    len(prefill_ops),
                'decode_ops_count':     len(decode_ops),
            },
            'ops_results':  ops_results,
        }
