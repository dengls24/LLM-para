"""
LLM-Para Core Analysis Engine
Comprehensive FLOP, memory, and performance analysis for Transformer-based LLMs.
Supports: GQA, MoE, RoPE, SwiGLU, FlashAttention, RMSNorm, DeepSeek MLA, etc.
"""

import math
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any


@dataclass
class OpResult:
    phase: str
    operation: str
    category: str       # QKV / Attention / FFN / Norm / Embed / Other
    flops: float
    param_count: float
    input_bytes: float
    weight_bytes: float
    output_bytes: float
    total_bytes: float
    density: float      # Op/Byte (arithmetic intensity)
    note: str = ""

    def to_dict(self):
        d = asdict(self)
        d['density_str'] = f"{self.density:.2f}" if self.density > 0 else "-"
        return d


def _byte_size(shape: tuple, bitwidth: int) -> float:
    n = 1
    for s in shape:
        n *= s
    return n * bitwidth / 8


class LLMAnalyzer:
    """
    Comprehensive LLM Inference Analyzer.
    Computes per-operator FLOPs, parameter counts, and memory access patterns
    for both Prefill and Decode phases across all Transformer layers.
    """

    def __init__(self, config: dict):
        self.config = config
        self._ops: List[OpResult] = []

    def _add(self, phase: str, name: str, category: str, flops: float,
             param_count: float, input_shape: tuple, weight_shape: Optional[tuple],
             output_shape: tuple, act_bits: int, weight_bits: int, note: str = ""):
        B_in = _byte_size(input_shape, act_bits)
        B_w = _byte_size(weight_shape, weight_bits) if weight_shape else 0.0
        B_out = _byte_size(output_shape, act_bits)
        B_total = B_in + B_w + B_out
        density = flops / B_total if B_total > 0 else 0.0
        self._ops.append(OpResult(
            phase=phase, operation=name, category=category,
            flops=flops, param_count=param_count,
            input_bytes=B_in, weight_bytes=B_w,
            output_bytes=B_out, total_bytes=B_total,
            density=round(density, 4), note=note
        ))

    def analyze(self) -> List[dict]:
        cfg = self.config
        h = cfg['hidden_size']
        n = cfg['num_heads']
        kv_n = cfg.get('num_key_value_heads', n)
        d = h // n          # per-head dimension
        b = cfg['batch_size']
        s = cfg['seq_len']
        L = cfg['num_layers']
        intermediate_size = cfg.get('intermediate_size', 4 * h)
        vocab_size = cfg.get('vocab_size', 32000)

        q = cfg['quant_config']
        a_bit = q['activation']
        w_attn = q['weight_attn']
        w_ffn = q['weight_ffn']
        kv_bit = q['kv_cache']
        rope_bit = q.get('rope_bit', 32)
        norm_bit = q.get('norm_bit', a_bit)

        use_rope = cfg.get('rope_theta') is not None
        use_moe = bool(cfg.get('num_experts_per_tok'))
        use_gate_ffn = cfg.get('use_gate_ffn', False)
        use_flash_attn = cfg.get('use_flash_attn', False)
        use_mla = cfg.get('use_mla', False)      # DeepSeek Multi-head Latent Attention
        use_rmsnorm = cfg.get('use_rmsnorm', True)

        n_exp_tok = cfg.get('num_experts_per_tok', 1)
        n_exp_total = cfg.get('num_local_experts', 1)
        rope_theta = cfg.get('rope_theta', 10000.0)
        rope_scaling = cfg.get('rope_scaling_factor', 1.0)
        max_gen_len = cfg.get('max_gen_len', 2048)

        # MLA (DeepSeek-V2 style) parameters
        mla_kv_lora_rank = cfg.get('mla_kv_lora_rank', 512)
        mla_q_lora_rank = cfg.get('mla_q_lora_rank', 1536)
        mla_qk_nope_head_dim = cfg.get('mla_qk_nope_head_dim', 128)
        mla_qk_rope_head_dim = cfg.get('mla_qk_rope_head_dim', 64)
        mla_v_head_dim = cfg.get('mla_v_head_dim', 128)

        self._ops = []

        # ── Embedding (once, not per layer) ─────────────────────────────────
        embed_flops = b * s * 2  # trivial lookup, but loads weights
        self._add('Prefill', 'Token Embedding', 'Embed',
                  embed_flops, vocab_size * h,
                  (b, s), (vocab_size, h), (b, s, h),
                  a_bit, w_attn, note="Vocabulary embedding lookup")

        # ── Per-phase analysis ───────────────────────────────────────────────
        for phase in ['Prefill', 'Decode']:
            seq = s if phase == 'Prefill' else 1
            hist = 0 if phase == 'Prefill' else s
            ctx = seq + hist    # total context (for attention)

            # ── Pre-attention Norm ────────────────────────────────────────
            norm_flops = b * seq * h * (4 if use_rmsnorm else 6)
            self._add(phase, f"{'RMSNorm' if use_rmsnorm else 'LayerNorm'} (pre-Attn)",
                      'Norm', norm_flops, h,
                      (b, seq, h), (h,), (b, seq, h),
                      a_bit, norm_bit,
                      note="Pre-attention normalization")

            # ── Attention ─────────────────────────────────────────────────
            if use_mla:
                self._analyze_mla(phase, seq, hist, ctx, b, h, n, d,
                                  a_bit, w_attn, kv_bit, rope_bit,
                                  mla_kv_lora_rank, mla_q_lora_rank,
                                  mla_qk_nope_head_dim, mla_qk_rope_head_dim,
                                  mla_v_head_dim, use_rope, rope_theta, rope_scaling)
            else:
                self._analyze_standard_attention(
                    phase, seq, hist, ctx, b, h, n, kv_n, d,
                    a_bit, w_attn, kv_bit, rope_bit,
                    use_rope, rope_theta, rope_scaling,
                    use_flash_attn)

            # ── Post-attention Norm ───────────────────────────────────────
            self._add(phase, f"{'RMSNorm' if use_rmsnorm else 'LayerNorm'} (pre-FFN)",
                      'Norm', norm_flops, h,
                      (b, seq, h), (h,), (b, seq, h),
                      a_bit, norm_bit,
                      note="Pre-FFN normalization")

            # ── FFN ───────────────────────────────────────────────────────
            if use_moe:
                self._analyze_moe_ffn(phase, seq, b, h, intermediate_size,
                                      n_exp_tok, n_exp_total,
                                      a_bit, w_ffn)
            else:
                self._analyze_dense_ffn(phase, seq, b, h, intermediate_size,
                                        use_gate_ffn, a_bit, w_ffn)

        # ── LM Head (output projection) ──────────────────────────────────────
        lmhead_flops = 2 * b * 1 * h * vocab_size
        self._add('Output', 'LM Head', 'Embed',
                  lmhead_flops, vocab_size * h,
                  (b, 1, h), (h, vocab_size), (b, 1, vocab_size),
                  a_bit, w_attn,
                  note="Final projection to vocabulary logits")

        # ── Scale by number of layers ─────────────────────────────────────────
        results = []
        for op in self._ops:
            rd = op.to_dict()
            if op.phase not in ('Output',) and op.operation not in ('Token Embedding',):
                rd['flops_total'] = op.flops * L
                rd['param_count_total'] = op.param_count * L
                rd['total_bytes_total'] = op.total_bytes * L
            else:
                rd['flops_total'] = op.flops
                rd['param_count_total'] = op.param_count
                rd['total_bytes_total'] = op.total_bytes
            rd['num_layers'] = L
            results.append(rd)

        return results

    def _analyze_standard_attention(self, phase, seq, hist, ctx, b, h, n, kv_n, d,
                                     a_bit, w_attn, kv_bit, rope_bit,
                                     use_rope, rope_theta, rope_scaling, use_flash_attn):
        kv_h = int(h * kv_n / n)   # total KV hidden dim

        # Q projection
        self._add(phase, 'Q Projection', 'QKV',
                  2 * b * seq * h * h, h * h,
                  (b, seq, h), (h, h), (b, seq, h),
                  a_bit, w_attn)
        # K projection
        self._add(phase, 'K Projection', 'QKV',
                  2 * b * seq * h * kv_h, h * kv_h,
                  (b, seq, h), (h, kv_h), (b, seq, kv_h),
                  a_bit, w_attn, note=f"GQA: {kv_n}/{n} KV heads")
        # V projection
        self._add(phase, 'V Projection', 'QKV',
                  2 * b * seq * h * kv_h, h * kv_h,
                  (b, seq, h), (h, kv_h), (b, seq, kv_h),
                  a_bit, w_attn, note=f"GQA: {kv_n}/{n} KV heads")

        # RoPE
        if use_rope:
            self._add(phase, 'RoPE-Q', 'RoPE',
                      b * seq * n * (d // 2) * 4, 0,
                      (b, n, seq, d), (seq, d), (b, n, seq, d),
                      a_bit, rope_bit, note=f"θ={rope_theta}, scale={rope_scaling}")
            self._add(phase, 'RoPE-K', 'RoPE',
                      b * seq * kv_n * (d // 2) * 4, 0,
                      (b, kv_n, seq, d), (seq, d), (b, kv_n, seq, d),
                      a_bit, rope_bit, note=f"θ={rope_theta}, scale={rope_scaling}")

        # Attention computation
        if use_flash_attn:
            # FlashAttention: fused kernel, much lower memory traffic
            flash_flops = 4 * b * n * seq * ctx * d
            self._add(phase, 'FlashAttention (fused)', 'Attention',
                      flash_flops, 0,
                      (b, n, seq, d), (b, kv_n, ctx, d), (b, n, seq, d),
                      a_bit, kv_bit, note=f"FA2 fused, ctx={ctx}")
        else:
            # QK^T
            self._add(phase, 'Q×Kᵀ (Score)', 'Attention',
                      2 * b * n * seq * ctx * d, 0,
                      (b, n, seq, d), (b, kv_n, d, ctx), (b, n, seq, ctx),
                      a_bit, kv_bit, note=f"GQA ctx={ctx}")
            # Softmax
            self._add(phase, 'Softmax', 'Attention',
                      b * n * seq * ctx * 5, 0,
                      (b, n, seq, ctx), None, (b, n, seq, ctx),
                      a_bit, a_bit, note="exp+sum+div (online softmax)")
            # AV
            self._add(phase, 'Attn×V', 'Attention',
                      2 * b * n * seq * ctx * d, 0,
                      (b, n, seq, ctx), (b, kv_n, ctx, d), (b, n, seq, d),
                      a_bit, kv_bit, note=f"GQA ctx={ctx}")

        # Output projection
        self._add(phase, 'O Projection', 'QKV',
                  2 * b * seq * h * h, h * h,
                  (b, seq, h), (h, h), (b, seq, h),
                  a_bit, w_attn)

    def _analyze_mla(self, phase, seq, hist, ctx, b, h, n, d,
                     a_bit, w_attn, kv_bit, rope_bit,
                     kv_lora_rank, q_lora_rank, qk_nope_dim, qk_rope_dim, v_dim,
                     use_rope, rope_theta, rope_scaling):
        """DeepSeek-V2 Multi-head Latent Attention (MLA)"""
        # Q: two-stage projection (lora decomposition)
        self._add(phase, 'Q Down-Proj (MLA)', 'QKV',
                  2 * b * seq * h * q_lora_rank, h * q_lora_rank,
                  (b, seq, h), (h, q_lora_rank), (b, seq, q_lora_rank),
                  a_bit, w_attn, note="MLA Q low-rank projection")
        self._add(phase, 'Q Up-Proj (MLA)', 'QKV',
                  2 * b * seq * q_lora_rank * n * (qk_nope_dim + qk_rope_dim),
                  q_lora_rank * n * (qk_nope_dim + qk_rope_dim),
                  (b, seq, q_lora_rank), (q_lora_rank, n * (qk_nope_dim + qk_rope_dim)),
                  (b, seq, n * (qk_nope_dim + qk_rope_dim)),
                  a_bit, w_attn, note="MLA Q head expansion")
        # KV: compressed latent
        self._add(phase, 'KV Compress (MLA)', 'QKV',
                  2 * b * seq * h * kv_lora_rank, h * kv_lora_rank,
                  (b, seq, h), (h, kv_lora_rank), (b, seq, kv_lora_rank),
                  a_bit, kv_bit, note="MLA KV compression (cached)")
        self._add(phase, 'KV Expand (MLA)', 'QKV',
                  2 * b * seq * kv_lora_rank * n * (qk_nope_dim + v_dim),
                  kv_lora_rank * n * (qk_nope_dim + v_dim),
                  (b, seq, kv_lora_rank), (kv_lora_rank, n * (qk_nope_dim + v_dim)),
                  (b, seq, n * (qk_nope_dim + v_dim)),
                  a_bit, w_attn, note="MLA KV head expansion")
        # RoPE on rope part of Q/K
        if use_rope:
            self._add(phase, 'RoPE-Q (MLA)', 'RoPE',
                      b * seq * n * (qk_rope_dim // 2) * 4, 0,
                      (b, n, seq, qk_rope_dim), (seq, qk_rope_dim), (b, n, seq, qk_rope_dim),
                      a_bit, rope_bit, note=f"θ={rope_theta}")
        # Attention computation (simplified: use nope+rope dim)
        eff_d = qk_nope_dim + qk_rope_dim
        self._add(phase, 'Q×Kᵀ (MLA)', 'Attention',
                  2 * b * n * seq * ctx * eff_d, 0,
                  (b, n, seq, eff_d), (b, n, eff_d, ctx), (b, n, seq, ctx),
                  a_bit, kv_bit, note=f"MLA ctx={ctx}")
        self._add(phase, 'Softmax (MLA)', 'Attention',
                  b * n * seq * ctx * 5, 0,
                  (b, n, seq, ctx), None, (b, n, seq, ctx),
                  a_bit, a_bit, note="online softmax")
        self._add(phase, 'Attn×V (MLA)', 'Attention',
                  2 * b * n * seq * ctx * v_dim, 0,
                  (b, n, seq, ctx), (b, n, ctx, v_dim), (b, n, seq, v_dim),
                  a_bit, kv_bit, note=f"MLA ctx={ctx}")
        # Output
        self._add(phase, 'O Projection (MLA)', 'QKV',
                  2 * b * seq * n * v_dim * h, n * v_dim * h,
                  (b, seq, n * v_dim), (n * v_dim, h), (b, seq, h),
                  a_bit, w_attn)

    def _analyze_moe_ffn(self, phase, seq, b, h, intermediate_size,
                          n_exp_tok, n_exp_total, a_bit, w_ffn):
        # Router gate
        self._add(phase, 'MoE Router', 'FFN',
                  2 * b * seq * h * n_exp_total, h * n_exp_total,
                  (b, seq, h), (h, n_exp_total), (b, seq, n_exp_total),
                  a_bit, w_ffn, note=f"Top-{n_exp_tok} of {n_exp_total} experts")
        # Expert params loaded (decode: only active, prefill: all)
        param_experts = n_exp_tok if phase == 'Decode' else n_exp_total
        # FFN Up+Gate (SwiGLU style for MoE)
        ffn1_flops = (2 * b * seq * h * intermediate_size * 2 * n_exp_tok
                      + b * seq * n_exp_tok * intermediate_size)
        self._add(phase, 'MoE FFN-Up+Gate (SwiGLU)', 'FFN',
                  ffn1_flops, h * intermediate_size * 2 * param_experts,
                  (b, seq * n_exp_tok, h),
                  (param_experts, h, intermediate_size * 2),
                  (b, seq * n_exp_tok, intermediate_size),
                  a_bit, w_ffn,
                  note=f"{n_exp_tok}/{n_exp_total} experts active, SwiGLU")
        # FFN Down
        ffn2_flops = 2 * b * seq * intermediate_size * h * n_exp_tok
        self._add(phase, 'MoE FFN-Down', 'FFN',
                  ffn2_flops, intermediate_size * h * param_experts,
                  (b, seq * n_exp_tok, intermediate_size),
                  (param_experts, intermediate_size, h),
                  (b, seq * n_exp_tok, h),
                  a_bit, w_ffn,
                  note=f"{n_exp_tok}/{n_exp_total} experts active")

    def _analyze_dense_ffn(self, phase, seq, b, h, intermediate_size,
                            use_gate_ffn, a_bit, w_ffn):
        if use_gate_ffn:
            # SwiGLU: W_up + W_gate fused
            ffn1_flops = (2 * b * seq * h * intermediate_size * 2
                          + b * seq * intermediate_size)
            self._add(phase, 'FFN-Up+Gate (SwiGLU)', 'FFN',
                      ffn1_flops, h * intermediate_size * 2,
                      (b, seq, h), (2, h, intermediate_size), (b, seq, intermediate_size),
                      a_bit, w_ffn, note="SwiGLU: SiLU(W₁x) ⊙ (W_gate·x)")
        else:
            self._add(phase, 'FFN-Up', 'FFN',
                      2 * b * seq * h * intermediate_size, h * intermediate_size,
                      (b, seq, h), (h, intermediate_size), (b, seq, intermediate_size),
                      a_bit, w_ffn)

        self._add(phase, 'FFN-Down', 'FFN',
                  2 * b * seq * intermediate_size * h, intermediate_size * h,
                  (b, seq, intermediate_size), (intermediate_size, h), (b, seq, h),
                  a_bit, w_ffn)

    def get_summary(self, results: List[dict]) -> dict:
        cfg = self.config
        h = cfg['hidden_size']
        n = cfg['num_heads']
        kv_n = cfg.get('num_key_value_heads', n)
        d = h // n
        L = cfg['num_layers']
        s = cfg['seq_len']
        b = cfg['batch_size']
        max_gen_len = cfg.get('max_gen_len', 2048)
        q = cfg['quant_config']
        kv_bit = q['kv_cache']
        w_ffn = q['weight_ffn']
        w_attn = q['weight_attn']

        prefill = [r for r in results if r['phase'] == 'Prefill']
        decode = [r for r in results if r['phase'] == 'Decode']

        prefill_flops = sum(r['flops_total'] for r in prefill)
        decode_flops = sum(r['flops_total'] for r in decode)
        total_flops = sum(r['flops_total'] for r in results)

        total_params = sum(r['param_count_total'] for r in prefill
                           if r['param_count'] > 0)
        # Use average weight bitwidth for model size
        avg_w_bit = (w_attn + w_ffn) / 2
        model_size_bytes = total_params * avg_w_bit / 8

        # KV cache
        kv_bytes_per_token = L * kv_n * d * 2 * (kv_bit / 8)
        kv_prefill_bytes = kv_bytes_per_token * s * b
        kv_decode_bytes = kv_bytes_per_token * max_gen_len * b
        kv_max_bytes = kv_bytes_per_token * (s + max_gen_len) * b

        # Phase breakdown by category
        categories = {}
        for r in results:
            cat = r['category']
            if cat not in categories:
                categories[cat] = {'prefill_flops': 0, 'decode_flops': 0}
            if r['phase'] == 'Prefill':
                categories[cat]['prefill_flops'] += r['flops_total']
            elif r['phase'] == 'Decode':
                categories[cat]['decode_flops'] += r['flops_total']

        return {
            'total_flops': total_flops,
            'prefill_flops': prefill_flops,
            'decode_flops': decode_flops,
            'total_params': total_params,
            'model_size_gb': model_size_bytes / (1024 ** 3),
            'kv_bytes_per_token': kv_bytes_per_token,
            'kv_prefill_mb': kv_prefill_bytes / (1024 ** 2),
            'kv_decode_mb': kv_decode_bytes / (1024 ** 2),
            'kv_max_mb': kv_max_bytes / (1024 ** 2),
            'kv_max_gb': kv_max_bytes / (1024 ** 3),
            'gqa_ratio': kv_n / n,
            'categories': categories,
            'num_layers': L,
            'seq_len': s,
            'batch_size': b,
            'max_gen_len': max_gen_len,
        }

    def get_roofline_data(self, results: List[dict], hardware: dict) -> dict:
        """
        Compute Roofline model data for given hardware.
        Returns structured data for frontend chart rendering.
        """
        peak_perf = hardware['peak_performance']    # FLOP/s
        mem_bw = hardware['memory_bandwidth']        # Byte/s
        ridge_point = peak_perf / mem_bw             # FLOP/Byte

        points = []
        for r in results:
            if r['density'] <= 0 or r['total_bytes'] <= 0:
                continue
            density = r['density']
            # Attainable performance on roofline
            attainable = min(density * mem_bw, peak_perf)
            points.append({
                'operation': r['operation'],
                'phase': r['phase'],
                'category': r['category'],
                'density': density,
                'attainable_perf': attainable,
                'flops_total': r['flops_total'],
                'total_bytes_total': r['total_bytes_total'],
                'is_memory_bound': density < ridge_point,
                'bound': 'Memory' if density < ridge_point else 'Compute',
                'efficiency': attainable / peak_perf,
                'note': r.get('note', ''),
            })

        return {
            'ridge_point': ridge_point,
            'peak_performance': peak_perf,
            'memory_bandwidth': mem_bw,
            'hardware_name': hardware['name'],
            'points': points,
        }
