"""
Speculative Decoding Analytical Model (WIP)

Models the performance of speculative decoding where a small "draft" model
proposes γ tokens that a large "target" model verifies in a single forward pass.

Key formulas:
  - Expected accepted tokens per step:  E[accepted] = (1 - α^(γ+1)) / (1 - α)
    where α = per-token acceptance rate, γ = speculation length
  - Latency per step = γ × T_draft_decode + T_target_verify(γ)
  - Speedup = E[accepted] × T_target_decode / latency_per_step
"""

from analyzer import LLMAnalyzer


# Draft model presets (small models suitable as draft)
DRAFT_MODEL_PRESETS = {
    'GPT-2 Small (117M)': {
        'hidden_size': 768, 'num_heads': 12, 'num_layers': 12,
        'intermediate_size': 3072, 'vocab_size': 50257,
        'use_gate_ffn': False, 'use_rmsnorm': False,
    },
    'GPT-2 XL (1.5B)': {
        'hidden_size': 1600, 'num_heads': 25, 'num_layers': 48,
        'intermediate_size': 6400, 'vocab_size': 50257,
        'use_gate_ffn': False, 'use_rmsnorm': False,
    },
    'Phi-3 Mini (3.8B)': {
        'hidden_size': 3072, 'num_heads': 32, 'num_layers': 32,
        'intermediate_size': 8192, 'vocab_size': 32064,
        'num_key_value_heads': 32,
        'use_gate_ffn': True, 'use_rmsnorm': True,
        'rope_theta': 10000.0, 'rope_scaling_factor': 1.0,
    },
    'LLaMA-2 7B': {
        'hidden_size': 4096, 'num_heads': 32, 'num_layers': 32,
        'intermediate_size': 11008, 'vocab_size': 32000,
        'num_key_value_heads': 32,
        'use_gate_ffn': True, 'use_rmsnorm': True,
        'rope_theta': 10000.0, 'rope_scaling_factor': 1.0,
    },
    'LLaMA-3 8B (Draft)': {
        'hidden_size': 4096, 'num_heads': 32, 'num_layers': 32,
        'intermediate_size': 14336, 'vocab_size': 128256,
        'num_key_value_heads': 8,
        'use_gate_ffn': True, 'use_rmsnorm': True,
        'rope_theta': 500000.0, 'rope_scaling_factor': 1.0,
    },
}


class SpeculativeAnalyzer:
    """Analytical model for speculative decoding performance."""

    def __init__(self, target_config, draft_config, hardware):
        self.target_cfg = target_config
        self.draft_cfg = draft_config
        self.hw = hardware

        # Ensure draft config has required fields from target
        for key in ['seq_len', 'batch_size', 'max_gen_len', 'quant_config']:
            if key not in self.draft_cfg:
                self.draft_cfg[key] = self.target_cfg.get(key)

        self.target_analyzer = LLMAnalyzer(self.target_cfg)
        self.draft_analyzer = LLMAnalyzer(self.draft_cfg)

        # Pre-compute base analysis
        self.target_results = self.target_analyzer.analyze()
        self.target_summary = self.target_analyzer.get_summary(self.target_results)
        self.draft_results = self.draft_analyzer.analyze()
        self.draft_summary = self.draft_analyzer.get_summary(self.draft_results)

    def _decode_latency_s(self, summary, label='target'):
        """Estimate single-token decode latency (memory-bound approximation)."""
        # Decode is memory-bound: latency ≈ model_bytes / bandwidth
        # hw['memory_bandwidth'] is already in B/s (e.g. 3.35e12)
        bw = self.hw['memory_bandwidth']
        model_bytes = summary['model_size_gb'] * 1e9
        return model_bytes / bw if bw > 0 else float('inf')

    def _verify_latency_s(self, gamma):
        """
        Target verification of γ tokens: one forward pass processing γ tokens.
        Approximation: like a mini-prefill of length γ.
        Cost ≈ γ × compute_per_token (compute-bound if γ large enough, else memory-bound).
        For first-order: verify ≈ max(γ × decode_flops / peak, model_bytes / bw)
        """
        bw = self.hw['memory_bandwidth']
        peak = self.hw['peak_performance']  # Already in FLOPS (e.g. 67e12)
        model_bytes = self.target_summary['model_size_gb'] * 1e9
        decode_flops = self.target_summary['decode_flops']

        # Memory-bound component: must load weights once
        t_mem = model_bytes / bw if bw > 0 else float('inf')
        # Compute-bound component: γ tokens of compute
        t_comp = (gamma * decode_flops) / peak if peak > 0 else float('inf')
        return max(t_mem, t_comp)

    def compute_expected_tokens(self, gamma, alpha):
        """
        Expected number of tokens generated per speculative step.
        E[tokens] = (1 - α^(γ+1)) / (1 - α)  for α < 1
        When α = 1, E[tokens] = γ + 1
        """
        if alpha >= 1.0:
            return gamma + 1
        if alpha <= 0.0:
            return 1
        return (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)

    def compute_step_latency(self, gamma):
        """
        Total wall-clock latency for one speculative step.
        = γ × T_draft_decode + T_target_verify(γ)
        """
        t_draft = self._decode_latency_s(self.draft_summary, 'draft')
        t_verify = self._verify_latency_s(gamma)
        return gamma * t_draft + t_verify

    def compute_speedup(self, gamma, alpha):
        """
        Speedup over vanilla autoregressive target decoding.
        speedup = E[tokens] / (step_latency / T_target_decode)
        """
        t_target = self._decode_latency_s(self.target_summary, 'target')
        if t_target <= 0:
            return 1.0
        expected = self.compute_expected_tokens(gamma, alpha)
        step_lat = self.compute_step_latency(gamma)
        # Vanilla: E[tokens] tokens would take E[tokens] × T_target
        vanilla_time = expected * t_target
        return vanilla_time / step_lat if step_lat > 0 else 1.0

    def compute_memory(self):
        """Memory overhead: target-only vs target+draft."""
        target_gb = self.target_summary['model_size_gb']
        draft_gb = self.draft_summary['model_size_gb']
        target_kv = self.target_summary['kv_max_gb']
        draft_kv = self.draft_summary['kv_max_gb']
        return {
            'target_model_gb': round(target_gb, 3),
            'draft_model_gb': round(draft_gb, 3),
            'target_kv_gb': round(target_kv, 3),
            'draft_kv_gb': round(draft_kv, 3),
            'total_vanilla_gb': round(target_gb + target_kv, 3),
            'total_speculative_gb': round(target_gb + draft_gb + target_kv + draft_kv, 3),
            'overhead_gb': round(draft_gb + draft_kv, 3),
            'overhead_pct': round((draft_gb + draft_kv) / (target_gb + target_kv) * 100, 1)
                if (target_gb + target_kv) > 0 else 0,
        }

    def compute_energy(self, gamma, alpha):
        """
        Energy per output token comparison.
        Vanilla: E_target_decode per token
        Speculative: (γ × E_draft_decode + E_target_verify) / E[tokens]
        Using power ≈ TDP (memory-bound ≈ full chip active)
        """
        tdp_w = self.hw.get('tdp_w', 200)
        t_target = self._decode_latency_s(self.target_summary)
        t_draft = self._decode_latency_s(self.draft_summary)
        t_verify = self._verify_latency_s(gamma)

        e_vanilla_per_token = tdp_w * t_target  # Joules
        step_energy = tdp_w * (gamma * t_draft + t_verify)
        expected = self.compute_expected_tokens(gamma, alpha)
        e_spec_per_token = step_energy / expected if expected > 0 else step_energy

        return {
            'vanilla_j_per_token': round(e_vanilla_per_token, 6),
            'speculative_j_per_token': round(e_spec_per_token, 6),
            'energy_ratio': round(e_spec_per_token / e_vanilla_per_token, 3)
                if e_vanilla_per_token > 0 else 1.0,
        }

    def sweep_gamma_alpha(self, gammas=None, alphas=None):
        """Sweep γ and α to produce speedup/energy curves."""
        if gammas is None:
            gammas = list(range(1, 17))
        if alphas is None:
            alphas = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

        curves = []
        for alpha in alphas:
            points = []
            for gamma in gammas:
                sp = self.compute_speedup(gamma, alpha)
                energy = self.compute_energy(gamma, alpha)
                expected = self.compute_expected_tokens(gamma, alpha)
                points.append({
                    'gamma': gamma,
                    'speedup': round(sp, 3),
                    'expected_tokens': round(expected, 2),
                    'energy_ratio': energy['energy_ratio'],
                })
            curves.append({
                'alpha': alpha,
                'points': points,
            })
        return {'gammas': gammas, 'alphas': alphas, 'curves': curves}

    def run_full_analysis(self, gamma=5, alpha=0.8):
        """Run complete speculative decoding analysis."""
        t_target = self._decode_latency_s(self.target_summary)
        t_draft = self._decode_latency_s(self.draft_summary)

        speedup = self.compute_speedup(gamma, alpha)
        expected = self.compute_expected_tokens(gamma, alpha)
        step_lat = self.compute_step_latency(gamma)
        memory = self.compute_memory()
        energy = self.compute_energy(gamma, alpha)
        sweep = self.sweep_gamma_alpha()

        tokens_per_sec_vanilla = 1.0 / t_target if t_target > 0 else 0
        tokens_per_sec_spec = expected / step_lat if step_lat > 0 else 0

        return {
            'params': {
                'gamma': gamma,
                'alpha': alpha,
            },
            'target_summary': {
                'model_size_gb': round(self.target_summary['model_size_gb'], 3),
                'total_params': self.target_summary['total_params'],
                'decode_flops': self.target_summary['decode_flops'],
                'decode_latency_ms': round(t_target * 1000, 3),
                'tokens_per_sec': round(tokens_per_sec_vanilla, 2),
            },
            'draft_summary': {
                'model_size_gb': round(self.draft_summary['model_size_gb'], 3),
                'total_params': self.draft_summary['total_params'],
                'decode_flops': self.draft_summary['decode_flops'],
                'decode_latency_ms': round(t_draft * 1000, 3),
            },
            'speedup': round(speedup, 3),
            'expected_tokens_per_step': round(expected, 2),
            'step_latency_ms': round(step_lat * 1000, 3),
            'tokens_per_sec_vanilla': round(tokens_per_sec_vanilla, 2),
            'tokens_per_sec_speculative': round(tokens_per_sec_spec, 2),
            'memory': memory,
            'energy': energy,
            'sweep': sweep,
        }
