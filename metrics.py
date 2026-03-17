"""
LLM-Para Extended Metrics Module
=================================
Implements multi-dimensional performance analysis for LLM inference:

1. Energy Roofline Model
   - Based on: "Power and Energy-efficiency Roofline Model for GPUs"
     (Ghane et al., ISPASS 2018)
   - Models power as a weighted mix of compute + memory utilization fractions
   - Energy efficiency (GFLOPS/W) as function of arithmetic intensity

2. TCO (Total Cost of Ownership)
   - Hardware amortization over lifetime
   - Electricity cost over lifetime
   - TCO per EFLOP (unit: $/EFLOP)

3. CO2e Carbon Emissions
   - Operational carbon: electricity × regional carbon intensity
   - Embodied carbon: chip manufacture (kg CO2e), amortized per compute

4. Memory Capacity Analysis
   - Model weight footprint (all layers × quant bits)
   - KV cache footprint (seq_len + gen_len)
   - Activation footprint (peak per layer)
   - Fit-in-memory check per tier

Reference architectures:
- LLMCompass (Zhang et al., ISCA 2024): arxiv.org/abs/2312.03134
- Hardware Co-Design Scaling Laws (Sun et al., 2026)
- Cambricon-LLM (Yu et al., 2024): chiplet-based hybrid
"""

import math
from typing import Dict, List, Optional, Tuple


# ─── Carbon intensity database (gCO2e/kWh) ──────────────────────────────────
CARBON_INTENSITY_GRID = {
    "US_Average":       386,
    "US_California":    200,
    "US_Texas":         450,
    "EU_Average":       255,
    "France":           58,     # heavy nuclear
    "Germany":          350,
    "China":            550,
    "Iceland":          28,     # geothermal
    "Global_Average":   420,
}

# ─── Energy Roofline Model ───────────────────────────────────────────────────

def energy_roofline_point(
    flops: float,
    total_bytes: float,
    density: float,
    hardware: dict,
) -> dict:
    """
    Compute energy-efficiency attainable performance at a given arithmetic intensity.

    Energy Roofline (Ghane et al. 2018):
        - Compute-bound regime:  E_eff = peak_FLOPS/W
        - Memory-bound regime:   E_eff = I × BW / P_active
        - P_active = TDP × (α × util_compute + (1-α) × util_memory)

    where α = compute_power_frac, I = arithmetic intensity (FLOP/Byte)
    """
    peak_perf = hardware['peak_performance']    # FLOP/s
    bw        = hardware['memory_bandwidth']     # Byte/s
    tdp       = hardware.get('tdp_w', 300.0)
    alpha     = hardware.get('compute_power_frac', 0.55)   # fraction of TDP for compute
    ridge     = peak_perf / bw

    if total_bytes <= 0:
        return {'energy_efficiency_gflops_per_w': 0.0, 'bound': 'N/A',
                'power_w': 0.0, 'energy_j': 0.0, 'time_s': 0.0}

    t_compute = flops / peak_perf
    t_memory  = total_bytes / bw
    t_total   = max(t_compute, t_memory)

    # Utilization fractions
    if density < ridge:
        u_compute = density / ridge   # partial compute utilization
        u_memory  = 1.0
    else:
        u_compute = 1.0
        u_memory  = ridge / density

    # Active power
    p_active = tdp * (alpha * u_compute + (1.0 - alpha) * u_memory)
    p_active = max(p_active, tdp * 0.05)  # idle power floor (5%)

    energy_j      = t_total * p_active
    energy_eff    = flops / energy_j if energy_j > 0 else 0.0  # FLOP/J = W
    energy_eff_gw = energy_eff / 1e9   # GFLOP/J (= GFLOPS/W at roofline)

    # Peak achievable energy efficiency (at ridge point)
    peak_energy_eff = peak_perf / tdp / 1e9   # GFLOPS/W

    return {
        'energy_efficiency_gflops_per_w': energy_eff_gw,
        'peak_energy_efficiency_gflops_per_w': peak_energy_eff,
        'normalized_energy_eff': energy_eff_gw / peak_energy_eff if peak_energy_eff > 0 else 0.0,
        'power_w': p_active,
        'energy_j': energy_j,
        'time_s': t_total,
        'bound': 'Memory' if density < ridge else 'Compute',
        'util_compute': u_compute,
        'util_memory': u_memory,
    }


def build_energy_roofline_curve(hardware: dict, n_points: int = 300) -> dict:
    """
    Build the Energy Roofline curve (GFLOPS/W vs Arithmetic Intensity).
    Returns x/y arrays for frontend plotting.
    """
    peak_perf = hardware['peak_performance']
    bw        = hardware['memory_bandwidth']
    tdp       = hardware.get('tdp_w', 300.0)
    alpha     = hardware.get('compute_power_frac', 0.55)
    ridge     = peak_perf / bw

    # x-axis: log-spaced from 0.01 to 10000 FLOP/Byte
    x_vals = [0.01 * (10000 / 0.01) ** (i / (n_points - 1)) for i in range(n_points)]
    y_vals = []

    peak_eff = peak_perf / tdp / 1e9  # GFLOPS/W ceiling

    for I in x_vals:
        if I < ridge:
            # Memory-bound: full memory util, partial compute
            u_c, u_m = I / ridge, 1.0
        else:
            # Compute-bound: full compute util, partial memory
            u_c, u_m = 1.0, ridge / I

        p = tdp * (alpha * u_c + (1.0 - alpha) * u_m)
        attain_perf = min(I * bw, peak_perf)
        eff = attain_perf / p / 1e9 if p > 0 else 0.0
        y_vals.append(eff)

    return {
        'x': x_vals,
        'y': y_vals,
        'ridge_point': ridge,
        'peak_efficiency_gflops_per_w': peak_eff,
        'tdp_w': tdp,
    }


# ─── TCO Analysis ────────────────────────────────────────────────────────────

def compute_tco(
    hardware: dict,
    lifetime_years: float = 3.0,
    utilization: float = 0.50,
    electricity_price_per_kwh: float = 0.10,
    pue: float = 1.3,   # Power Usage Effectiveness (datacenter overhead)
) -> dict:
    """
    TCO = Hardware amortization + Electricity cost (over lifetime).

    Reference: LLMCompass Section V, Hardware Co-Design Scaling Laws (Sun 2026)

    Args:
        pue: Power Usage Effectiveness (1.0 = no overhead, 1.5 = typical DC)
    """
    cost_usd  = hardware.get('cost_usd', 5000.0)
    tdp_w     = hardware.get('tdp_w', 300.0)
    peak_perf = hardware['peak_performance']  # FLOP/s

    lifetime_hours   = lifetime_years * 365.25 * 24.0
    lifetime_seconds = lifetime_hours * 3600.0

    # Total compute capacity over lifetime
    total_flops = peak_perf * lifetime_seconds * utilization  # FLOPs

    # ── Hardware amortization ────────────────────────────────────────────────
    hw_depreciation = cost_usd  # full cost amortized
    # Per EFLOP (1e18 FLOP)
    hw_cost_per_eflop = hw_depreciation / (total_flops / 1e18) if total_flops > 0 else float('inf')

    # ── Electricity cost ─────────────────────────────────────────────────────
    energy_kwh_active  = tdp_w / 1000.0 * lifetime_hours * utilization
    energy_kwh_total   = energy_kwh_active * pue     # include cooling/infra
    electricity_cost   = energy_kwh_total * electricity_price_per_kwh
    energy_cost_per_eflop = electricity_cost / (total_flops / 1e18) if total_flops > 0 else float('inf')

    total_tco         = cost_usd + electricity_cost
    tco_per_eflop     = hw_cost_per_eflop + energy_cost_per_eflop

    # ── Performance-per-dollar ───────────────────────────────────────────────
    perf_per_dollar   = peak_perf / cost_usd / 1e9  # GFLOPS/$

    # ── Energy cost fraction ──────────────────────────────────────────────────
    energy_fraction   = electricity_cost / total_tco if total_tco > 0 else 0.0

    return {
        'hardware_cost_usd':       cost_usd,
        'electricity_cost_usd':    electricity_cost,
        'total_tco_usd':           total_tco,
        'hw_cost_per_eflop_usd':   hw_cost_per_eflop,
        'energy_cost_per_eflop_usd': energy_cost_per_eflop,
        'tco_per_eflop_usd':       tco_per_eflop,
        'energy_fraction':         energy_fraction,
        'perf_per_dollar_gflops':  perf_per_dollar,
        'lifetime_years':          lifetime_years,
        'utilization':             utilization,
        'electricity_kwh':         energy_kwh_total,
        'pue':                     pue,
        'total_flops':             total_flops,
    }


# ─── CO2e Carbon Emissions ───────────────────────────────────────────────────

def compute_co2e(
    hardware: dict,
    flops_analyzed: float,
    region: str = 'Global_Average',
    lifetime_years: float = 3.0,
    utilization: float = 0.50,
) -> dict:
    """
    Carbon footprint analysis: operational + embodied (Scope 1+2+3).

    Based on methodology in:
    - Hardware Co-Design Scaling Laws (Sun et al. 2026)
    - MLPerf Power measurement framework
    """
    tdp_w         = hardware.get('tdp_w', 300.0)
    peak_perf     = hardware['peak_performance']
    carbon_mfg    = hardware.get('carbon_mfg_kgco2e', 100.0)   # kg CO2e
    ci_grid       = CARBON_INTENSITY_GRID.get(region, 420)      # gCO2e/kWh

    # ── Operational CO2e ─────────────────────────────────────────────────────
    # Time to execute flops_analyzed at full throughput
    t_analyzed_s    = flops_analyzed / peak_perf
    energy_kwh_op   = tdp_w * t_analyzed_s / 3600.0 / 1000.0
    op_co2e_g       = energy_kwh_op * ci_grid            # gCO2e
    op_co2e_kg      = op_co2e_g / 1000.0

    # ── Embodied CO2e (amortized) ─────────────────────────────────────────────
    # Total lifetime compute capacity
    lifetime_flops  = peak_perf * lifetime_years * 365.25 * 24.0 * 3600.0 * utilization
    # Fraction of lifetime used by this workload
    frac            = flops_analyzed / lifetime_flops if lifetime_flops > 0 else 0.0
    embodied_co2e_kg = carbon_mfg * frac

    total_co2e_kg   = op_co2e_kg + embodied_co2e_kg

    # ── Per-EFLOP metrics ─────────────────────────────────────────────────────
    op_co2e_per_eflop       = op_co2e_g / (flops_analyzed / 1e18) if flops_analyzed > 0 else 0.0
    embodied_co2e_per_eflop = (embodied_co2e_kg * 1000) / (flops_analyzed / 1e18) if flops_analyzed > 0 else 0.0
    total_co2e_per_eflop    = op_co2e_per_eflop + embodied_co2e_per_eflop

    return {
        'operational_co2e_kg':    op_co2e_kg,
        'embodied_co2e_kg':       embodied_co2e_kg,
        'total_co2e_kg':          total_co2e_kg,
        'op_co2e_per_eflop_g':    op_co2e_per_eflop,
        'emb_co2e_per_eflop_g':   embodied_co2e_per_eflop,
        'total_co2e_per_eflop_g': total_co2e_per_eflop,
        'carbon_intensity_gco2_kwh': ci_grid,
        'region':                 region,
        'energy_kwh':             energy_kwh_op,
        'op_fraction':            op_co2e_kg / total_co2e_kg if total_co2e_kg > 0 else 0.0,
    }


# ─── Memory Capacity Analysis ────────────────────────────────────────────────

def compute_memory_footprint(
    model_config: dict,
    results: List[dict],
) -> dict:
    """
    Compute peak memory footprint for LLM inference.
    Breaks down: weights + KV cache + activations.
    """
    cfg   = model_config
    h     = cfg['hidden_size']
    n     = cfg['num_heads']
    kv_n  = cfg.get('num_key_value_heads', n)
    d     = h // n
    L     = cfg['num_layers']
    s     = cfg['seq_len']
    b     = cfg['batch_size']
    mgl   = cfg.get('max_gen_len', 2048)
    q     = cfg['quant_config']

    w_attn  = q['weight_attn']
    w_ffn   = q['weight_ffn']
    a_bit   = q['activation']
    kv_bit  = q['kv_cache']

    intermediate = cfg.get('intermediate_size', 4 * h)
    vocab_size   = cfg.get('vocab_size', 32000)
    n_exp_tok    = cfg.get('num_experts_per_tok', 1)
    n_exp_total  = cfg.get('num_local_experts', 1)

    # ── Weight footprint ─────────────────────────────────────────────────────
    # Per layer
    w_q        = h * h * w_attn / 8
    w_kv       = h * int(h * kv_n / n) * 2 * w_attn / 8  # K+V
    w_o        = h * h * w_attn / 8

    if cfg.get('num_experts_per_tok'):
        # MoE: FFN shared by all experts (prefill loads all, decode loads active)
        w_ffn_layer = intermediate * h * 3 * w_ffn / 8 * n_exp_total  # up+gate+down × experts
    elif cfg.get('use_gate_ffn'):
        w_ffn_layer = intermediate * h * 3 * w_ffn / 8  # up + gate + down
    else:
        w_ffn_layer = intermediate * h * 2 * w_ffn / 8  # up + down

    w_norm     = h * 2 * a_bit / 8   # 2 norms per layer

    w_per_layer = w_q + w_kv + w_o + w_ffn_layer + w_norm
    w_total     = w_per_layer * L + vocab_size * h * w_attn / 8 * 2  # +embedding+lm_head

    # ── KV Cache footprint ───────────────────────────────────────────────────
    bytes_per_token = L * kv_n * d * 2 * (kv_bit / 8)  # K + V
    kv_prefill      = bytes_per_token * s * b
    kv_max          = bytes_per_token * (s + mgl) * b

    # ── Activation footprint (peak per layer) ────────────────────────────────
    # Largest intermediate: attention scores (b, n, s, s) or FFN intermediate
    act_attn   = b * n * s * s * (a_bit / 8)        # full attention matrix
    act_ffn    = b * s * intermediate * (a_bit / 8)  # FFN intermediate
    act_peak   = max(act_attn, act_ffn)

    # ── Total peak (prefill) ─────────────────────────────────────────────────
    total_prefill = w_total + kv_prefill + act_peak

    # ── Total peak (decode) ──────────────────────────────────────────────────
    # Only 1 token activation + full KV cache
    act_decode    = b * 1 * h * (a_bit / 8)
    total_decode  = w_total + kv_max + act_decode

    peak_memory   = max(total_prefill, total_decode)

    return {
        'weights_gb':           w_total / 1e9,
        'weights_per_layer_mb': w_per_layer / 1e6,
        'kv_prefill_gb':        kv_prefill / 1e9,
        'kv_max_gb':            kv_max / 1e9,
        'kv_bytes_per_token':   bytes_per_token,
        'activation_prefill_gb': act_peak / 1e9,
        'total_prefill_gb':     total_prefill / 1e9,
        'total_decode_gb':      total_decode / 1e9,
        'peak_memory_gb':       peak_memory / 1e9,
    }


def check_memory_fits(
    footprint: dict,
    hardware: dict,
) -> dict:
    """
    Check if model fits in each memory tier.
    Returns fit status and overflow amount.
    """
    w_gb    = footprint['weights_gb']
    kv_gb   = footprint['kv_max_gb']
    act_gb  = footprint['activation_prefill_gb']
    peak_gb = footprint['peak_memory_gb']

    result = {'layers_fit': {}, 'recommendation': ''}

    if hardware.get('is_heterogeneous'):
        tiers = hardware['memory_tiers']
        # Check layer-by-layer placement
        sram_cap_gb  = tiers['SRAM']['capacity'] / 1e9
        dram_cap_gb  = tiers['DRAM']['capacity'] / 1e9
        flash_cap_gb = tiers['Flash']['capacity'] / 1e9

        result['sram_capacity_gb']  = sram_cap_gb
        result['dram_capacity_gb']  = dram_cap_gb
        result['flash_capacity_gb'] = flash_cap_gb

        # Best-effort placement strategy:
        # SRAM: active layer weights + activations
        # DRAM: KV cache + overflow weights
        # Flash: rest of weights
        layer_w_mb = footprint['weights_per_layer_mb'] / 1000  # in GB
        result['layers_in_sram']   = max(0, int(sram_cap_gb / layer_w_mb))
        result['layers_in_dram']   = max(0, int(dram_cap_gb / layer_w_mb))
        remaining_w = max(0, w_gb - sram_cap_gb - dram_cap_gb)
        result['weights_in_flash_gb'] = remaining_w
        result['kv_fits_dram']     = (kv_gb + remaining_w) <= dram_cap_gb
        result['model_fits_flash'] = w_gb <= flash_cap_gb
        result['bottleneck_tier']  = (
            'SRAM' if result['layers_in_sram'] > 0 else
            'DRAM' if w_gb <= dram_cap_gb else 'Flash'
        )
        cap = hardware['memory_capacity'] / 1e9
    else:
        cap = hardware.get('memory_capacity', 0) / 1e9

    result['hardware_capacity_gb'] = cap
    result['model_fits']           = peak_gb <= cap
    result['weights_fit']          = w_gb <= cap
    result['with_kv_fits']         = (w_gb + kv_gb) <= cap
    result['overflow_gb']          = max(0.0, peak_gb - cap)
    result['utilization_pct']      = min(100.0, peak_gb / cap * 100) if cap > 0 else 100.0

    if result['model_fits']:
        result['recommendation'] = '✅ Model fits in memory'
    elif result['weights_fit']:
        result['recommendation'] = '⚠️ Weights fit, but KV cache may overflow'
    else:
        result['recommendation'] = f'❌ Model too large by {result["overflow_gb"]:.1f} GB — use quantization or offloading'

    return result


# ─── Full metrics pipeline ────────────────────────────────────────────────────

def run_full_metrics(
    results: List[dict],
    summary: dict,
    hardware: dict,
    model_config: dict,
    tco_params: Optional[dict] = None,
    co2_region: str = 'Global_Average',
) -> dict:
    """
    Run all extended metrics analyses and return combined report.
    """
    tco_p = tco_params or {}

    # Per-operator energy metrics
    energy_points = []
    for r in results:
        if r['density'] > 0 and r['flops_total'] > 0:
            ep = energy_roofline_point(
                r['flops_total'], r['total_bytes_total'], r['density'], hardware)
            ep['operation'] = r['operation']
            ep['phase']     = r['phase']
            ep['category']  = r['category']
            ep['density']   = r['density']
            energy_points.append(ep)

    # Energy roofline curve
    energy_curve = build_energy_roofline_curve(hardware)

    # TCO
    tco = compute_tco(
        hardware,
        lifetime_years          = tco_p.get('lifetime_years', 3.0),
        utilization             = tco_p.get('utilization', 0.5),
        electricity_price_per_kwh = tco_p.get('electricity_price', 0.10),
        pue                     = tco_p.get('pue', 1.3),
    )

    # CO2e
    total_flops = summary.get('total_flops', 1e12)
    co2 = compute_co2e(hardware, total_flops, region=co2_region)

    # Memory footprint
    footprint = compute_memory_footprint(model_config, results)
    fit       = check_memory_fits(footprint, hardware)

    # Summary stats across operators
    total_energy_j = sum(ep['energy_j'] for ep in energy_points)
    avg_power_w    = sum(ep['power_w'] for ep in energy_points) / max(1, len(energy_points))
    avg_eff        = sum(ep['energy_efficiency_gflops_per_w'] for ep in energy_points) / max(1, len(energy_points))
    mem_bound_ops  = sum(1 for ep in energy_points if ep['bound'] == 'Memory')
    cmp_bound_ops  = sum(1 for ep in energy_points if ep['bound'] == 'Compute')
    peak_eff       = hardware['peak_performance'] / hardware.get('tdp_w', 300) / 1e9

    return {
        'energy_points':   energy_points,
        'energy_curve':    energy_curve,
        'tco':             tco,
        'co2e':            co2,
        'memory':          {**footprint, **fit},
        'summary': {
            'total_energy_j':            total_energy_j,
            'avg_power_w':               avg_power_w,
            'avg_energy_eff_gflops_w':   avg_eff,
            'peak_energy_eff_gflops_w':  peak_eff,
            'mem_bound_ops':             mem_bound_ops,
            'compute_bound_ops':         cmp_bound_ops,
            'memory_fits':               fit['model_fits'],
            'peak_memory_gb':            footprint['peak_memory_gb'],
            'tco_usd':                   tco['total_tco_usd'],
            'co2e_kg':                   co2['total_co2e_kg'],
        }
    }
