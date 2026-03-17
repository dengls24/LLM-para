"""
LLM-Para Design Space Exploration (DSE) Engine
===============================================
Systematically sweeps hardware design parameters to find Pareto-optimal
configurations for LLM inference under multiple objectives:

  - Performance: throughput (TFLOP/s attainable)
  - Energy efficiency: GFLOPS/W
  - TCO: $/EFLOP
  - Carbon: gCO2e/EFLOP
  - Memory fit: can the model fit?

Inspired by:
  - LLMCompass (Zhang et al. ISCA 2024): arxiv.org/abs/2312.03134
  - Hardware Co-Design Scaling Laws (Sun et al. 2026)

Usage:
  from dse import DSEEngine
  engine = DSEEngine(model_config)
  results = engine.run(dse_params)
  pareto = engine.pareto_frontier(results, ['perf', 'energy_eff'])
"""

import math
import itertools
from typing import List, Dict, Tuple, Optional
from analyzer import LLMAnalyzer
from metrics import compute_tco, compute_co2e, compute_memory_footprint, check_memory_fits


# ─── DSE Parameter Presets ────────────────────────────────────────────────────

DSE_PRESETS = {
    "Quick Scan (3×3 grid)": {
        "peak_performance_tflops": [1, 10, 100],
        "memory_bandwidth_gbs":    [10, 100, 1000],
        "memory_capacity_gb":      [4, 16, 64],
        "tdp_w":                   [20, 100, 400],
        "cost_usd":                [500, 5000, 30000],
        "n_points": 27,
    },
    "Edge/Mobile DSE": {
        "peak_performance_tflops": [0.5, 1, 2, 5, 10, 20],
        "memory_bandwidth_gbs":    [5, 10, 20, 50, 100],
        "memory_capacity_gb":      [2, 4, 8, 16],
        "tdp_w":                   [2, 5, 10, 20, 40],
        "cost_usd":                [50, 100, 300, 500, 1000],
    },
    "Datacenter GPU DSE": {
        "peak_performance_tflops": [10, 20, 50, 100, 200, 500],
        "memory_bandwidth_gbs":    [100, 500, 1000, 2000, 5000],
        "memory_capacity_gb":      [16, 40, 80, 160, 320],
        "tdp_w":                   [100, 200, 400, 700],
        "cost_usd":                [3000, 10000, 20000, 50000],
    },
    "PIM / Near-Memory DSE": {
        "peak_performance_tflops": [0.1, 0.5, 1, 2, 5],
        "memory_bandwidth_gbs":    [50, 100, 200, 500, 1000, 2000],
        "memory_capacity_gb":      [8, 32, 128, 512],
        "tdp_w":                   [5, 10, 20, 50],
        "cost_usd":                [100, 300, 800, 2000],
    },
    "Custom": {
        "peak_performance_tflops": [1, 5, 10, 50, 100],
        "memory_bandwidth_gbs":    [10, 50, 100, 500, 1000],
        "memory_capacity_gb":      [4, 16, 64, 256],
        "tdp_w":                   [10, 50, 200, 500],
        "cost_usd":                [200, 1000, 5000, 20000],
    },
}


class DSEEngine:
    """
    Design Space Exploration Engine.

    Sweeps hardware parameter combinations and evaluates each against
    the specified LLM workload using multiple performance metrics.
    """

    def __init__(self, model_config: dict):
        self.model_cfg  = model_config
        # Pre-analyze the model once (results are hardware-independent)
        self._analyzer  = LLMAnalyzer(model_config)
        self._ops       = self._analyzer.analyze()
        self._summary   = self._analyzer.get_summary(self._ops)

    def _make_hw(self, peak_tflops, bw_gbs, cap_gb, tdp_w, cost_usd,
                  tech_nm=7, carbon_mfg=80, compute_frac=0.55, name=None) -> dict:
        name = name or f"{peak_tflops:.1f}T/{bw_gbs:.0f}G/{cap_gb:.0f}GB"
        return {
            'name':                name,
            'category':            'DSE',
            'peak_performance':    peak_tflops * 1e12,
            'peak_performance_fp16': peak_tflops * 2e12,
            'memory_bandwidth':    bw_gbs * 1e9,
            'memory_capacity':     cap_gb * 1e9,
            'tdp_w':               tdp_w,
            'cost_usd':            cost_usd,
            'tech_node_nm':        tech_nm,
            'carbon_mfg_kgco2e':   carbon_mfg,
            'compute_power_frac':  compute_frac,
        }

    def _evaluate_hw(self, hw: dict, tco_params: dict, co2_region: str) -> Optional[dict]:
        """Evaluate a single hardware point against the model workload."""
        try:
            peak  = hw['peak_performance']
            bw    = hw['memory_bandwidth']
            ridge = peak / bw

            # ── Performance (roofline) ───────────────────────────────────────
            total_attain_flops = 0.0
            total_ideal_flops  = 0.0
            mem_bound_cnt      = 0
            cmp_bound_cnt      = 0
            for r in self._ops:
                d = r['density']
                if d <= 0:
                    continue
                attain = min(d * bw, peak)
                total_attain_flops += attain * r.get('num_layers', 1)
                total_ideal_flops  += peak   * r.get('num_layers', 1)
                if d < ridge:
                    mem_bound_cnt += 1
                else:
                    cmp_bound_cnt += 1

            avg_attain_tflops = total_attain_flops / max(1, len(self._ops)) / 1e12

            # ── Energy efficiency ────────────────────────────────────────────
            tdp   = hw.get('tdp_w', 300.0)
            alpha = hw.get('compute_power_frac', 0.55)
            total_energy_j = 0.0
            for r in self._ops:
                d = r['density']
                if d <= 0:
                    continue
                u_c = min(1.0, d / ridge) if ridge > 0 else 0.0
                u_m = min(1.0, ridge / d) if d > 0 else 1.0
                p   = tdp * (alpha * u_c + (1 - alpha) * u_m)
                t   = max(r['flops_total'] / peak, r['total_bytes_total'] / bw) if peak > 0 and bw > 0 else 0.0
                total_energy_j += t * p * r.get('num_layers', 1)

            total_flops = self._summary.get('total_flops', 1e12)
            energy_eff  = total_flops / total_energy_j / 1e9 if total_energy_j > 0 else 0.0

            # ── TCO ──────────────────────────────────────────────────────────
            tco = compute_tco(
                hw,
                lifetime_years            = tco_params.get('lifetime_years', 3.0),
                utilization               = tco_params.get('utilization', 0.5),
                electricity_price_per_kwh = tco_params.get('electricity_price', 0.10),
                pue                       = tco_params.get('pue', 1.3),
            )

            # ── CO2e ─────────────────────────────────────────────────────────
            co2 = compute_co2e(hw, total_flops, region=co2_region)

            # ── Memory fit ───────────────────────────────────────────────────
            fp  = compute_memory_footprint(self.model_cfg, self._ops)
            fit = check_memory_fits(fp, hw)

            # ── Aggregated score ─────────────────────────────────────────────
            # Pareto-relevant scalars (all "higher is better" or "lower is better"):
            # Normalize for radar chart: higher = better for all
            perf_norm    = min(1.0, avg_attain_tflops / (peak / 1e12))
            eff_norm     = min(1.0, energy_eff / 100.0)   # relative to 100 GFLOPS/W
            cost_norm    = min(1.0, 10000.0 / max(1, hw['cost_usd']))
            co2_norm     = min(1.0, 1000.0 / max(1, co2['total_co2e_per_eflop_g']))
            mem_norm     = 1.0 if fit['model_fits'] else 0.3

            return {
                # Hardware params
                'name':                    hw['name'],
                'peak_performance_tflops': peak / 1e12,
                'memory_bandwidth_gbs':    bw / 1e9,
                'memory_capacity_gb':      hw['memory_capacity'] / 1e9,
                'tdp_w':                   tdp,
                'cost_usd':                hw['cost_usd'],
                'ridge_point':             ridge,

                # Performance metrics
                'avg_attain_tflops':       avg_attain_tflops,
                'mem_bound_ops':           mem_bound_cnt,
                'compute_bound_ops':       cmp_bound_cnt,
                'perf_efficiency':         total_attain_flops / max(1, total_ideal_flops),

                # Energy
                'energy_efficiency_gflops_per_w': energy_eff,
                'total_energy_j':          total_energy_j,

                # TCO
                'tco_per_eflop_usd':       tco['tco_per_eflop_usd'],
                'hw_cost_per_eflop_usd':   tco['hw_cost_per_eflop_usd'],
                'energy_cost_per_eflop_usd': tco['energy_cost_per_eflop_usd'],

                # CO2e
                'co2e_per_eflop_g':        co2['total_co2e_per_eflop_g'],
                'co2e_total_kg':           co2['total_co2e_kg'],

                # Memory
                'model_fits_memory':       fit['model_fits'],
                'peak_memory_gb':          fp['peak_memory_gb'],
                'memory_utilization_pct':  fit['utilization_pct'],

                # Normalized scores [0–1] for radar / Pareto charts
                'score_performance':       perf_norm,
                'score_energy':            eff_norm,
                'score_cost':              cost_norm,
                'score_carbon':            co2_norm,
                'score_memory':            mem_norm,
                'overall_score':           (perf_norm + eff_norm + cost_norm + co2_norm + mem_norm) / 5.0,
            }
        except Exception:
            return None

    def run(
        self,
        dse_params: dict,
        tco_params: Optional[dict] = None,
        co2_region: str = 'Global_Average',
        max_points: int = 500,
    ) -> dict:
        """
        Run DSE sweep.

        Args:
            dse_params: dict with keys:
                peak_performance_tflops, memory_bandwidth_gbs,
                memory_capacity_gb, tdp_w, cost_usd
            tco_params: TCO calculation parameters
            co2_region: grid carbon intensity region
            max_points: cap total sweep points (uniform subsampling if exceeded)
        """
        tco_p = tco_params or {}

        # Build grid
        keys  = ['peak_performance_tflops', 'memory_bandwidth_gbs',
                  'memory_capacity_gb', 'tdp_w', 'cost_usd']
        grids = [dse_params.get(k, [10]) for k in keys]
        combos = list(itertools.product(*grids))

        # Subsample if too many
        if len(combos) > max_points:
            step = len(combos) // max_points
            combos = combos[::step]

        results = []
        for combo in combos:
            pt, bw, cap, tdp, cost = combo
            hw = self._make_hw(pt, bw, cap, tdp, cost)
            ev = self._evaluate_hw(hw, tco_p, co2_region)
            if ev:
                results.append(ev)

        if not results:
            return {'points': [], 'pareto_perf_cost': [], 'pareto_perf_energy': []}

        # Pareto frontiers
        pareto_pc  = self._pareto_2d(results, 'avg_attain_tflops', 'tco_per_eflop_usd',
                                      max1=True, max2=False)
        pareto_pe  = self._pareto_2d(results, 'avg_attain_tflops', 'energy_efficiency_gflops_per_w',
                                      max1=True, max2=True)
        pareto_pco = self._pareto_2d(results, 'avg_attain_tflops', 'co2e_per_eflop_g',
                                      max1=True, max2=False)

        # Best points
        best_perf   = max(results, key=lambda r: r['avg_attain_tflops'])
        best_eff    = max(results, key=lambda r: r['energy_efficiency_gflops_per_w'])
        best_tco    = min(results, key=lambda r: r['tco_per_eflop_usd'])
        best_carbon = min(results, key=lambda r: r['co2e_per_eflop_g'])
        best_overall = max(results, key=lambda r: r['overall_score'])

        return {
            'points':              results,
            'pareto_perf_cost':    pareto_pc,
            'pareto_perf_energy':  pareto_pe,
            'pareto_perf_carbon':  pareto_pco,
            'best': {
                'performance':     best_perf,
                'energy':          best_eff,
                'tco':             best_tco,
                'carbon':          best_carbon,
                'overall':         best_overall,
            },
            'stats': {
                'total_points':    len(results),
                'fits_memory_pct': sum(1 for r in results if r['model_fits_memory']) / len(results) * 100,
                'pareto_pc_size':  len(pareto_pc),
                'pareto_pe_size':  len(pareto_pe),
            }
        }

    @staticmethod
    def _pareto_2d(points: list, obj1: str, obj2: str,
                    max1: bool = True, max2: bool = True) -> list:
        """
        Find Pareto frontier for two objectives.
        max1/max2: True = maximize, False = minimize.
        """
        def dominates(a, b):
            v1a, v1b = a[obj1], b[obj1]
            v2a, v2b = a[obj2], b[obj2]
            better1 = (v1a >= v1b) if max1 else (v1a <= v1b)
            better2 = (v2a >= v2b) if max2 else (v2a <= v2b)
            strict1 = (v1a >  v1b) if max1 else (v1a <  v1b)
            strict2 = (v2a >  v2b) if max2 else (v2a <  v2b)
            return better1 and better2 and (strict1 or strict2)

        pareto = []
        for p in points:
            if not any(dominates(q, p) for q in points):
                pareto.append(p)

        # Sort by obj1
        pareto.sort(key=lambda r: r[obj1], reverse=max1)
        return pareto

    def sensitivity_analysis(
        self,
        base_hw: dict,
        param: str,
        multipliers: List[float],
        tco_params: Optional[dict] = None,
        co2_region: str = 'Global_Average',
    ) -> List[dict]:
        """
        Vary one hardware parameter while holding others fixed.
        Useful for: 'How much does doubling bandwidth help?'

        Args:
            param: one of 'peak_performance', 'memory_bandwidth',
                          'memory_capacity', 'tdp_w', 'cost_usd'
        """
        tco_p = tco_params or {}
        results = []
        base_val = base_hw.get(param, 1.0)

        for m in multipliers:
            hw = dict(base_hw)
            hw[param] = base_val * m
            hw['name'] = f"{hw['name']} ({m:.1f}×{param.replace('_',' ')})"
            ev = self._evaluate_hw(hw, tco_p, co2_region)
            if ev:
                ev['multiplier'] = m
                ev['param_value'] = base_val * m
                ev['param_name']  = param
                results.append(ev)

        return results
