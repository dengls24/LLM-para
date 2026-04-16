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
import random
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


# ─── Physical Constraints for Hardware DSE ──────────────────────────────────

AREA_CONSTRAINTS = {
    'compute_mm2_per_tflops': 1.5,
    'sram_mm2_per_gb': 12.0,
    'hbm_mm2_per_gb': 0.3,
    'total_die_area_mm2': 800.0,
}

POWER_CONSTRAINTS = {
    'compute_w_per_tflops': 4.0,
    'memory_w_per_gbs': 0.015,
    'static_power_frac': 0.10,
}


def check_physical_feasibility(peak_tflops, bw_gbs, cap_gb, tdp_w,
                                area_budget=None, power_budget=None):
    """
    Check whether a hardware configuration is physically feasible
    under area and power constraints.
    Returns (feasible: bool, violations: dict).
    """
    area_budget = area_budget or AREA_CONSTRAINTS['total_die_area_mm2']

    compute_area = peak_tflops * AREA_CONSTRAINTS['compute_mm2_per_tflops']
    memory_area = cap_gb * AREA_CONSTRAINTS['hbm_mm2_per_gb']
    total_area = compute_area + memory_area

    compute_power = peak_tflops * POWER_CONSTRAINTS['compute_w_per_tflops']
    memory_power = bw_gbs * POWER_CONSTRAINTS['memory_w_per_gbs']
    static_power = tdp_w * POWER_CONSTRAINTS['static_power_frac']
    required_active_power = compute_power + memory_power + static_power

    violations = {}
    feasible = True

    if total_area > area_budget:
        feasible = False
        violations['area'] = total_area - area_budget

    pw_budget = power_budget or tdp_w
    if required_active_power > pw_budget:
        feasible = False
        violations['power'] = required_active_power - pw_budget

    max_bw_per_gb = 200.0
    if cap_gb > 0 and bw_gbs / cap_gb > max_bw_per_gb:
        feasible = False
        violations['bw_cap_ratio'] = bw_gbs / cap_gb - max_bw_per_gb

    return feasible, violations


# ─── NSGA-II Multi-Objective DSE ─────────────────────────────────────────────

class Individual:
    """A single design point in the NSGA-II population."""
    __slots__ = ('genes', 'objectives', 'constraints', 'rank', 'crowding',
                 'feasible', 'result')

    def __init__(self, genes):
        self.genes = genes
        self.objectives = None
        self.constraints = None
        self.rank = None
        self.crowding = 0.0
        self.feasible = False
        self.result = None


class NSGAII_DSE:
    """
    NSGA-II based multi-objective DSE for LLM inference hardware.
    Supports continuous parameter ranges with physical area/power constraints.

    Reference: Deb et al. "A Fast and Elitist Multiobjective Genetic Algorithm:
    NSGA-II" IEEE TEC, 2002.
    """

    DEFAULT_RANGES = {
        'peak_performance_tflops': (0.5, 500.0),
        'memory_bandwidth_gbs':    (5.0, 5000.0),
        'memory_capacity_gb':      (2.0, 512.0),
        'tdp_w':                   (2.0, 750.0),
        'cost_usd':                (50.0, 50000.0),
    }

    GENE_KEYS = ['peak_performance_tflops', 'memory_bandwidth_gbs',
                 'memory_capacity_gb', 'tdp_w', 'cost_usd']

    def __init__(self, model_config: dict):
        self.model_cfg = model_config
        self._analyzer = LLMAnalyzer(model_config)
        self._ops = self._analyzer.analyze()
        self._summary = self._analyzer.get_summary(self._ops)
        self._fp = compute_memory_footprint(model_config, self._ops)
        self._dse_engine = DSEEngine(model_config)

    def _random_individual(self, ranges):
        genes = []
        for key in self.GENE_KEYS:
            lo, hi = ranges[key]
            if hi / lo > 10:
                val = math.exp(random.uniform(math.log(lo), math.log(hi)))
            else:
                val = random.uniform(lo, hi)
            genes.append(val)
        return Individual(genes)

    def _evaluate(self, ind, tco_params, co2_region, use_constraints):
        pt, bw, cap, tdp, cost = ind.genes
        if use_constraints:
            feasible, violations = check_physical_feasibility(pt, bw, cap, tdp)
            ind.feasible = feasible
            ind.constraints = violations
        else:
            ind.feasible = True
            ind.constraints = {}
        hw = self._dse_engine._make_hw(pt, bw, cap, tdp, cost)
        ev = self._dse_engine._evaluate_hw(hw, tco_params, co2_region)
        if ev is None:
            ind.objectives = (0.0, float('inf'), float('inf'))
            ind.feasible = False
            ind.result = None
            return
        ind.result = ev
        ind.objectives = (
            ev['avg_attain_tflops'],
            ev['energy_efficiency_gflops_per_w'],
            ev['tco_per_eflop_usd'],
        )

    def _dominates(self, a, b, directions):
        if not a.feasible and not b.feasible:
            return False
        if a.feasible and not b.feasible:
            return True
        if not a.feasible and b.feasible:
            return False
        better_any = False
        for av, bv, d in zip(a.objectives, b.objectives, directions):
            if d > 0:
                if av < bv: return False
                if av > bv: better_any = True
            else:
                if av > bv: return False
                if av < bv: better_any = True
        return better_any
# PLACEHOLDER_NSGAII_2

    def _fast_non_dominated_sort(self, population, directions):
        n = len(population)
        domination_count = [0] * n
        dominated_set = [[] for _ in range(n)]
        fronts = [[]]
        for i in range(n):
            for j in range(n):
                if i == j: continue
                if self._dominates(population[i], population[j], directions):
                    dominated_set[i].append(j)
                elif self._dominates(population[j], population[i], directions):
                    domination_count[i] += 1
            if domination_count[i] == 0:
                population[i].rank = 0
                fronts[0].append(i)
        k = 0
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        population[j].rank = k + 1
                        next_front.append(j)
            k += 1
            fronts.append(next_front)
        if not fronts[-1]:
            fronts.pop()
        return fronts

    def _crowding_distance(self, population, front_indices):
        n = len(front_indices)
        if n <= 2:
            for i in front_indices:
                population[i].crowding = float('inf')
            return
        for i in front_indices:
            population[i].crowding = 0.0
        num_obj = len(population[front_indices[0]].objectives)
        for m in range(num_obj):
            sorted_idx = sorted(front_indices,
                                key=lambda i: population[i].objectives[m])
            obj_min = population[sorted_idx[0]].objectives[m]
            obj_max = population[sorted_idx[-1]].objectives[m]
            spread = obj_max - obj_min
            if spread == 0: continue
            population[sorted_idx[0]].crowding = float('inf')
            population[sorted_idx[-1]].crowding = float('inf')
            for k in range(1, n - 1):
                prev_val = population[sorted_idx[k - 1]].objectives[m]
                next_val = population[sorted_idx[k + 1]].objectives[m]
                population[sorted_idx[k]].crowding += (next_val - prev_val) / spread

    def _tournament_select(self, population, tournament_size=2):
        candidates = random.sample(range(len(population)), tournament_size)
        best = candidates[0]
        for c in candidates[1:]:
            p_c, p_b = population[c], population[best]
            if p_c.rank < p_b.rank:
                best = c
            elif p_c.rank == p_b.rank and p_c.crowding > p_b.crowding:
                best = c
        return population[best]

    def _sbx_crossover(self, p1, p2, ranges, eta=20.0):
        c1_genes, c2_genes = [], []
        for i, key in enumerate(self.GENE_KEYS):
            lo, hi = ranges[key]
            if random.random() < 0.5:
                x1, x2 = p1.genes[i], p2.genes[i]
                if abs(x1 - x2) < 1e-14:
                    c1_genes.append(x1); c2_genes.append(x2); continue
                u = random.random()
                beta = (2.0 * u) ** (1.0 / (eta + 1.0)) if u <= 0.5 else (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                c1 = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
                c2 = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)
                c1_genes.append(max(lo, min(hi, c1)))
                c2_genes.append(max(lo, min(hi, c2)))
            else:
                c1_genes.append(p1.genes[i]); c2_genes.append(p2.genes[i])
        return Individual(c1_genes), Individual(c2_genes)

    def _polynomial_mutation(self, ind, ranges, eta=20.0, prob=None):
        n_genes = len(self.GENE_KEYS)
        prob = prob or (1.0 / n_genes)
        for i, key in enumerate(self.GENE_KEYS):
            if random.random() < prob:
                lo, hi = ranges[key]
                delta = hi - lo
                if delta < 1e-14: continue
                u = random.random()
                deltaq = (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0 if u < 0.5 else 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0))
                ind.genes[i] = max(lo, min(hi, ind.genes[i] + deltaq * delta))
# PLACEHOLDER_NSGAII_3

    def run(self, param_ranges=None, population_size=200, generations=50,
            use_constraints=True, tco_params=None, co2_region='Global_Average',
            objectives='perf_tco', seed=None):
        """Run NSGA-II optimization."""
        if seed is not None:
            random.seed(seed)
        ranges = dict(self.DEFAULT_RANGES)
        if param_ranges:
            ranges.update(param_ranges)
        tco_p = tco_params or {}
        directions = (1, 1, -1)  # max perf, max energy_eff, min tco

        population = [self._random_individual(ranges) for _ in range(population_size)]
        for ind in population:
            self._evaluate(ind, tco_p, co2_region, use_constraints)

        init_fronts = self._fast_non_dominated_sort(population, directions)
        for front in init_fronts:
            self._crowding_distance(population, front)

        history = []
        for gen in range(generations):
            offspring = []
            while len(offspring) < population_size:
                p1 = self._tournament_select(population)
                p2 = self._tournament_select(population)
                c1, c2 = self._sbx_crossover(p1, p2, ranges)
                self._polynomial_mutation(c1, ranges)
                self._polynomial_mutation(c2, ranges)
                offspring.extend([c1, c2])
            offspring = offspring[:population_size]
            for ind in offspring:
                self._evaluate(ind, tco_p, co2_region, use_constraints)
            combined = population + offspring
            fronts = self._fast_non_dominated_sort(combined, directions)
            for front in fronts:
                self._crowding_distance(combined, front)
            population = []
            for front in fronts:
                if len(population) + len(front) <= population_size:
                    population.extend([combined[i] for i in front])
                else:
                    remaining = population_size - len(population)
                    sorted_front = sorted(front, key=lambda i: combined[i].crowding, reverse=True)
                    population.extend([combined[i] for i in sorted_front[:remaining]])
                    break
            feasible_count = sum(1 for p in population if p.feasible)
            best_perf = max((p.objectives[0] for p in population if p.feasible), default=0)
            history.append({'generation': gen, 'feasible_count': feasible_count, 'best_perf_tflops': best_perf})

        all_results = []
        for ind in population:
            if ind.result is not None:
                r = dict(ind.result)
                r['nsga2_rank'] = ind.rank
                r['nsga2_crowding'] = ind.crowding
                r['nsga2_feasible'] = ind.feasible
                all_results.append(r)

        feasible_results = [r for r in all_results if r.get('nsga2_feasible', True)]
        pareto_pc = DSEEngine._pareto_2d(feasible_results, 'avg_attain_tflops', 'tco_per_eflop_usd', max1=True, max2=False)
        pareto_pe = DSEEngine._pareto_2d(feasible_results, 'avg_attain_tflops', 'energy_efficiency_gflops_per_w', max1=True, max2=True)
        pareto_pco = DSEEngine._pareto_2d(feasible_results, 'avg_attain_tflops', 'co2e_per_eflop_g', max1=True, max2=False)

        return {
            'points': all_results,
            'pareto_perf_cost': pareto_pc,
            'pareto_perf_energy': pareto_pe,
            'pareto_perf_carbon': pareto_pco,
            'stats': {
                'total_points': len(all_results),
                'feasible_pct': sum(1 for r in all_results if r.get('nsga2_feasible', True)) / max(1, len(all_results)) * 100,
                'pareto_pc_size': len(pareto_pc),
                'pareto_pe_size': len(pareto_pe),
                'generations': generations,
                'population_size': population_size,
                'constrained': use_constraints,
            },
            'history': history,
        }

    def run_stability_check(self, param_ranges=None, n_runs=3, population_size=200,
                            generations=50, use_constraints=True, tco_params=None,
                            co2_region='Global_Average'):
        """Run NSGA-II multiple times and compute hypervolume stability."""
        hypervolumes = []
        all_pareto_sizes = []
        for run_idx in range(n_runs):
            result = self.run(param_ranges=param_ranges, population_size=population_size,
                              generations=generations, use_constraints=use_constraints,
                              tco_params=tco_params, co2_region=co2_region, seed=run_idx * 42 + 7)
            hv = self._hypervolume_2d(result['pareto_perf_cost'],
                                      'avg_attain_tflops', 'tco_per_eflop_usd',
                                      ref_point=(0.0, 1e6), max1=True, max2=False)
            hypervolumes.append(hv)
            all_pareto_sizes.append(len(result['pareto_perf_cost']))
        mean_hv = sum(hypervolumes) / len(hypervolumes)
        std_hv = (sum((h - mean_hv) ** 2 for h in hypervolumes) / len(hypervolumes)) ** 0.5
        cv_hv = std_hv / mean_hv * 100 if mean_hv > 0 else 0.0
        return {
            'n_runs': n_runs, 'hypervolumes': hypervolumes,
            'mean_hypervolume': mean_hv, 'std_hypervolume': std_hv,
            'cv_hypervolume_pct': cv_hv, 'pareto_sizes': all_pareto_sizes,
        }

    @staticmethod
    def _hypervolume_2d(pareto_points, obj1, obj2, ref_point, max1=True, max2=False):
        """Compute 2D hypervolume indicator."""
        if not pareto_points:
            return 0.0
        pts = []
        for p in pareto_points:
            v1 = p[obj1] if max1 else -p[obj1]
            v2 = p[obj2] if max2 else -p[obj2]
            pts.append((v1, v2))
        ref1 = ref_point[0] if max1 else -ref_point[0]
        ref2 = ref_point[1] if max2 else -ref_point[1]
        pts.sort(key=lambda x: x[0], reverse=True)
        hv = 0.0
        prev_y = ref2
        for x, y in pts:
            if x <= ref1 or y <= ref2: continue
            hv += (x - ref1) * (y - prev_y)
            prev_y = max(prev_y, y)
        return abs(hv)
