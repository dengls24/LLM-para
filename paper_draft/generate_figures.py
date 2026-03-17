"""
LLM-Para Paper Figure Generator
================================
Generates all publication-quality figures for the paper.
Output: paper_draft/figures/  (PDF + PNG at 300 DPI)

Figures produced:
  fig1_roofline.pdf      - Classical Roofline (3 models × 2 phases × H100)
  fig2_energy_roofline.pdf - Energy Roofline (GFLOPS/W vs intensity)
  fig3_hetero_decode.pdf  - Decode throughput across storage tiers
  fig4_dse_pareto.pdf    - DSE Pareto frontier (perf vs TCO)
  fig5_memory_breakdown.pdf - Memory footprint breakdown
  fig6_flops_breakdown.pdf  - FLOPs by operator category
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

from analyzer import LLMAnalyzer
from metrics import run_full_metrics, build_energy_roofline_curve, CARBON_INTENSITY_GRID
from hetero import HeteroAnalyzer
from dse import DSEEngine
from configs import MODEL_CONFIGS, HARDWARE_CONFIGS

# ── Publication style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'DejaVu Serif'],
    'font.size':         9,
    'axes.titlesize':    10,
    'axes.labelsize':    9,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'legend.fontsize':   8,
    'figure.dpi':        300,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth':    0.8,
    'grid.linewidth':    0.4,
    'grid.alpha':        0.4,
    'lines.linewidth':   1.2,
})

os.makedirs('figures', exist_ok=True)

# ── Color palette (colorblind-safe) ───────────────────────────────────────────
COLORS = {
    'blue':   '#1f77b4',
    'orange': '#ff7f0e',
    'green':  '#2ca02c',
    'red':    '#d62728',
    'purple': '#9467bd',
    'brown':  '#8c564b',
    'pink':   '#e377c2',
    'gray':   '#7f7f7f',
    'olive':  '#bcbd22',
    'cyan':   '#17becf',
}
CAT_COLORS = {
    'QKV':       COLORS['blue'],
    'Attention': COLORS['red'],
    'FFN':       COLORS['green'],
    'Norm':      COLORS['orange'],
    'Embed':     COLORS['purple'],
    'RoPE':      COLORS['cyan'],
    'Other':     COLORS['gray'],
}
PHASE_MARKERS = {'Prefill': 'o', 'Decode': '^', 'Output': 's'}

# ── Utility ────────────────────────────────────────────────────────────────────
def save_fig(fig, name, tight=True):
    path_pdf = f'figures/{name}.pdf'
    path_png = f'figures/{name}.png'
    if tight:
        fig.tight_layout()
    fig.savefig(path_pdf, format='pdf')
    fig.savefig(path_png, format='png')
    print(f'  Saved: {path_pdf}  +  {path_png}')
    plt.close(fig)

def si_fmt(v, unit=''):
    if v >= 1e18: return f'{v/1e18:.1f}E{unit}'
    if v >= 1e15: return f'{v/1e15:.1f}P{unit}'
    if v >= 1e12: return f'{v/1e12:.1f}T{unit}'
    if v >= 1e9:  return f'{v/1e9:.1f}G{unit}'
    if v >= 1e6:  return f'{v/1e6:.1f}M{unit}'
    return f'{v:.1f}{unit}'

# ═══════════════════════════════════════════════════════════════════════════════
print('Generating Figure 1: Classical Roofline (multi-model)...')
# ═══════════════════════════════════════════════════════════════════════════════
HW_KEY = 'NVIDIA H100 SXM'
hw = HARDWARE_CONFIGS[HW_KEY]
peak = hw['peak_performance']   # FP32 67e12
bw   = hw['memory_bandwidth']   # 3.35e12
ridge = peak / bw

MODELS_RF = ['LLaMA-3 8B', 'Mixtral 8x7B', 'DeepSeek-V2 (MLA+MoE)']
MODEL_STYLES = {
    'LLaMA-3 8B':           {'color': COLORS['blue'],   'ls': '-'},
    'Mixtral 8x7B':         {'color': COLORS['orange'], 'ls': '--'},
    'DeepSeek-V2 (MLA+MoE)':{'color': COLORS['green'],  'ls': ':'},
}

fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
for ax_idx, phase in enumerate(['Prefill', 'Decode']):
    ax = axes[ax_idx]
    ax.set_xscale('log'); ax.set_yscale('log')

    # Roofline ceiling
    x_roof = np.logspace(-2, 4, 500)
    y_roof = np.minimum(x_roof * bw, peak)
    ax.plot(x_roof, y_roof / 1e12, 'k-', lw=1.8, label='Roofline', zorder=5)

    # Ridge annotation
    ax.axvline(ridge, color='k', lw=0.8, ls='--', alpha=0.5)
    ax.text(ridge * 1.08, peak / 1e12 * 0.5,
            f'Ridge\n{ridge:.0f} F/B', fontsize=7, va='center')
    ax.fill_betweenx([1e-3, peak / 1e12], 0, ridge,
                     alpha=0.04, color='#aaaaff')
    ax.text(ridge * 0.3, peak / 1e12 * 0.06,
            'Memory\nBound', fontsize=7, ha='center', color='#444488')
    ax.text(ridge * 3, peak / 1e12 * 0.3,
            'Compute\nBound', fontsize=7, ha='center', color='#444444')

    for mname in MODELS_RF:
        a = LLMAnalyzer(MODEL_CONFIGS[mname])
        ops = a.analyze()
        pts = [r for r in ops if r['phase'] == phase and r['density'] > 0]
        style = MODEL_STYLES[mname]
        for r in pts:
            att = min(r['density'] * bw, peak) / 1e12
            ax.scatter(r['density'], att,
                       c=CAT_COLORS.get(r['category'], COLORS['gray']),
                       marker=PHASE_MARKERS[phase], s=28, zorder=4,
                       edgecolors=style['color'], linewidths=0.8, alpha=0.85)

    ax.set_xlabel('Arithmetic Intensity (FLOP/Byte)')
    ax.set_ylabel('Attainable Performance (TFLOP/s)')
    ax.set_title(f'({chr(97+ax_idx)}) {phase} Phase — {HW_KEY}')
    ax.set_xlim(0.05, 1000); ax.set_ylim(1e-3, peak / 1e12 * 2)
    ax.grid(True, which='both'); ax.grid(True, which='minor', alpha=0.15)

# Category legend
cat_handles = [mpatches.Patch(color=v, label=k) for k, v in CAT_COLORS.items()]
axes[1].legend(handles=cat_handles, title='Category', loc='lower right',
               ncol=2, fontsize=7, title_fontsize=7)
save_fig(fig, 'fig1_roofline')

# ═══════════════════════════════════════════════════════════════════════════════
print('Generating Figure 2: Energy Roofline (multi-hardware)...')
# ═══════════════════════════════════════════════════════════════════════════════
HW_ENERGY = {
    'H100 SXM':       'NVIDIA H100 SXM',
    'MI300X':         'AMD MI300X',
    'Apple M3 Ultra': 'Apple M3 Ultra',
    'Snapdragon 8G3': 'Snapdragon 8 Gen 3 NPU',
    'DRAM-PIM':       'DRAM-PIM (HBM-PIM)',
    'Cambricon-LLM':  'Cambricon-LLM (Chiplet)',
}
HW_COLORS = list(COLORS.values())[:len(HW_ENERGY)]

cfg_llama = MODEL_CONFIGS['LLaMA-3 8B']
a_llama   = LLMAnalyzer(cfg_llama)
ops_llama = a_llama.analyze()

fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))

# Left: Energy Roofline curves
ax = axes[0]
ax.set_xscale('log'); ax.set_yscale('log')
for (label, hw_key), color in zip(HW_ENERGY.items(), HW_COLORS):
    hw = HARDWARE_CONFIGS[hw_key]
    curve = build_energy_roofline_curve(hw, n_points=400)
    ax.plot(curve['x'], curve['y'], color=color, label=label, lw=1.2)
    # Mark ridge point
    ridge_hw = hw['peak_performance'] / hw['memory_bandwidth']
    ax.axvline(ridge_hw, color=color, lw=0.5, ls=':', alpha=0.5)

ax.set_xlabel('Arithmetic Intensity (FLOP/Byte)')
ax.set_ylabel('Energy Efficiency (GFLOPS/W)')
ax.set_title('(a) Energy Roofline Curves')
ax.legend(fontsize=7, loc='upper left')
ax.grid(True, which='both'); ax.grid(True, which='minor', alpha=0.15)
ax.set_xlim(0.01, 1000)

# Right: Scatter of LLaMA-3 8B decode ops on H100
ax2 = axes[1]
ax2.set_xscale('log'); ax2.set_yscale('log')
hw_h100 = HARDWARE_CONFIGS['NVIDIA H100 SXM']
m_h100  = run_full_metrics(ops_llama, a_llama.get_summary(ops_llama), hw_h100, cfg_llama)
curve_h = build_energy_roofline_curve(hw_h100, n_points=400)
ax2.plot(curve_h['x'], curve_h['y'], 'k-', lw=1.5, label='H100 Energy Ceiling', zorder=5)

decode_pts = [p for p in m_h100['energy_points'] if p['phase'] == 'Decode']
for p in decode_pts:
    ax2.scatter(p['density'], p['energy_efficiency_gflops_per_w'],
                c=CAT_COLORS.get(p['category'], COLORS['gray']),
                marker='^', s=35, edgecolors='k', linewidths=0.5, zorder=4, alpha=0.85)

ax2.set_xlabel('Arithmetic Intensity (FLOP/Byte)')
ax2.set_ylabel('Energy Efficiency (GFLOPS/W)')
ax2.set_title('(b) LLaMA-3 8B Decode on H100')
ax2.grid(True, which='both'); ax2.grid(True, which='minor', alpha=0.15)
cat_handles2 = [mpatches.Patch(color=v, label=k) for k, v in CAT_COLORS.items()
                if any(p['category'] == k for p in decode_pts)]
ax2.legend(handles=cat_handles2, title='Category', loc='upper left',
           fontsize=7, title_fontsize=7)

save_fig(fig, 'fig2_energy_roofline')

# ═══════════════════════════════════════════════════════════════════════════════
print('Generating Figure 3: Heterogeneous Architecture Analysis...')
# ═══════════════════════════════════════════════════════════════════════════════
HETERO_HW = {
    'Cambricon-LLM\n(Chiplet)':   'Cambricon-LLM (Chiplet)',
    'Flash-LLM\n(NAND Storage)':  'Flash-LLM (NAND Storage)',
    'NAND-PIM\n(Near-Storage)':   'NAND-PIM (Near-Storage)',
}
MODELS_HETERO = ['LLaMA-3 8B', 'Mixtral 8x7B']

fig, axes = plt.subplots(1, 2, figsize=(7, 3.3))

# Left: decode throughput bar chart
ax = axes[0]
x = np.arange(len(HETERO_HW))
width = 0.35
model_colors = [COLORS['blue'], COLORS['orange']]

for m_idx, mname in enumerate(MODELS_HETERO):
    tps_vals = []
    for hw_label, hw_key in HETERO_HW.items():
        hw = HARDWARE_CONFIGS[hw_key]
        ha = HeteroAnalyzer(MODEL_CONFIGS[mname], hw)
        r  = ha.run_full_analysis()
        tps_vals.append(r['throughput']['decode_tokens_per_sec'])
    offset = (m_idx - 0.5) * width
    bars = ax.bar(x + offset, tps_vals, width, label=mname,
                  color=model_colors[m_idx], alpha=0.85, edgecolor='k', lw=0.5)
    for bar, v in zip(bars, tps_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05,
                f'{v:.1f}', ha='center', va='bottom', fontsize=7)

ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels(list(HETERO_HW.keys()), fontsize=8)
ax.set_ylabel('Decode Throughput (tokens/s)')
ax.set_title('(a) Decode Throughput by Storage Tier')
ax.legend(fontsize=7)
ax.grid(True, axis='y', alpha=0.4)
ax.set_ylim(0.1, 1000)

# Right: memory placement breakdown (Cambricon-LLM, LLaMA-3 8B)
ax2 = axes[1]
hw_c = HARDWARE_CONFIGS['Cambricon-LLM (Chiplet)']
ha_c = HeteroAnalyzer(MODEL_CONFIGS['LLaMA-3 8B'], hw_c)
r_c  = ha_c.run_full_analysis()
ps   = r_c['placement']['summary']

categories = ['Weights\n(SRAM)', 'Weights\n(DRAM)', 'Weights\n(Flash)',
              'KV Cache\n(DRAM)', 'Activations\n(SRAM)']
sizes = [
    ps['weights_in_sram_gb'],
    ps['weights_in_dram_gb'],
    ps['weights_in_flash_gb'],
    ps['kv_cache_gb'],
    ps['activations_gb'],
]
tier_colors = [COLORS['green'], COLORS['blue'], COLORS['orange'],
               COLORS['cyan'],  COLORS['green']]
tier_hatch  = ['',              '',             '///',
               '',              '///']

bars2 = ax2.bar(categories, sizes, color=tier_colors, hatch=tier_hatch,
                edgecolor='k', linewidth=0.6, alpha=0.85)
for bar, v in zip(bars2, sizes):
    if v > 0.05:
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.05,
                 f'{v:.2f} GB', ha='center', va='bottom', fontsize=7)

# Capacity lines
for tier, (cap, color, label) in {
    'SRAM':  (hw_c['memory_tiers']['SRAM']['capacity']/1e9,  COLORS['green'],  'SRAM cap.'),
    'DRAM':  (hw_c['memory_tiers']['DRAM']['capacity']/1e9,  COLORS['blue'],   'DRAM cap.'),
    'Flash': (hw_c['memory_tiers']['Flash']['capacity']/1e9, COLORS['orange'], 'Flash cap.'),
}.items():
    ax2.axhline(cap, color=color, lw=1, ls='--', alpha=0.7, label=label)

ax2.set_ylabel('Memory Size (GB)')
ax2.set_title('(b) Data Placement: LLaMA-3 8B\non Cambricon-LLM Chiplet')
ax2.legend(fontsize=6.5, loc='upper right')
ax2.tick_params(axis='x', labelsize=7)
ax2.grid(True, axis='y', alpha=0.3)

save_fig(fig, 'fig3_hetero')

# ═══════════════════════════════════════════════════════════════════════════════
print('Generating Figure 4: DSE Pareto Frontier...')
# ═══════════════════════════════════════════════════════════════════════════════
cfg_dse = MODEL_CONFIGS['LLaMA-3 8B']
engine  = DSEEngine(cfg_dse)
dse_params = {
    'peak_performance_tflops': [0.5, 1, 5, 10, 50, 100, 500],
    'memory_bandwidth_gbs':    [10, 50, 100, 500, 1000, 3000],
    'memory_capacity_gb':      [4, 16, 64, 256],
    'tdp_w':                   [10, 50, 200, 700],
    'cost_usd':                [200, 1000, 5000, 30000],
}
dse_result = engine.run(dse_params, max_points=400)

fig, axes = plt.subplots(1, 2, figsize=(7, 3.3))

# Left: Performance vs TCO
ax = axes[0]
all_pts   = dse_result['points']
pareto_pc = dse_result['pareto_perf_cost']
pareto_names = {p['name'] for p in pareto_pc}

fits   = [p for p in all_pts if p['model_fits_memory'] and p['name'] not in pareto_names]
nofits = [p for p in all_pts if not p['model_fits_memory']]

ax.scatter([p['tco_per_eflop_usd']   for p in nofits],
           [p['avg_attain_tflops']    for p in nofits],
           c='#dddddd', s=8, label='No fit', zorder=1, alpha=0.6, edgecolors='none')
ax.scatter([p['tco_per_eflop_usd']   for p in fits],
           [p['avg_attain_tflops']    for p in fits],
           c=COLORS['blue'], s=12, label='Fits memory', zorder=2, alpha=0.5, edgecolors='none')
ax.scatter([p['tco_per_eflop_usd']   for p in pareto_pc],
           [p['avg_attain_tflops']    for p in pareto_pc],
           c=COLORS['red'], s=55, marker='*', label='Pareto frontier', zorder=5,
           edgecolors='darkred', linewidths=0.6)
# Connect Pareto pts
px = [p['tco_per_eflop_usd'] for p in pareto_pc]
py = [p['avg_attain_tflops'] for p in pareto_pc]
ax.plot(px, py, 'r--', lw=0.9, alpha=0.7, zorder=4)

# Mark real hardware
for hw_key, label in [('NVIDIA H100 SXM', 'H100'), ('AMD MI300X', 'MI300X'),
                       ('Apple M3 Ultra', 'M3U'), ('DRAM-PIM (HBM-PIM)', 'HBM-PIM')]:
    hw = HARDWARE_CONFIGS[hw_key]
    peak = hw['peak_performance']; bw_hw = hw['memory_bandwidth']
    cost = hw['cost_usd']; tdp_hw = hw['tdp_w']
    ridge_hw = peak / bw_hw
    # approximate attainable perf for LLaMA decode (density ~0.5 F/B average)
    avg_density = 0.5
    att = min(avg_density * bw_hw, peak) / 1e12
    life_flops = peak * 3 * 365.25 * 24 * 3600 * 0.5
    tco = (cost + tdp_hw/1000*3*365.25*24*0.5*1.3*0.10) / (life_flops/1e18)
    ax.scatter(tco, att, marker='D', s=50, c=COLORS['purple'],
               edgecolors='black', linewidths=0.7, zorder=6)
    ax.annotate(label, (tco, att), fontsize=6.5, xytext=(5, 3),
                textcoords='offset points')

ax.set_xscale('log')
ax.set_xlabel('TCO per EFLOP (USD)')
ax.set_ylabel('Attainable Performance (TFLOPS)')
ax.set_title('(a) Performance–Cost Pareto')
ax.legend(fontsize=7, loc='upper left')
ax.grid(True, which='both', alpha=0.3)

# Right: Performance vs Energy Efficiency
ax2 = axes[1]
pareto_pe = dse_result['pareto_perf_energy']
pareto_pe_names = {p['name'] for p in pareto_pe}
others_pe = [p for p in all_pts if p['name'] not in pareto_pe_names and p['model_fits_memory']]

ax2.scatter([p['energy_efficiency_gflops_per_w'] for p in others_pe],
            [p['avg_attain_tflops']               for p in others_pe],
            c=COLORS['blue'], s=10, alpha=0.4, edgecolors='none', label='Feasible')
ax2.scatter([p['energy_efficiency_gflops_per_w'] for p in pareto_pe],
            [p['avg_attain_tflops']               for p in pareto_pe],
            c=COLORS['green'], s=55, marker='*', zorder=5,
            edgecolors='darkgreen', linewidths=0.6, label='Pareto frontier')
px2 = [p['energy_efficiency_gflops_per_w'] for p in pareto_pe]
py2 = [p['avg_attain_tflops']               for p in pareto_pe]
ax2.plot(px2, py2, 'g--', lw=0.9, alpha=0.7, zorder=4)

ax2.set_xlabel('Energy Efficiency (GFLOPS/W)')
ax2.set_ylabel('Attainable Performance (TFLOPS)')
ax2.set_title('(b) Performance–Energy Pareto')
ax2.legend(fontsize=7)
ax2.grid(True, which='both', alpha=0.3)

save_fig(fig, 'fig4_dse_pareto')

# ═══════════════════════════════════════════════════════════════════════════════
print('Generating Figure 5: FLOPs Breakdown...')
# ═══════════════════════════════════════════════════════════════════════════════
MODELS_BREAKDOWN = ['LLaMA-3 8B', 'Mixtral 8x7B', 'DeepSeek-V2 (MLA+MoE)']
CATS = ['QKV', 'Attention', 'FFN', 'Norm', 'Embed', 'RoPE']

fig, axes = plt.subplots(2, 3, figsize=(7, 4.5))

for m_idx, mname in enumerate(MODELS_BREAKDOWN):
    a = LLMAnalyzer(MODEL_CONFIGS[mname])
    ops = a.analyze()
    for ph_idx, phase in enumerate(['Prefill', 'Decode']):
        ax = axes[ph_idx][m_idx]
        phase_ops = [r for r in ops if r['phase'] == phase]
        cat_flops = {}
        for r in phase_ops:
            cat = r['category']
            cat_flops[cat] = cat_flops.get(cat, 0) + r['flops_total']
        total = sum(cat_flops.values())
        cats_present = [c for c in CATS if c in cat_flops]
        vals = [cat_flops.get(c, 0) / total * 100 for c in cats_present]
        colors = [CAT_COLORS[c] for c in cats_present]

        wedges, texts, autotexts = ax.pie(
            vals, labels=None, colors=colors, autopct='%1.0f%%',
            pctdistance=0.75, startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 0.5},
            textprops={'fontsize': 7},
        )
        for at in autotexts:
            at.set_fontsize(6.5)
            if float(at.get_text().replace('%', '')) < 5:
                at.set_visible(False)
        short = mname.replace('DeepSeek-V2 (MLA+MoE)', 'DeepSeek-V2').replace(' 8B', '').replace(' 8x7B', '')
        title = f'{short}\n({phase})\nTotal: {si_fmt(total, "FLOP")}'
        ax.set_title(title, fontsize=7.5, pad=2)

# Shared legend
legend_handles = [mpatches.Patch(facecolor=CAT_COLORS[c], label=c,
                                  edgecolor='k', linewidth=0.4) for c in CATS]
fig.legend(handles=legend_handles, loc='lower center', ncol=6,
           fontsize=8, title='Operator Category', title_fontsize=8,
           bbox_to_anchor=(0.5, -0.02))
fig.text(0.5, 0.52, '——— Prefill Phase ———', ha='center', fontsize=8.5, fontweight='bold')
fig.text(0.5, 0.01, '——— Decode Phase ———', ha='center', fontsize=8.5, fontweight='bold')

save_fig(fig, 'fig5_flops_breakdown')

# ═══════════════════════════════════════════════════════════════════════════════
print('Generating Figure 6: Memory & Quantization Impact...')
# ═══════════════════════════════════════════════════════════════════════════════
from metrics import compute_memory_footprint, check_memory_fits

cfg_base = dict(MODEL_CONFIGS['LLaMA-3 8B'])
quant_scenarios = {
    'FP16 (16-bit)':    {'weight_attn': 16, 'weight_ffn': 16, 'kv_cache': 16,
                          'activation': 16, 'rope_bit': 32},
    'INT8-W (8-bit W)': {'weight_attn': 8,  'weight_ffn': 8,  'kv_cache': 8,
                          'activation': 16, 'rope_bit': 32},
    'INT4-W (4-bit W)': {'weight_attn': 4,  'weight_ffn': 4,  'kv_cache': 8,
                          'activation': 16, 'rope_bit': 32},
    'INT4+KV4':         {'weight_attn': 4,  'weight_ffn': 4,  'kv_cache': 4,
                          'activation': 16, 'rope_bit': 16},
    'BitNet (2-bit)':   {'weight_attn': 2,  'weight_ffn': 2,  'kv_cache': 4,
                          'activation': 8,  'rope_bit': 16},
}
hw_targets = {
    'H100 (80GB)':          80,
    'RTX 4090 (24GB)':      24,
    'Cambricon DRAM (16GB)':16,
    'Edge Device (8GB)':    8,
    'Mobile (4GB)':         4,
}

fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))

# Left: stacked bar memory footprint by quantization
ax = axes[0]
labels  = list(quant_scenarios.keys())
w_vals  = []
kv_vals = []
act_vals = []
for qname, qcfg in quant_scenarios.items():
    c = dict(cfg_base); c['quant_config'] = qcfg
    a2 = LLMAnalyzer(c); ops2 = a2.analyze()
    fp = compute_memory_footprint(c, ops2)
    w_vals.append(fp['weights_gb'])
    kv_vals.append(fp['kv_max_gb'])
    act_vals.append(fp['activation_prefill_gb'])

x = np.arange(len(labels))
b1 = ax.bar(x, w_vals,  label='Weights',    color=COLORS['blue'],   alpha=0.85, edgecolor='k', lw=0.4)
b2 = ax.bar(x, kv_vals, label='KV Cache',   color=COLORS['orange'], alpha=0.85, edgecolor='k', lw=0.4,
            bottom=w_vals)
b3 = ax.bar(x, act_vals,label='Activations',color=COLORS['green'],  alpha=0.85, edgecolor='k', lw=0.4,
            bottom=[w+k for w, k in zip(w_vals, kv_vals)])

# Hardware capacity lines
hw_colors_mem = [COLORS['purple'], COLORS['red'], COLORS['cyan'], COLORS['brown'], COLORS['pink']]
for (hwname, cap), color in zip(hw_targets.items(), hw_colors_mem):
    ax.axhline(cap, color=color, lw=1.0, ls='--', alpha=0.8, label=hwname)

ax.set_xticks(x); ax.set_xticklabels(labels, rotation=22, ha='right', fontsize=7)
ax.set_ylabel('Memory Footprint (GB)')
ax.set_title('(a) Memory by Quantization\n(LLaMA-3 8B, seq=2048)')
ax.legend(fontsize=6.5, loc='upper right', ncol=1)
ax.grid(True, axis='y', alpha=0.35)

# Right: decode throughput on Cambricon-LLM vs quantization
ax2 = axes[1]
tps_cambricon = []
for qname, qcfg in quant_scenarios.items():
    c = dict(cfg_base); c['quant_config'] = qcfg
    hw_c2 = HARDWARE_CONFIGS['Cambricon-LLM (Chiplet)']
    ha2 = HeteroAnalyzer(c, hw_c2)
    r2  = ha2.run_full_analysis()
    tps_cambricon.append(r2['throughput']['decode_tokens_per_sec'])

bar_colors = [CAT_COLORS['Attention'] if v < 5 else CAT_COLORS['FFN'] if v < 30 else CAT_COLORS['QKV']
              for v in tps_cambricon]
bars3 = ax2.bar(x, tps_cambricon, color=bar_colors, edgecolor='k', lw=0.5, alpha=0.85)
for bar, v in zip(bars3, tps_cambricon):
    ax2.text(bar.get_x() + bar.get_width()/2, v + 0.5,
             f'{v:.1f}', ha='center', va='bottom', fontsize=7.5)

ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=22, ha='right', fontsize=7)
ax2.set_ylabel('Decode Throughput (tokens/s)')
ax2.set_title('(b) Decode Throughput on\nCambricon-LLM vs. Quantization')
ax2.grid(True, axis='y', alpha=0.35)
ax2.set_ylim(0, max(tps_cambricon) * 1.25)

# Color legend for (b)
handles_b = [mpatches.Patch(color=COLORS['red'], label='< 5 tok/s (poor)'),
             mpatches.Patch(color=COLORS['green'], label='5–30 tok/s'),
             mpatches.Patch(color=COLORS['blue'], label='>30 tok/s (good)')]
ax2.legend(handles=handles_b, fontsize=6.5, loc='upper left')

save_fig(fig, 'fig6_memory_quant')

# ═══════════════════════════════════════════════════════════════════════════════
print('Generating Figure 7: TCO & CO2e Comparison...')
# ═══════════════════════════════════════════════════════════════════════════════
from metrics import compute_tco, compute_co2e

HW_COMPARE = {
    'H100 SXM':     'NVIDIA H100 SXM',
    'A100 (80G)':   'NVIDIA A100 SXM (80GB)',
    'RTX 4090':     'NVIDIA RTX 4090',
    'MI300X':       'AMD MI300X',
    'M3 Ultra':     'Apple M3 Ultra',
    'Gaudi 3':      'Intel Gaudi 3',
    'SD8G3 NPU':    'Snapdragon 8 Gen 3 NPU',
    'HBM-PIM':      'DRAM-PIM (HBM-PIM)',
    'Cambricon':    'Cambricon-LLM (Chiplet)',
}
hw_labels = list(HW_COMPARE.keys())
x_hw = np.arange(len(hw_labels))

tco_per_eflop_vals   = []
co2_op_vals          = []
co2_emb_vals         = []
perf_per_dollar      = []

for hw_key in HW_COMPARE.values():
    hw = HARDWARE_CONFIGS[hw_key]
    t  = compute_tco(hw)
    co = compute_co2e(hw, hw['peak_performance'], region='Global_Average')
    tco_per_eflop_vals.append(t['tco_per_eflop_usd'])
    co2_op_vals.append(co['op_co2e_per_eflop_g'])
    co2_emb_vals.append(co['emb_co2e_per_eflop_g'])
    perf_per_dollar.append(t['perf_per_dollar_gflops'])

fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4))

# Left: TCO/EFLOP
ax = axes[0]
ax.bar(x_hw, tco_per_eflop_vals, color=[COLORS['blue']]*5 + [COLORS['green']]*2 + [COLORS['orange']]*2,
       edgecolor='k', lw=0.4, alpha=0.85)
ax.set_xticks(x_hw); ax.set_xticklabels(hw_labels, rotation=30, ha='right', fontsize=7)
ax.set_ylabel('TCO per EFLOP (USD)')
ax.set_title('(a) TCO per EFLOP\n(3-yr, 50% utilization, $0.10/kWh)')
ax.set_yscale('log')
ax.grid(True, axis='y', alpha=0.35)

# Right: CO2e stacked (op + embodied)
ax2 = axes[1]
ax2.bar(x_hw, co2_op_vals,  label='Operational CO₂e', color=COLORS['red'],    alpha=0.85, edgecolor='k', lw=0.4)
ax2.bar(x_hw, co2_emb_vals, label='Embodied CO₂e',    color=COLORS['orange'], alpha=0.85, edgecolor='k', lw=0.4,
        bottom=co2_op_vals)
ax2.set_xticks(x_hw); ax2.set_xticklabels(hw_labels, rotation=30, ha='right', fontsize=7)
ax2.set_ylabel('CO₂e per EFLOP (g CO₂e)')
ax2.set_title('(b) Carbon Footprint per EFLOP\n(Global Average Grid)')
ax2.set_yscale('log')
ax2.legend(fontsize=7)
ax2.grid(True, axis='y', alpha=0.35)

save_fig(fig, 'fig7_tco_co2')

# ═══════════════════════════════════════════════════════════════════════════════
print('\nAll figures saved to paper_draft/figures/')
print('Summary:')
for f in sorted(os.listdir('figures')):
    path = f'figures/{f}'
    size = os.path.getsize(path)
    print(f'  {path:45s}  {size/1024:.1f} KB')
