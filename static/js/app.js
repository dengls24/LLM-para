/* =========================================================
   LLM-Para Frontend Application
   ========================================================= */

'use strict';

// ── State ─────────────────────────────────────────────────────────────────────
const state = {
  models: [],
  hardware: [],
  constants: {},
  currentResults: null,
  currentSummary: null,
  currentConfig: null,
  charts: {},
  sortField: null,
  sortAsc: true,
  phaseFilter: 'all',
  catFilter: '',
  quantConfig: { activation: 16, weight_attn: 16, weight_ffn: 16, kv_cache: 16, rope_bit: 32 },
};

// ── Utility ───────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const fmt = {
  flops(v) {
    if (!v) return '0';
    if (v >= 1e18) return (v / 1e18).toFixed(2) + ' EFLOPs';
    if (v >= 1e15) return (v / 1e15).toFixed(2) + ' PFLOPs';
    if (v >= 1e12) return (v / 1e12).toFixed(2) + ' TFLOPs';
    if (v >= 1e9)  return (v / 1e9).toFixed(2)  + ' GFLOPs';
    if (v >= 1e6)  return (v / 1e6).toFixed(2)  + ' MFLOPs';
    return v.toFixed(0) + ' FLOPs';
  },
  bytes(v) {
    if (!v) return '0 B';
    if (v >= 1e12) return (v / 1e12).toFixed(2) + ' TB';
    if (v >= 1e9)  return (v / 1e9).toFixed(2)  + ' GB';
    if (v >= 1e6)  return (v / 1e6).toFixed(2)  + ' MB';
    if (v >= 1e3)  return (v / 1e3).toFixed(2)  + ' KB';
    return v.toFixed(0) + ' B';
  },
  params(v) {
    if (!v) return '0';
    if (v >= 1e12) return (v / 1e12).toFixed(2) + 'T';
    if (v >= 1e9)  return (v / 1e9).toFixed(2)  + 'B';
    if (v >= 1e6)  return (v / 1e6).toFixed(2)  + 'M';
    if (v >= 1e3)  return (v / 1e3).toFixed(2)  + 'K';
    return v.toFixed(0);
  },
  perf(v) {  // FLOP/s
    if (!v) return '0';
    if (v >= 1e15) return (v / 1e15).toFixed(1) + ' PFLOP/s';
    if (v >= 1e12) return (v / 1e12).toFixed(1) + ' TFLOP/s';
    if (v >= 1e9)  return (v / 1e9).toFixed(1)  + ' GFLOP/s';
    return v.toFixed(0) + ' FLOP/s';
  },
  bw(v) {   // Byte/s
    if (!v) return '0';
    if (v >= 1e12) return (v / 1e12).toFixed(2) + ' TB/s';
    if (v >= 1e9)  return (v / 1e9).toFixed(2)  + ' GB/s';
    return (v / 1e6).toFixed(0) + ' MB/s';
  },
};

function showError(msg) {
  $('errorMsg').textContent = msg;
  $('errorToast').classList.remove('hidden');
  setTimeout(() => $('errorToast').classList.add('hidden'), 6000);
}

function destroyChart(key) {
  if (state.charts[key]) {
    state.charts[key].destroy();
    delete state.charts[key];
  }
}

// ── Boot ──────────────────────────────────────────────────────────────────────
async function init() {
  try {
    const [modelsRes, hwRes, constRes] = await Promise.all([
      fetch('/api/models').then(r => r.json()),
      fetch('/api/hardware').then(r => r.json()),
      fetch('/api/constants').then(r => r.json()),
    ]);
    state.models = modelsRes.models;
    state.hardware = hwRes.hardware;
    state.constants = constRes;

    populateModelSelect();
    populateHardwareSelects();
    setupEventListeners();
    setupCollapsibles();
    setupBitButtons();
  } catch (e) {
    showError('Failed to load server data: ' + e.message);
  }
}

// ── Populate selects ──────────────────────────────────────────────────────────
function populateModelSelect() {
  const sel = $('modelPreset');
  const groups = {};
  state.models.forEach(m => {
    const group = m.name.split(' ')[0];
    if (!groups[group]) groups[group] = [];
    groups[group].push(m);
  });
  Object.entries(groups).forEach(([grp, items]) => {
    const og = document.createElement('optgroup');
    og.label = grp;
    items.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m.name;
      opt.textContent = m.name;
      og.appendChild(opt);
    });
    sel.appendChild(og);
  });
}

function populateHardwareSelects() {
  const sels = [$('hardwarePreset'), $('rooflineHW')];
  const groups = {};
  state.hardware.forEach(h => {
    if (!groups[h.category]) groups[h.category] = [];
    groups[h.category].push(h);
  });
  sels.forEach(sel => {
    if (!sel) return;
    sel.innerHTML = '<option value="">— Select Hardware —</option>';
    Object.entries(groups).forEach(([grp, items]) => {
      const og = document.createElement('optgroup');
      og.label = grp;
      items.forEach(h => {
        const opt = document.createElement('option');
        opt.value = h.key;
        opt.textContent = h.name;
        og.appendChild(opt);
      });
      sel.appendChild(og);
    });
  });
}

// ── Event Listeners ───────────────────────────────────────────────────────────
function setupEventListeners() {
  // Preset model selection
  $('modelPreset').addEventListener('change', e => {
    const m = state.models.find(x => x.name === e.target.value);
    if (m) fillFormFromConfig(m.config);
  });

  // Hardware preset
  $('hardwarePreset').addEventListener('change', e => {
    const hw = state.hardware.find(h => h.key === e.target.value);
    if (hw) {
      $('hw-info').classList.remove('hidden');
      $('hw-peak-fp32').textContent = fmt.perf(hw.peak_performance);
      $('hw-peak-fp16').textContent = fmt.perf(hw.peak_performance_fp16);
      $('hw-bw').textContent = fmt.bw(hw.memory_bandwidth);
      $('hw-cap').textContent = fmt.bytes(hw.memory_capacity);
      const ridge = hw.peak_performance / hw.memory_bandwidth;
      $('hw-ridge').textContent = ridge.toFixed(1) + ' FLOP/B';
    } else {
      $('hw-info').classList.add('hidden');
    }
  });

  // Roofline hardware change
  $('rooflineHW').addEventListener('change', () => {
    if (state.currentResults) renderRoofline(state.currentResults);
  });

  // Analyze button
  $('analyzeBtn').addEventListener('click', runAnalysis);

  // Tabs
  document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      tab.classList.add('active');
      $('panel-' + tab.dataset.tab).classList.add('active');
      if (tab.dataset.tab === 'roofline' && state.currentResults) renderRoofline(state.currentResults);
      if (tab.dataset.tab === 'charts' && state.currentResults) renderCharts(state.currentResults, state.currentSummary);
      if (tab.dataset.tab === 'memory' && state.currentResults) renderMemory(state.currentResults, state.currentSummary, state.currentConfig);
    });
  });

  // Phase filters
  document.querySelectorAll('.filter-btn[data-phase]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.filter-btn[data-phase]').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.phaseFilter = btn.dataset.phase;
      renderTable(state.currentResults);
    });
  });

  // Category filter
  $('categoryFilter').addEventListener('change', e => {
    state.catFilter = e.target.value;
    renderTable(state.currentResults);
  });

  // Table sort headers
  document.querySelectorAll('th[data-sort]').forEach(th => {
    th.addEventListener('click', () => {
      const field = th.dataset.sort;
      if (state.sortField === field) state.sortAsc = !state.sortAsc;
      else { state.sortField = field; state.sortAsc = false; }
      renderTable(state.currentResults);
    });
  });

  // MoE toggle
  $('use_moe').addEventListener('change', e => {
    $('moe-params').classList.toggle('hidden', !e.target.checked);
  });

  // MLA toggle
  $('use_mla').addEventListener('change', e => {
    $('mla-params').classList.toggle('hidden', !e.target.checked);
  });

  // RoPE toggle
  $('use_rope').addEventListener('change', e => {
    $('rope-params').classList.toggle('hidden', !e.target.checked);
  });

  // Export
  $('exportCSV').addEventListener('click', exportCSV);
  $('exportJSON').addEventListener('click', exportJSON);
}

function setupCollapsibles() {
  document.querySelectorAll('.toggle-btn[data-target]').forEach(btn => {
    const body = $(btn.dataset.target);
    const chevron = btn.querySelector('.chevron');
    btn.addEventListener('click', () => {
      const isHidden = body.classList.toggle('hidden');
      if (chevron) chevron.classList.toggle('open', !isHidden);
    });
  });
}

function setupBitButtons() {
  document.querySelectorAll('.bit-buttons').forEach(group => {
    group.querySelectorAll('.bit-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        group.querySelectorAll('.bit-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        state.quantConfig[group.dataset.field] = parseInt(btn.dataset.val);
      });
    });
  });
}

// ── Fill form from preset ──────────────────────────────────────────────────────
function fillFormFromConfig(cfg) {
  const setVal = (id, v) => { if ($(id)) $(id).value = v; };

  setVal('hidden_size', cfg.hidden_size);
  setVal('num_heads', cfg.num_heads);
  setVal('num_key_value_heads', cfg.num_key_value_heads || cfg.num_heads);
  setVal('num_layers', cfg.num_layers);
  setVal('intermediate_size', cfg.intermediate_size || 4 * cfg.hidden_size);
  setVal('vocab_size', cfg.vocab_size || 32000);
  setVal('seq_len', cfg.seq_len || 2048);
  setVal('batch_size', cfg.batch_size || 1);
  setVal('max_gen_len', cfg.max_gen_len || 4096);

  // Flags
  $('use_gate_ffn').checked = cfg.use_gate_ffn || false;
  $('use_rmsnorm').checked = cfg.use_rmsnorm !== false;
  $('use_flash_attn').checked = cfg.use_flash_attn || false;

  // RoPE
  const hasRope = !!cfg.rope_theta;
  $('use_rope').checked = hasRope;
  $('rope-params').classList.toggle('hidden', !hasRope);
  if (hasRope) {
    setVal('rope_theta', cfg.rope_theta);
    setVal('rope_scaling_factor', cfg.rope_scaling_factor || 1.0);
  }

  // MoE
  const hasMoE = !!cfg.num_experts_per_tok;
  $('use_moe').checked = hasMoE;
  $('moe-params').classList.toggle('hidden', !hasMoE);
  if (hasMoE) {
    setVal('num_experts_per_tok', cfg.num_experts_per_tok);
    setVal('num_local_experts', cfg.num_local_experts);
  }

  // MLA
  const hasMLA = !!cfg.use_mla;
  $('use_mla').checked = hasMLA;
  $('mla-params').classList.toggle('hidden', !hasMLA);
  if (hasMLA) {
    setVal('mla_kv_lora_rank', cfg.mla_kv_lora_rank || 512);
    setVal('mla_q_lora_rank', cfg.mla_q_lora_rank || 1536);
    setVal('mla_qk_nope_head_dim', cfg.mla_qk_nope_head_dim || 128);
    setVal('mla_qk_rope_head_dim', cfg.mla_qk_rope_head_dim || 64);
    setVal('mla_v_head_dim', cfg.mla_v_head_dim || 128);
  }

  // Quantization
  if (cfg.quant_config) {
    const q = cfg.quant_config;
    Object.entries(q).forEach(([field, val]) => {
      const group = document.querySelector(`.bit-buttons[data-field="${field}"]`);
      if (!group) return;
      group.querySelectorAll('.bit-btn').forEach(b => b.classList.remove('active'));
      const active = group.querySelector(`.bit-btn[data-val="${val}"]`);
      if (active) active.classList.add('active');
      state.quantConfig[field] = val;
    });
  }
}

// ── Build config from form ─────────────────────────────────────────────────────
function buildConfig() {
  const g = id => parseInt($(id).value) || 0;
  const gf = id => parseFloat($(id).value) || 0;
  const gb = id => $(id).checked;

  const cfg = {
    hidden_size: g('hidden_size'),
    num_heads: g('num_heads'),
    num_key_value_heads: g('num_key_value_heads'),
    num_layers: g('num_layers'),
    intermediate_size: g('intermediate_size'),
    vocab_size: g('vocab_size'),
    seq_len: g('seq_len'),
    batch_size: g('batch_size'),
    max_gen_len: g('max_gen_len'),
    use_gate_ffn: gb('use_gate_ffn'),
    use_rmsnorm: gb('use_rmsnorm'),
    use_flash_attn: gb('use_flash_attn'),
    quant_config: { ...state.quantConfig },
  };

  if (gb('use_rope')) {
    cfg.rope_theta = gf('rope_theta');
    cfg.rope_scaling_factor = gf('rope_scaling_factor');
  }

  if (gb('use_moe')) {
    cfg.num_experts_per_tok = g('num_experts_per_tok');
    cfg.num_local_experts = g('num_local_experts');
  }

  if (gb('use_mla')) {
    cfg.use_mla = true;
    cfg.mla_kv_lora_rank = g('mla_kv_lora_rank');
    cfg.mla_q_lora_rank = g('mla_q_lora_rank');
    cfg.mla_qk_nope_head_dim = g('mla_qk_nope_head_dim');
    cfg.mla_qk_rope_head_dim = g('mla_qk_rope_head_dim');
    cfg.mla_v_head_dim = g('mla_v_head_dim');
  }

  // Add hardware key
  cfg.hardware_key = $('hardwarePreset').value || $('rooflineHW').value || '';

  return cfg;
}

// ── Run Analysis ──────────────────────────────────────────────────────────────
async function runAnalysis() {
  const btn = $('analyzeBtn');
  const btnText = $('analyzeBtnText');
  const spinner = $('analyzeSpinner');
  btn.disabled = true;
  btnText.classList.add('hidden');
  spinner.classList.remove('hidden');

  try {
    const cfg = buildConfig();
    state.currentConfig = cfg;

    const res = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(cfg),
    });
    const data = await res.json();
    if (!data.success) throw new Error(data.error || 'Analysis failed');

    state.currentResults = data.results;
    state.currentSummary = data.summary;

    showResults(data.results, data.summary, cfg);
  } catch (e) {
    showError(e.message);
  } finally {
    btn.disabled = false;
    btnText.classList.remove('hidden');
    spinner.classList.add('hidden');
  }
}

// ── Render Results ────────────────────────────────────────────────────────────
function showResults(results, summary, cfg) {
  $('emptyState').classList.add('hidden');
  const container = $('resultsContainer');
  container.classList.remove('hidden');
  container.classList.add('fade-in');

  // Model name label
  const preset = $('modelPreset').value;
  $('result-model-name').textContent = preset || 'Custom Model';

  // Summary cards
  $('val-total-flops').textContent = fmt.flops(summary.total_flops);
  $('val-params').textContent = fmt.params(summary.total_params);
  $('sub-params').textContent = summary.model_size_gb.toFixed(2) + ' GB';
  $('val-kv').textContent = fmt.bytes(summary.kv_max_mb * 1024 * 1024);
  $('sub-kv').textContent = `${summary.kv_prefill_mb.toFixed(0)} MB prefill`;
  $('val-gqa').textContent = summary.gqa_ratio.toFixed(2) + '×';
  $('val-prefill').textContent = fmt.flops(summary.prefill_flops);
  $('val-decode').textContent = fmt.flops(summary.decode_flops);

  // Table (active tab)
  renderTable(results);

  // Charts if tab is active
  const activeTab = document.querySelector('.tab.active')?.dataset?.tab;
  if (activeTab === 'roofline') renderRoofline(results);
  if (activeTab === 'charts') renderCharts(results, summary);
  if (activeTab === 'memory') renderMemory(results, summary, cfg);

  // Also update roofline HW selector to match sidebar
  const hwKey = $('hardwarePreset').value;
  if (hwKey) $('rooflineHW').value = hwKey;
}

// ── Table ──────────────────────────────────────────────────────────────────────
function renderTable(results) {
  if (!results) return;
  const tbody = $('opsTableBody');
  let rows = [...results];

  // Filter by phase
  if (state.phaseFilter !== 'all') rows = rows.filter(r => r.phase === state.phaseFilter);
  // Filter by category
  if (state.catFilter) rows = rows.filter(r => r.category === state.catFilter);
  // Sort
  if (state.sortField) {
    rows.sort((a, b) => {
      const diff = (a[state.sortField] || 0) - (b[state.sortField] || 0);
      return state.sortAsc ? diff : -diff;
    });
  }

  const colors = state.constants.category_colors || {};
  tbody.innerHTML = rows.map(r => {
    const catColor = colors[r.category] || '#888';
    const densityClass = r.density > 100 ? 'density-high' : r.density > 10 ? 'density-mid' : 'density-low';
    return `<tr>
      <td><span class="phase-badge phase-${r.phase}">${r.phase}</span></td>
      <td style="font-weight:500">${r.operation}</td>
      <td><span class="cat-badge" style="color:${catColor};border-color:${catColor}40;background:${catColor}15">${r.category}</span></td>
      <td class="num-cell">${fmt.flops(r.flops_total)}</td>
      <td class="num-cell">${fmt.params(r.param_count_total)}</td>
      <td class="num-cell">${fmt.bytes(r.total_bytes_total)}</td>
      <td class="num-cell ${densityClass}">${r.density > 0 ? r.density.toFixed(1) : '—'}</td>
      <td class="note-cell" title="${r.note}">${r.note || '—'}</td>
    </tr>`;
  }).join('');
}

// ── Roofline ──────────────────────────────────────────────────────────────────
async function renderRoofline(results) {
  const hwKey = $('rooflineHW').value;
  if (!hwKey) return;

  const hw = state.hardware.find(h => h.key === hwKey);
  if (!hw) return;

  const ridgePoint = hw.peak_performance / hw.memory_bandwidth;

  // Prepare scatter points
  const catColors = state.constants.category_colors || {};
  const phaseAlpha = { Prefill: 1.0, Decode: 0.7, Output: 0.5 };
  const phaseShape = { Prefill: 'circle', Decode: 'triangle', Output: 'rectRot' };

  // Group by category for datasets
  const catDatasets = {};
  const usedPhases = new Set();

  results.forEach(r => {
    if (r.density <= 0) return;
    const density = r.density;
    const attain = Math.min(density * hw.memory_bandwidth, hw.peak_performance);
    if (!catDatasets[r.category]) {
      catDatasets[r.category] = { Prefill: [], Decode: [], Output: [] };
    }
    catDatasets[r.category][r.phase] = catDatasets[r.category][r.phase] || [];
    catDatasets[r.category][r.phase].push({
      x: density, y: attain,
      operation: r.operation,
      flops: r.flops_total,
      density: density,
      bound: density < ridgePoint ? 'Memory' : 'Compute',
    });
    usedPhases.add(r.phase);
  });

  // Build Chart.js datasets
  const datasets = [];
  Object.entries(catDatasets).forEach(([cat, phases]) => {
    const color = catColors[cat] || '#888';
    Object.entries(phases).forEach(([phase, pts]) => {
      if (!pts.length) return;
      datasets.push({
        label: `${cat} (${phase})`,
        data: pts,
        backgroundColor: color + (phase === 'Prefill' ? 'dd' : '88'),
        borderColor: color,
        borderWidth: 1.5,
        pointStyle: phaseShape[phase] || 'circle',
        pointRadius: 7,
        pointHoverRadius: 10,
      });
    });
  });

  // Build roofline line
  const xMin = 0.01, xMax = Math.max(...results.filter(r=>r.density>0).map(r=>r.density)) * 3 || ridgePoint * 10;
  const rooflinePoints = [];
  for (let i = 0; i <= 200; i++) {
    const x = xMin * Math.pow(xMax / xMin, i / 200);
    rooflinePoints.push({ x, y: Math.min(x * hw.memory_bandwidth, hw.peak_performance) });
  }

  datasets.unshift({
    label: 'Roofline',
    data: rooflinePoints,
    type: 'line',
    borderColor: '#ff6b6b',
    borderWidth: 2,
    pointRadius: 0,
    fill: false,
    tension: 0,
    order: -1,
  });

  destroyChart('roofline');

  const ctx = $('rooflineChart').getContext('2d');
  state.charts.roofline = new Chart(ctx, {
    type: 'scatter',
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          type: 'logarithmic',
          title: {
            display: true, text: 'Arithmetic Intensity (FLOP/Byte)',
            color: '#9098b8', font: { family: 'JetBrains Mono, monospace', size: 12 }
          },
          grid: { color: '#2a2f45' },
          ticks: { color: '#9098b8' },
        },
        y: {
          type: 'logarithmic',
          title: {
            display: true, text: 'Attainable Performance (FLOP/s)',
            color: '#9098b8', font: { family: 'JetBrains Mono, monospace', size: 12 }
          },
          grid: { color: '#2a2f45' },
          ticks: {
            color: '#9098b8',
            callback: v => fmt.perf(v),
          },
        },
      },
      plugins: {
        legend: {
          position: 'top',
          labels: { color: '#9098b8', font: { size: 11 }, boxWidth: 12, padding: 12 },
        },
        tooltip: {
          backgroundColor: '#1c2030',
          borderColor: '#333a52',
          borderWidth: 1,
          titleColor: '#e8eaf2',
          bodyColor: '#9098b8',
          callbacks: {
            title: items => {
              const d = items[0]?.raw;
              return d?.operation || items[0]?.dataset?.label;
            },
            label: item => {
              const d = item.raw;
              if (d.operation) {
                return [
                  `Intensity: ${d.x?.toFixed ? d.x.toFixed(2) : d.x} FLOP/B`,
                  `Perf: ${fmt.perf(d.y)}`,
                  `FLOPs: ${fmt.flops(d.flops)}`,
                  `Bound: ${d.bound}`,
                ];
              }
              return `${item.dataset.label}`;
            },
          },
        },
        annotation: {
          annotations: {
            ridgeLine: {
              type: 'line',
              xMin: ridgePoint, xMax: ridgePoint,
              borderColor: '#5b7eff55',
              borderWidth: 1.5,
              borderDash: [6, 4],
              label: {
                display: true,
                content: `Ridge: ${ridgePoint.toFixed(1)} FLOP/B`,
                color: '#5b7eff',
                backgroundColor: 'transparent',
                font: { size: 10, family: 'JetBrains Mono, monospace' },
                position: 'start',
              },
            },
          },
        },
      },
    },
  });

  // Roofline stats
  const memOps = results.filter(r => r.density > 0 && r.density < ridgePoint).length;
  const cmpOps = results.filter(r => r.density >= ridgePoint).length;
  const avgEff = results.filter(r => r.density > 0)
    .reduce((acc, r) => acc + Math.min(r.density * hw.memory_bandwidth, hw.peak_performance) / hw.peak_performance, 0)
    / Math.max(1, results.filter(r => r.density > 0).length);

  $('rs-ridge').textContent = ridgePoint.toFixed(1) + ' FLOP/B';
  $('rs-mem-ops').textContent = memOps;
  $('rs-cmp-ops').textContent = cmpOps;
  $('rs-avg-eff').textContent = (avgEff * 100).toFixed(1) + '%';
  $('rooflineStats').classList.remove('hidden');

  // Legend
  renderRooflineLegend(catColors);
}

function renderRooflineLegend(catColors) {
  const legend = $('rooflineLegend');
  legend.innerHTML = Object.entries(catColors).map(([cat, color]) =>
    `<div class="legend-item"><div class="legend-dot" style="background:${color}"></div>${cat}</div>`
  ).join('');
}

// ── Analysis Charts ────────────────────────────────────────────────────────────
function renderCharts(results, summary) {
  const catColors = state.constants.category_colors || {};
  const cats = Object.keys(catColors);

  // 1. FLOPs by category
  const catFlops = {};
  cats.forEach(c => { catFlops[c] = { prefill: 0, decode: 0 }; });
  results.forEach(r => {
    if (!catFlops[r.category]) catFlops[r.category] = { prefill: 0, decode: 0 };
    if (r.phase === 'Prefill') catFlops[r.category].prefill += r.flops_total;
    else if (r.phase === 'Decode') catFlops[r.category].decode += r.flops_total;
  });

  destroyChart('catFlop');
  const catLabels = cats.filter(c => catFlops[c] && (catFlops[c].prefill + catFlops[c].decode) > 0);
  state.charts.catFlop = new Chart($('catFlopChart').getContext('2d'), {
    type: 'bar',
    data: {
      labels: catLabels,
      datasets: [
        {
          label: 'Prefill',
          data: catLabels.map(c => catFlops[c].prefill),
          backgroundColor: catLabels.map(c => (catColors[c] || '#888') + 'cc'),
          borderColor: catLabels.map(c => catColors[c] || '#888'),
          borderWidth: 1,
        },
        {
          label: 'Decode',
          data: catLabels.map(c => catFlops[c].decode),
          backgroundColor: catLabels.map(c => (catColors[c] || '#888') + '55'),
          borderColor: catLabels.map(c => catColors[c] || '#888'),
          borderWidth: 1,
          borderDash: [4, 2],
        },
      ],
    },
    options: chartOpts('FLOPs', fmt.flops),
  });

  // 2. Memory by category
  const catMem = {};
  results.forEach(r => {
    if (!catMem[r.category]) catMem[r.category] = 0;
    catMem[r.category] += r.total_bytes_total;
  });
  destroyChart('catMem');
  const memLabels = cats.filter(c => catMem[c] > 0);
  state.charts.catMem = new Chart($('catMemChart').getContext('2d'), {
    type: 'doughnut',
    data: {
      labels: memLabels,
      datasets: [{
        data: memLabels.map(c => catMem[c]),
        backgroundColor: memLabels.map(c => (catColors[c] || '#888') + 'cc'),
        borderColor: memLabels.map(c => catColors[c] || '#888'),
        borderWidth: 1.5,
      }],
    },
    options: {
      ...darkChartOpts(),
      plugins: {
        ...darkChartOpts().plugins,
        tooltip: {
          ...darkChartOpts().plugins.tooltip,
          callbacks: { label: item => `${item.label}: ${fmt.bytes(item.raw)}` },
        },
      },
    },
  });

  // 3. Prefill vs Decode phase comparison
  const phaseFlops = { Prefill: 0, Decode: 0, Output: 0 };
  results.forEach(r => { phaseFlops[r.phase] = (phaseFlops[r.phase] || 0) + r.flops_total; });

  destroyChart('phase');
  state.charts.phase = new Chart($('phaseChart').getContext('2d'), {
    type: 'bar',
    data: {
      labels: ['Prefill', 'Decode (×1 token)', 'Output'],
      datasets: [{
        label: 'FLOPs',
        data: [phaseFlops.Prefill, phaseFlops.Decode, phaseFlops.Output],
        backgroundColor: ['#4e8df5cc', '#52c41acc', '#a78bfacc'],
        borderColor: ['#4e8df5', '#52c41a', '#a78bfa'],
        borderWidth: 1.5,
      }],
    },
    options: chartOpts('FLOPs', fmt.flops),
  });

  // 4. Density distribution
  const densityVals = results.filter(r => r.density > 0).map(r => ({
    x: r.density, y: r.flops_total,
    op: r.operation, phase: r.phase, cat: r.category,
  }));

  destroyChart('density');
  state.charts.density = new Chart($('densityChart').getContext('2d'), {
    type: 'scatter',
    data: {
      datasets: [{
        label: 'Operations',
        data: densityVals,
        backgroundColor: densityVals.map(d => (catColors[d.cat] || '#888') + 'bb'),
        borderColor: densityVals.map(d => catColors[d.cat] || '#888'),
        borderWidth: 1,
        pointRadius: 6,
        pointHoverRadius: 9,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          type: 'logarithmic',
          title: { display: true, text: 'Arithmetic Intensity (FLOP/B)', color: '#9098b8' },
          grid: { color: '#2a2f45' },
          ticks: { color: '#9098b8' },
        },
        y: {
          type: 'logarithmic',
          title: { display: true, text: 'FLOPs', color: '#9098b8' },
          grid: { color: '#2a2f45' },
          ticks: { color: '#9098b8', callback: v => fmt.flops(v) },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          ...darkChartOpts().plugins.tooltip,
          callbacks: {
            title: items => items[0]?.raw?.op,
            label: item => [
              `Intensity: ${item.raw.x.toFixed(2)} FLOP/B`,
              `FLOPs: ${fmt.flops(item.raw.y)}`,
              `Phase: ${item.raw.phase}`,
            ],
          },
        },
      },
    },
  });
}

function chartOpts(yLabel, yFmt) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: { grid: { color: '#2a2f45' }, ticks: { color: '#9098b8', font: { size: 11 } } },
      y: {
        grid: { color: '#2a2f45' },
        ticks: { color: '#9098b8', font: { size: 11 }, callback: yFmt },
        title: { display: true, text: yLabel, color: '#9098b8' },
      },
    },
    ...darkChartOpts(),
  };
}

function darkChartOpts() {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: { color: '#9098b8', font: { size: 11 }, boxWidth: 12, padding: 10 },
      },
      tooltip: {
        backgroundColor: '#1c2030', borderColor: '#333a52', borderWidth: 1,
        titleColor: '#e8eaf2', bodyColor: '#9098b8',
      },
    },
  };
}

// ── Memory Analysis ────────────────────────────────────────────────────────────
function renderMemory(results, summary, cfg) {
  // Weights bar
  const maxMem = 200; // GB reference
  const pct = Math.min(100, (summary.model_size_gb / maxMem) * 100);
  $('memWeightsBar').style.width = pct + '%';
  $('memWeightsStats').innerHTML = `
    ${fmt.params(summary.total_params)} parameters &nbsp;|&nbsp;
    ${summary.model_size_gb.toFixed(2)} GB (@ avg quant)
  `;

  // KV Cache timeline
  const tokenCounts = [];
  const kvBytes = [];
  const maxTok = (cfg?.seq_len || 2048) + (cfg?.max_gen_len || 4096);
  const step = Math.max(1, Math.floor(maxTok / 30));
  for (let t = 0; t <= maxTok; t += step) {
    tokenCounts.push(t);
    kvBytes.push(summary.kv_bytes_per_token * t * (cfg?.batch_size || 1));
  }

  destroyChart('kv');
  state.charts.kv = new Chart($('kvCacheChart').getContext('2d'), {
    type: 'line',
    data: {
      labels: tokenCounts,
      datasets: [{
        label: 'KV Cache',
        data: kvBytes,
        borderColor: '#22d3ee',
        backgroundColor: '#22d3ee22',
        fill: true,
        tension: 0.2,
        pointRadius: 0,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: { display: true, text: 'Context Length (tokens)', color: '#9098b8' },
          grid: { color: '#2a2f45' },
          ticks: { color: '#9098b8', maxTicksLimit: 6, font: { size: 10 } },
        },
        y: {
          title: { display: true, text: 'KV Cache Size', color: '#9098b8' },
          grid: { color: '#2a2f45' },
          ticks: { color: '#9098b8', callback: fmt.bytes, font: { size: 10 } },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          ...darkChartOpts().plugins.tooltip,
          callbacks: {
            label: item => `KV Cache: ${fmt.bytes(item.raw)}`,
          },
        },
      },
    },
  });

  // Per-op memory (top 12 by memory)
  const sorted = [...results].sort((a, b) => b.total_bytes_total - a.total_bytes_total).slice(0, 12);
  const catColors = state.constants.category_colors || {};

  destroyChart('opMem');
  state.charts.opMem = new Chart($('opMemChart').getContext('2d'), {
    type: 'bar',
    data: {
      labels: sorted.map(r => `${r.operation} (${r.phase})`),
      datasets: [
        {
          label: 'Input',
          data: sorted.map(r => r.input_bytes * (r.num_layers || 1)),
          backgroundColor: '#4e8df5aa',
          stack: 'mem',
        },
        {
          label: 'Weight',
          data: sorted.map(r => r.weight_bytes * (r.num_layers || 1)),
          backgroundColor: '#52c41aaa',
          stack: 'mem',
        },
        {
          label: 'Output',
          data: sorted.map(r => r.output_bytes * (r.num_layers || 1)),
          backgroundColor: '#fb923caa',
          stack: 'mem',
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: 'y',
      scales: {
        x: {
          stacked: true,
          grid: { color: '#2a2f45' },
          ticks: { color: '#9098b8', callback: fmt.bytes, font: { size: 10 } },
        },
        y: {
          stacked: true,
          grid: { color: '#2a2f45' },
          ticks: { color: '#9098b8', font: { size: 10 } },
        },
      },
      plugins: {
        legend: {
          labels: { color: '#9098b8', font: { size: 11 }, boxWidth: 12 },
        },
        tooltip: {
          ...darkChartOpts().plugins.tooltip,
          callbacks: { label: item => `${item.dataset.label}: ${fmt.bytes(item.raw)}` },
        },
      },
    },
  });
}

// ── Export ────────────────────────────────────────────────────────────────────
async function exportCSV() {
  if (!state.currentResults) return;
  const res = await fetch('/api/export/csv', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ results: state.currentResults }),
  });
  downloadBlob(await res.blob(), 'llm_para_analysis.csv');
}

async function exportJSON() {
  if (!state.currentResults) return;
  const res = await fetch('/api/export/json', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ results: state.currentResults, summary: state.currentSummary }),
  });
  downloadBlob(await res.blob(), 'llm_para_analysis.json');
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ── Bootstrap ──────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', init);
