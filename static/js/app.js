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

// ── Theme ──────────────────────────────────────────────────────────────────────
function getThemeColors() {
  const s = getComputedStyle(document.documentElement);
  return {
    grid:   s.getPropertyValue('--border').trim(),
    tick:   s.getPropertyValue('--text-secondary').trim(),
    muted:  s.getPropertyValue('--text-muted').trim(),
    primary:s.getPropertyValue('--text-primary').trim(),
    bg:     s.getPropertyValue('--bg-elevated').trim(),
    border: s.getPropertyValue('--border-light').trim(),
  };
}
function initTheme() {
  const t = localStorage.getItem('llmpara-theme') || 'dark';
  applyTheme(t);
}
function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  const icon = $('themeIcon');
  if (icon) icon.textContent = theme === 'dark' ? '\u2600\uFE0F' : '\uD83C\uDF19';
  localStorage.setItem('llmpara-theme', theme);
  refreshChartColors();
}
function toggleTheme() {
  const cur = document.documentElement.getAttribute('data-theme') || 'dark';
  applyTheme(cur === 'dark' ? 'light' : 'dark');
}
function refreshChartColors() {
  if (!state.currentResults) return;
  const active = document.querySelector('.tab.active')?.dataset?.tab;
  if (active === 'roofline') renderRoofline(state.currentResults);
  if (active === 'charts')   renderCharts(state.currentResults, state.currentSummary);
  if (active === 'memory')   renderMemory(state.currentResults, state.currentSummary, state.currentConfig);
}

// ── Boot ──────────────────────────────────────────────────────────────────────
async function init() {
  initTheme();
  try {
    const [modelsRes, hwRes, constRes, regionsRes, dsePresetsRes, heteroHwRes, parallelHwRes, draftModelsRes] = await Promise.all([
      fetch('/api/models').then(r => r.json()),
      fetch('/api/hardware').then(r => r.json()),
      fetch('/api/constants').then(r => r.json()),
      fetch('/api/metrics/regions').then(r => r.json()),
      fetch('/api/dse/presets').then(r => r.json()),
      fetch('/api/hetero/hardware').then(r => r.json()),
      fetch('/api/parallelism/hardware').then(r => r.json()),
      fetch('/api/speculative/draft-models').then(r => r.json()),
    ]);
    state.models      = modelsRes.models;
    state.hardware    = hwRes.hardware;
    state.constants   = constRes;
    state.co2Regions  = regionsRes.regions || [];
    state.dsePresets  = dsePresetsRes.preset_params || {};
    state.heteroHW    = heteroHwRes.hardware || [];
    state.parallelHW  = parallelHwRes.hardware || [];
    state.draftModels = draftModelsRes.draft_models || [];

    populateModelSelect();
    populateHardwareSelects();
    populateExtendedSelects();
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

function _buildHWOptions(hw, includeEmpty = true) {
  const groups = {};
  hw.forEach(h => {
    const cat = h.category || 'Other';
    if (!groups[cat]) groups[cat] = [];
    groups[cat].push(h);
  });
  return { groups };
}

function _fillHWSelect(sel, groups, emptyLabel = '— Select Hardware —', defaultKey = '') {
  if (!sel) return;
  sel.innerHTML = emptyLabel ? `<option value="">${emptyLabel}</option>` : '';
  Object.entries(groups).forEach(([grp, items]) => {
    const og = document.createElement('optgroup');
    og.label = grp;
    items.forEach(h => {
      const opt = document.createElement('option');
      opt.value = h.key;
      opt.textContent = h.name;
      if (h.key === defaultKey) opt.selected = true;
      og.appendChild(opt);
    });
    sel.appendChild(og);
  });
}

function populateHardwareSelects() {
  const { groups } = _buildHWOptions(state.hardware);
  ['hardwarePreset', 'rooflineHW'].forEach(id => _fillHWSelect($(id), groups));
}

function populateExtendedSelects() {
  // Energy & TCO hardware selects (all hardware)
  const { groups } = _buildHWOptions(state.hardware);
  ['energyHW', 'tcoHW'].forEach(id => {
    _fillHWSelect($(id), groups, '— Select Hardware —', 'NVIDIA H100 SXM');
  });

  // Hetero hardware only
  if (state.heteroHW && state.heteroHW.length > 0) {
    const sel = $('heteroHW');
    if (sel) {
      sel.innerHTML = '<option value="">— Select Hetero Architecture —</option>';
      state.heteroHW.forEach(h => {
        const opt = document.createElement('option');
        opt.value = h.key;
        opt.textContent = `${h.name} [${h.tiers.join('+')}]`;
        sel.appendChild(opt);
      });
      // Show/update tier config when selection changes
      sel.addEventListener('change', () => {
        const hw = state.heteroHW.find(x => x.key === sel.value);
        if (hw && hw.memory_tiers) {
          fillTierConfig(hw.memory_tiers);
          $('heteroTierConfig').classList.remove('hidden');
        } else {
          $('heteroTierConfig').classList.add('hidden');
        }
      });
    }
  }

  // CO2 region selects
  const regions = state.co2Regions || [];
  ['tcoRegion', 'dseRegion'].forEach(id => {
    const sel = $(id);
    if (!sel) return;
    sel.innerHTML = '';
    regions.forEach(r => {
      const opt = document.createElement('option');
      opt.value = r.key;
      opt.textContent = `${r.key.replace(/_/g, ' ')} (${r.gco2_kwh} gCO₂/kWh)`;
      if (r.key === 'Global_Average') opt.selected = true;
      sel.appendChild(opt);
    });
  });

  // DSE presets
  const dsePresets = state.dsePresets || {};
  const dseSel = $('dsePreset');
  if (dseSel) {
    dseSel.innerHTML = '';
    Object.keys(dsePresets).forEach((name, i) => {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name;
      if (i === 0) opt.selected = true;
      dseSel.appendChild(opt);
    });
    // Render initial DSE params
    renderDSEParamCards(dsePresets[Object.keys(dsePresets)[0]]);
    dseSel.addEventListener('change', () => {
      renderDSEParamCards(dsePresets[dseSel.value] || {});
    });
  }

  // Parallel hardware
  const parallelHW = state.parallelHW || [];
  const parSel = $('parallelHW');
  if (parSel && parallelHW.length > 0) {
    parSel.innerHTML = '<option value="">— Select Multi-Chip Hardware —</option>';
    parallelHW.forEach(h => {
      const opt = document.createElement('option');
      opt.value = h.key;
      opt.textContent = `${h.name} (${h.num_devices} devices, ${h.inter_chip_bw_gbs} GB/s)`;
      parSel.appendChild(opt);
    });
  }

  // Speculative decoding: hardware select (all hardware) + draft model select
  const specHWSel = $('specHW');
  if (specHWSel) {
    _fillHWSelect(specHWSel, groups, '— Select Hardware —', 'NVIDIA H100 SXM');
  }
  const draftSel = $('specDraftModel');
  if (draftSel && state.draftModels) {
    draftSel.innerHTML = '';
    state.draftModels.forEach((d, i) => {
      const opt = document.createElement('option');
      opt.value = d.name;
      opt.textContent = `${d.name} (h=${d.hidden_size}, L=${d.num_layers})`;
      if (i === 0) opt.selected = true;
      draftSel.appendChild(opt);
    });
  }
}

function renderDSEParamCards(params) {
  const grid = $('dseParamsGrid');
  if (!grid) return;
  grid.innerHTML = '';
  const paramMeta = {
    peak_performance_tflops: { label: 'Peak Performance', unit: 'TFLOPS' },
    memory_bandwidth_gbs:    { label: 'Memory Bandwidth', unit: 'GB/s' },
    memory_capacity_gb:      { label: 'Memory Capacity', unit: 'GB' },
    tdp_w:                   { label: 'TDP', unit: 'W' },
    cost_usd:                { label: 'Cost', unit: '$' },
  };
  Object.entries(params).forEach(([key, vals]) => {
    if (!Array.isArray(vals)) return;
    const meta = paramMeta[key] || { label: key, unit: '' };
    const card = document.createElement('div');
    card.className = 'dse-param-card';
    card.innerHTML = `
      <div class="param-name">${meta.label}</div>
      <div class="dse-param-values">${vals.map(v => meta.unit === '$' ? '$' + v.toLocaleString() : v + ' ' + meta.unit).join(' · ')}</div>
    `;
    grid.appendChild(card);
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
    const isCustom = e.target.value && (e.target.value.startsWith('Custom') ||
                     e.target.value.includes('PIM'));
    if (hw) {
      $('hw-info').classList.remove('hidden');
      $('hw-peak-fp32').textContent = fmt.perf(hw.peak_performance);
      $('hw-peak-fp16').textContent = fmt.perf(hw.peak_performance_fp16);
      $('hw-bw').textContent = fmt.bw(hw.memory_bandwidth);
      $('hw-cap').textContent = fmt.bytes(hw.memory_capacity);
      const ridge = hw.peak_performance / hw.memory_bandwidth;
      $('hw-ridge').textContent = ridge.toFixed(1) + ' FLOP/B';
      // Show custom edit fields for Custom/PIM hardware
      if (isCustom) {
        $('hw-custom-params').classList.remove('hidden');
        $('hw_custom_peak').value = (hw.peak_performance / 1e12).toFixed(2);
        $('hw_custom_peak_fp16').value = ((hw.peak_performance_fp16 || hw.peak_performance) / 1e12).toFixed(2);
        $('hw_custom_bw').value = (hw.memory_bandwidth / 1e9).toFixed(0);
        $('hw_custom_cap').value = (hw.memory_capacity / 1e9).toFixed(1);
        $('hw_custom_tdp').value = hw.tdp_w || 200;
        $('hw_custom_cost').value = hw.cost_usd || 2000;
      } else {
        $('hw-custom-params').classList.add('hidden');
      }
    } else {
      $('hw-info').classList.add('hidden');
      $('hw-custom-params').classList.add('hidden');
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
      if (tab.dataset.tab === 'charts'   && state.currentResults) renderCharts(state.currentResults, state.currentSummary);
      if (tab.dataset.tab === 'memory'   && state.currentResults) renderMemory(state.currentResults, state.currentSummary, state.currentConfig);
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

  // Extended tab buttons
  $('runEnergyBtn') && $('runEnergyBtn').addEventListener('click', runEnergyAnalysis);
  $('runTCOBtn')    && $('runTCOBtn').addEventListener('click', runTCOAnalysis);
  $('runHeteroBtn') && $('runHeteroBtn').addEventListener('click', runHeteroAnalysis);
  $('runDSEBtn')    && $('runDSEBtn').addEventListener('click', runDSE);
  $('runParallelBtn') && $('runParallelBtn').addEventListener('click', runParallelAnalysis);
  $('runSpecBtn') && $('runSpecBtn').addEventListener('click', runSpeculativeAnalysis);
  $('themeToggle') && $('themeToggle').addEventListener('click', toggleTheme);

  // Apply custom hardware params button
  $('applyCustomHW') && $('applyCustomHW').addEventListener('click', () => {
    const peak = parseFloat($('hw_custom_peak').value) * 1e12;
    const peakFp16 = parseFloat($('hw_custom_peak_fp16').value) * 1e12;
    const bw = parseFloat($('hw_custom_bw').value) * 1e9;
    const cap = parseFloat($('hw_custom_cap').value) * 1e9;
    // Update display
    $('hw-peak-fp32').textContent = fmt.perf(peak);
    $('hw-peak-fp16').textContent = fmt.perf(peakFp16);
    $('hw-bw').textContent = fmt.bw(bw);
    $('hw-cap').textContent = fmt.bytes(cap);
    $('hw-ridge').textContent = (peak / bw).toFixed(1) + ' FLOP/B';
    // Store custom overrides in state
    state.customHW = {
      peak_performance: peak,
      peak_performance_fp16: peakFp16,
      memory_bandwidth: bw,
      memory_capacity: cap,
      tdp_w: parseFloat($('hw_custom_tdp').value) || 200,
      cost_usd: parseFloat($('hw_custom_cost').value) || 2000,
    };
  });

  // Parallel hardware select: update form defaults
  if ($('parallelHW')) {
    $('parallelHW').addEventListener('change', () => {
      const hw = (state.parallelHW || []).find(h => h.key === $('parallelHW').value);
      if (hw) {
        $('parTP').value = hw.num_devices;
        $('parBW').value = hw.inter_chip_bw_gbs;
        $('parTopology').value = hw.topology || 'ring';
      }
    });
  }
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

  // Include custom hardware overrides if set
  if (state.customHW && (cfg.hardware_key.startsWith('Custom') || cfg.hardware_key.includes('PIM'))) {
    cfg.custom_hardware = state.customHW;
  }

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

  const c = getThemeColors();
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
            color: c.tick, font: { family: 'JetBrains Mono, monospace', size: 12 }
          },
          grid: { color: c.grid },
          ticks: { color: c.tick },
        },
        y: {
          type: 'logarithmic',
          title: {
            display: true, text: 'Attainable Performance (FLOP/s)',
            color: c.tick, font: { family: 'JetBrains Mono, monospace', size: 12 }
          },
          grid: { color: c.grid },
          ticks: {
            color: c.tick,
            callback: v => fmt.perf(v),
          },
        },
      },
      plugins: {
        legend: {
          position: 'top',
          labels: { color: c.tick, font: { size: 11 }, boxWidth: 12, padding: 12 },
        },
        tooltip: {
          backgroundColor: c.bg,
          borderColor: c.border,
          borderWidth: 1,
          titleColor: c.primary,
          bodyColor: c.tick,
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
  const c = getThemeColors();

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
          title: { display: true, text: 'Arithmetic Intensity (FLOP/B)', color: c.tick },
          grid: { color: c.grid },
          ticks: { color: c.tick },
        },
        y: {
          type: 'logarithmic',
          title: { display: true, text: 'FLOPs', color: c.tick },
          grid: { color: c.grid },
          ticks: { color: c.tick, callback: v => fmt.flops(v) },
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
  const c = getThemeColors();
  return {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: { grid: { color: c.grid }, ticks: { color: c.tick, font: { size: 11 } } },
      y: {
        grid: { color: c.grid },
        ticks: { color: c.tick, font: { size: 11 }, callback: yFmt },
        title: { display: true, text: yLabel, color: c.tick },
      },
    },
    ...darkChartOpts(),
  };
}

function darkChartOpts() {
  const c = getThemeColors();
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: { color: c.tick, font: { size: 11 }, boxWidth: 12, padding: 10 },
      },
      tooltip: {
        backgroundColor: c.bg, borderColor: c.border, borderWidth: 1,
        titleColor: c.primary, bodyColor: c.tick,
      },
    },
  };
}

// ── Memory Analysis ────────────────────────────────────────────────────────────
function renderMemory(results, summary, cfg) {
  const c = getThemeColors();
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
          title: { display: true, text: 'Context Length (tokens)', color: c.tick },
          grid: { color: c.grid },
          ticks: { color: c.tick, maxTicksLimit: 6, font: { size: 10 } },
        },
        y: {
          title: { display: true, text: 'KV Cache Size', color: c.tick },
          grid: { color: c.grid },
          ticks: { color: c.tick, callback: fmt.bytes, font: { size: 10 } },
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
          grid: { color: c.grid },
          ticks: { color: c.tick, callback: fmt.bytes, font: { size: 10 } },
        },
        y: {
          stacked: true,
          grid: { color: c.grid },
          ticks: { color: c.tick, font: { size: 10 } },
        },
      },
      plugins: {
        legend: {
          labels: { color: c.tick, font: { size: 11 }, boxWidth: 12 },
        },
        tooltip: {
          ...darkChartOpts().plugins.tooltip,
          callbacks: { label: item => `${item.dataset.label}: ${fmt.bytes(item.raw)}` },
        },
      },
    },
  });
}

// ── Energy Roofline Tab ───────────────────────────────────────────────────────

async function runEnergyAnalysis() {
  if (!state.currentResults || !state.currentSummary || !state.currentConfig) {
    showError('Run an analysis first, then switch to this tab.');
    return;
  }
  const hwKey = $('energyHW').value;
  if (!hwKey) { showError('Select a hardware platform.'); return; }

  const btn = $('runEnergyBtn');
  btn.disabled = true; btn.textContent = 'Computing…';

  try {
    const res = await fetch('/api/metrics', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        config: state.currentConfig,
        results: state.currentResults,
        summary: state.currentSummary,
        hardware_key: hwKey,
        tco_params: {},
        co2_region: 'Global_Average',
      }),
    });
    const data = await res.json();
    if (!data.success) { showError(data.error); return; }

    const m = data.metrics;
    const s = m.summary;

    // Stats row
    $('es-peak-eff').textContent  = s.peak_energy_eff_gflops_w?.toFixed(1) + ' GFLOPS/W' || '—';
    $('es-avg-eff').textContent   = s.avg_energy_eff_gflops_w?.toFixed(2)  + ' GFLOPS/W' || '—';
    $('es-tdp').textContent       = m.tco ? (m.tco.hardware_cost_usd ? '—' : '—') : '—';
    const hw = state.hardware.find(h => h.key === hwKey);
    if (hw) {
      $('es-tdp').textContent = hw.tdp_w ? hw.tdp_w + ' W' : '—';
    }
    $('es-total-energy').textContent = s.total_energy_j < 1 ?
      (s.total_energy_j * 1000).toFixed(2) + ' mJ' :
      s.total_energy_j.toFixed(4) + ' J';
    $('es-mem-bound').textContent  = s.mem_bound_ops;
    $('es-cmp-bound').textContent  = s.compute_bound_ops;

    renderEnergyRooflineChart(m);
    renderOpEnergyChart(m.energy_points);
    renderPowerUtilChart(m.energy_points);

  } catch(e) {
    showError('Energy analysis failed: ' + e.message);
  } finally {
    btn.disabled = false; btn.textContent = 'Compute';
  }
}

function renderEnergyRooflineChart(m) {
  const curve  = m.energy_curve;
  const pts    = m.energy_points;
  const colors = state.constants.category_colors || {};
  const catList = [...new Set(pts.map(p => p.category))];
  const c = getThemeColors();

  const datasets = [
    {
      label: 'Energy Roofline Ceiling',
      data: curve.x.map((x, i) => ({ x, y: curve.y[i] })),
      type: 'line',
      borderColor: '#5b7eff',
      borderWidth: 2,
      pointRadius: 0,
      fill: false,
      tension: 0.2,
      order: 0,
    },
  ];

  const phaseShape = { Prefill: 'circle', Decode: 'triangle', Output: 'rect' };
  for (const cat of catList) {
    const catPts = pts.filter(p => p.category === cat);
    datasets.push({
      label: cat,
      data: catPts.map(p => ({ x: p.density, y: p.energy_efficiency_gflops_per_w,
        label: p.operation, phase: p.phase })),
      backgroundColor: (colors[cat] || '#888') + 'cc',
      borderColor:     colors[cat] || '#888',
      borderWidth: 1.5,
      pointRadius: 6,
      pointStyle: catPts[0] ? (phaseShape[catPts[0].phase] || 'circle') : 'circle',
      type: 'scatter',
      order: 1,
    });
  }

  // Ridge point annotation
  const ridge = curve.ridge_point;
  const annotations = {
    ridgeLine: {
      type: 'line', scaleID: 'x', value: ridge,
      borderColor: '#fb923c88', borderWidth: 1, borderDash: [6, 4],
      label: {
        content: `Ridge: ${ridge.toFixed(1)} F/B`, display: true,
        position: 'start', color: '#fb923c',
        font: { size: 10, family: 'JetBrains Mono' },
      },
    },
  };

  destroyChart('energyRoofline');
  state.charts.energyRoofline = new Chart($('energyRooflineChart').getContext('2d'), {
    type: 'scatter',
    data: { datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: {
          type: 'logarithmic',
          title: { display: true, text: 'Arithmetic Intensity (FLOP/Byte)', color: c.tick },
          grid: { color: c.grid },
          ticks: { color: c.tick, font: { size: 10 } },
        },
        y: {
          type: 'logarithmic',
          title: { display: true, text: 'Energy Efficiency (GFLOPS/W)', color: c.tick },
          grid: { color: c.grid },
          ticks: { color: c.tick, font: { size: 10 } },
        },
      },
      plugins: {
        legend: { labels: { color: c.tick, font: { size: 11 }, boxWidth: 12 } },
        annotation: { annotations },
        tooltip: {
          ...darkChartOpts().plugins.tooltip,
          callbacks: {
            label: item => {
              const d = item.raw;
              return d.label ?
                [`${d.label} (${d.phase})`, `Intensity: ${d.x?.toFixed(2)} F/B`, `Efficiency: ${item.raw.y?.toFixed(2)} GFLOPS/W`] :
                `${item.raw.y?.toFixed(2)} GFLOPS/W`;
            },
          },
        },
      },
    },
  });
}

function renderOpEnergyChart(pts) {
  const sorted = [...pts].sort((a, b) => b.energy_j - a.energy_j).slice(0, 14);
  const colors = state.constants.category_colors || {};
  const c = getThemeColors();
  destroyChart('opEnergy');
  state.charts.opEnergy = new Chart($('opEnergyChart').getContext('2d'), {
    type: 'bar',
    data: {
      labels: sorted.map(p => `${p.operation.substring(0,16)} (${p.phase.substring(0,1)})`),
      datasets: [{
        label: 'Energy (mJ)',
        data: sorted.map(p => p.energy_j * 1000),
        backgroundColor: sorted.map(p => (colors[p.category] || '#888') + 'cc'),
        borderColor:     sorted.map(p => colors[p.category] || '#888'),
        borderWidth: 1,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false, indexAxis: 'y',
      scales: {
        x: { grid: { color: c.grid }, ticks: { color: c.tick, font: { size: 10 },
          callback: v => v.toFixed(2) + ' mJ' } },
        y: { grid: { color: c.grid }, ticks: { color: c.tick, font: { size: 9 } } },
      },
      plugins: {
        legend: { display: false },
        tooltip: { ...darkChartOpts().plugins.tooltip,
          callbacks: { label: i => `${i.raw.toFixed(3)} mJ` } },
      },
    },
  });
}

function renderPowerUtilChart(pts) {
  const avgComputeUtil = pts.reduce((s, p) => s + (p.util_compute || 0), 0) / Math.max(1, pts.length);
  const avgMemUtil     = pts.reduce((s, p) => s + (p.util_memory  || 0), 0) / Math.max(1, pts.length);
  const idle           = Math.max(0, 1 - Math.max(avgComputeUtil, avgMemUtil));
  const c = getThemeColors();

  destroyChart('powerUtil');
  state.charts.powerUtil = new Chart($('powerUtilChart').getContext('2d'), {
    type: 'doughnut',
    data: {
      labels: ['Compute Active', 'Memory Active', 'Idle'],
      datasets: [{
        data: [avgComputeUtil * 100, avgMemUtil * 100, idle * 100],
        backgroundColor: ['#5b7eff99', '#22d3ee99', '#2a2f45'],
        borderColor:     ['#5b7eff',   '#22d3ee',   '#333a52'],
        borderWidth: 2,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { position: 'bottom', labels: { color: c.tick, font: { size: 11 }, boxWidth: 14 } },
        tooltip: { ...darkChartOpts().plugins.tooltip,
          callbacks: { label: i => `${i.label}: ${i.raw.toFixed(1)}%` } },
      },
    },
  });
}

// ── TCO & CO2e Tab ────────────────────────────────────────────────────────────

async function runTCOAnalysis() {
  if (!state.currentResults || !state.currentConfig) {
    showError('Run an analysis first.'); return;
  }
  const hwKey = $('tcoHW').value;
  if (!hwKey) { showError('Select a hardware platform.'); return; }

  const btn = $('runTCOBtn');
  btn.disabled = true; btn.textContent = 'Computing…';

  const tcoParams = {
    lifetime_years:    parseFloat($('tcoLifetime').value) || 3,
    utilization:       (parseFloat($('tcoUtil').value) || 50) / 100,
    electricity_price: parseFloat($('tcoElec').value) || 0.10,
    pue:               parseFloat($('tcoPUE').value)  || 1.3,
  };
  const region = $('tcoRegion').value || 'Global_Average';

  try {
    const res = await fetch('/api/metrics', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        config: state.currentConfig,
        results: state.currentResults,
        summary: state.currentSummary,
        hardware_key: hwKey,
        tco_params: tcoParams,
        co2_region: region,
      }),
    });
    const data = await res.json();
    if (!data.success) { showError(data.error); return; }

    const m = data.metrics;
    const t = m.tco; const c = m.co2e;

    // Stats
    $('ts-hw-cost').textContent   = '$' + t.hardware_cost_usd.toLocaleString();
    $('ts-e-cost').textContent    = '$' + t.electricity_cost_usd.toFixed(0).toLocaleString();
    $('ts-total-tco').textContent = '$' + t.total_tco_usd.toFixed(0).toLocaleString();
    $('ts-tco-eflop').textContent = '$' + t.tco_per_eflop_usd.toFixed(2) + '/EFLOP';
    $('ts-co2-op').textContent    = c.operational_co2e_kg.toFixed(4) + ' kg';
    $('ts-co2-emb').textContent   = c.embodied_co2e_kg.toFixed(4) + ' kg';
    $('ts-kwh').textContent       = t.electricity_kwh.toFixed(1) + ' kWh';
    $('ts-gflops-dollar').textContent = t.perf_per_dollar_gflops.toFixed(1);

    renderTCOCharts(m, hwKey);

  } catch(e) {
    showError('TCO analysis failed: ' + e.message);
  } finally {
    btn.disabled = false; btn.textContent = 'Compute TCO';
  }
}

function renderTCOCharts(m, selectedHW) {
  const t = m.tco; const co2 = m.co2e;
  const c = getThemeColors();

  // TCO Breakdown (pie)
  destroyChart('tcoBreakdown');
  state.charts.tcoBreakdown = new Chart($('tcoBreakdownChart').getContext('2d'), {
    type: 'doughnut',
    data: {
      labels: ['Hardware Cost', 'Electricity Cost'],
      datasets: [{
        data: [t.hardware_cost_usd, t.electricity_cost_usd],
        backgroundColor: ['#5b7eff99', '#fb923c99'],
        borderColor:     ['#5b7eff',   '#fb923c'],
        borderWidth: 2,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { position: 'bottom', labels: { color: c.tick, font: { size: 11 }, boxWidth: 14 } },
        tooltip: { ...darkChartOpts().plugins.tooltip,
          callbacks: { label: i => `${i.label}: $${i.raw.toLocaleString()}` } },
      },
    },
  });

  // CO2e Breakdown (pie)
  destroyChart('co2');
  state.charts.co2 = new Chart($('co2Chart').getContext('2d'), {
    type: 'doughnut',
    data: {
      labels: ['Operational CO₂e', 'Embodied CO₂e'],
      datasets: [{
        data: [co2.operational_co2e_kg, co2.embodied_co2e_kg],
        backgroundColor: ['#4ade8099', '#f8717199'],
        borderColor:     ['#4ade80',   '#f87171'],
        borderWidth: 2,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { position: 'bottom', labels: { color: c.tick, font: { size: 11 }, boxWidth: 14 } },
        tooltip: { ...darkChartOpts().plugins.tooltip,
          callbacks: { label: i => `${i.label}: ${i.raw.toFixed(4)} kg CO₂e` } },
      },
    },
  });

  // Multi-HW TCO/EFLOP comparison (use representative hardware)
  const hwSample = ['NVIDIA H100 SXM','NVIDIA A100 SXM (80GB)','NVIDIA RTX 4090',
    'AMD MI300X', 'Apple M3 Ultra', 'Snapdragon 8 Gen 3 NPU',
    'DRAM-PIM (HBM-PIM)', 'Cambricon-LLM (Chiplet)'];
  const hwData = state.hardware.filter(h => hwSample.includes(h.key) || h.key === selectedHW)
    .slice(0, 10);

  // TCO/EFLOP = cost/(lifetime_flops)  — approximate from config
  const tcoPerHW = hwData.map(h => {
    const peak = h.peak_performance || 1e12;
    const cost = h.cost_usd || 5000;
    const tdp  = h.tdp_w || 300;
    const lifeFlops = peak * 3 * 365.25 * 24 * 3600 * 0.5;
    const energyKwh = tdp / 1000 * 3 * 365.25 * 24 * 0.5 * 1.3;
    const tcoTotal  = cost + energyKwh * 0.10;
    return tcoTotal / (lifeFlops / 1e18);
  });

  destroyChart('tcoHW');
  state.charts.tcoHW = new Chart($('tcoHWChart').getContext('2d'), {
    type: 'bar',
    data: {
      labels: hwData.map(h => h.name.replace('NVIDIA ', '').replace('Apple ', '')),
      datasets: [{
        label: 'TCO / EFLOP (USD)',
        data: tcoPerHW,
        backgroundColor: hwData.map((h, i) => ['#5b7eff','#4ade80','#fb923c','#f87171',
          '#22d3ee','#a78bfa','#facc15','#e879f9','#34d399','#60a5fa'][i % 10] + '99'),
        borderWidth: 1,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { grid: { color: c.grid }, ticks: { color: c.tick, font: { size: 9 } } },
        y: { grid: { color: c.grid }, ticks: { color: c.tick, font: { size: 10 },
          callback: v => '$' + v.toFixed(0) } },
      },
      plugins: {
        legend: { display: false },
        tooltip: { ...darkChartOpts().plugins.tooltip,
          callbacks: { label: i => `$${i.raw.toFixed(2)} / EFLOP` } },
      },
    },
  });

  // Energy Efficiency vs $/GFLOPS scatter
  const effCostData = hwData.map(h => {
    const peak = h.peak_performance || 1e12;
    const tdp  = h.tdp_w  || 300;
    const cost = h.cost_usd || 5000;
    return {
      x: cost / (peak / 1e9),         // $/GFLOPS
      y: peak / tdp / 1e9,             // GFLOPS/W
      label: h.name.replace('NVIDIA ', '').replace('Apple ', ''),
    };
  });

  destroyChart('effCost');
  state.charts.effCost = new Chart($('effCostChart').getContext('2d'), {
    type: 'scatter',
    data: {
      datasets: [{
        label: 'Hardware Platforms',
        data: effCostData,
        backgroundColor: '#5b7effaa',
        borderColor:     '#5b7eff',
        borderWidth: 1.5,
        pointRadius: 8,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: {
          type: 'logarithmic',
          title: { display: true, text: '$/GFLOPS (lower = better)', color: c.tick },
          grid: { color: c.grid },
          ticks: { color: c.tick, font: { size: 10 }, callback: v => '$' + v.toFixed(2) },
        },
        y: {
          title: { display: true, text: 'Energy Efficiency (GFLOPS/W)', color: c.tick },
          grid: { color: c.grid },
          ticks: { color: c.tick, font: { size: 10 } },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: { ...darkChartOpts().plugins.tooltip,
          callbacks: {
            label: i => [i.raw.label, `$${i.raw.x.toFixed(3)}/GFLOPS`, `${i.raw.y.toFixed(2)} GFLOPS/W`],
          },
        },
      },
    },
  });
}

// ── Heterogeneous Architecture Tab ────────────────────────────────────────────

async function runHeteroAnalysis() {
  if (!state.currentConfig) { showError('Run an analysis first.'); return; }
  const hwKey = $('heteroHW').value;
  if (!hwKey) { showError('Select a heterogeneous hardware.'); return; }

  const btn = $('runHeteroBtn');
  btn.disabled = true; btn.textContent = 'Analyzing…';
  $('heteroEmpty').classList.add('hidden');
  $('heteroResults').classList.add('hidden');

  try {
    const res = await fetch('/api/hetero', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        config: state.currentConfig,
        hardware_key: hwKey,
        memory_tiers: $('heteroTierConfig').classList.contains('hidden') ? null : readTierConfig(),
      }),
    });
    const data = await res.json();
    if (!data.success) { showError(data.error); $('heteroEmpty').classList.remove('hidden'); return; }

    const h = data.hetero;
    const ps = h.placement.summary;
    const tp = h.throughput;

    // Stats
    const tierClass = { SRAM: 'tier-sram', DRAM: 'tier-dram', Flash: 'tier-flash' };
    $('hs-bottleneck').innerHTML = `<span class="${tierClass[ps.decode_bottleneck] || ''}">${ps.decode_bottleneck}</span>`;
    $('hs-tps').textContent    = tp.decode_tokens_per_sec.toFixed(2) + ' tok/s';
    $('hs-ttft').textContent   = tp.ttft_ms.toFixed(1) + ' ms';
    $('hs-sram-util').textContent = ps.sram_utilization_pct.toFixed(1) + '%';
    $('hs-dram-util').textContent = ps.dram_utilization_pct.toFixed(1) + '%';
    $('hs-flash-spills').textContent = ps.flash_spills ? '⚠️ Yes' : '✅ No';

    renderHeteroPlacementChart(ps);
    renderHeteroThroughputChart(tp);
    renderHeteroOpsChart(h.hetero_ops);
    renderHeteroTable(h.placement.placement);

    $('heteroResults').classList.remove('hidden');
  } catch(e) {
    showError('Hetero analysis failed: ' + e.message);
    $('heteroEmpty').classList.remove('hidden');
  } finally {
    btn.disabled = false; btn.textContent = 'Analyze Placement';
  }
}

function renderHeteroPlacementChart(ps) {
  const sramW = ps.weights_in_sram_gb || 0;
  const dramW = ps.weights_in_dram_gb || 0;
  const flashW = ps.weights_in_flash_gb || 0;
  const kv   = ps.kv_cache_gb || 0;
  const act  = ps.activations_gb || 0;
  const c = getThemeColors();

  destroyChart('heteroPlacement');
  state.charts.heteroPlacement = new Chart($('heteroPlacementChart').getContext('2d'), {
    type: 'bar',
    data: {
      labels: ['SRAM', 'DRAM', 'Flash'],
      datasets: [
        { label: 'Weights', data: [sramW, dramW, flashW],
          backgroundColor: '#5b7eff99', borderColor: '#5b7eff', borderWidth: 1.5, stack: 'a' },
        { label: 'KV Cache', data: [0, kv, 0],
          backgroundColor: '#22d3ee99', borderColor: '#22d3ee', borderWidth: 1.5, stack: 'a' },
        { label: 'Activations', data: [act, 0, 0],
          backgroundColor: '#4ade8099', borderColor: '#4ade80', borderWidth: 1.5, stack: 'a' },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { stacked: true, grid: { color: c.grid }, ticks: { color: c.tick, font: { size: 11 } } },
        y: { stacked: true, grid: { color: c.grid }, ticks: { color: c.tick, font: { size: 10 },
          callback: v => v.toFixed(1) + ' GB' } },
      },
      plugins: {
        legend: { labels: { color: c.tick, font: { size: 11 }, boxWidth: 12 } },
        tooltip: { ...darkChartOpts().plugins.tooltip,
          callbacks: { label: i => `${i.dataset.label}: ${i.raw.toFixed(2)} GB` } },
      },
    },
  });
}

function renderHeteroThroughputChart(tp) {
  const tiers = ['SRAM', 'DRAM', 'Flash'].filter(t => tp[`tokens_per_sec_${t}`] !== undefined);
  const c = getThemeColors();
  destroyChart('heteroThroughput');
  state.charts.heteroThroughput = new Chart($('heteroThroughputChart').getContext('2d'), {
    type: 'bar',
    data: {
      labels: tiers.map(t => t + '\n(' + (tp[`tokens_per_sec_${t}`] > 1000 ? '>1K' : '') + ')'),
      datasets: [{
        label: 'Tokens/s',
        data: tiers.map(t => tp[`tokens_per_sec_${t}`]),
        backgroundColor: ['#4ade8099', '#22d3ee99', '#fb923c99'],
        borderColor:     ['#4ade80',   '#22d3ee',   '#fb923c'],
        borderWidth: 1.5,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { grid: { color: c.grid }, ticks: { color: c.tick, font: { size: 11 } } },
        y: { type: 'logarithmic', grid: { color: c.grid },
          ticks: { color: c.tick, font: { size: 10 }, callback: v => v.toFixed(0) } },
      },
      plugins: {
        legend: { display: false },
        tooltip: { ...darkChartOpts().plugins.tooltip,
          callbacks: { label: i => `${i.raw.toFixed(2)} tokens/s` } },
      },
    },
  });
}

function renderHeteroOpsChart(heteroOps) {
  const ops = heteroOps.filter(op => op.phase === 'Decode').slice(0, 12);
  const tierColors = { SRAM: '#4ade80', DRAM: '#22d3ee', Flash: '#fb923c' };
  const c = getThemeColors();
  destroyChart('heteroOps');
  state.charts.heteroOps = new Chart($('heteroOpsChart').getContext('2d'), {
    type: 'bar',
    data: {
      labels: ops.map(o => o.operation.substring(0, 20)),
      datasets: [
        {
          label: 'Effective BW (Decode)',
          data: ops.map(o => o.effective_bw_gbs),
          backgroundColor: ops.map(o => (tierColors[o.data_tier] || '#888') + '99'),
          borderColor:     ops.map(o => tierColors[o.data_tier] || '#888'),
          borderWidth: 1.5,
          stack: 'a',
        },
        {
          label: 'Ideal BW',
          data: ops.map(o => o.ideal_perf_gflops ? Math.max(o.effective_bw_gbs * o.slowdown_vs_ideal, o.effective_bw_gbs) : 0),
          backgroundColor: '#5b7eff22',
          borderColor:     '#5b7eff',
          borderWidth: 1,
          stack: 'b',
        },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { grid: { color: c.grid }, ticks: { color: c.tick, font: { size: 9 } } },
        y: { grid: { color: c.grid }, ticks: { color: c.tick, font: { size: 10 },
          callback: v => v.toFixed(0) + ' GB/s' } },
      },
      plugins: {
        legend: { labels: { color: c.tick, font: { size: 11 }, boxWidth: 12 } },
        tooltip: { ...darkChartOpts().plugins.tooltip,
          callbacks: { label: i => `${i.dataset.label}: ${i.raw.toFixed(1)} GB/s` } },
      },
    },
  });
}

function renderHeteroTable(placement) {
  const tbody = $('heteroPlacementBody');
  tbody.innerHTML = '';
  const tierClass = { SRAM: 'tier-sram', DRAM: 'tier-dram', Flash: 'tier-flash' };
  for (const p of placement) {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><span class="${tierClass[p.tier] || ''}">${p.tier}</span></td>
      <td>${p.data_type}</td>
      <td>${p.size_gb.toFixed(2)}</td>
      <td>${(p.bandwidth / 1e12).toFixed(2)} TB/s</td>
      <td>${p.energy_pj_per_byte}</td>
    `;
    tbody.appendChild(tr);
  }
}

// ── DSE Explorer Tab ──────────────────────────────────────────────────────────

async function runDSE() {
  if (!state.currentConfig) { showError('Run an analysis first.'); return; }
  const btn = $('runDSEBtn');
  btn.disabled = true; btn.textContent = 'Sweeping…';
  $('dseEmpty').classList.add('hidden');
  $('dseResults').classList.add('hidden');

  const presetName = $('dsePreset').value;
  const region     = $('dseRegion').value || 'Global_Average';
  const maxPts     = parseInt($('dseMaxPoints').value) || 300;
  const dsePresets = state.dsePresets || {};
  const dseParams  = dsePresets[presetName] || dsePresets[Object.keys(dsePresets)[0]];

  try {
    const res = await fetch('/api/dse/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        config: state.currentConfig,
        dse_params: dseParams,
        tco_params: { lifetime_years: 3, utilization: 0.5, electricity_price: 0.10, pue: 1.3 },
        co2_region: region,
        max_points: maxPts,
      }),
    });
    const data = await res.json();
    if (!data.success) { showError(data.error); $('dseEmpty').classList.remove('hidden'); return; }

    const d = data.dse;
    const s = d.stats;

    // Stats
    $('ds-total').textContent    = s.total_points;
    $('ds-fit-pct').textContent  = s.fits_memory_pct.toFixed(1) + '%';
    $('ds-pareto-pc').textContent = s.pareto_pc_size;
    $('ds-best-perf').textContent = d.best.performance.avg_attain_tflops.toFixed(1) + ' TFLOPS';
    $('ds-best-eff').textContent  = d.best.energy.energy_efficiency_gflops_per_w.toFixed(1) + ' GFLOPS/W';
    $('ds-best-tco').textContent  = '$' + d.best.tco.tco_per_eflop_usd.toFixed(1) + '/EFLOP';

    renderDSEParetoChart(d);
    renderDSEParetoEnergyChart(d);
    renderDSEParetoTable(d.pareto_perf_cost);

    $('dseResults').classList.remove('hidden');
  } catch(e) {
    showError('DSE failed: ' + e.message);
    $('dseEmpty').classList.remove('hidden');
  } finally {
    btn.disabled = false; btn.textContent = '🎯 Run DSE';
  }
}

function renderDSEParetoChart(d) {
  const all    = d.points;
  const pareto = new Set(d.pareto_perf_cost.map(p => p.name));

  const allPts    = all.filter(p => !pareto.has(p.name));
  const paretoPts = d.pareto_perf_cost;
  const fitPts    = allPts.filter(p =>  p.model_fits_memory);
  const noFitPts  = allPts.filter(p => !p.model_fits_memory);
  const c = getThemeColors();

  destroyChart('dsePareto');
  state.charts.dsePareto = new Chart($('dseParetoChart').getContext('2d'), {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'All Pts (no fit)',
          data: noFitPts.map(p => ({ x: p.tco_per_eflop_usd, y: p.avg_attain_tflops,
            name: p.name, eff: p.energy_efficiency_gflops_per_w })),
          backgroundColor: '#f8717122',
          borderColor: '#f87171',
          borderWidth: 0, pointRadius: 4,
        },
        {
          label: 'All Pts (fits memory)',
          data: fitPts.map(p => ({ x: p.tco_per_eflop_usd, y: p.avg_attain_tflops,
            name: p.name, eff: p.energy_efficiency_gflops_per_w })),
          backgroundColor: '#5b7eff44',
          borderColor: '#5b7eff',
          borderWidth: 0, pointRadius: 4,
        },
        {
          label: '⭐ Pareto Frontier',
          data: paretoPts.map(p => ({ x: p.tco_per_eflop_usd, y: p.avg_attain_tflops,
            name: p.name, eff: p.energy_efficiency_gflops_per_w })),
          backgroundColor: '#fb923c',
          borderColor: '#fb923c',
          borderWidth: 2, pointRadius: 8,
          showLine: true,
          borderDash: [],
          tension: 0.1,
          type: 'line',
          fill: false,
          order: 0,
        },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: {
          type: 'logarithmic',
          title: { display: true, text: 'TCO per EFLOP (USD) — lower is better', color: c.tick },
          grid: { color: c.grid },
          ticks: { color: c.tick, font: { size: 10 }, callback: v => '$' + v.toFixed(0) },
        },
        y: {
          title: { display: true, text: 'Attainable Performance (TFLOPS) — higher is better', color: c.tick },
          grid: { color: c.grid },
          ticks: { color: c.tick, font: { size: 10 } },
        },
      },
      plugins: {
        legend: { labels: { color: c.tick, font: { size: 11 }, boxWidth: 12 } },
        tooltip: { ...darkChartOpts().plugins.tooltip,
          callbacks: {
            label: i => [i.raw.name, `TCO: $${i.raw.x?.toFixed(2)}/EFLOP`,
              `Perf: ${i.raw.y?.toFixed(2)} TFLOPS`, `Eff: ${i.raw.eff?.toFixed(1)} GFLOPS/W`],
          },
        },
      },
    },
  });
}

function renderDSEParetoEnergyChart(d) {
  const all    = d.points;
  const pareto = new Set(d.pareto_perf_energy.map(p => p.name));
  const paretoPts = d.pareto_perf_energy;
  const otherPts  = all.filter(p => !pareto.has(p.name));
  const c = getThemeColors();

  destroyChart('dseParetoEnergy');
  state.charts.dseParetoEnergy = new Chart($('dseParetoEnergyChart').getContext('2d'), {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'Other points',
          data: otherPts.map(p => ({ x: p.energy_efficiency_gflops_per_w, y: p.avg_attain_tflops,
            name: p.name })),
          backgroundColor: '#5b7eff33',
          borderColor: '#5b7eff',
          borderWidth: 0, pointRadius: 3,
        },
        {
          label: '⭐ Pareto Frontier',
          data: paretoPts.map(p => ({ x: p.energy_efficiency_gflops_per_w, y: p.avg_attain_tflops,
            name: p.name })),
          backgroundColor: '#4ade80',
          borderColor: '#4ade80',
          borderWidth: 2, pointRadius: 7,
          showLine: true, type: 'line', fill: false, tension: 0.1, order: 0,
        },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: {
          title: { display: true, text: 'Energy Efficiency (GFLOPS/W)', color: c.tick },
          grid: { color: c.grid },
          ticks: { color: c.tick, font: { size: 10 } },
        },
        y: {
          title: { display: true, text: 'Performance (TFLOPS)', color: c.tick },
          grid: { color: c.grid },
          ticks: { color: c.tick, font: { size: 10 } },
        },
      },
      plugins: {
        legend: { labels: { color: c.tick, font: { size: 11 }, boxWidth: 12 } },
        tooltip: { ...darkChartOpts().plugins.tooltip,
          callbacks: { label: i => [i.raw.name,
            `Eff: ${i.raw.x?.toFixed(2)} GFLOPS/W`, `Perf: ${i.raw.y?.toFixed(2)} TFLOPS`] },
        },
      },
    },
  });
}

function renderDSEParetoTable(pareto) {
  const tbody = $('dseParetoBody');
  tbody.innerHTML = '';
  for (const [i, p] of pareto.entries()) {
    const tr = document.createElement('tr');
    if (i === 0) tr.classList.add('pareto-row');
    tr.innerHTML = `
      <td>${i === 0 ? '<span class="pareto-star">⭐</span> ' : ''}${p.name}</td>
      <td>${p.peak_performance_tflops.toFixed(1)}</td>
      <td>${p.memory_bandwidth_gbs.toFixed(0)}</td>
      <td>${p.memory_capacity_gb.toFixed(0)}</td>
      <td>${p.tdp_w}</td>
      <td>$${p.cost_usd.toLocaleString()}</td>
      <td>${p.avg_attain_tflops.toFixed(2)}</td>
      <td>${p.energy_efficiency_gflops_per_w.toFixed(2)}</td>
      <td>$${p.tco_per_eflop_usd.toFixed(2)}</td>
      <td>${p.co2e_per_eflop_g.toFixed(1)}</td>
      <td class="${p.model_fits_memory ? 'fits-yes' : 'fits-no'}">${p.model_fits_memory ? '✅' : '❌'}</td>
    `;
    tbody.appendChild(tr);
  }
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

// ── Tier Config Helpers ────────────────────────────────────────────────────────

function fillTierConfig(tiers) {
  if (tiers.SRAM) {
    $('tier_sram_bw').value = (tiers.SRAM.bandwidth / 1e12).toFixed(2);
    $('tier_sram_cap').value = (tiers.SRAM.capacity / 1e9).toFixed(3);
    $('tier_sram_energy').value = tiers.SRAM.energy_per_byte_pj;
    $('tier_sram_latency').value = tiers.SRAM.latency_ns;
  }
  if (tiers.DRAM) {
    $('tier_dram_bw').value = (tiers.DRAM.bandwidth / 1e9).toFixed(0);
    $('tier_dram_cap').value = (tiers.DRAM.capacity / 1e9).toFixed(1);
    $('tier_dram_energy').value = tiers.DRAM.energy_per_byte_pj;
    $('tier_dram_latency').value = tiers.DRAM.latency_ns;
  }
  if (tiers.Flash) {
    $('tier_flash_bw').value = (tiers.Flash.bandwidth / 1e9).toFixed(1);
    $('tier_flash_cap').value = (tiers.Flash.capacity / 1e9).toFixed(0);
    $('tier_flash_energy').value = tiers.Flash.energy_per_byte_pj;
    $('tier_flash_latency').value = tiers.Flash.latency_ns;
  }
}

function readTierConfig() {
  return {
    SRAM: {
      bandwidth: parseFloat($('tier_sram_bw').value) * 1e12,
      capacity: parseFloat($('tier_sram_cap').value) * 1e9,
      energy_per_byte_pj: parseFloat($('tier_sram_energy').value),
      latency_ns: parseFloat($('tier_sram_latency').value),
    },
    DRAM: {
      bandwidth: parseFloat($('tier_dram_bw').value) * 1e9,
      capacity: parseFloat($('tier_dram_cap').value) * 1e9,
      energy_per_byte_pj: parseFloat($('tier_dram_energy').value),
      latency_ns: parseFloat($('tier_dram_latency').value),
    },
    Flash: {
      bandwidth: parseFloat($('tier_flash_bw').value) * 1e9,
      capacity: parseFloat($('tier_flash_cap').value) * 1e9,
      energy_per_byte_pj: parseFloat($('tier_flash_energy').value),
      latency_ns: parseFloat($('tier_flash_latency').value),
    },
  };
}

// ── Multi-Chip Parallelism Tab ────────────────────────────────────────────────

async function runParallelAnalysis() {
  if (!state.currentConfig) { showError('Run an analysis first.'); return; }
  const hwKey = $('parallelHW').value;
  if (!hwKey) { showError('Select a multi-chip hardware.'); return; }

  const btn = $('runParallelBtn');
  btn.disabled = true; btn.textContent = 'Analyzing…';
  $('parallelEmpty').classList.add('hidden');
  $('parallelResults').classList.add('hidden');

  try {
    const res = await fetch('/api/parallelism', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        config: state.currentConfig,
        hardware_key: hwKey,
        tp_degree: parseInt($('parTP').value) || 8,
        pp_degree: parseInt($('parPP').value) || 1,
        dp_degree: parseInt($('parDP').value) || 1,
        inter_chip_bw_gbs: parseFloat($('parBW').value) || 900,
        topology: $('parTopology').value || 'all_to_all',
      }),
    });
    const data = await res.json();
    if (!data.success) { showError(data.error); $('parallelEmpty').classList.remove('hidden'); return; }

    const p = data.parallelism;
    const s = p.summary;
    const mem = p.per_device_memory;
    const tp = p.throughput;

    // Stats
    $('ps-strategy').textContent = s.strategy;
    $('ps-tps').textContent = tp.decode_tokens_per_sec.toFixed(2) + ' tok/s';
    $('ps-speedup').textContent = tp.speedup.toFixed(2) + 'x';
    $('ps-efficiency').textContent = tp.parallel_efficiency_pct.toFixed(1) + '%';
    $('ps-overhead').textContent = tp.comm_overhead_pct.toFixed(1) + '%';
    $('ps-weight-dev').textContent = mem.weights_gb.toFixed(2) + ' GB';
    $('ps-kv-dev').textContent = mem.kv_cache_gb.toFixed(2) + ' GB';
    $('ps-fits').innerHTML = mem.fits ?
      '<span class="fits-yes">Yes (' + mem.utilization_pct.toFixed(0) + '%)</span>' :
      '<span class="fits-no">No (' + mem.utilization_pct.toFixed(0) + '%)</span>';

    renderParallelMemChart(mem);
    renderParallelScalingChart(tp.scaling);
    renderParallelCommChart(tp);
    renderParallelEffChart(tp.scaling);
    renderParallelShardTable(p.sharding, mem);

    $('parallelResults').classList.remove('hidden');
  } catch(e) {
    showError('Parallelism analysis failed: ' + e.message);
    $('parallelEmpty').classList.remove('hidden');
  } finally {
    btn.disabled = false; btn.textContent = 'Analyze Sharding';
  }
}

function renderParallelMemChart(mem) {
  const c = getThemeColors();
  destroyChart('parallelMem');
  state.charts.parallelMem = new Chart($('parallelMemChart').getContext('2d'), {
    type: 'doughnut',
    data: {
      labels: ['Weights', 'KV Cache', 'Activations', 'Free'],
      datasets: [{
        data: [mem.weights_gb, mem.kv_cache_gb, mem.activations_gb,
               Math.max(0, mem.device_capacity_gb - mem.total_per_device_gb)],
        backgroundColor: ['#5b7eff99', '#22d3ee99', '#4ade8099', '#2a2f4555'],
        borderColor:     ['#5b7eff',   '#22d3ee',   '#4ade80',   '#333a52'],
        borderWidth: 2,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { position: 'bottom', labels: { color: c.tick, font: { size: 11 }, boxWidth: 14 } },
        tooltip: { ...darkChartOpts().plugins.tooltip,
          callbacks: { label: i => `${i.label}: ${i.raw.toFixed(2)} GB` } },
      },
    },
  });
}

function renderParallelScalingChart(scaling) {
  const c = getThemeColors();
  destroyChart('parallelScaling');
  state.charts.parallelScaling = new Chart($('parallelScalingChart').getContext('2d'), {
    type: 'line',
    data: {
      labels: scaling.map(s => s.devices),
      datasets: [
        {
          label: 'Actual Tokens/s',
          data: scaling.map(s => s.tokens_per_sec),
          borderColor: '#5b7eff',
          backgroundColor: '#5b7eff33',
          fill: true,
          tension: 0.2,
          pointRadius: 4,
        },
        {
          label: 'Ideal Scaling',
          data: scaling.map(s => scaling[0].tokens_per_sec * s.devices),
          borderColor: '#4ade8055',
          borderDash: [6, 4],
          pointRadius: 0,
          fill: false,
        },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { title: { display: true, text: 'Number of Devices', color: c.tick },
             grid: { color: c.grid }, ticks: { color: c.tick } },
        y: { title: { display: true, text: 'Tokens/s (Decode)', color: c.tick },
             grid: { color: c.grid }, ticks: { color: c.tick } },
      },
      plugins: {
        legend: { labels: { color: c.tick, font: { size: 11 }, boxWidth: 12 } },
        tooltip: { ...darkChartOpts().plugins.tooltip,
          callbacks: { label: i => `${i.dataset.label}: ${i.raw.toFixed(2)} tok/s` } },
      },
    },
  });
}

function renderParallelCommChart(tp) {
  const c = getThemeColors();
  destroyChart('parallelComm');
  state.charts.parallelComm = new Chart($('parallelCommChart').getContext('2d'), {
    type: 'bar',
    data: {
      labels: ['Per Layer'],
      datasets: [
        { label: 'Compute (\u00b5s)', data: [tp.compute_time_per_layer_us],
          backgroundColor: '#5b7eff99', borderColor: '#5b7eff', borderWidth: 1.5 },
        { label: 'Communication (\u00b5s)', data: [tp.comm_time_per_layer_us],
          backgroundColor: '#fb923c99', borderColor: '#fb923c', borderWidth: 1.5 },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { grid: { color: c.grid }, ticks: { color: c.tick } },
        y: { grid: { color: c.grid },
             ticks: { color: c.tick, callback: v => v.toFixed(0) + ' \u00b5s' },
             title: { display: true, text: 'Time per Layer', color: c.tick } },
      },
      plugins: {
        legend: { labels: { color: c.tick, font: { size: 11 }, boxWidth: 12 } },
        tooltip: { ...darkChartOpts().plugins.tooltip,
          callbacks: { label: i => `${i.dataset.label}: ${i.raw.toFixed(1)} \u00b5s` } },
      },
    },
  });
}

function renderParallelEffChart(scaling) {
  const c = getThemeColors();
  destroyChart('parallelEff');
  state.charts.parallelEff = new Chart($('parallelEffChart').getContext('2d'), {
    type: 'line',
    data: {
      labels: scaling.map(s => s.devices),
      datasets: [
        {
          label: 'Parallel Efficiency',
          data: scaling.map(s => s.efficiency_pct),
          borderColor: '#22d3ee',
          backgroundColor: '#22d3ee33',
          fill: true,
          tension: 0.2,
          pointRadius: 4,
        },
        {
          label: 'Ideal (100%)',
          data: scaling.map(() => 100),
          borderColor: '#4ade8055',
          borderDash: [6, 4],
          pointRadius: 0,
          fill: false,
        },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { title: { display: true, text: 'Number of Devices', color: c.tick },
             grid: { color: c.grid }, ticks: { color: c.tick } },
        y: { title: { display: true, text: 'Efficiency (%)', color: c.tick },
             grid: { color: c.grid }, ticks: { color: c.tick },
             min: 0, max: 110 },
      },
      plugins: {
        legend: { labels: { color: c.tick, font: { size: 11 }, boxWidth: 12 } },
        tooltip: { ...darkChartOpts().plugins.tooltip,
          callbacks: { label: i => `${i.dataset.label}: ${i.raw.toFixed(1)}%` } },
      },
    },
  });
}

function renderParallelShardTable(sharding, mem) {
  const tbody = $('parallelShardBody');
  tbody.innerHTML = '';
  const rows = [
    { type: 'Weights', total: fmt.bytes(sharding.total_weight_bytes),
      perDev: sharding.weight_per_device_gb.toFixed(2) + ' GB',
      shard: `Split by TP=${sharding.strategy.split(',')[0].split('=')[1]} and PP` },
    { type: 'KV Cache', total: '—',
      perDev: mem.kv_cache_gb.toFixed(2) + ' GB',
      shard: `KV heads/device: ${sharding.kv_heads_per_device}` },
    { type: 'Activations', total: '—',
      perDev: mem.activations_gb.toFixed(2) + ' GB',
      shard: 'Per micro-batch' },
  ];
  for (const r of rows) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td style="font-weight:500">${r.type}</td><td>${r.total}</td><td>${r.perDev}</td><td class="note-cell">${r.shard}</td>`;
    tbody.appendChild(tr);
  }
}

// ── Speculative Decoding Analysis ────────────────────────────────────────────

async function runSpeculativeAnalysis() {
  if (!state.currentConfig) { showError('Run a model analysis first (sidebar).'); return; }
  const hwKey = $('specHW')?.value;
  if (!hwKey) { showError('Select a hardware platform.'); return; }
  const draftName = $('specDraftModel')?.value;
  if (!draftName) { showError('Select a draft model.'); return; }

  const draftModel = state.draftModels.find(d => d.name === draftName);
  if (!draftModel) { showError('Draft model not found.'); return; }

  const gamma = parseInt($('specGamma')?.value || '5');
  const alpha = parseFloat($('specAlpha')?.value || '0.80');

  const btn = $('runSpecBtn');
  if (btn) { btn.disabled = true; btn.textContent = 'Analyzing…'; }
  try {
    const cfg = buildConfig();
    const res = await fetch('/api/speculative', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        config: cfg,
        draft_config: draftModel.config,
        hardware_key: hwKey,
        gamma, alpha,
      }),
    });
    const data = await res.json();
    if (!data.success) throw new Error(data.error || 'Speculative analysis failed');

    const s = data.speculative;
    // Stats
    $('sp-speedup').textContent = s.speedup.toFixed(2) + '×';
    $('sp-tps-vanilla').textContent = s.tokens_per_sec_vanilla.toFixed(1);
    $('sp-tps-spec').textContent = s.tokens_per_sec_speculative.toFixed(1);
    $('sp-expected').textContent = s.expected_tokens_per_step.toFixed(2);
    $('sp-step-lat').textContent = s.step_latency_ms.toFixed(2) + ' ms';
    $('sp-mem-overhead').textContent = '+' + s.memory.overhead_gb.toFixed(2) + ' GB (' + s.memory.overhead_pct.toFixed(0) + '%)';
    $('sp-energy-ratio').textContent = s.energy.energy_ratio.toFixed(2) + '×';

    renderSpecSpeedupChart(s.sweep);
    renderSpecMemChart(s.memory);
    renderSpecEnergyChart(s.energy, s.target_summary, s.draft_summary);
    renderSpecAcceptChart(s.sweep);
    renderSpecCompTable(s);

    $('specResults').classList.remove('hidden');
    $('specEmpty').classList.add('hidden');
  } catch (e) {
    showError(e.message);
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = 'Analyze Speculative Decoding'; }
  }
}

function renderSpecSpeedupChart(sweep) {
  destroyChart('specSpeedup');
  const tc = getThemeColors();
  const colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899'];
  const datasets = sweep.curves.map((c, i) => ({
    label: `α = ${c.alpha}`,
    data: c.points.map(p => ({ x: p.gamma, y: p.speedup })),
    borderColor: colors[i % colors.length],
    backgroundColor: colors[i % colors.length] + '22',
    borderWidth: 2,
    tension: 0.3,
    fill: false,
    pointRadius: 3,
  }));
  state.charts.specSpeedup = new Chart($('specSpeedupChart').getContext('2d'), {
    type: 'line',
    data: { datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { type: 'linear', title: { display: true, text: 'Speculation Length (γ)', color: tc.text },
             ticks: { color: tc.text, stepSize: 1 }, grid: { color: tc.grid } },
        y: { title: { display: true, text: 'Speedup ×', color: tc.text },
             ticks: { color: tc.text }, grid: { color: tc.grid }, min: 0 },
      },
      plugins: {
        legend: { labels: { color: tc.text } },
        tooltip: { callbacks: { label: ctx => `α=${sweep.curves[ctx.datasetIndex].alpha}: ${ctx.parsed.y.toFixed(2)}×` } },
      },
    },
  });
}

function renderSpecMemChart(mem) {
  destroyChart('specMem');
  const tc = getThemeColors();
  state.charts.specMem = new Chart($('specMemChart').getContext('2d'), {
    type: 'bar',
    data: {
      labels: ['Vanilla (Target Only)', 'Speculative (Target+Draft)'],
      datasets: [
        { label: 'Model Weights', data: [mem.target_model_gb, mem.target_model_gb + mem.draft_model_gb],
          backgroundColor: ['#3b82f6', '#3b82f6'] },
        { label: 'KV Cache', data: [mem.target_kv_gb, mem.target_kv_gb + mem.draft_kv_gb],
          backgroundColor: ['#10b981', '#10b981'] },
        { label: 'Draft Overhead', data: [0, mem.overhead_gb],
          backgroundColor: ['transparent', '#ef444488'] },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { stacked: true, ticks: { color: tc.text }, grid: { color: tc.grid } },
        y: { stacked: true, title: { display: true, text: 'Memory (GB)', color: tc.text },
             ticks: { color: tc.text }, grid: { color: tc.grid } },
      },
      plugins: { legend: { labels: { color: tc.text } } },
    },
  });
}

function renderSpecEnergyChart(energy, target, draft) {
  destroyChart('specEnergy');
  const tc = getThemeColors();
  const vanillaJ = energy.vanilla_j_per_token * 1000; // mJ
  const specJ = energy.speculative_j_per_token * 1000;
  state.charts.specEnergy = new Chart($('specEnergyChart').getContext('2d'), {
    type: 'bar',
    data: {
      labels: ['Vanilla Decode', 'Speculative Decode'],
      datasets: [{
        label: 'Energy per Token (mJ)',
        data: [vanillaJ, specJ],
        backgroundColor: [vanillaJ <= specJ ? '#10b981' : '#ef4444', specJ <= vanillaJ ? '#10b981' : '#f59e0b'],
        borderWidth: 1,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      indexAxis: 'y',
      scales: {
        x: { title: { display: true, text: 'Energy per Token (mJ)', color: tc.text },
             ticks: { color: tc.text }, grid: { color: tc.grid } },
        y: { ticks: { color: tc.text }, grid: { color: tc.grid } },
      },
      plugins: { legend: { display: false } },
    },
  });
}

function renderSpecAcceptChart(sweep) {
  destroyChart('specAccept');
  const tc = getThemeColors();
  const colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899'];
  const datasets = sweep.curves.map((c, i) => ({
    label: `α = ${c.alpha}`,
    data: c.points.map(p => ({ x: p.gamma, y: p.expected_tokens })),
    borderColor: colors[i % colors.length],
    borderWidth: 2,
    tension: 0.3,
    fill: false,
    pointRadius: 3,
  }));
  state.charts.specAccept = new Chart($('specAcceptChart').getContext('2d'), {
    type: 'line',
    data: { datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { type: 'linear', title: { display: true, text: 'Speculation Length (γ)', color: tc.text },
             ticks: { color: tc.text, stepSize: 1 }, grid: { color: tc.grid } },
        y: { title: { display: true, text: 'E[tokens/step]', color: tc.text },
             ticks: { color: tc.text }, grid: { color: tc.grid }, min: 0 },
      },
      plugins: { legend: { labels: { color: tc.text } } },
    },
  });
}

function renderSpecCompTable(s) {
  const tbody = $('specCompBody');
  if (!tbody) return;
  tbody.innerHTML = '';
  const fmt = (v) => typeof v === 'number' ? (v >= 1e6 ? (v/1e9).toFixed(2)+'B' : v >= 1e3 ? (v/1e6).toFixed(2)+'M' : v.toFixed(2)) : v;
  const rows = [
    { m: 'Parameters', t: fmt(s.target_summary.total_params), d: fmt(s.draft_summary.total_params), sp: '—' },
    { m: 'Model Size (GB)', t: s.target_summary.model_size_gb.toFixed(2), d: s.draft_summary.model_size_gb.toFixed(2), sp: s.memory.total_speculative_gb.toFixed(2) },
    { m: 'Decode Latency (ms)', t: s.target_summary.decode_latency_ms.toFixed(2), d: s.draft_summary.decode_latency_ms.toFixed(2), sp: s.step_latency_ms.toFixed(2) + ' /step' },
    { m: 'Tokens/s', t: s.tokens_per_sec_vanilla.toFixed(1), d: '—', sp: s.tokens_per_sec_speculative.toFixed(1) },
    { m: 'Speedup', t: '1.00×', d: '—', sp: s.speedup.toFixed(2) + '×' },
    { m: 'Energy/Token (mJ)', t: (s.energy.vanilla_j_per_token*1000).toFixed(2), d: '—', sp: (s.energy.speculative_j_per_token*1000).toFixed(2) },
    { m: 'Memory Overhead', t: '0 GB', d: '—', sp: '+' + s.memory.overhead_gb.toFixed(2) + ' GB (' + s.memory.overhead_pct.toFixed(0) + '%)' },
  ];
  for (const r of rows) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td style="font-weight:500">${r.m}</td><td>${r.t}</td><td>${r.d}</td><td>${r.sp}</td>`;
    tbody.appendChild(tr);
  }
}

// ── Bootstrap ──────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', init);
