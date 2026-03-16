# LLM-Para: Transformer Computation & Roofline Analyzer

<div align="center">

### 🌐 [**llm-para.onrender.com**](https://llm-para.onrender.com) — Live Demo, No Install Needed

[![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-llm--para.onrender.com-5b7eff?style=for-the-badge)](https://llm-para.onrender.com)

</div>

---

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Flask](https://img.shields.io/badge/backend-Flask-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Web](https://img.shields.io/badge/web-Chart.js-ff6384.svg)](https://www.chartjs.org/)

A comprehensive **web-based** toolchain for analyzing **computation complexity**, **memory access patterns**, and **hardware performance bottlenecks** in large Transformer-based language models. Features an interactive Roofline model visualization and detailed per-operator breakdown for the complete inference pipeline.

> **Try it now →** https://llm-para.onrender.com

## 🚀 Features

### 📊 Comprehensive Operator Coverage
Beyond standard Transformer analysis, LLM-Para models every operator in the inference pipeline:

| Operator | Description |
|---|---|
| **Token Embedding** | Vocabulary lookup memory cost |
| **RMSNorm / LayerNorm** | Pre/post attention and FFN normalization |
| **Q / K / V Projection** | Linear projections with GQA support |
| **RoPE-Q / RoPE-K** | Rotary Position Embedding FLOPs |
| **Q×Kᵀ (Score)** | Attention score computation |
| **Softmax** | Online softmax with exp/sum/div |
| **Attn×V** | Attention output aggregation |
| **FlashAttention** | Fused kernel with reduced memory traffic |
| **O Projection** | Output attention projection |
| **MoE Router** | Expert routing gate computation |
| **MoE FFN-Up+Gate** | Sparse MoE SwiGLU up-projection |
| **MoE FFN-Down** | Sparse MoE down-projection |
| **FFN-Up+Gate (SwiGLU)** | Dense gated feed-forward |
| **FFN-Down** | Dense FFN down-projection |
| **MLA (DeepSeek)** | Multi-head Latent Attention KV compression |
| **LM Head** | Final vocabulary projection |

### 🏗️ Architecture Support
- ✅ **Grouped Query Attention (GQA)** — LLaMA-3, Mistral, Qwen2
- ✅ **Mixture of Experts (MoE)** — Mixtral, Qwen2-MoE, DeepSeek
- ✅ **Multi-head Latent Attention (MLA)** — DeepSeek-V2, DeepSeek-R1
- ✅ **Rotary Position Encoding (RoPE)** — with theta and scaling factor
- ✅ **SwiGLU / Gated FFN** — LLaMA, Mistral, Gemma
- ✅ **FlashAttention** — fused kernel memory analysis
- ✅ **RMSNorm / LayerNorm** — configurable norm type
- ✅ **Quantization** — per-component bit-widths (2/4/8/16/32-bit)

### 🖥️ Hardware Platform Library
| Category | Platforms |
|---|---|
| **NVIDIA GPU** | H100 SXM/PCIe, A100 (40/80GB), A10, RTX 4090/4080 |
| **AMD GPU** | MI300X, MI250X |
| **Apple Silicon** | M3 Ultra, M2 Ultra, M2 Max |
| **Intel** | Gaudi 3, Xeon Platinum 8480+ |
| **Mobile NPU** | Snapdragon 8 Gen 3/2, Dimensity 9300 |
| **PIM** | DRAM-PIM (HBM-PIM), NAND-PIM (HILOS), SRAM-PIM |

### 🌐 Web Interface
- Real-time computation with interactive parameter controls
- Interactive Roofline model chart (log-log scale, per operator)
- Per-category FLOPs and memory breakdown charts
- Arithmetic intensity scatter visualization
- KV cache timeline analysis
- Per-operator memory decomposition (input/weight/output)
- CSV and JSON export

## 📦 Installation

```bash
git clone https://github.com/dengls24/LLM-para.git
cd LLM-para
pip install -r requirements.txt
```

## 🚀 Quick Start

### Web Interface
```bash
python app.py
# Open http://localhost:5000
```

### Command Line (batch analysis)
```bash
python cli.py --model "LLaMA-3 8B" --hardware "NVIDIA H100 SXM" --output results.csv
```

### Python API
```python
from analyzer import LLMAnalyzer

config = {
    "hidden_size": 4096,
    "num_heads": 32,
    "num_key_value_heads": 8,       # GQA
    "num_layers": 32,
    "intermediate_size": 14336,
    "vocab_size": 128256,
    "seq_len": 2048,
    "batch_size": 1,
    "max_gen_len": 4096,
    "use_gate_ffn": True,           # SwiGLU
    "use_rmsnorm": True,
    "rope_theta": 500000.0,         # RoPE
    "rope_scaling_factor": 1.0,
    "quant_config": {
        "activation": 16,
        "weight_attn": 16,
        "weight_ffn": 16,
        "kv_cache": 16,
        "rope_bit": 32,
    },
}

analyzer = LLMAnalyzer(config)
results = analyzer.analyze()
summary = analyzer.get_summary(results)

print(f"Total FLOPs: {summary['total_flops']:.2e}")
print(f"Parameters:  {summary['total_params']:.2e}")
print(f"Model Size:  {summary['model_size_gb']:.2f} GB")
print(f"KV Cache:    {summary['kv_max_mb']:.0f} MB (max)")

# Roofline analysis for H100
from configs import HARDWARE_CONFIGS
hw = HARDWARE_CONFIGS["NVIDIA H100 SXM"]
roofline = analyzer.get_roofline_data(results, hw)
print(f"Ridge Point: {roofline['ridge_point']:.1f} FLOP/B")
```

### MoE Example (Mixtral 8x7B)
```python
config = {
    "hidden_size": 4096,
    "num_heads": 32,
    "num_key_value_heads": 8,
    "num_layers": 32,
    "intermediate_size": 14336,
    "num_experts_per_tok": 2,        # Top-2 routing
    "num_local_experts": 8,          # 8 total experts
    "use_gate_ffn": True,
    "rope_theta": 1000000.0,
    "quant_config": {"activation": 16, "weight_attn": 16,
                     "weight_ffn": 16, "kv_cache": 16, "rope_bit": 32},
    # ... other params
}
```

### MLA Example (DeepSeek-V2)
```python
config = {
    "use_mla": True,
    "mla_kv_lora_rank": 512,
    "mla_q_lora_rank": 1536,
    "mla_qk_nope_head_dim": 128,
    "mla_qk_rope_head_dim": 64,
    "mla_v_head_dim": 128,
    # ... other params
}
```

## 📁 Project Structure

```
LLM-para/
├── app.py              # Flask web server & REST API
├── analyzer.py         # Core analysis engine (all operators)
├── configs.py          # Model & hardware preset configs
├── cli.py              # Command-line interface
├── requirements.txt
├── README.md
└── static/
    ├── index.html      # Web UI
    ├── css/
    │   └── style.css   # Dark theme stylesheet
    └── js/
        └── app.js      # Frontend application logic
```

## 🔌 REST API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/models` | List preset model configs |
| `GET` | `/api/hardware` | List hardware platforms |
| `POST` | `/api/analyze` | Run full analysis |
| `POST` | `/api/roofline` | Get roofline data for given HW |
| `POST` | `/api/export/csv` | Download results as CSV |
| `POST` | `/api/export/json` | Download results as JSON |
| `POST` | `/api/compare` | Compare multiple model configs |

## 📊 Preset Model Library

| Family | Models |
|---|---|
| GPT-2 | Small (117M), XL (1.5B) |
| LLaMA-2 | 7B, 13B, 70B |
| LLaMA-3 | 8B, 70B, 405B |
| Mixtral | 8x7B, 8x22B |
| Qwen2 | 7B, 72B, MoE 57B-A14B |
| DeepSeek | V2 (MLA+MoE), R1 671B |
| Gemma-2 | 9B, 27B |
| Phi-3 | Mini (3.8B) |
| BitNet | b1.58 3B |

## 🔬 Methodology

### FLOP Counting
For a standard Transformer layer (per layer, per token in decode):

| Operation | FLOPs |
|---|---|
| Q/K/V Proj | `2 × b × s × h × h` (K/V scaled by GQA ratio) |
| RoPE | `4 × b × s × n × d/2` per Q/K |
| Q×Kᵀ | `2 × b × n × s × ctx × d` |
| Softmax | `5 × b × n × s × ctx` |
| Attn×V | `2 × b × n × s × ctx × d` |
| O Proj | `2 × b × s × h × h` |
| FFN SwiGLU | `2 × b × s × h × ffn_size × 2 + b × s × ffn_size` |
| RMSNorm | `4 × b × s × h` per norm |

### Memory Model
Memory traffic = input bytes + weight bytes + output bytes, with bit-width per component.
Arithmetic Intensity = FLOPs / Total Bytes.

### Roofline
Attainable performance = `min(intensity × BW, peak_FLOP/s)`.
Memory-bound: `intensity < ridge_point = peak / BW`.
Compute-bound: `intensity ≥ ridge_point`.

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

## 📚 Citation

```bibtex
@software{llm_para_2024,
  title={LLM-Para: Transformer Computation \& Roofline Analyzer},
  author={dengls24},
  year={2024},
  url={https://github.com/dengls24/LLM-para},
  note={Comprehensive per-operator FLOPs and Roofline analysis for LLM inference}
}
```

## 🔗 Related Work

- [Roofline Model](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf)
- [LLM-Viewer](https://github.com/hahnyuan/LLM-Viewer)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [DeepSeek-V2 MLA](https://arxiv.org/abs/2405.04434)
