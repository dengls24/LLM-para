# LLM-para: Transformer Computation & Roofline Analyzer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)

A comprehensive toolchain for analyzing **computation complexity**, **memory access patterns**, and **performance bottlenecks** in large Transformer-based language models. This project provides detailed FLOP analysis and Roofline model visualization to help optimize LLM inference performance.

## 🚀 Key Features

### 📊 Transformer Analysis (`all.py`)
- **Multi-phase Analysis**: Comprehensive FLOP and memory analysis for:
  - **Prefill Phase**: Processing input sequences
  - **Decode Phase**: Generating first token
  - **Decode Last Phase**: Generating final token with full context
- **Advanced Architecture Support**:
  - ✅ **Grouped Query Attention (GQA)**: Efficient attention with fewer KV heads
  - ✅ **Mixture of Experts (MoE)**: Sparse expert routing and computation
  - ✅ **Rotary Position Embedding (RoPE)**: With scaling factor support
  - ✅ **Gated Feed-Forward Networks**: SwiGLU and similar activations
  - ✅ **Flexible Quantization**: Configurable bit-widths for different components
- **Detailed Metrics**:
  - Operation-level FLOP counting
  - Memory bandwidth requirements
  - Computational density (Op/Byte) analysis
  - Parameter count tracking
- **Output Formats**:
  - CSV reports with detailed metrics
  - Visualization charts for performance analysis

### 🎯 Roofline Visualization (`roofline_model.py`)
- **Advanced Roofline Plots**: 
  - Broken x-axis design for clear memory-bound vs compute-bound visualization
  - Multiple hardware configuration support
  - Custom markers for different operations and phases
- **Hardware Configurations**:
  - NVIDIA A100, H100
  - Smartphone NPUs (Snapdragon 8 Gen 2)
  - Processing-in-Memory (DRAM-PIM, NAND-PIM)
  - Intel Xeon, Apple M2 Ultra, AMD MI250X
- **Visualization Features**:
  - Separate legend generation
  - Operation type and phase differentiation
  - Ridge point and performance ceiling annotations

---

## 📦 Installation

### Requirements
```bash
pip install numpy pandas matplotlib
```

### Quick Setup
```bash
git clone <repository-url>
cd cal_para
pip install -r requirements.txt  # if available
```

---

## 🔧 Usage

### 1. Transformer Model Analysis

```python
from all import compute_and_plot_transformer_analysis

# Example: Analyze LLaMA-3 8B model
compute_and_plot_transformer_analysis(
    hidden_size=4096,
    num_heads=32,
    seq_len=2048,
    batch_size=1,
    num_layers=32,
    intermediate_size=11008,
    num_key_value_heads=8,  # GQA configuration
    quant_config={
        "activation": 16,
        "weight_attn": 16, 
        "weight_ffn": 16,
        "kv_cache": 16,
        "rope_bit": 32
    },
    use_rope=True,
    rope_theta=500000.0,
    output_csv_path="llama3_8b_analysis.csv"
)
```

### 2. MoE Model Analysis (Mixtral-8x7B)

```python
# Example: Analyze Mixtral-8x7B model
compute_and_plot_transformer_analysis(
    hidden_size=4096,
    num_heads=32,
    seq_len=2048,
    batch_size=1,
    num_layers=32,
    intermediate_size=14336,
    num_key_value_heads=8,
    num_experts_per_tok=2,  # MoE: 2 experts per token
    num_local_experts=8,    # MoE: 8 total experts
    quant_config={
        "activation": 16,
        "weight_attn": 16,
        "weight_ffn": 16, 
        "kv_cache": 16,
        "rope_bit": 32
    },
    use_rope=True,
    rope_theta=1000000.0,
    output_csv_path="mixtral_8x7b_analysis.csv"
)
```

### 3. Roofline Model Visualization

```python
from roofline_model import analyze_from_csv

# Define hardware configuration
hardware_config = {
    'name': 'NVIDIA A100',
    'peak_performance': 19.5e12,  # FLOP/s (FP32)
    'memory_bandwidth': 1.555e12,  # Byte/s (HBM2)
}

# Generate Roofline plot
analyze_from_csv(
    csv_path="llama3_8b_analysis.csv",
    hardware_config=hardware_config,
    output_path="llama3_roofline.png"
)
```

---

## 📋 Supported Models & Architectures

| Model Family | GQA | MoE | RoPE | Gated FFN | Status |
|--------------|-----|-----|------|-----------|--------|
| **LLaMA/LLaMA-2** | ❌ | ❌ | ✅ | ✅ | ✅ Supported |
| **LLaMA-3** | ✅ | ❌ | ✅ | ✅ | ✅ Supported |
| **Mixtral-8x7B** | ✅ | ✅ | ✅ | ✅ | ✅ Supported |
| **BitNet** | ✅ | ❌ | ✅ | ✅ | ✅ Supported |
| **Custom Models** | ✅ | ✅ | ✅ | ✅ | ✅ Configurable |

---

## 🖥️ Hardware Configurations

The tool supports analysis for various hardware platforms:

| Hardware | Peak Performance | Memory Bandwidth | Use Case |
|----------|------------------|------------------|----------|
| **NVIDIA A100** | 19.5 TFLOP/s | 1.555 TB/s | Data Center |
| **NVIDIA H100** | 67 TFLOP/s | 3.35 TB/s | High-Performance |
| **Snapdragon 8 Gen 2** | 4.35 TOPS | 51.2 GB/s | Mobile/Edge |
| **DRAM-PIM** | 0.8 TFLOP/s | 0.8 TB/s | Near-Memory Computing |
| **NAND-PIM** | 0.2 TFLOP/s | 0.2 TB/s | Storage Computing |
| **Intel Xeon Platinum** | 2.8 TFLOP/s | 204.8 GB/s | CPU-based |
| **Apple M2 Ultra** | 27.2 TFLOP/s | 800 GB/s | Unified Memory |
| **AMD MI250X** | 47.9 TFLOP/s | 3.28 TB/s | HPC/AI |

---

## 📊 Output Examples

### CSV Analysis Report
The tool generates detailed CSV reports with columns:
- **Phase**: Prefill/Decode/Decode_Last
- **Operation**: Specific computation (QKV, Attention, FFN, etc.)
- **FLOPs**: Floating-point operations count
- **Param Count**: Parameter count for the operation
- **Memory Metrics**: Input/Output/Weight bytes
- **Density**: Computational intensity (Op/Byte)

### Roofline Visualization
- **Main Plot**: `roofline_model.png` - Complete Roofline analysis
- **Legend**: `roofline_model_legend.png` - Detailed operation legend

---

## 🔬 Advanced Features

### Quantization Analysis
Configure different bit-widths for various components:
```python
quant_config = {
    "activation": 16,      # Activation precision
    "weight_attn": 4,     # Attention weight precision  
    "weight_ffn": 8,      # FFN weight precision
    "kv_cache": 8,        # KV cache precision
    "rope_bit": 32        # RoPE computation precision
}
```

### RoPE Configuration
```python
# Standard RoPE
use_rope=True
rope_theta=10000.0
rope_scaling_factor=1.0

# Extended context with scaling
rope_theta=500000.0
rope_scaling_factor=4.0
```

### MoE Analysis
```python
# Mixtral-style MoE
num_experts_per_tok=2    # Active experts per token
num_local_experts=8      # Total expert count
```

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📚 Citation

If you use this tool in your research, please cite:

```bibtex
@software{llm_para_analyzer,
  title={LLM-para: Transformer Computation & Roofline Analyzer},
  author={[Your Name]},
  year={2024},
  url={https://github.com/[username]/LLM-para}
}
```

---

## 🔗 Related Work

- [Roofline Model Paper](https://doi.org/10.1145/1498765.1498785)
- [Transformer Architecture](https://arxiv.org/abs/1706.03762)
- [LLaMA Models](https://arxiv.org/abs/2302.13971)
- [Mixtral MoE](https://arxiv.org/abs/2401.04088)

---


