# LLM-para
Calculate LLM parameters for Roofline...


# Transformer Computation & Roofline Analyzer

This project provides a complete toolchain for analyzing **computation**, **memory access**, and **efficiency bottlenecks** in large Transformer-based language models. It includes both:
1. A **model analysis script** to compute FLOPs, parameter sizes, and KV cache usage.
2. A **visualization script** that draws **Roofline performance models** from output data.

---

## 🔍 Features

### ✅ Transformer Analysis Script
- Analyze FLOPs and memory access in:
  - Prefill
  - Decode
  - Decode (Last Token)
- Supports:
  - GQA / MoE (doing)
  - RoPE & scaling
  - Gated FFNs
  - Arbitrary quantization config
- Outputs CSV with detailed operation-level metrics
- Plots per-phase FLOPs/memory bar chart

### ✅ Roofline Visualization Script
- Draws Roofline plots with **broken x-axis** for memory-bound vs compute-bound zones
- Overlays each model operation and phase with custom markers
- Compatible with any CSV output from the analysis script
- Generates two images:
  - `roofline_model.png`
  - `roofline_model_legend.png`

---

## 📦 Requirements

```bash
pip install numpy pandas matplotlib

