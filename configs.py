"""
LLM-Para Model & Hardware Configurations
Comprehensive preset configurations for popular LLM architectures and hardware platforms.
"""

# ─── Model Presets ───────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    # ── GPT Series ────────────────────────────────────────────────────────────
    "GPT-2 Small (117M)": {
        "hidden_size": 768, "num_heads": 12, "num_layers": 12,
        "intermediate_size": 3072, "vocab_size": 50257,
        "seq_len": 1024, "batch_size": 1, "max_gen_len": 512,
        "use_gate_ffn": False, "use_rmsnorm": False,
        "quant_config": {"activation": 16, "weight_attn": 16, "weight_ffn": 16,
                         "kv_cache": 16, "rope_bit": 32},
    },
    "GPT-2 XL (1.5B)": {
        "hidden_size": 1600, "num_heads": 25, "num_layers": 48,
        "intermediate_size": 6400, "vocab_size": 50257,
        "seq_len": 1024, "batch_size": 1, "max_gen_len": 512,
        "use_gate_ffn": False, "use_rmsnorm": False,
        "quant_config": {"activation": 16, "weight_attn": 16, "weight_ffn": 16,
                         "kv_cache": 16, "rope_bit": 32},
    },

    # ── LLaMA Series ─────────────────────────────────────────────────────────
    "LLaMA-2 7B": {
        "hidden_size": 4096, "num_heads": 32, "num_key_value_heads": 32,
        "num_layers": 32, "intermediate_size": 11008, "vocab_size": 32000,
        "seq_len": 2048, "batch_size": 1, "max_gen_len": 4096,
        "use_gate_ffn": True, "use_rmsnorm": True,
        "rope_theta": 10000.0, "rope_scaling_factor": 1.0,
        "quant_config": {"activation": 16, "weight_attn": 16, "weight_ffn": 16,
                         "kv_cache": 16, "rope_bit": 32},
    },
    "LLaMA-2 13B": {
        "hidden_size": 5120, "num_heads": 40, "num_key_value_heads": 40,
        "num_layers": 40, "intermediate_size": 13824, "vocab_size": 32000,
        "seq_len": 2048, "batch_size": 1, "max_gen_len": 4096,
        "use_gate_ffn": True, "use_rmsnorm": True,
        "rope_theta": 10000.0, "rope_scaling_factor": 1.0,
        "quant_config": {"activation": 16, "weight_attn": 16, "weight_ffn": 16,
                         "kv_cache": 16, "rope_bit": 32},
    },
    "LLaMA-2 70B": {
        "hidden_size": 8192, "num_heads": 64, "num_key_value_heads": 8,
        "num_layers": 80, "intermediate_size": 28672, "vocab_size": 32000,
        "seq_len": 4096, "batch_size": 1, "max_gen_len": 4096,
        "use_gate_ffn": True, "use_rmsnorm": True,
        "rope_theta": 10000.0, "rope_scaling_factor": 1.0,
        "quant_config": {"activation": 16, "weight_attn": 16, "weight_ffn": 16,
                         "kv_cache": 16, "rope_bit": 32},
    },
    "LLaMA-3 8B": {
        "hidden_size": 4096, "num_heads": 32, "num_key_value_heads": 8,
        "num_layers": 32, "intermediate_size": 14336, "vocab_size": 128256,
        "seq_len": 2048, "batch_size": 1, "max_gen_len": 8192,
        "use_gate_ffn": True, "use_rmsnorm": True,
        "rope_theta": 500000.0, "rope_scaling_factor": 1.0,
        "quant_config": {"activation": 16, "weight_attn": 16, "weight_ffn": 16,
                         "kv_cache": 16, "rope_bit": 32},
    },
    "LLaMA-3 70B": {
        "hidden_size": 8192, "num_heads": 64, "num_key_value_heads": 8,
        "num_layers": 80, "intermediate_size": 28672, "vocab_size": 128256,
        "seq_len": 4096, "batch_size": 1, "max_gen_len": 8192,
        "use_gate_ffn": True, "use_rmsnorm": True,
        "rope_theta": 500000.0, "rope_scaling_factor": 1.0,
        "quant_config": {"activation": 16, "weight_attn": 16, "weight_ffn": 16,
                         "kv_cache": 16, "rope_bit": 32},
    },
    "LLaMA-3 405B": {
        "hidden_size": 16384, "num_heads": 128, "num_key_value_heads": 8,
        "num_layers": 126, "intermediate_size": 53248, "vocab_size": 128256,
        "seq_len": 4096, "batch_size": 1, "max_gen_len": 8192,
        "use_gate_ffn": True, "use_rmsnorm": True,
        "rope_theta": 500000.0, "rope_scaling_factor": 1.0,
        "quant_config": {"activation": 16, "weight_attn": 16, "weight_ffn": 16,
                         "kv_cache": 16, "rope_bit": 32},
    },

    # ── Mixtral MoE ──────────────────────────────────────────────────────────
    "Mixtral 8x7B": {
        "hidden_size": 4096, "num_heads": 32, "num_key_value_heads": 8,
        "num_layers": 32, "intermediate_size": 14336, "vocab_size": 32000,
        "seq_len": 2048, "batch_size": 1, "max_gen_len": 4096,
        "use_gate_ffn": True, "use_rmsnorm": True,
        "num_experts_per_tok": 2, "num_local_experts": 8,
        "rope_theta": 1000000.0, "rope_scaling_factor": 1.0,
        "quant_config": {"activation": 16, "weight_attn": 16, "weight_ffn": 16,
                         "kv_cache": 16, "rope_bit": 32},
    },
    "Mixtral 8x22B": {
        "hidden_size": 6144, "num_heads": 48, "num_key_value_heads": 8,
        "num_layers": 56, "intermediate_size": 16384, "vocab_size": 32000,
        "seq_len": 4096, "batch_size": 1, "max_gen_len": 4096,
        "use_gate_ffn": True, "use_rmsnorm": True,
        "num_experts_per_tok": 2, "num_local_experts": 8,
        "rope_theta": 1000000.0, "rope_scaling_factor": 1.0,
        "quant_config": {"activation": 16, "weight_attn": 16, "weight_ffn": 16,
                         "kv_cache": 16, "rope_bit": 32},
    },

    # ── Qwen Series ──────────────────────────────────────────────────────────
    "Qwen2 7B": {
        "hidden_size": 3584, "num_heads": 28, "num_key_value_heads": 4,
        "num_layers": 28, "intermediate_size": 18944, "vocab_size": 151936,
        "seq_len": 4096, "batch_size": 1, "max_gen_len": 8192,
        "use_gate_ffn": True, "use_rmsnorm": True,
        "rope_theta": 1000000.0, "rope_scaling_factor": 1.0,
        "quant_config": {"activation": 16, "weight_attn": 16, "weight_ffn": 16,
                         "kv_cache": 16, "rope_bit": 32},
    },
    "Qwen2 72B": {
        "hidden_size": 8192, "num_heads": 64, "num_key_value_heads": 8,
        "num_layers": 80, "intermediate_size": 29568, "vocab_size": 151936,
        "seq_len": 4096, "batch_size": 1, "max_gen_len": 8192,
        "use_gate_ffn": True, "use_rmsnorm": True,
        "rope_theta": 1000000.0, "rope_scaling_factor": 1.0,
        "quant_config": {"activation": 16, "weight_attn": 16, "weight_ffn": 16,
                         "kv_cache": 16, "rope_bit": 32},
    },
    "Qwen2-MoE 57B-A14B": {
        "hidden_size": 3584, "num_heads": 28, "num_key_value_heads": 4,
        "num_layers": 28, "intermediate_size": 2560, "vocab_size": 151936,
        "seq_len": 4096, "batch_size": 1, "max_gen_len": 8192,
        "use_gate_ffn": True, "use_rmsnorm": True,
        "num_experts_per_tok": 4, "num_local_experts": 64,
        "rope_theta": 1000000.0, "rope_scaling_factor": 1.0,
        "quant_config": {"activation": 16, "weight_attn": 16, "weight_ffn": 16,
                         "kv_cache": 16, "rope_bit": 32},
    },

    # ── DeepSeek Series ──────────────────────────────────────────────────────
    "DeepSeek-V2 (MLA+MoE)": {
        "hidden_size": 5120, "num_heads": 128, "num_key_value_heads": 128,
        "num_layers": 60, "intermediate_size": 1536, "vocab_size": 102400,
        "seq_len": 4096, "batch_size": 1, "max_gen_len": 8192,
        "use_gate_ffn": True, "use_rmsnorm": True, "use_mla": True,
        "mla_kv_lora_rank": 512, "mla_q_lora_rank": 1536,
        "mla_qk_nope_head_dim": 128, "mla_qk_rope_head_dim": 64, "mla_v_head_dim": 128,
        "num_experts_per_tok": 6, "num_local_experts": 160,
        "rope_theta": 10000.0, "rope_scaling_factor": 1.0,
        "quant_config": {"activation": 16, "weight_attn": 16, "weight_ffn": 16,
                         "kv_cache": 16, "rope_bit": 32},
    },
    "DeepSeek-R1 671B": {
        "hidden_size": 7168, "num_heads": 128, "num_key_value_heads": 128,
        "num_layers": 61, "intermediate_size": 2048, "vocab_size": 129280,
        "seq_len": 4096, "batch_size": 1, "max_gen_len": 8192,
        "use_gate_ffn": True, "use_rmsnorm": True, "use_mla": True,
        "mla_kv_lora_rank": 512, "mla_q_lora_rank": 1536,
        "mla_qk_nope_head_dim": 128, "mla_qk_rope_head_dim": 64, "mla_v_head_dim": 128,
        "num_experts_per_tok": 8, "num_local_experts": 256,
        "rope_theta": 10000.0, "rope_scaling_factor": 1.0,
        "quant_config": {"activation": 16, "weight_attn": 16, "weight_ffn": 16,
                         "kv_cache": 16, "rope_bit": 32},
    },

    # ── Gemma Series ─────────────────────────────────────────────────────────
    "Gemma-2 9B": {
        "hidden_size": 3584, "num_heads": 16, "num_key_value_heads": 8,
        "num_layers": 42, "intermediate_size": 14336, "vocab_size": 256000,
        "seq_len": 4096, "batch_size": 1, "max_gen_len": 8192,
        "use_gate_ffn": True, "use_rmsnorm": True,
        "rope_theta": 10000.0, "rope_scaling_factor": 1.0,
        "quant_config": {"activation": 16, "weight_attn": 16, "weight_ffn": 16,
                         "kv_cache": 16, "rope_bit": 32},
    },
    "Gemma-2 27B": {
        "hidden_size": 4608, "num_heads": 32, "num_key_value_heads": 16,
        "num_layers": 46, "intermediate_size": 36864, "vocab_size": 256000,
        "seq_len": 4096, "batch_size": 1, "max_gen_len": 8192,
        "use_gate_ffn": True, "use_rmsnorm": True,
        "rope_theta": 10000.0, "rope_scaling_factor": 1.0,
        "quant_config": {"activation": 16, "weight_attn": 16, "weight_ffn": 16,
                         "kv_cache": 16, "rope_bit": 32},
    },

    # ── Phi Series ───────────────────────────────────────────────────────────
    "Phi-3 Mini (3.8B)": {
        "hidden_size": 3072, "num_heads": 32, "num_key_value_heads": 32,
        "num_layers": 32, "intermediate_size": 8192, "vocab_size": 32064,
        "seq_len": 4096, "batch_size": 1, "max_gen_len": 4096,
        "use_gate_ffn": True, "use_rmsnorm": True,
        "rope_theta": 10000.0, "rope_scaling_factor": 1.0,
        "quant_config": {"activation": 16, "weight_attn": 16, "weight_ffn": 16,
                         "kv_cache": 16, "rope_bit": 32},
    },

    # ── BitNet ───────────────────────────────────────────────────────────────
    "BitNet b1.58 3B": {
        "hidden_size": 3200, "num_heads": 25, "num_key_value_heads": 5,
        "num_layers": 26, "intermediate_size": 8640, "vocab_size": 32000,
        "seq_len": 4096, "batch_size": 1, "max_gen_len": 4096,
        "use_gate_ffn": True, "use_rmsnorm": True,
        "rope_theta": 500000.0, "rope_scaling_factor": 1.0,
        "quant_config": {"activation": 8, "weight_attn": 2, "weight_ffn": 2,
                         "kv_cache": 4, "rope_bit": 16},
    },
}

# ─── Hardware Presets ─────────────────────────────────────────────────────────

HARDWARE_CONFIGS = {
    # ── NVIDIA GPUs ──────────────────────────────────────────────────────────
    "NVIDIA H100 SXM": {
        "name": "NVIDIA H100 SXM", "category": "GPU",
        "peak_performance": 67e12,      # FP32 TFLOP/s
        "peak_performance_fp16": 989e12, # FP16 TFLOP/s (with sparsity)
        "memory_bandwidth": 3.35e12,    # HBM3 TB/s
        "memory_capacity": 80e9,        # 80 GB HBM3
    },
    "NVIDIA H100 PCIe": {
        "name": "NVIDIA H100 PCIe", "category": "GPU",
        "peak_performance": 51e12,
        "peak_performance_fp16": 756e12,
        "memory_bandwidth": 2.0e12,
        "memory_capacity": 80e9,
    },
    "NVIDIA A100 SXM (80GB)": {
        "name": "NVIDIA A100 SXM (80GB)", "category": "GPU",
        "peak_performance": 19.5e12,
        "peak_performance_fp16": 312e12,
        "memory_bandwidth": 2.0e12,     # HBM2e
        "memory_capacity": 80e9,
    },
    "NVIDIA A100 SXM (40GB)": {
        "name": "NVIDIA A100 SXM (40GB)", "category": "GPU",
        "peak_performance": 19.5e12,
        "peak_performance_fp16": 312e12,
        "memory_bandwidth": 1.555e12,
        "memory_capacity": 40e9,
    },
    "NVIDIA A10": {
        "name": "NVIDIA A10", "category": "GPU",
        "peak_performance": 31.2e12,
        "peak_performance_fp16": 125e12,
        "memory_bandwidth": 600e9,
        "memory_capacity": 24e9,
    },
    "NVIDIA RTX 4090": {
        "name": "NVIDIA RTX 4090", "category": "GPU",
        "peak_performance": 82.6e12,
        "peak_performance_fp16": 165.2e12,
        "memory_bandwidth": 1008e9,     # GDDR6X
        "memory_capacity": 24e9,
    },
    "NVIDIA RTX 4080": {
        "name": "NVIDIA RTX 4080", "category": "GPU",
        "peak_performance": 48.7e12,
        "peak_performance_fp16": 97.4e12,
        "memory_bandwidth": 717e9,
        "memory_capacity": 16e9,
    },

    # ── AMD GPUs ─────────────────────────────────────────────────────────────
    "AMD MI300X": {
        "name": "AMD MI300X", "category": "GPU",
        "peak_performance": 163.4e12,
        "peak_performance_fp16": 1307e12,
        "memory_bandwidth": 5.3e12,     # HBM3
        "memory_capacity": 192e9,
    },
    "AMD MI250X": {
        "name": "AMD MI250X", "category": "GPU",
        "peak_performance": 47.9e12,
        "peak_performance_fp16": 383e12,
        "memory_bandwidth": 3.28e12,
        "memory_capacity": 128e9,
    },

    # ── Apple Silicon ─────────────────────────────────────────────────────────
    "Apple M3 Ultra": {
        "name": "Apple M3 Ultra", "category": "CPU/NPU",
        "peak_performance": 54.4e12,
        "peak_performance_fp16": 54.4e12,
        "memory_bandwidth": 800e9,
        "memory_capacity": 192e9,
    },
    "Apple M2 Ultra": {
        "name": "Apple M2 Ultra", "category": "CPU/NPU",
        "peak_performance": 27.2e12,
        "peak_performance_fp16": 27.2e12,
        "memory_bandwidth": 800e9,
        "memory_capacity": 192e9,
    },
    "Apple M2 Max": {
        "name": "Apple M2 Max", "category": "CPU/NPU",
        "peak_performance": 13.6e12,
        "peak_performance_fp16": 13.6e12,
        "memory_bandwidth": 400e9,
        "memory_capacity": 96e9,
    },

    # ── Intel ─────────────────────────────────────────────────────────────────
    "Intel Gaudi 3": {
        "name": "Intel Gaudi 3", "category": "AI Accelerator",
        "peak_performance": 99.5e12,
        "peak_performance_fp16": 1835e12,
        "memory_bandwidth": 3.7e12,
        "memory_capacity": 96e9,
    },
    "Intel Xeon Platinum 8480+": {
        "name": "Intel Xeon Platinum 8480+", "category": "CPU",
        "peak_performance": 3.84e12,
        "peak_performance_fp16": 3.84e12,
        "memory_bandwidth": 307e9,      # DDR5
        "memory_capacity": 4096e9,      # configurable
    },

    # ── Mobile / Edge ─────────────────────────────────────────────────────────
    "Snapdragon 8 Gen 3 NPU": {
        "name": "Snapdragon 8 Gen 3 NPU", "category": "Mobile NPU",
        "peak_performance": 45e12,       # INT8 TOPS
        "peak_performance_fp16": 10e12,
        "memory_bandwidth": 77e9,        # LPDDR5X
        "memory_capacity": 16e9,
    },
    "Snapdragon 8 Gen 2 NPU": {
        "name": "Snapdragon 8 Gen 2 NPU", "category": "Mobile NPU",
        "peak_performance": 18e12,
        "peak_performance_fp16": 4.35e12,
        "memory_bandwidth": 51.2e9,
        "memory_capacity": 12e9,
    },
    "MediaTek Dimensity 9300 NPU": {
        "name": "MediaTek Dimensity 9300 NPU", "category": "Mobile NPU",
        "peak_performance": 35e12,
        "peak_performance_fp16": 8e12,
        "memory_bandwidth": 77e9,
        "memory_capacity": 16e9,
    },

    # ── Processing-In-Memory (PIM) ───────────────────────────────────────────
    "DRAM-PIM (HBM-PIM)": {
        "name": "DRAM-PIM (HBM-PIM)", "category": "PIM",
        "peak_performance": 0.8e12,
        "peak_performance_fp16": 1.6e12,
        "memory_bandwidth": 1.2e12,
        "memory_capacity": 32e9,
    },
    "NAND-PIM (HILOS)": {
        "name": "NAND-PIM (HILOS)", "category": "PIM",
        "peak_performance": 0.2e12,
        "peak_performance_fp16": 0.4e12,
        "memory_bandwidth": 0.2e12,
        "memory_capacity": 128e9,       # large NAND capacity
    },
    "SRAM-PIM (MRAM)": {
        "name": "SRAM-PIM (MRAM)", "category": "PIM",
        "peak_performance": 2.0e12,
        "peak_performance_fp16": 4.0e12,
        "memory_bandwidth": 4.0e12,
        "memory_capacity": 8e9,
    },

    # ── Custom ────────────────────────────────────────────────────────────────
    "Custom Hardware": {
        "name": "Custom Hardware", "category": "Custom",
        "peak_performance": 10e12,
        "peak_performance_fp16": 20e12,
        "memory_bandwidth": 1e12,
        "memory_capacity": 32e9,
    },
}

# ─── Category colors for frontend ─────────────────────────────────────────────

CATEGORY_COLORS = {
    "QKV":       "#4e8df5",   # blue
    "Attention": "#f55f4e",   # red-orange
    "FFN":       "#52c41a",   # green
    "Norm":      "#fa8c16",   # orange
    "Embed":     "#722ed1",   # purple
    "RoPE":      "#13c2c2",   # cyan
    "Other":     "#8c8c8c",   # grey
}

PHASE_SHAPES = {
    "Prefill":  "circle",
    "Decode":   "triangle",
    "Output":   "rect",
}
