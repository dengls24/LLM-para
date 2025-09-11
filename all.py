"""
Transformer Model Analysis Tool
==============================

This is a tool for analyzing computational complexity (FLOPs) and memory usage of Transformer models.
Supports various modern Transformer architecture features, including:
- Grouped Query Attention (GQA)
- Mixture of Experts (MoE)
- Rotary Position Embedding (RoPE)
- Gated Feed-Forward Networks (Gated FFN)
- Quantization configuration analysis

Main Features:
1. Calculate FLOPs for Prefill and Decode phases
2. Analyze memory access patterns and bandwidth requirements
3. Evaluate KV cache usage
4. Generate detailed performance analysis reports

Author: [Your Name]
Version: 1.0
License: MIT
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def compute_and_plot_transformer_analysis(
    hidden_size: int,
    num_heads: int,
    seq_len: int,
    batch_size: int,
    num_layers: int,
    quant_config: dict,
    intermediate_size: int = None,  # FFN intermediate layer size, default is 4*hidden_size
    num_key_value_heads: int = None,  # Number of KV heads in GQA, default equals num_heads
    num_experts_per_tok: int = None,    # Number of experts per token in MoE, typically 2
    num_local_experts: int = None,      # Total number of experts in MoE, typically 8
    out_l: int = 0,
    max_gen_len: int = 4096,  # Maximum generation length
    output_csv_path: str = "true_density_transformer_analysis.csv",
    use_gate_ffn: bool = False,  # Whether to use gated FFN (like SwiGLU)
    use_rope: bool = True,  # Whether to use rotary position embedding
    rope_theta: float = 10000.0,  # RoPE base frequency parameter
    rope_scaling_factor: float = 1.0  # RoPE scaling factor for length extrapolation
):
    """
    Calculate and analyze computational complexity and memory usage of Transformer models
    
    Parameters:
    ----------
    hidden_size : int
        Hidden dimension (usually denoted as d_model or h)
    num_heads : int
        Number of attention heads
    seq_len : int
        Input sequence length
    batch_size : int
        Batch size
    num_layers : int
        Number of Transformer layers
    quant_config : dict
        Quantization configuration containing the following keys:
        - "activation": Activation bit width
        - "weight_attn": Attention weight bit width
        - "weight_ffn": FFN weight bit width
        - "kv_cache": KV cache bit width
        - "rope_bit": RoPE computation bit width
    intermediate_size : int, optional
        FFN intermediate layer size, default is 4*hidden_size
    num_key_value_heads : int, optional
        Number of KV heads in GQA, default equals num_heads (standard attention)
    num_experts_per_tok : int, optional
        Number of experts activated per token in MoE, typically 2
    num_local_experts : int, optional
        Total number of experts in MoE, typically 8
    out_l : int
        Output length offset
    max_gen_len : int
        Maximum generation length
    output_csv_path : str
        Output CSV file path
    use_gate_ffn : bool
        Whether to use gated FFN (like SwiGLU activation function)
    use_rope : bool
        Whether to use rotary position embedding
    rope_theta : float
        RoPE base frequency parameter
    rope_scaling_factor : float
        RoPE scaling factor for length extrapolation
    
    Returns:
    -------
    None
        Function generates CSV reports and visualization charts
    """
    
    # Calculate basic parameters
    d = hidden_size // num_heads  # Dimension per attention head
    b, s, h, n, L = batch_size, seq_len, hidden_size, num_heads, num_layers
    o_l = out_l

    # Check if MoE is enabled
    use_moe = False
    if num_experts_per_tok is not None:
        use_moe = True
    print("use_moe", use_moe)

    # Set default parameters
    if intermediate_size is None:
        intermediate_size = 4 * h  # Standard FFN intermediate layer size
    
    if num_key_value_heads is None:
        num_key_value_heads = num_heads  # Standard attention, KV heads equal Q heads
    
    kv_n = num_key_value_heads  # KV heads abbreviation
    
    def byte_size(shape, bitwidth):
        """
        Calculate byte size of tensor with given shape and bit width
        
        Parameters:
        ----------
        shape : tuple
            Tensor shape
        bitwidth : int
            Data bit width
            
        Returns:
        -------
        int
            Byte size
        """
        num_elements = 1
        for dim in shape:
            num_elements *= dim
        return num_elements * bitwidth / 8

    # List to store analysis data
    data = []

    def add_row(phase, name, input1, input2, output,
                flops, param_count,
                input_shape, weight_shape, output_shape,
                act_bits, weight_bits, note=""):
        """
        Add a row of analysis data
        
        Parameters:
        ----------
        phase : str
            Computation phase (Prefill/Decode/Decode_Last)
        name : str
            Operation name
        input1, input2, output : str
            Input/output shape descriptions
        flops : int
            Number of floating-point operations
        param_count : int
            Parameter count
        input_shape, weight_shape, output_shape : tuple
            Actual tensor shapes
        act_bits, weight_bits : int
            Activation and weight bit widths
        note : str
            Additional notes
        """
        
        # Calculate byte sizes for each part
        B_input = byte_size(input_shape, act_bits)
        B_output = byte_size(output_shape, act_bits)
        B_weight = byte_size(weight_shape, weight_bits) if weight_shape else 0

        B_total = B_input + B_output + B_weight
        # Calculate computational density (FLOPs per Byte)
        density = "-" if B_total == 0 else round(flops / B_total, 2)

        data.append({
            "Phase": phase,
            "Operation": name,
            "FLOPs": flops,
            "Param Count": param_count,
            "Input1 Bytes": B_input,
            "Input2 Bytes": B_weight,
            "Output Bytes": B_output,
            "Total Bytes": B_total,
            "Density (Op/Byte)": density,
            "Input Shape": input_shape,
            "Weight Shape": weight_shape,
            "Output Shape": output_shape
        })

    # Extract bit width settings from quantization config
    a_bit = quant_config["activation"]     # Activation bit width
    w_attn = quant_config["weight_attn"]   # Attention weight bit width
    w_ffn = quant_config["weight_ffn"]     # FFN weight bit width
    kv_bit = quant_config["kv_cache"]      # KV cache bit width
    rope_bit = quant_config["rope_bit"]    # RoPE computation bit width

    # ==========================================
    # PREFILL Phase Analysis
    # ==========================================
    phase = "Prefill"
    seq = s  # Prefill phase processes complete sequence

    # Query projection computation (using full number of attention heads)
    flops = 2 * b * seq * h * h  # Matrix multiplication FLOPs = 2 * M * N * K
    param_count = h * h
    add_row(phase, "xW_Q", "(b,s,h)", "(h,h)", "(b,s,h)",
            flops, param_count, (b, seq, h), (h, h), (b, seq, h), a_bit, w_attn)
    
    # Key and Value projection computation (considering GQA, KV heads may be fewer than Q heads)
    for name in ['W_K', 'W_V']:
        # In GQA, K and V parameter count and computation are reduced proportionally
        flops = 2 * b * seq * h * h * (kv_n / n)
        param_count = h * h * (kv_n / n)
        add_row(phase, f"x{name}", "(b,s,h)", f"(h,h*{kv_n/n})", f"(b,s,h*{kv_n/n})",
                flops, param_count, (b, seq, h), (h, h * (kv_n / n)), (b, seq, h * (kv_n / n)), a_bit, w_attn)
    
    # RoPE position encoding computation
    if use_rope:
        # RoPE is applied to Q and K, performing rotation transformation for each position and head
        # Each position needs to compute sin and cos, then apply rotation transformation
        # For each position and each head's dimension pair (dim_i, dim_i+1), 4 FLOPs are needed (2 multiplications, 2 additions)
        # Total: seq positions, n Q heads, kv_n K heads, d/2 dimension pairs
        
        # Q RoPE computation
        rope_q_flops = b * seq * n * (d // 2) * 4  # 4 = 2 multiplications + 2 additions
        add_row(phase, "RoPE-Q", "(b,n,s,d)", "(s,d)", "(b,n,s,d)",
                rope_q_flops, 0, (b, n, seq, d), (seq, d), (b, n, seq, d), a_bit, rope_bit,
                note=f"RoPE theta={rope_theta}, scaling={rope_scaling_factor}")
        
        # K RoPE computation
        rope_k_flops = b * seq * kv_n * (d // 2) * 4
        add_row(phase, "RoPE-K", "(b,kv_n,s,d)", "(s,d)", "(b,kv_n,s,d)",
                rope_k_flops, 0, (b, kv_n, seq, d), (seq, d), (b, kv_n, seq, d), a_bit, rope_bit,
                note=f"RoPE theta={rope_theta}, scaling={rope_scaling_factor}")

    # Attention score computation Q × K^T (considering GQA)
    add_row(phase, "Q × Kᵀ", "(b,n,s,d)", "(b,kv_n,d,s)", "(b,n,s,s)",
            2 * b * n * seq * seq * d, 0,
            (b, n, seq, d), (b,kv_n, seq, d), (b, n, seq, seq), a_bit, kv_bit,
            note=f"GQA: {n} Q heads, {kv_n} KV heads")

    # Attention weighted sum Attn × V (considering GQA)
    add_row(phase, "Attn × V", "(b,n,s,s)", "(b,kv_n,s,d)", "(b,n,s,d)",
            2 * b * n * seq * seq * d, 0,
            (b, n, seq, seq), (b, kv_n, seq, d), (b, n, seq, d), a_bit, kv_bit,
            note=f"GQA: {n} Q heads, {kv_n} KV heads")

    # Output projection computation
    add_row(phase, "xW_O", "(b,s,h)", "(h,h)", "(b,s,h)",
            2 * b * seq * h * h, h * h,
            (b, seq, h), (h, h), (b, seq, h), a_bit, w_attn)
    
    # FFN computation (distinguish between MoE and standard FFN)
    if use_moe:
        # MoE routing computation: decide which experts each token uses
        moe_router_flops = b * seq * h * num_local_experts * 2
        param_count = h * num_local_experts
        add_row(phase, "Router", "(b,s,h)", f"(h,{num_local_experts})", f"(b,s,{num_local_experts})", 
                moe_router_flops, param_count, 
                (b,seq,h), (h,num_local_experts), (b,seq,num_local_experts), a_bit, w_ffn)
        
        # MoE FFN-1 computation (up + gate projection)
        FFN_1_moe = b * seq * h * intermediate_size * 2 * 2 * num_experts_per_tok  # Two matrix multiplications * experts per token
        FFN_1_moe += b * seq * num_experts_per_tok * intermediate_size  # SiLU(W₁x) ⊙ (W_gate*x) element-wise multiplication

        param_count = h * intermediate_size * 2 * num_local_experts  # Two weight matrices * all experts
        add_row(phase, "FFN-1(with Moe)", f"(b,{s}*num_experts_per_tok,h)", f"(num_local_experts,h,{intermediate_size}*2)", f"(b,{s}*num_experts_per_tok,{intermediate_size})", 
                FFN_1_moe, param_count, 
                (b, seq * num_experts_per_tok, h), (h, intermediate_size * 2 * num_local_experts), (b, seq * num_experts_per_tok, intermediate_size ), a_bit, w_ffn)
        
        # MoE FFN-2 computation (down projection)
        FFN_2_moe = b * seq * intermediate_size * h * 2 * num_experts_per_tok
        param_count = intermediate_size * h * num_local_experts
        add_row(phase, "FFN-2(with Moe)", f"(b, {s}*num_experts_per_tok, {intermediate_size})", f"(num_local_experts, {intermediate_size}, h)", f"(b,{s}*num_experts_per_tok,h)",
                FFN_2_moe, param_count,
                (b, seq * num_experts_per_tok, intermediate_size), (intermediate_size, h, num_local_experts), (b, seq * num_experts_per_tok, h), a_bit, w_ffn)
    else:
        # Standard FFN computation
        if use_gate_ffn:
            # Gated FFN (like SwiGLU): need to compute two projections W₁ and W_gate
            gate_flops = 2 * b * seq * h * intermediate_size * 2  # Two matrix multiplications
            gate_flops += b * seq * intermediate_size  # SiLU(W₁x) ⊙ (W_gate*x) element-wise multiplication
            gate_param_count = h * intermediate_size * 2  # Two weight matrices
            
            add_row(phase, "FFN-1 (with Gate)", "(b,s,h)", f"(h,{intermediate_size}*2)", f"(b,s,{intermediate_size})",
                    gate_flops, gate_param_count, 
                    (b, seq, h), (h, intermediate_size * 2), (b, seq, intermediate_size), a_bit, w_ffn,
                    note="Contains Gate mechanism")
        else:
            # Standard FFN-1 computation (up projection)
            add_row(phase, "FFN-1", "(b,s,h)", f"(h,{intermediate_size})", f"(b,s,{intermediate_size})",
                    2 * b * seq * h * intermediate_size, h * intermediate_size,
                    (b, seq, h), (h, intermediate_size), (b, seq, intermediate_size), a_bit, w_ffn)

        # FFN-2 computation (down projection)
        add_row(phase, "FFN-2", f"(b,s,{intermediate_size})", f"({intermediate_size},h)", "(b,s,h)",
                2 * b * seq * h * intermediate_size, intermediate_size * h,
                (b, seq, intermediate_size), (intermediate_size, h), (b, seq, h), a_bit, w_ffn)

    # ==========================================
    # DECODE Phase Analysis (generating first token)
    # ==========================================
    phase = "Decode"
    seq = 1  # Decode phase generates one token at a time
    hist = s  # Historical sequence length
    o_l = 0   # Initial generation token position
    
    # Query projection computation
    flops = 2 * b * seq * h * h
    param_count = h * h
    add_row(phase, "xW_Q", "(b,1,h)", "(h,h)", "(b,1,h)",
            flops, param_count, (b, seq, h), (h, h), (b, seq, h), a_bit, w_attn)
    
    # Key and Value projection computation (considering GQA)
    for name in ['W_K', 'W_V']:
        flops = 2 * b * seq * h * h * (kv_n / n)
        param_count = h * h * (kv_n / n)
        add_row(phase, f"x{name}", "(b,1,h)", f"(h,h*{kv_n/n})", f"(b,1,h*{kv_n/n})",
                flops, param_count, (b, seq, h), (h, h * (kv_n / n)), (b, seq, h * (kv_n / n)), a_bit, w_attn)
    
    # RoPE computation - Decode phase
    if use_rope:
        # Q RoPE computation - only process current token
        rope_q_flops = b * 1 * n * (d // 2) * 4
        add_row(phase, "RoPE-Q", "(b,n,1,d)", "(1,d)", "(b,n,1,d)",
                rope_q_flops, 0, (b, n, 1, d), (1, d), (b, n, 1, d), a_bit, rope_bit,
                note=f"RoPE theta={rope_theta}, scaling={rope_scaling_factor}, position={s+o_l}")
        
        # K RoPE computation - only process current token
        rope_k_flops = b * 1 * kv_n * (d // 2) * 4
        add_row(phase, "RoPE-K", "(b,kv_n,1,d)", "(1,d)", "(b,kv_n,1,d)",
                rope_k_flops, 0, (b, kv_n, 1, d), (1, d), (b, kv_n, 1, d), a_bit, rope_bit,
                note=f"RoPE theta={rope_theta}, scaling={rope_scaling_factor}, position={s+o_l}")

    # Attention computation - current token with all historical tokens
    add_row(phase, "Q × Kᵀ", "(b,n,1,d)", "(b,kv_n,d,s+o_l+1)", "(b,n,1,s+o_l+1)",
            2 * b * n * hist * d, 0,
            (b, n, 1, d), (b, kv_n, d, hist + 1), (b, n, 1, hist + 1), a_bit, kv_bit,
            note=f"GQA: {n} Q heads, {kv_n} KV heads, history length: {hist}")

    add_row(phase, "Attn × V", "(b,n,1,s+o_l+1)", "(b,kv_n,s+o_l+1,d)", "(b,n,1,d)",
            2 * b * n * (hist + 1) * d, 0,
            (b, n, 1, hist + 1), (b, kv_n, hist + 1, d), (b, n, 1, d), a_bit, kv_bit,
            note=f"GQA: {n} Q heads, {kv_n} KV heads, history length: {hist}")

    # Output projection computation
    add_row(phase, "xW_O", "(b,1,h)", "(h,h)", "(b,1,h)",
            2 * b * seq * h * h, h * h,
            (b, seq, h), (h, h), (b, seq, h), a_bit, w_attn)
    
    # FFN computation (similar to Prefill phase, but sequence length is 1)
    if use_moe:
        # MoE routing computation
        moe_router_flops = b * seq * h * num_local_experts * 2
        param_count = h * num_local_experts
        add_row(phase, "Router", "(b,s,h)", f"(h,{num_local_experts})", f"(b,s,{num_local_experts})", 
                moe_router_flops, param_count, 
                (b,seq,h), (h,num_local_experts), (b,seq,num_local_experts), a_bit, w_ffn)
        
        # MoE FFN computation
        FFN_1_moe = b * seq * h * intermediate_size * 2 * 2 * num_experts_per_tok
        FFN_1_moe += b * seq * num_experts_per_tok * intermediate_size

        param_count = h * intermediate_size * 2 * num_experts_per_tok  # Decode phase only uses activated experts
        add_row(phase, "FFN-1(with Moe)", f"(b,{s}*num_experts_per_tok,h)", f"(num_experts_per_tok,h,{intermediate_size}*2)", f"(b,{s}*num_experts_per_tok,{intermediate_size})", 
                FFN_1_moe, param_count, 
                (b, seq * num_experts_per_tok, h), (h, intermediate_size * 2 * num_experts_per_tok), (b, seq * num_experts_per_tok, intermediate_size ), a_bit, w_ffn)
        
        FFN_2_moe = b * seq * intermediate_size * h * 2 * num_experts_per_tok
        param_count = intermediate_size * h * num_experts_per_tok
        add_row(phase, "FFN-2(with Moe)", f"(b, {s}*num_experts_per_tok, {intermediate_size})", f"(num_experts_per_tok, {intermediate_size}, h)", f"(b,{s}*num_experts_per_tok,h)",
                FFN_2_moe, param_count,
                (b, seq * num_experts_per_tok, intermediate_size), (intermediate_size, h, num_experts_per_tok), (b, seq * num_experts_per_tok, h), a_bit, w_ffn)
    else:
        # Standard FFN computation
        if use_gate_ffn:
            gate_flops = 2 * b * seq * h * intermediate_size * 2
            gate_flops += b * seq * intermediate_size
            gate_param_count = h * intermediate_size * 2
            
            add_row(phase, "FFN-1 (with Gate)", "(b,s,h)", f"(h,{intermediate_size}*2)", f"(b,s,{intermediate_size})",
                    gate_flops, gate_param_count, 
                    (b, seq, h), (h, intermediate_size * 2), (b, seq, intermediate_size), a_bit, w_ffn,
                    note="Contains Gate mechanism")
        else:
            add_row(phase, "FFN-1", "(b,s,h)", f"(h,{intermediate_size})", f"(b,s,{intermediate_size})",
                    2 * b * seq * h * intermediate_size, h * intermediate_size,
                    (b, seq, h), (h, intermediate_size), (b, seq, intermediate_size), a_bit, w_ffn)

        add_row(phase, "FFN-2", f"(b,1,{intermediate_size})", f"({intermediate_size},h)", "(b,1,h)",
                2 * b * seq * h * intermediate_size, intermediate_size * h,
                (b, seq, intermediate_size), (intermediate_size, h), (b, seq, h), a_bit, w_ffn)

    # ==========================================
    # DECODE_LAST Phase Analysis (generating last token)
    # ==========================================
    phase = "Decode_Last"
    o_l = max_gen_len - 1  # Position of last generated token
    hist = s + o_l  # History length = original sequence + generated tokens
    
    # Computation process same as Decode phase, but with longer history
    # Query projection computation
    flops = 2 * b * seq * h * h
    param_count = h * h
    add_row(phase, "xW_Q", "(b,1,h)", "(h,h)", "(b,1,h)",
            flops, param_count, (b, seq, h), (h, h), (b, seq, h), a_bit, w_attn)
    
    # Key and Value projection computation
    for name in ['W_K', 'W_V']:
        flops = 2 * b * seq * h * h * (kv_n / n)
        param_count = h * h * (kv_n / n)
        add_row(phase, f"x{name}", "(b,1,h)", f"(h,h*{kv_n/n})", f"(b,1,h*{kv_n/n})",
                flops, param_count, (b, seq, h), (h, h * (kv_n / n)), (b, seq, h * (kv_n / n)), a_bit, w_attn)
    
    # RoPE computation - last position
    if use_rope:
        rope_q_flops = b * 1 * n * (d // 2) * 4
        add_row(phase, "RoPE-Q", "(b,n,1,d)", "(1,d)", "(b,n,1,d)",
                rope_q_flops, 0, (b, n, 1, d), (1, d), (b, n, 1, d), a_bit, rope_bit,
                note=f"RoPE theta={rope_theta}, scaling={rope_scaling_factor}, position={s+o_l}")
        
        rope_k_flops = b * 1 * kv_n * (d // 2) * 4
        add_row(phase, "RoPE-K", "(b,kv_n,1,d)", "(1,d)", "(b,kv_n,1,d)",
                rope_k_flops, 0, (b, kv_n, 1, d), (1, d), (b, kv_n, 1, d), a_bit, rope_bit,
                note=f"RoPE theta={rope_theta}, scaling={rope_scaling_factor}, position={s+o_l}")

    # Attention computation - with longer history sequence
    add_row(phase, "Q × Kᵀ", "(b,n,1,d)", "(b,kv_n,d,s+o_l+1)", "(b,n,1,s+o_l+1)",
            2 * b * n * hist * d, 0,
            (b, n, 1, d), (b, kv_n, d, hist + 1), (b, n, 1, hist + 1), a_bit, kv_bit,
            note=f"GQA: {n} Q heads, {kv_n} KV heads, history length: {hist}")

    add_row(phase, "Attn × V", "(b,n,1,s+o_l+1)", "(b,kv_n,s+o_l+1,d)", "(b,n,1,d)",
            2 * b * n * (hist + 1) * d, 0,
            (b, n, 1, hist + 1), (b, kv_n, hist + 1, d), (b, n, 1, d), a_bit, kv_bit,
            note=f"GQA: {n} Q heads, {kv_n} KV heads, history length: {hist}")

    # Output projection computation
    add_row(phase, "xW_O", "(b,1,h)", "(h,h)", "(b,1,h)",
            2 * b * seq * h * h, h * h,
            (b, seq, h), (h, h), (b, seq, h), a_bit, w_attn)

    # FFN computation (same as previous phases)
    if use_moe:
        moe_router_flops = b * seq * h * num_local_experts * 2
        param_count = h * num_local_experts
        add_row(phase, "Router", "(b,s,h)", f"(h,{num_local_experts})", f"(b,s,{num_local_experts})", 
                moe_router_flops, param_count, 
                (b,seq,h), (h,num_local_experts), (b,seq,num_local_experts), a_bit, w_ffn)
        
        FFN_1_moe = b * seq * h * intermediate_size * 2 * 2 * num_experts_per_tok
        FFN_1_moe += b * seq * num_experts_per_tok * intermediate_size

        param_count = h * intermediate_size * 2 * num_experts_per_tok
        add_row(phase, "FFN-1(with Moe)", f"(b,{s}*num_experts_per_tok,h)", f"(num_experts_per_tok,h,{intermediate_size}*2)", f"(b,{s}*num_experts_per_tok,{intermediate_size})", 
                FFN_1_moe, param_count, 
                (b, seq * num_experts_per_tok, h), (h, intermediate_size * 2 * num_experts_per_tok), (b, seq * num_experts_per_tok, intermediate_size ), a_bit, w_ffn)
        
        FFN_2_moe = b * seq * intermediate_size * h * 2 * num_experts_per_tok
        param_count = intermediate_size * h * num_experts_per_tok
        add_row(phase, "FFN-2(with Moe)", f"(b, {s}*num_experts_per_tok, {intermediate_size})", f"(num_experts_per_tok, {intermediate_size}, h)", f"(b,{s}*num_experts_per_tok,h)",
                FFN_2_moe, param_count,
                (b, seq * num_experts_per_tok, intermediate_size), (intermediate_size, h, num_experts_per_tok), (b, seq * num_experts_per_tok, h), a_bit, w_ffn)
    else:
        if use_gate_ffn:
            gate_flops = 2 * b * seq * h * intermediate_size * 2
            gate_flops += b * seq * intermediate_size
            gate_param_count = h * intermediate_size * 2
            
            add_row(phase, "FFN-1 (with Gate)", "(b,s,h)", f"(h,{intermediate_size}*2)", f"(b,s,{intermediate_size})",
                    gate_flops, gate_param_count, 
                    (b, seq, h), (h, intermediate_size * 2), (b, seq, intermediate_size), a_bit, w_ffn,
                    note="Contains Gate mechanism")
        else:
            add_row(phase, "FFN-1", "(b,s,h)", f"(h,{intermediate_size})", f"(b,s,{intermediate_size})",
                    2 * b * seq * h * intermediate_size, h * intermediate_size,
                    (b, seq, h), (h, intermediate_size), (b, seq, intermediate_size), a_bit, w_ffn)

        add_row(phase, "FFN-2", f"(b,1,{intermediate_size})", f"({intermediate_size},h)", "(b,1,h)",
                2 * b * seq * h * intermediate_size, intermediate_size * h,
                (b, seq, intermediate_size), (intermediate_size, h), (b, seq, h), a_bit, w_ffn)

    # ==========================================
    # Data Processing and Analysis
    # ==========================================
    
    # Convert data to DataFrame
    df = pd.DataFrame(data)
    df.to_csv(output_csv_path, index=False)
    
    # Calculate overall statistics (multiply by number of layers)
    df['FLOPs'] = df['FLOPs'] * L
    df['Total Bytes'] = df['Total Bytes'] * L

    # Calculate total parameter count (only count Prefill phase to avoid duplication)
    prefill_df = df[df['Phase'] == 'Prefill']
    total_params = prefill_df['Param Count'].sum() * L
    # Convert parameter count to GB, considering weight bit width
    total_params_gb = (total_params * w_ffn / 8) / (1024**3)

    # Calculate total FLOPs and memory access
    total_flops = df['FLOPs'].sum()
    total_bytes = df['Total Bytes'].sum()

    # Print overall analysis results
    print(f"\n🔢 Total {L}-Layer Summary:")
    print(f"  FLOPs        = {total_flops:.2e} Op")
    print(f"  Param Count  = {total_params:.2e} ({total_params_gb:.2f} GB @{w_ffn}bit)")
    print(f"  Memory Access= {total_bytes / (1024**2):.2f} MB")

    # ==========================================
    # KV Cache Analysis
    # ==========================================
    
    # Calculate KV cache size per token (considering GQA)
    kv_cache_bytes_per_token = L * kv_n * d * 2 * (kv_bit / 8)  # Key + Value, only consider kv_n heads
    kv_cache_total = kv_cache_bytes_per_token * s * b  # KV cache for original sequence

    # Calculate KV cache during generation
    gen_kv_cache_total = kv_cache_bytes_per_token * max_gen_len * b  # KV cache for generated tokens
    max_kv_cache = kv_cache_bytes_per_token * (s + max_gen_len) * b  # Maximum KV cache (original + generated)
    
    print(f"\n💾 KV Cache Analysis:")
    print(f"  Per token: {kv_cache_bytes_per_token:.1f} Bytes")
    print(f"  Prefill seq: {kv_cache_total / (1024 ** 2):.2f} MB")
    print(f"  Generation: {gen_kv_cache_total / (1024 ** 2):.2f} MB")
    print(f"  Max total: {max_kv_cache / (1024 ** 2):.2f} MB (sequence length: {s + max_gen_len})")
    print(f"  GQA ratio: {kv_n}/{n} = {kv_n/n:.2f}x reduction")
    
    # Calculate computation for generating last token
    last_token_df = df[df['Phase'] == 'Decode_Last']
    last_token_flops = last_token_df['FLOPs'].sum() * L
    last_token_bytes = last_token_df['Total Bytes'].sum() * L
    print(f"\n🔢 Generating Last Token (position {max_gen_len}):")
    print(f"  FLOPs        = {last_token_flops:.2e} Op")
    print(f"  Memory Access= {last_token_bytes / (1024**2):.2f} MB")
    print(f"  History length= {s + max_gen_len - 1} tokens")

    # ==========================================
    # RoPE Analysis
    # ==========================================
    
    if use_rope:
        print(f"\n🔄 RoPE Analysis:")
        print(f"  Base theta: {rope_theta}")
        print(f"  Scaling factor: {rope_scaling_factor}")
        print(f"  Max position: {s + max_gen_len - 1}")
        
        # Calculate theoretical effective context length for RoPE
        max_freq = 1.0 / (rope_theta * rope_scaling_factor)
        effective_context_length = int(np.pi / (max_freq * np.pi / (d // 2)))
        
        print(f"  Max frequency: {max_freq:.6f}")
        print(f"  Theoretical effective context length: ~{effective_context_length}")
        
        # Check if sequence length exceeds effective range
        if s + max_gen_len > effective_context_length:
            print(f"  ⚠️ Warning: Current sequence length({s + max_gen_len}) exceeds theoretical effective context length({effective_context_length}), may cause performance degradation")
        else:
            print(f"  ✅ Current sequence length({s + max_gen_len}) is within theoretical effective context length({effective_context_length})")

    # ==========================================
    # Visualization
    # ==========================================
    
    # Group statistics by phase for FLOPs and memory access
    grouped = df.groupby('Phase')[['FLOPs', 'Total Bytes']].sum()
    grouped['FLOPs'] /= 1e9  # Convert to GFLOPs
    grouped['Total Bytes'] /= 1024 ** 2  # Convert to MB

    # Create dual-axis bar chart
    ax = grouped.plot(kind='bar', secondary_y='Total Bytes', figsize=(10, 5))
    ax.set_ylabel("FLOPs (G)")
    ax.right_ax.set_ylabel("Memory Access (MB)")
    ax.set_title(f"Phase-wise Computation vs Memory ({L} Layers)")
    plt.tight_layout()
    plt.savefig("phasewise_flops_memory.png")
    plt.show()

    print(f"\n✅ CSV saved to '{output_csv_path}', chart saved to 'phasewise_flops_memory.png'")


# ==========================================
# Example Usage
# ==========================================

# The following are some pre-configured model examples, uncomment to run analysis

# BitNet model example
# compute_and_plot_transformer_analysis(
#     hidden_size=2560,
#     num_heads=20,
#     seq_len=4096,
#     batch_size=1,
#     num_layers=30,
#     intermediate_size=6912,
#     num_key_value_heads=5,
#     out_l=0,
#     max_gen_len=4096,
#     quant_config={
#         "activation": 8,
#         "kv_cache": 4,
#         "weight_ffn": 2,
#         "weight_attn": 2,
#         "rope_bit": 16
#     },
#     use_gate_ffn=True,
#     use_rope=True,
#     rope_theta=500000.0,
#     rope_scaling_factor=1.0,
#     output_csv_path="bitnet_analysis.csv"
# )

#LLaMA-3 8B model example
compute_and_plot_transformer_analysis(
    hidden_size=4096,
    num_heads=32,
    seq_len=4096,
    batch_size=1,
    num_layers=32,
    intermediate_size=14336,
    num_key_value_heads=8,  # GQA configuration
    out_l=0,
    max_gen_len=4096,
    quant_config={
        "activation": 16,
        "kv_cache": 16,
        "weight_ffn": 16,
        "weight_attn": 16,
        "rope_bit": 16
    },
    use_gate_ffn=True,
    use_rope=True,
    rope_theta=500000.0,
    rope_scaling_factor=1.0,
    output_csv_path="llama3_8b_analysis.csv"
)

# Mixtral-8x7B model analysis (currently running example)
# compute_and_plot_transformer_analysis(
#     # Mixtral-8x7B configuration
#     hidden_size=4096,
#     num_heads=32,
#     seq_len=4096,
#     batch_size=1,
#     num_layers=32,
#     quant_config={
#         "activation": 16,
#         "kv_cache": 16,
#         "weight_ffn": 16,
#         "weight_attn": 16,
#         "rope_bit": 16
#     },
#     intermediate_size=14336,
#     num_key_value_heads=8,  # GQA configuration
#     num_experts_per_tok=2,   # Set to None to disable MoE
#     num_local_experts=8,
#     out_l=0,
#     max_gen_len=4096,
#     use_gate_ffn=True,  # Use SwiGLU activation
#     use_rope=True,      # Use RoPE position encoding
#     rope_theta=500000.0,
#     rope_scaling_factor=1.0,
#     output_csv_path="Mixtral-8x7B_density_transformer_analysis.csv"
# )