import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from config import CONFIGS

# User selects the required model
model_name = "llama3-8B_BF16"   #Mixtral_8x7B / llama3-8B_BF16 / 

data = []

def byte_size(shape, bitwidth):
    num_elements = 1
    for dim in shape:
        num_elements *= dim
    return num_elements * bitwidth / 8

def add_row(phase, name, input1, input2, output,
            flops, param_count,
            input_shape, weight_shape, output_shape,
            act_bits, weight_bits, note=""):

    B_input = byte_size(input_shape, act_bits)
    B_output = byte_size(output_shape, act_bits)
    B_weight = byte_size(weight_shape, weight_bits) if weight_shape else 0

    B_total = B_input + B_output + B_weight
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
        "input_shape": input_shape,
        "weight_shape": weight_shape
    })

def compute_and_plot_transformer_analysis(
    hidden_size: int,
    num_heads: int,
    seq_len: int,
    batch_size: int,
    num_layers: int,
    quant_config: dict,
    intermediate_size: int = None,  # New parameter
    num_key_value_heads: int = None,  # New GQA parameter
    num_experts_per_tok: int = None,    # New MoE parameter  2
    num_local_experts: int = None,      # New MoE parameter  8
    max_gen_len: int = 4096,  # New parameter, maximum generation length
    output_csv_path: str = "transformer_analysis.csv",
    use_gate_ffn: bool = False,  # New parameter, whether to use gate mechanism in FFN
    rope_theta: float = None,  # New parameter, RoPE theta parameter
    rope_scaling_factor: float = None  # New parameter, RoPE scaling factor for length extrapolation
):
    d = hidden_size // num_heads
    b, s, h, n, L = batch_size, seq_len, hidden_size, num_heads, num_layers

    use_rope = False
    if rope_theta is not None:
        use_rope = True

    use_moe = False
    # If MoE parameters are provided, enable MoE computation
    if num_experts_per_tok is not None:
        use_moe = True

    # If intermediate_size is not specified, default to 4*h
    if intermediate_size is None:
        intermediate_size = 4 * h
    
    # If num_key_value_heads is not specified, default to num_heads (standard attention)
    if num_key_value_heads is None:
        num_key_value_heads = num_heads
    
    # Number of KV heads
    kv_n = num_key_value_heads
    
    # Quant settings
    a_bit = quant_config["activation"]
    w_attn = quant_config["weight_attn"]
    w_ffn = quant_config["weight_ffn"]
    kv_bit = quant_config["kv_cache"]
    rope_bit = quant_config["rope_bit"]

    for phase in ["Prefill","Decode"]:

        if phase == "Prefill" :
            seq = s
            hist = 0
        else:
            seq = 1
            hist = s

        # For W_Q, use full num_heads
        flops = 2 * b * seq * h * h
        param_count = h * h
        add_row(phase, "xW_Q", f"(b,s,h)", "(h,h)", "(b,s,h)",
                flops, param_count, (b, seq, h), (h, h), (b, seq, h), a_bit, w_attn)
        
        # For W_K and W_V, use num_key_value_heads
        for name in ['W_K', 'W_V']:
            # For GQA, K and V parameter count and computation will be reduced
            flops = 2 * b * seq * h * h * (kv_n / n)
            param_count = h * h * (kv_n / n)
            add_row(phase, f"x{name}", "(b,s,h)", f"(h,h*{kv_n/n})", "(b,s,h)",
                    flops, param_count, (b, seq, h), (h, int(h * (kv_n / n))), (b, seq, int(h * (kv_n / n))), a_bit, w_attn)
            
        # Add RoPE computation
        if use_rope:
            # RoPE is applied to Q and K, computing rotation operations for each head
            # Each position needs to compute sin and cos, then apply rotation transformation
            # For each position and each dimension pair (dim_i, dim_i+1) of each head, 4 FLOPs are needed (2 multiplications, 2 additions)
            # Total: seq positions, n Q heads, kv_n K heads, d/2 dimension pairs
            
            # RoPE computation for Q
            rope_q_flops = b * seq * n * (d // 2) * 4  # 4 = 2 multiplications + 2 additions
            add_row(phase, "RoPE-Q", "(b,n,s,d)", "(s,d)", "(b,n,s,d) ",
                    rope_q_flops, 0, (b, n, seq, d), (seq, d), (b, n, seq, d), a_bit, rope_bit,
                    note=f"RoPE theta={rope_theta}, scaling={rope_scaling_factor}")
            
            # RoPE computation for K
            rope_k_flops = b * seq * kv_n * (d // 2) * 4  # 4 = 2 multiplications + 2 additions
            add_row(phase, "RoPE-K", "(b,kv_n,s,d)", "(s,d)", "(b,kv_n,s,d)",
                    rope_k_flops, 0, (b, kv_n, seq, d), (seq, d), (b, kv_n, seq, d), a_bit, rope_bit,
                    note=f"RoPE theta={rope_theta}, scaling={rope_scaling_factor}")
            
        # Attention computation - considering GQA
        add_row(phase, "Q × Kᵀ", "(b,n,s,d)", "(b,kv_n,d,hist+s)", "(b,n,s,hist+s)",
                2 * b * n * seq * (seq + hist) * d, 0,
                (b, n, seq, d), (b,kv_n, d, seq + hist), (b, n, seq, seq + hist), a_bit, kv_bit,
                note=f"GQA: {n} Q heads, {kv_n} KV heads")

        add_row(phase, "Attn × V", "(b,n,s,hist+s)", "(b,kv_n,hist+s,d)", "(b,n,s,d)",
                2 * b * n * seq * (seq + hist) * d, 0,
                (b, n, seq, seq + hist), (b, kv_n, seq + hist, d), (b, n, seq, d), a_bit, kv_bit,
                note=f"GQA: {n} Q heads, {kv_n} KV heads")

        add_row(phase, "xW_O", "(b,s,h)", "(h,h)", "(b,s,h)",
                2 * b * seq * h * h, h * h,
                (b, seq, h), (h, h), (b, seq, h), a_bit, w_attn)
            
        # FFN computation
        if use_moe :
            # Router computation overhead
            moe_router_flops = b * seq * h * num_local_experts * 2
            param_count = h * num_local_experts
            add_row(phase, "Router", "(b,s,h)", f"(h,{num_local_experts})", f"(b,s,{num_local_experts})", 
                    moe_router_flops, param_count, 
                    (b,seq,h), (h,num_local_experts), (b,seq,num_local_experts), a_bit, w_ffn)

            if phase == "Prefill":
                # FFN-1(up + gate)(with MoE)
                FFN_1_moe = b * seq * h * intermediate_size * 2 * 2 * num_experts_per_tok # * two matrix multiplications * each token computes num_experts_per_tok times
                # Additional element-wise multiplication
                FFN_1_moe += b * seq * num_experts_per_tok * intermediate_size  # SiLU(W₁x) ⊙ (W_gate*x)

                param_count = h * intermediate_size * 2 * num_local_experts  # two weight matrices * assume prefill uses all experts
                add_row(phase, "FFN-1(with MoE)", f"(b,{s}*num_experts_per_tok,h)", f"(num_local_experts,h,{intermediate_size}*2)", f"(b,{s}*num_experts_per_tok,{intermediate_size})", 
                        FFN_1_moe, param_count, 
                        (b, seq * num_experts_per_tok, h), (h, intermediate_size * 2 * num_local_experts), (b, seq * num_experts_per_tok, intermediate_size ), a_bit, w_ffn)

                # FFN-2(with MoE)
                FFN_2_moe = b * seq * intermediate_size * h * 2 * num_experts_per_tok
                param_count = intermediate_size * h * num_local_experts
                add_row(phase, "FFN-2(with MoE)", f"(b, {s}*num_experts_per_tok, {intermediate_size})", f"(num_local_experts, {intermediate_size}, h)", f"(b,{s}*num_experts_per_tok,h)",
                        FFN_2_moe, param_count,
                        (b, seq * num_experts_per_tok, intermediate_size), (num_local_experts,intermediate_size, h), (b, seq * num_experts_per_tok, h), a_bit, w_ffn)
            else : # "Decode"
                # FFN-1(up + gate)(with MoE)
                FFN_1_moe = b * seq * h * intermediate_size * 2 * 2 * num_experts_per_tok # * two matrix multiplications * each token computes num_experts_per_tok times
                # Additional element-wise multiplication
                FFN_1_moe += b * seq * num_experts_per_tok * intermediate_size  # SiLU(W₁x) ⊙ (W_gate*x)

                param_count = h * intermediate_size * 2 * num_experts_per_tok  # two weight matrices * decode phase only uses num_experts_per_tok experts
                add_row(phase, "FFN-1(with MoE)", f"(b,{s}*num_experts_per_tok,h)", f"(num_experts_per_tok,h,{intermediate_size}*2)", f"(b,{s}*num_experts_per_tok,{intermediate_size})", 
                        FFN_1_moe, param_count, 
                        (b, seq * num_experts_per_tok, h), (h, intermediate_size * 2 * num_experts_per_tok), (b, seq * num_experts_per_tok, intermediate_size ), a_bit, w_ffn)
                
                # FFN-2(with MoE)
                FFN_2_moe = b * seq * intermediate_size * h * 2 * num_experts_per_tok
                param_count = intermediate_size * h * num_experts_per_tok
                add_row(phase, "FFN-2(with MoE)", f"(b, {s}*num_experts_per_tok, {intermediate_size})", f"(num_experts_per_tok, {intermediate_size}, h)", f"(b,{s}*num_experts_per_tok,h)",
                        FFN_2_moe, param_count,
                        (b, seq * num_experts_per_tok, intermediate_size), (num_experts_per_tok,intermediate_size, h), (b, seq * num_experts_per_tok, h), a_bit, w_ffn) 
        else : # Without MoE
            # Modify FFN-1, use intermediate_size instead of 4*h
            # Modify FFN-1, consider gate mechanism
            if use_gate_ffn:
                # Compute FLOPs for both W₁ and W_gate matrices
                gate_flops = 2 * b * seq * h * intermediate_size * 2  # two matrix multiplications
                # Additional element-wise multiplication
                gate_flops += b * seq * intermediate_size  # SiLU(W₁x) ⊙ (W_gate*x)
                gate_param_count = h * intermediate_size * 2  # two weight matrices
                
                add_row(phase, "FFN-1 (with Gate)", "(b,s,h)", f"(h,{intermediate_size}*2)", f"(b,s,{intermediate_size})",
                        gate_flops, gate_param_count, 
                        (b, seq, h), (2, h, intermediate_size), (b, seq, intermediate_size), a_bit, w_ffn,
                        note="Includes Gate mechanism")
            else:
                # Original FFN-1 computation
                add_row(phase, "FFN-1", "(b,s,h)", f"(h,{intermediate_size})", f"(b,s,{intermediate_size})",
                        2 * b * seq * h * intermediate_size, h * intermediate_size,
                        (b, seq, h), (h, intermediate_size), (b, seq, intermediate_size), a_bit, w_ffn)

            # Modify FFN-2, use intermediate_size instead of 4*h
            add_row(phase, "FFN-2", f"(b,s,{intermediate_size})", f"({intermediate_size},h)", "(b,s,h)",
                    2 * b * seq * h * intermediate_size, intermediate_size * h,
                    (b, seq, intermediate_size), (intermediate_size, h), (b, seq, h), a_bit, w_ffn)

    # Convert existing data to a Pandas DataFrame for subsequent processing and saving
    df = pd.DataFrame(data)
    df.to_csv(output_csv_path, index=False)
    # === Summary ===

    df['FLOPs'] = df['FLOPs'] * L
    df['Total Bytes'] = df['Total Bytes'] * L

    # Only calculate parameter count for prefill phase, and convert to GB units considering weight bitwidth
    prefill_df = df[df['Phase'] == 'Prefill']
    total_params = prefill_df['Param Count'].sum() * L
    # Convert parameter count to GB, considering weight bitwidth
    total_params_gb = (total_params * w_ffn / 8) / (1024**3)  # Use FFN weight bitwidth for conversion

    total_flops = df['FLOPs'].sum()
    total_bytes = df['Total Bytes'].sum()

    print(f"\n🔢 Total {L}-Layer Summary:")
    print(f"  FLOPs        = {total_flops:.2e} Op")
    print(f"  Param Count  = {total_params:.2e} ({total_params_gb:.2f} GB @{w_ffn}bit)")
    print(f"  Memory Access= {total_bytes / (1024**2):.2f} MB")

    # === KV Cache Write Analysis ===
    # Modify KV cache calculation, considering GQA
    kv_cache_bytes_per_token = L * kv_n * d * 2 * (kv_bit / 8)  # Key + Value, only consider kv_n heads
    kv_cache_total = kv_cache_bytes_per_token * s * b  # KV cache for original sequence

    # Calculate KV cache during generation process
    gen_kv_cache_total = kv_cache_bytes_per_token * max_gen_len * b  # KV cache for generated tokens
    max_kv_cache = kv_cache_bytes_per_token * (s + max_gen_len) * b  # Maximum KV cache (original + generated)
    
    print(f"\n💾 KV Cache Analysis:")
    print(f"  Per token: {kv_cache_bytes_per_token:.1f} Bytes")
    print(f"  Prefill seq: {kv_cache_total / (1024 ** 2):.2f} MB")
    print(f"  Generation: {gen_kv_cache_total / (1024 ** 2):.2f} MB")
    print(f"  Max total: {max_kv_cache / (1024 ** 2):.2f} MB (sequence length: {s + max_gen_len})")
    print(f"  GQA ratio: {kv_n}/{n} = {kv_n/n:.2f}x reduction")

    # 计算生成最后一个token的计算量
    # last_token_df = df[df['Phase'] == 'Decode_Last']
    # last_token_flops = last_token_df['FLOPs'].sum() * L
    # last_token_bytes = last_token_df['Total Bytes'].sum() * L
    # print(f"\n🔢 生成最后一个Token (位置 {max_gen_len}):")
    # print(f"  FLOPs        = {last_token_flops:.2e} Op")
    # print(f"  Memory Access= {last_token_bytes / (1024**2):.2f} MB")
    # print(f"  历史长度      = {s + max_gen_len - 1} tokens")

    # Add RoPE-related analysis
    if use_rope:
        print(f"\n🔄 RoPE Analysis:")
        print(f"  Base theta: {rope_theta}")
        print(f"  Scaling factor: {rope_scaling_factor}")
        print(f"  Max position: {s + max_gen_len - 1}")
        
        # Calculate maximum frequency
        max_freq = 1.0 / (rope_theta * rope_scaling_factor)
        nyquist_freq = 0.5  # Nyquist frequency
        
        # Calculate RoPE's effective context length
        effective_context_length = int(np.pi / (max_freq * np.pi / (d // 2)))
        
        print(f"  Max frequency: {max_freq:.6f}")
        print(f"  Theoretical effective context length: ~{effective_context_length}")
        
        if s + max_gen_len > effective_context_length:
            print(f"  ⚠️ Warning: Current sequence length({s + max_gen_len}) exceeds theoretical effective context length({effective_context_length}), may cause performance degradation")
        else:
            print(f"  ✅ Current sequence length({s + max_gen_len}) is within theoretical effective context length({effective_context_length}) range")

    # === Visualization ===
    grouped = df.groupby('Phase')[['FLOPs', 'Total Bytes']].sum()
    grouped['FLOPs'] /= 1e9  # Convert to GFLOPs
    grouped['Total Bytes'] /= 1024 ** 2  # Convert to MB

    ax = grouped.plot(kind='bar', secondary_y='Total Bytes', figsize=(10, 5))
    ax.set_ylabel("FLOPs (G)")
    ax.right_ax.set_ylabel("Memory Access (MB)")
    ax.set_title(f"Phase-wise Computation vs Memory ({L} Layers)")
    plt.tight_layout()
    plt.savefig("phasewise_flops_memory.png")
    plt.show()

   # df.to_csv(output_csv_path, index=False)
    print(f"\n✅ CSV saved to '{output_csv_path}', chart saved to 'phasewise_flops_memory.png'")

if __name__ == "__main__":
    compute_and_plot_transformer_analysis(**CONFIGS[model_name])
        