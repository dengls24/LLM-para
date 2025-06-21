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
    intermediate_size: int = None,  # 新增参数
    num_key_value_heads: int = None,  # 新增GQA参数
    out_l: int = 0,
    max_gen_len: int = 4096,  # 新增参数，最大生成长度
    output_csv_path: str = "true_density_transformer_analysis.csv",
    use_gate_ffn: bool = False,  # 新增参数，是否使用gate机制的FFN
    use_rope: bool = True,  # 新增参数，是否使用RoPE
    rope_theta: float = 10000.0,  # 新增参数，RoPE的theta参数
    rope_scaling_factor: float = 1.0  # 新增参数，RoPE的缩放因子，用于长度外推
):
    d = hidden_size // num_heads
    b, s, h, n, L = batch_size, seq_len, hidden_size, num_heads, num_layers
    o_l = out_l
    # 如果未指定intermediate_size，则默认使用4*h
    if intermediate_size is None:
        intermediate_size = 4 * h
    
    # 如果未指定num_key_value_heads，则默认等于num_heads（标准注意力）
    if num_key_value_heads is None:
        num_key_value_heads = num_heads
    
    # KV头数
    kv_n = num_key_value_heads
    
    def byte_size(shape, bitwidth):
        num_elements = 1
        for dim in shape:
            num_elements *= dim
        return num_elements * bitwidth / 8

    data = []

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
            "Density (Op/Byte)": density
        })

    # Quant settings
    a_bit = quant_config["activation"]
    w_attn = quant_config["weight_attn"]
    w_ffn = quant_config["weight_ffn"]
    kv_bit = quant_config["kv_cache"]
    rope_bit = quant_config["rope_bit"]

    # === PREFILL PHASE ===
    phase = "Prefill"
    seq = s

    # 对于W_Q，使用完整的num_heads
    flops = 2 * b * seq * h * h
    param_count = h * h
    add_row(phase, "xW_Q", "(b,s,h)", "(h,h)", "(b,s,h)",
            flops, param_count, (b, seq, h), (h, h), (b, seq, h), a_bit, w_attn)
    
    # 对于W_K和W_V，使用num_key_value_heads
    for name in ['W_K', 'W_V']:
        # 对于GQA，K和V的参数量和计算量会减少
        flops = 2 * b * seq * h * h * (kv_n / n)
        param_count = h * h * (kv_n / n)
        add_row(phase, f"x{name}", "(b,s,h)", f"(h,h*{kv_n/n})", "(b,s,h)",
                flops, param_count, (b, seq, h), (h, h * (kv_n / n)), (b, seq, h), a_bit, w_attn)
    
    # 添加RoPE计算
    if use_rope:
        # RoPE应用于Q和K，计算每个头的旋转操作
        # 每个位置需要计算sin和cos，然后应用旋转变换
        # 对于每个位置和每个头的每个维度对(dim_i, dim_i+1)，需要4个FLOPs(2次乘法，2次加法)
        # 总共有seq个位置，n个Q头，kv_n个K头，d/2个维度对
        
        # Q的RoPE计算
        rope_q_flops = b * seq * n * (d // 2) * 4  # 4 = 2次乘法 + 2次加法
        add_row(phase, "RoPE-Q", "(b,n,s,d)", "(s,d)", "(b,n,s,d)",
                rope_q_flops, 0, (b, n, seq, d), (seq, d), (b, n, seq, d), a_bit, rope_bit,
                note=f"RoPE theta={rope_theta}, scaling={rope_scaling_factor}")
        
        # K的RoPE计算
        rope_k_flops = b * seq * kv_n * (d // 2) * 4  # 4 = 2次乘法 + 2次加法
        add_row(phase, "RoPE-K", "(b,kv_n,s,d)", "(s,d)", "(b,kv_n,s,d)",
                rope_k_flops, 0, (b, kv_n, seq, d), (seq, d), (b, kv_n, seq, d), a_bit, rope_bit,
                note=f"RoPE theta={rope_theta}, scaling={rope_scaling_factor}")

    # 注意力计算 - 考虑GQA
    add_row(phase, "Q × Kᵀ", "(b,n,s,d)", "(b,kv_n,d,s)", "(b,n,s,s)",
            2 * b * n * seq * seq * d, 0,
            (b, n, seq, d), (b,kv_n, seq, d), (b, n, seq, seq), a_bit, kv_bit,
            note=f"GQA: {n} Q heads, {kv_n} KV heads")

    add_row(phase, "Attn × V", "(b,n,s,s)", "(b,kv_n,s,d)", "(b,n,s,d)",
            2 * b * n * seq * seq * d, 0,
            (b, n, seq, seq), (b, kv_n, seq, d), (b, n, seq, d), a_bit, kv_bit,
            note=f"GQA: {n} Q heads, {kv_n} KV heads")

    add_row(phase, "xW_O", "(b,s,h)", "(h,h)", "(b,s,h)",
            2 * b * seq * h * h, h * h,
            (b, seq, h), (h, h), (b, seq, h), a_bit, w_attn)

    # 修改FFN-1，使用intermediate_size替代4*h
    # 修改FFN-1，考虑gate机制
    if use_gate_ffn:
        # 计算W₁和W_gate两个矩阵的FLOPs
        gate_flops = 2 * b * seq * h * intermediate_size * 2  # 两个矩阵乘法
        # 额外的逐元素乘法
        gate_flops += b * seq * intermediate_size  # SiLU(W₁x) ⊙ (W_gate*x)
        gate_param_count = h * intermediate_size * 2  # 两个权重矩阵
        
        add_row(phase, "FFN-1 (with Gate)", "(b,s,h)", f"(h,{intermediate_size}*2)", f"(b,s,{intermediate_size})",
                gate_flops, gate_param_count, 
                (b, seq, h), (h, intermediate_size * 2), (b, seq, intermediate_size), a_bit, w_ffn,
                note="包含Gate机制")
    else:
        # 原始FFN-1计算
        add_row(phase, "FFN-1", "(b,s,h)", f"(h,{intermediate_size})", f"(b,s,{intermediate_size})",
                2 * b * seq * h * intermediate_size, h * intermediate_size,
                (b, seq, h), (h, intermediate_size), (b, seq, intermediate_size), a_bit, w_ffn)

    # 修改FFN-2，使用intermediate_size替代4*h
    add_row(phase, "FFN-2", f"(b,s,{intermediate_size})", f"({intermediate_size},h)", "(b,s,h)",
            2 * b * seq * h * intermediate_size, intermediate_size * h,
            (b, seq, intermediate_size), (intermediate_size, h), (b, seq, h), a_bit, w_ffn)

    # === DECODE PHASE ===
    phase = "Decode"
    seq = 1 #生成一个token
    hist = s
    #print (hist)
    o_l = 0  # 初始生成token位置
    # 计算第一个token的decode阶段
    # 对于W_Q，使用完整的num_heads
    flops = 2 * b * seq * h * h
    param_count = h * h
    add_row(phase, "xW_Q", "(b,1,h)", "(h,h)", "(b,1,h)",
            flops, param_count, (b, seq, h), (h, h), (b, seq, h), a_bit, w_attn)
    
    # 对于W_K和W_V，使用num_key_value_heads
    for name in ['W_K', 'W_V']:
        # 对于GQA，K和V的参数量和计算量会减少
        flops = 2 * b * seq * h * h * (kv_n / n)
        param_count = h * h * (kv_n / n)
        add_row(phase, f"x{name}", "(b,1,h)", f"(h,h*{kv_n/n})", "(b,1,h)",
                flops, param_count, (b, seq, h), (h, h * (kv_n / n)), (b, seq, h), a_bit, w_attn)
    
    # 添加RoPE计算 - Decode阶段
    if use_rope:
        # Q的RoPE计算 - 只有一个token
        rope_q_flops = b * 1 * n * (d // 2) * 4  # 4 = 2次乘法 + 2次加法
        add_row(phase, "RoPE-Q", "(b,n,1,d)", "(1,d)", "(b,n,1,d)",
                rope_q_flops, 0, (b, n, 1, d), (1, d), (b, n, 1, d), a_bit, rope_bit,
                note=f"RoPE theta={rope_theta}, scaling={rope_scaling_factor}, 位置={s+o_l}")
        
        # K的RoPE计算 - 只有一个token
        rope_k_flops = b * 1 * kv_n * (d // 2) * 4  # 4 = 2次乘法 + 2次加法
        add_row(phase, "RoPE-K", "(b,kv_n,1,d)", "(1,d)", "(b,kv_n,1,d)",
                rope_k_flops, 0, (b, kv_n, 1, d), (1, d), (b, kv_n, 1, d), a_bit, rope_bit,
                note=f"RoPE theta={rope_theta}, scaling={rope_scaling_factor}, 位置={s+o_l}")

    # 注意力计算 - 考虑GQA和更长的历史
    add_row(phase, "Q × Kᵀ", "(b,n,1,d)", "(b,kv_n,d,s+o_l+1)", "(b,n,1,s+o_l+1)",
            2 * b * n * hist * d, 0,
            (b, n, 1, d), (b, kv_n, d, hist + 1), (b, n, 1, hist + 1), a_bit, kv_bit,
            note=f"GQA: {n} Q heads, {kv_n} KV heads, 历史长度: {hist}")

    add_row(phase, "Attn × V", "(b,n,1,s+o_l+1)", "(b,kv_n,s+o_l+1,d)", "(b,n,1,d)",
            2 * b * n * (hist + 1) * d, 0,
            (b, n, 1, hist + 1), (b, kv_n, hist + 1, d), (b, n, 1, d), a_bit, kv_bit,
            note=f"GQA: {n} Q heads, {kv_n} KV heads, 历史长度: {hist}")

    add_row(phase, "xW_O", "(b,1,h)", "(h,h)", "(b,1,h)",
            2 * b * seq * h * h, h * h,
            (b, seq, h), (h, h), (b, seq, h), a_bit, w_attn)
    
    # 修改FFN-1，使用intermediate_size替代4*h
    # 修改FFN-1，考虑gate机制
    if use_gate_ffn:
        # 计算W₁和W_gate两个矩阵的FLOPs
        gate_flops = 2 * b * seq * h * intermediate_size * 2  # 两个矩阵乘法
        # 额外的逐元素乘法
        gate_flops += b * seq * intermediate_size  # SiLU(W₁x) ⊙ (W_gate*x)
        gate_param_count = h * intermediate_size * 2  # 两个权重矩阵
        
        add_row(phase, "FFN-1 (with Gate)", "(b,s,h)", f"(h,{intermediate_size}*2)", f"(b,s,{intermediate_size})",
                gate_flops, gate_param_count, 
                (b, seq, h), (h, intermediate_size * 2), (b, seq, intermediate_size), a_bit, w_ffn,
                note="包含Gate机制")
    else:
        # 原始FFN-1计算
        add_row(phase, "FFN-1", "(b,s,h)", f"(h,{intermediate_size})", f"(b,s,{intermediate_size})",
                2 * b * seq * h * intermediate_size, h * intermediate_size,
                (b, seq, h), (h, intermediate_size), (b, seq, intermediate_size), a_bit, w_ffn)

    # 修改FFN-2，使用intermediate_size替代4*h
    add_row(phase, "FFN-2", f"(b,1,{intermediate_size})", f"({intermediate_size},h)", "(b,1,h)",
            2 * b * seq * h * intermediate_size, intermediate_size * h,
            (b, seq, intermediate_size), (intermediate_size, h), (b, seq, h), a_bit, w_ffn)

    # 计算最后一个token的decode阶段
    phase = "Decode_Last"
    o_l = max_gen_len - 1  # 最后一个生成token的位置
    hist = s + o_l  # 历史长度 = 原始序列 + 已生成的tokens
    
    # 对于W_Q，使用完整的num_heads
    flops = 2 * b * seq * h * h
    param_count = h * h
    add_row(phase, "xW_Q", "(b,1,h)", "(h,h)", "(b,1,h)",
            flops, param_count, (b, seq, h), (h, h), (b, seq, h), a_bit, w_attn)
    
    # 对于W_K和W_V，使用num_key_value_heads
    for name in ['W_K', 'W_V']:
        # 对于GQA，K和V的参数量和计算量会减少
        flops = 2 * b * seq * h * h * (kv_n / n)
        param_count = h * h * (kv_n / n)
        add_row(phase, f"x{name}", "(b,1,h)", f"(h,h*{kv_n/n})", "(b,1,h)",
                flops, param_count, (b, seq, h), (h, h * (kv_n / n)), (b, seq, h), a_bit, w_attn)
    
    # 添加RoPE计算 - Decode_Last阶段
    if use_rope:
        # Q的RoPE计算 - 只有一个token，但位置是最后一个
        rope_q_flops = b * 1 * n * (d // 2) * 4  # 4 = 2次乘法 + 2次加法
        add_row(phase, "RoPE-Q", "(b,n,1,d)", "(1,d)", "(b,n,1,d)",
                rope_q_flops, 0, (b, n, 1, d), (1, d), (b, n, 1, d), a_bit, rope_bit,
                note=f"RoPE theta={rope_theta}, scaling={rope_scaling_factor}, 位置={s+o_l}")
        
        # K的RoPE计算 - 只有一个token，但位置是最后一个
        rope_k_flops = b * 1 * kv_n * (d // 2) * 4  # 4 = 2次乘法 + 2次加法
        add_row(phase, "RoPE-K", "(b,kv_n,1,d)", "(1,d)", "(b,kv_n,1,d)",
                rope_k_flops, 0, (b, kv_n, 1, d), (1, d), (b, kv_n, 1, d), a_bit, rope_bit,
                note=f"RoPE theta={rope_theta}, scaling={rope_scaling_factor}, 位置={s+o_l}")

    # 注意力计算 - 考虑GQA和更长的历史
    add_row(phase, "Q × Kᵀ", "(b,n,1,d)", "(b,kv_n,d,s+o_l+1)", "(b,n,1,s+o_l+1)",
            2 * b * n * hist * d, 0,
            (b, n, 1, d), (b, kv_n, d, hist + 1), (b, n, 1, hist + 1), a_bit, kv_bit,
            note=f"GQA: {n} Q heads, {kv_n} KV heads, 历史长度: {hist}")

    add_row(phase, "Attn × V", "(b,n,1,s+o_l+1)", "(b,kv_n,s+o_l+1,d)", "(b,n,1,d)",
            2 * b * n * (hist + 1) * d, 0,
            (b, n, 1, hist + 1), (b, kv_n, hist + 1, d), (b, n, 1, d), a_bit, kv_bit,
            note=f"GQA: {n} Q heads, {kv_n} KV heads, 历史长度: {hist}")

    add_row(phase, "xW_O", "(b,1,h)", "(h,h)", "(b,1,h)",
            2 * b * seq * h * h, h * h,
            (b, seq, h), (h, h), (b, seq, h), a_bit, w_attn)

    # 修改FFN-1，使用intermediate_size替代4*h
    # 修改FFN-1，考虑gate机制
    if use_gate_ffn:
        # 计算W₁和W_gate两个矩阵的FLOPs
        gate_flops = 2 * b * seq * h * intermediate_size * 2  # 两个矩阵乘法
        # 额外的逐元素乘法
        gate_flops += b * seq * intermediate_size  # SiLU(W₁x) ⊙ (W_gate*x)
        gate_param_count = h * intermediate_size * 2  # 两个权重矩阵
        
        add_row(phase, "FFN-1 (with Gate)", "(b,s,h)", f"(h,{intermediate_size}*2)", f"(b,s,{intermediate_size})",
                gate_flops, gate_param_count, 
                (b, seq, h), (h, intermediate_size * 2), (b, seq, intermediate_size), a_bit, w_ffn,
                note="包含Gate机制")
    else:
        # 原始FFN-1计算
        add_row(phase, "FFN-1", "(b,s,h)", f"(h,{intermediate_size})", f"(b,s,{intermediate_size})",
                2 * b * seq * h * intermediate_size, h * intermediate_size,
                (b, seq, h), (h, intermediate_size), (b, seq, intermediate_size), a_bit, w_ffn)

    # 修改FFN-2，使用intermediate_size替代4*h
    add_row(phase, "FFN-2", f"(b,1,{intermediate_size})", f"({intermediate_size},h)", "(b,1,h)",
            2 * b * seq * h * intermediate_size, intermediate_size * h,
            (b, seq, intermediate_size), (intermediate_size, h), (b, seq, h), a_bit, w_ffn)

    df = pd.DataFrame(data)

    df.to_csv(output_csv_path, index=False)
    # === Summary ===

    df['FLOPs'] = df['FLOPs'] * L
    df['Total Bytes'] = df['Total Bytes'] * L

    # 只计算prefill阶段的参数量，并按照权重位宽转换为GB单位
    prefill_df = df[df['Phase'] == 'Prefill']
    total_params = prefill_df['Param Count'].sum() * L
    # 将参数量转换为GB，考虑权重位宽
    total_params_gb = (total_params * w_ffn / 8) / (1024**3)  # 使用FFN权重位宽进行转换

    total_flops = df['FLOPs'].sum()
    total_bytes = df['Total Bytes'].sum()

    print(f"\n🔢 Total {L}-Layer Summary:")
    print(f"  FLOPs        = {total_flops:.2e} Op")
    print(f"  Param Count  = {total_params:.2e} ({total_params_gb:.2f} GB @{w_ffn}bit)")
    print(f"  Memory Access= {total_bytes / (1024**2):.2f} MB")

    # === KV Cache Write Analysis ===
    # 修改KV缓存计算，考虑GQA
    kv_cache_bytes_per_token = L * kv_n * d * 2 * (kv_bit / 8)  # Key + Value，只考虑kv_n个头
    kv_cache_total = kv_cache_bytes_per_token * s * b  # 原始序列的KV缓存

    # 计算生成过程中的KV缓存
    gen_kv_cache_total = kv_cache_bytes_per_token * max_gen_len * b  # 生成tokens的KV缓存
    max_kv_cache = kv_cache_bytes_per_token * (s + max_gen_len) * b  # 最大KV缓存（原始+生成）
    
    print(f"\n💾 KV Cache Analysis:")
    print(f"  Per token: {kv_cache_bytes_per_token:.1f} Bytes")
    print(f"  Prefill seq: {kv_cache_total / (1024 ** 2):.2f} MB")
    print(f"  Generation: {gen_kv_cache_total / (1024 ** 2):.2f} MB")
    print(f"  Max total: {max_kv_cache / (1024 ** 2):.2f} MB (序列长度: {s + max_gen_len})")
    print(f"  GQA ratio: {kv_n}/{n} = {kv_n/n:.2f}x reduction")


#     print(f"\n💾 KV Cache Total Write:")
#     print(f"  Per token: {kv_cache_bytes_per_token:.1f} Bytes")
#     print(f"  Full seq : {kv_cache_total / (1024 ** 2):.2f} MB")
#     print(f"  GQA ratio: {kv_n}/{n} = {kv_n/n:.2f}x reduction")
    
    # 计算生成最后一个token的计算量
    last_token_df = df[df['Phase'] == 'Decode_Last']
    last_token_flops = last_token_df['FLOPs'].sum() * L
    last_token_bytes = last_token_df['Total Bytes'].sum() * L
    print(f"\n🔢 生成最后一个Token (位置 {max_gen_len}):")
    print(f"  FLOPs        = {last_token_flops:.2e} Op")
    print(f"  Memory Access= {last_token_bytes / (1024**2):.2f} MB")
    print(f"  历史长度      = {s + max_gen_len - 1} tokens")



#     # === Visualization ===
#     grouped = df.groupby('Phase')[['FLOPs', 'Total Bytes']].sum()
#     grouped['FLOPs'] /= 1e9  # Convert to GFLOPs
#     grouped['Total Bytes'] /= 1024 ** 2  # Convert to MB

#     ax = grouped.plot(kind='bar', secondary_y='Total Bytes', figsize=(10, 5))
#     ax.set_ylabel("FLOPs (G)")
#     ax.right_ax.set_ylabel("Memory Access (MB)")
#     ax.set_title(f"Phase-wise Computation vs Memory ({L} Layers)")
#     plt.tight_layout()
#     plt.savefig("phasewise_flops_memory.png")
#     plt.show()

#    # df.to_csv(output_csv_path, index=False)
#     print(f"\n✅ CSV saved to '{output_csv_path}', chart saved to 'phasewise_flops_memory.png'")
    
# # 运行分析，添加intermediate_size参数和num_key_value_heads参数
# # 根据BitNet配置文件，使用正确的参数
# compute_and_plot_transformer_analysis(
#     hidden_size=2560,
#     num_heads=20,
#     seq_len=4096,
#     batch_size=1,
#     num_layers=30,
#     intermediate_size=6912,  # 从BitNet配置文件中获取
#     num_key_value_heads=5,   # 从BitNet配置文件中获取
#     out_l = 0,
#     max_gen_len = 4096,      # 最大生成长度
#     quant_config={
#         "activation": 8,
#         "kv_cache": 4,
#         "weight_ffn": 2,
#         "weight_attn": 2
#     },
#     use_gate_ffn=True,  # 启用gate机制
#     use_rope=True,      # 启用RoPE
#     rope_theta=10000.0, # 标准RoPE theta参数
#     rope_scaling_factor=1.0, # 不进行缩放
#     output_csv_path="true_density_transformer_analysis.csv"
# )

    # 添加RoPE相关的分析
    if use_rope:
        print(f"\n🔄 RoPE分析:")
        print(f"  基础theta: {rope_theta}")
        print(f"  缩放因子: {rope_scaling_factor}")
        print(f"  最大位置: {s + max_gen_len - 1}")
        
        # 计算最大频率
        max_freq = 1.0 / (rope_theta * rope_scaling_factor)
        nyquist_freq = 0.5  # 奈奎斯特频率
        
        # 计算RoPE的有效上下文长度
        effective_context_length = int(np.pi / (max_freq * np.pi / (d // 2)))
        
        print(f"  最大频率: {max_freq:.6f}")
        print(f"  理论有效上下文长度: ~{effective_context_length}")
        
        if s + max_gen_len > effective_context_length:
            print(f"  ⚠️ 警告: 当前序列长度({s + max_gen_len})超过了理论有效上下文长度({effective_context_length})，可能导致性能下降")
        else:
            print(f"  ✅ 当前序列长度({s + max_gen_len})在理论有效上下文长度({effective_context_length})范围内")

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
    
# 运行分析，添加intermediate_size参数和num_key_value_heads参数
# 根据BitNet配置文件，使用正确的参数
# compute_and_plot_transformer_analysis(
#     hidden_size=2560,
#     num_heads=20,
#     seq_len=4096,
#     batch_size=1,
#     num_layers=30,
#     intermediate_size=6912,  # 从BitNet配置文件中获取
#     num_key_value_heads=5,   # 从BitNet配置文件中获取
#     out_l = 0,
#     max_gen_len = 4096,      # 最大生成长度
#     quant_config={
#         "activation": 8,
#         "kv_cache": 4,
#         "weight_ffn": 2,
#         "weight_attn": 2,
#         "rope_bit": 16
#     },
#     use_gate_ffn=True,  # 启用gate机制
#     use_rope=True,      # 启用RoPE
#     rope_theta=500000.0, # 标准RoPE theta参数
#     rope_scaling_factor=1.0, # 不进行缩放
#     output_csv_path="true_density_transformer_analysis.csv"
# )
# compute_and_plot_transformer_analysis(
#         #llama3-8B BF16
#     hidden_size=4096,
#     num_heads=32,
#     seq_len=4096,
#     batch_size=1,
#     num_layers=32,
#     intermediate_size=14336,  # 从BitNet配置文件中获取
#     num_key_value_heads=8,   # 从BitNet配置文件中获取
#     out_l = 0,
#     max_gen_len = 4096,      # 最大生成长度
#     quant_config={
#         "activation": 16,
#         "kv_cache": 16,
#         "weight_ffn": 16,
#         "weight_attn": 16,
#         "rope_bit": 16
#     },
#     use_gate_ffn=True,  # 启用gate机制
#     use_rope=True,      # 启用RoPE
#     rope_theta=500000.0, # 标准RoPE theta参数
#     rope_scaling_factor=1.0, # 不进行缩放
#     output_csv_path="llama-true_density_transformer_analysis.csv"
# )

compute_and_plot_transformer_analysis(
        #llama-65B FP16
    hidden_size=8192,
    num_heads=64,
    seq_len=4096,
    batch_size=1,
    num_layers=80,
    intermediate_size=22016,  # 从BitNet配置文件中获取
    num_key_value_heads=64,   # 从BitNet配置文件中获取
    out_l = 0,
    max_gen_len = 4096,      # 最大生成长度
    quant_config={
        "activation": 16,
        "kv_cache": 16,
        "weight_ffn": 16,
        "weight_attn": 16,
        "rope_bit": 16
    },
    use_gate_ffn=True,  # 启用gate机制
    use_rope=True,      # 启用RoPE
    rope_theta=500000.0, # 标准RoPE theta参数
    rope_scaling_factor=1.0, # 不进行缩放
    output_csv_path="llama65B-true_density_transformer_analysis.csv"
)



