CONFIGS = {
    # PE array configuration
    "PE_array" : {
        'Data_flow': "WS",
        'R': 32,
        'C': 32,
    },

    # Mixtral_8x7B configuration
    "Mixtral_8x7B" : {
        "hidden_size" : 4096,
        "num_heads" : 32,
        "seq_len" : 2048,
        "batch_size" : 1,
        "num_layers" : 32,
        "quant_config" : {
            "activation": 16,
            "kv_cache": 16,
            "weight_ffn": 16,
            "weight_attn": 16,
            "rope_bit": 16
        },
        "intermediate_size" : 14336,  
        "num_key_value_heads" : 8,   
        "num_experts_per_tok" : 2,   
        "num_local_experts" : 8,     
        "max_gen_len" : 4096,      # Maximum generation length
        "use_gate_ffn" : True,  # Enable gate mechanism
        "rope_theta" : 500000.0, # Standard RoPE theta parameter
        "rope_scaling_factor" : 1.0, # No scaling
        "output_csv_path" : "Mixtral-8x7B_transformer_analysis_with_moe_test.csv"
    },

    "llama3-8B_BF16" : {
        "hidden_size" : 4096,
        "num_heads" : 32,
        "seq_len" : 2048,
        "batch_size" : 1,
        "num_layers" : 32,
        "intermediate_size" : 14336,  # Obtained from BitNet configuration file
        "num_key_value_heads" : 8,   # Obtained from BitNet configuration file
        "max_gen_len" : 4096,      # Maximum generation length
        "quant_config" : {
            "activation": 16,
            "kv_cache": 16,
            "weight_ffn": 16,
            "weight_attn": 16,
            "rope_bit": 16
        },
        "use_gate_ffn" : True,  # Enable gate mechanism
        "rope_theta" : 500000.0, # Standard RoPE theta parameter
        "rope_scaling_factor" : 1.0, # No scaling
        "output_csv_path" : "llama3-8B_BF16.csv"
        }
}

# Run analysis, add intermediate_size parameter and num_key_value_heads parameter
# Use correct parameters according to BitNet configuration file
# compute_and_plot_transformer_analysis(
#     hidden_size=2560,
#     num_heads=20,
#     seq_len=4096,
#     batch_size=1,
#     num_layers=30,
#     intermediate_size=6912,  # Obtained from BitNet configuration file
#     num_key_value_heads=5,   # Obtained from BitNet configuration file
#     out_l = 0,
#     max_gen_len = 4096,      # Maximum generation length
#     quant_config={
#         "activation": 8,
#         "kv_cache": 4,
#         "weight_ffn": 2,
#         "weight_attn": 2,
#         "rope_bit": 16
#     },
#     use_gate_ffn=True,  # Enable gate mechanism
#     use_rope=True,      # Enable RoPE
#     rope_theta=500000.0, # Standard RoPE theta parameter
#     rope_scaling_factor=1.0, # No scaling
#     output_csv_path="true_density_transformer_analysis.csv"
# )

# compute_and_plot_transformer_analysis(
#         #llama-65B FP16
#     hidden_size=8192,
#     num_heads=64,
#     seq_len=4096,
#     batch_size=1,
#     num_layers=80,
#     intermediate_size=22016,  # Obtained from BitNet configuration file
#     num_key_value_heads=64,   # Obtained from BitNet configuration file
#     out_l = 0,
#     max_gen_len = 4096,      # Maximum generation length
#     quant_config={
#         "activation": 16,
#         "kv_cache": 16,
#         "weight_ffn": 16,
#         "weight_attn": 16,
#         "rope_bit": 16
#     },
#     use_gate_ffn=True,  # Enable gate mechanism
#     use_rope=True,      # Enable RoPE
#     rope_theta=500000.0, # Standard RoPE theta parameter
#     rope_scaling_factor=1.0, # No scaling
#     output_csv_path="llama65B-true_density_transformer_analysis.csv"
# )

# # Run analysis, add intermediate_size parameter and num_key_value_heads parameter
# # Use correct parameters according to BitNet configuration file
# compute_and_plot_transformer_analysis(
#     hidden_size=2560,
#     num_heads=20,
#     seq_len=4096,
#     batch_size=1,
#     num_layers=30,
#     intermediate_size=6912,  # Obtained from BitNet configuration file
#     num_key_value_heads=5,   # Obtained from BitNet configuration file
#     out_l = 0,
#     max_gen_len = 4096,      # Maximum generation length
#     quant_config={
#         "activation": 8,
#         "kv_cache": 4,
#         "weight_ffn": 2,
#         "weight_attn": 2
#     },
#     use_gate_ffn=True,  # Enable gate mechanism
#     use_rope=True,      # Enable RoPE
#     rope_theta=10000.0, # Standard RoPE theta parameter
#     rope_scaling_factor=1.0, # No scaling
#     output_csv_path="true_density_transformer_analysis.csv"
# )
