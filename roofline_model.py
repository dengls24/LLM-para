import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib import ticker

def plot_roofline_model(df, hardware_config, output_path="roofline_model.png"):
    """
    Plot Roofline model for a single hardware configuration with broken axis
    
    Parameters:
    df - DataFrame containing computational density data
    hardware_config - Single hardware configuration including name, peak performance and memory bandwidth
    output_path - Output image path
    """
    # Create figure with broken axis
    fig = plt.figure(figsize=(12, 8))
    
    # Extract hardware parameters
    name = hardware_config['name']
    peak_performance = hardware_config['peak_performance']  # FLOP/s
    memory_bandwidth = hardware_config['memory_bandwidth']  # Byte/s
    
    # Calculate ridge point
    ridge_point = peak_performance / memory_bandwidth
    
    # Set x-axis range to ensure coverage of all data points and ridge point
    valid_density = []
    for _, row in df.iterrows():
        if row['Density (Op/Byte)'] != '-':
            valid_density.append(float(row['Density (Op/Byte)']))
    
    if valid_density:
        min_density = min(min(valid_density) * 0.5, ridge_point * 0.5)
        max_density = max(max(valid_density) * 1.5, ridge_point * 1.5)
    else:
        min_density = ridge_point * 0.1
        max_density = ridge_point * 10
    
    # Create broken axis
    # First part: from 0 to ridge point
    ax1 = fig.add_subplot(121)
    # Second part: from ridge point to maximum value
    ax2 = fig.add_subplot(122, sharey=ax1)
    
    # Hide y-axis labels on the right subplot
    plt.setp(ax2.get_yticklabels(), visible=False)
    
    # Set x-axis range
    ax1.set_xlim(0, ridge_point * 1.5)  # First part shows up to ridge point
    ax2.set_xlim(ridge_point * 1, max_density)  # Second part starts from ridge point
    
    # Set y-axis range
    ax1.set_ylim(0, peak_performance * 1.2)
    ax2.set_ylim(0, peak_performance * 1.2)
    
    # Create x-axis data points
    x_range1 = np.linspace(0, ridge_point * 1.5, 500)
    x_range2 = np.linspace(ridge_point * 1, max_density, 500)
    
    # Plot memory bandwidth bound line and compute capability ceiling line
    memory_bound1 = [min(x * memory_bandwidth, peak_performance) for x in x_range1]
    memory_bound2 = [min(x * memory_bandwidth, peak_performance) for x in x_range2]
    
    # Plot Roofline
    ax1.plot(x_range1, memory_bound1, 'r-', linewidth=2)
    ax2.plot(x_range2, memory_bound2, 'r-', linewidth=2)
    
    # Plot compute capability ceiling line
    ax1.axhline(y=peak_performance, linestyle='--', color='g', linewidth=2)
    ax2.axhline(y=peak_performance, linestyle='--', color='g', linewidth=2)
    
    # Mark ridge point
    # ax1.scatter([ridge_point], [peak_performance], marker='o', s=100, color='blue', zorder=10)
    # ax2.scatter([ridge_point], [peak_performance], marker='o', s=100, color='blue', zorder=10)
    
    # Add region labels
    ax1.text(ridge_point * 0.3, peak_performance * 0.5, "Memory\nBound", color='red', fontsize=12)
    ax2.text(max_density * 0.7, peak_performance * 0.9, "Compute\nBound", color='green', fontsize=12)
    
    # Mark ridge point coordinates
    ax1.annotate(f"I_max = {ridge_point:.2f}", 
                xy=(ridge_point, 0), 
                xytext=(0, -20),
                textcoords='offset points',
                ha='center',
                fontsize=10)
    
    # Mark peak performance
    ax1.annotate(f"π = {peak_performance:.2e}", 
                xy=(0, peak_performance), 
                xytext=(-40, 0),
                textcoords='offset points',
                va='center',
                fontsize=10)
    
    # Mark memory bandwidth
    ax1.annotate(f"β = {memory_bandwidth:.2e}", 
                xy=(0, 0), 
                xytext=(20, 20),
                textcoords='offset points',
                fontsize=10)
    
    # Define operation type and phase markers and colors
    operation_markers = {
        'QKV': 's',  # Square
        'RoPE': 'p',  # Pentagon
        'Attention': 'o',  # Circle
        'Output': 'v',  # Triangle
        'FFN-1': 'd',  # Diamond
        'FFN-2': 'h'   # Hexagon
    }
    
    operation_colors = {
        'QKV': 'orange',
        'RoPE': 'purple',
        'Attention': 'red',
        'Output': 'green',
        'FFN-1': 'blue',
        'FFN-2': 'cyan'
    }
    
    phase_markers = {
        'Prefill': '^',  # Up triangle
        'Decode': 'o',   # Circle
        'Decode_Last': 's'  # Square
    }
    
    # Create empty list for legend
    legend_elements = []
    
    # Create mapping of operation types and phases
    operation_map = {}
    
    # Plot data points
    for _, row in df.iterrows():
        if row['Density (Op/Byte)'] == '-':
            continue
            
        density = float(row['Density (Op/Byte)'])
        operation = row['Operation']
        phase = row['Phase']
        flops = row['FLOPs']
        
        # Determine operation type
        op_type = None
        if any(x in operation for x in ['xW_Q', 'xW_K', 'xW_V', 'W_Q', 'W_K', 'W_V']):
            op_type = 'QKV'
        elif any(x in operation for x in ['RoPE-Q', 'RoPE-K']):
            op_type = 'RoPE'
        elif any(x in operation for x in ['Q × K', 'Q × Kᵀ', 'Attn × V']):
            op_type = 'Attention'
        elif 'xW_O' in operation or 'W_O' in operation:
            op_type = 'Output'
        elif 'FFN-1' in operation:
            op_type = 'FFN-1'
        elif 'FFN-2' in operation:
            op_type = 'FFN-2'
        else:
            op_type = 'Other'
        
        # Get marker and color
        marker = phase_markers.get(phase, 'x')  # Select marker based on phase
        color = operation_colors.get(op_type, 'gray')  # Select color based on operation type
        
        # Calculate actual performance on roofline
        if density < ridge_point:
            # Memory bandwidth bound
            attainable_perf = density * memory_bandwidth
        else:
            # Compute capability bound
            attainable_perf = peak_performance
        
        # Select correct subplot and ensure points are plotted
        if density <= ridge_point * 1.5:
            ax = ax1
            ax.scatter(density, attainable_perf, marker=marker, s=80, 
                      color=color, edgecolors='black', alpha=0.7)
        else:
            ax = ax2
            ax.scatter(density, attainable_perf, marker=marker, s=80, 
                      color=color, edgecolors='black', alpha=0.7)
        
        # Record operation and phase combination for legend
        key = f"{op_type}_{phase}"
        if key not in operation_map:
            operation_map[key] = {
                'op_type': op_type,
                'phase': phase,
                'color': color,
                'marker': marker,
                'operation': operation
            }
    
    # Add break marks
    d = .015  # Size of break lines
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1-d, 1+d), (-d, +d), **kwargs)
    ax1.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
    
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    ax2.plot((-d, +d), (1-d, 1+d), **kwargs)
    
    # Create legends
    # 1. Operation type legend
    op_legend = []
    for op_type, color in operation_colors.items():
        op_legend.append(mpatches.Patch(color=color, label=op_type))
    
    # 2. Phase legend
    phase_legend = []
    for phase, marker in phase_markers.items():
        if phase == 'Decode_Last':
            phase_display = 'Decode (Last Token)'
        else:
            phase_display = phase
        phase_legend.append(plt.Line2D([0], [0], marker=marker, color='black', linestyle='None', 
                                      markersize=8, label=phase_display))
    
    # 3. Specific operation legend
    op_phase_legend = []
    for key, info in operation_map.items():
        op_type = info['op_type']
        phase = info['phase']
        marker = info['marker']
        color = info['color']
        operation = info['operation']
        
        if phase == 'Decode_Last':
            phase_display = 'Decode (Last)'
        else:
            phase_display = phase
            
        label = f"{operation} ({phase_display})"
        op_phase_legend.append(plt.Line2D([0], [0], marker=marker, color=color, linestyle='None', 
                                         markersize=8, label=label))
    
    # Set chart properties
    ax1.set_xlabel('Operational Intensity (I) [FLOP/Byte]')
    ax2.set_xlabel('Operational Intensity (I) [FLOP/Byte]')
    ax1.set_ylabel('Attainable Performance (P) [FLOP/s]')
    fig.suptitle(f'Roofline Model - {name}', fontsize=16)
    
    # Add grid
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # Add legends with grouped display
    # 1. Operation type legend
    legend1 = ax1.legend(handles=op_legend, loc='upper left', title='Operation Types')
    ax1.add_artist(legend1)
    
    # 2. Phase legend
    legend2 = ax2.legend(handles=phase_legend, loc='upper left', title='Phases')
    ax2.add_artist(legend2)
    
    # Modified section for saving charts in plot_roofline_model function
    
    # Before modification:
    # 3. Create separate legend figure
    fig_legend = plt.figure(figsize=(12, 3))
    fig_legend.legend(handles=op_phase_legend, loc='center', ncol=3, title='Operations by Phase')
    fig_legend.tight_layout()
    fig_legend.savefig(output_path.replace('.png', '_legend.png'))
    
    # Save main chart
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    
    # After modification:
    # 3. Create separate legend figure
    legend_path = output_path.replace('.png', '_legend.png')
    fig_legend = plt.figure(figsize=(12, 3))
    fig_legend.legend(handles=op_phase_legend, loc='center', ncol=3, title='Operations by Phase')
    fig_legend.tight_layout()
    fig_legend.savefig(legend_path)
    plt.close(fig_legend)  # Close legend figure
    
    # Save main chart
    fig.tight_layout()
    fig.savefig(output_path)
    plt.show()
    
    print(f"\n✅ Roofline model saved to '{output_path}'")
    print(f"✅ Legend saved to '{legend_path}'")

def analyze_from_csv(csv_path, hardware_config, output_path="Mixtral-8x7B.png"):
    """
    Read data from CSV file and analyze
    
    Parameters:
    csv_path - CSV file path
    hardware_config - Hardware configuration
    output_path - Output image path
    """
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Plot Roofline model
        plot_roofline_model(df, hardware_config, output_path)
        
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

# Example usage
if __name__ == "__main__":
    # Define multiple hardware configurations
    hardware_configs = {
        'NVIDIA_A100': {
            'name': 'NVIDIA A100',
            'peak_performance': 19.5e12,  # FLOP/s (FP32)
            'memory_bandwidth': 1.555e12,  # Byte/s (HBM2)
        },
        'NVIDIA_H100': {
            'name': 'NVIDIA H100',
            'peak_performance': 67e12,  # FLOP/s (FP32)
            'memory_bandwidth': 3.35e12,  # Byte/s (HBM3)
        },
        'Smartphone_NPU': {
            'name': 'Smartphone NPU (Snapdragon 8 Gen 2)',
            'peak_performance': 4.35e12,  # FLOP/s (INT8)
            'memory_bandwidth': 51.2e9,  # Byte/s (LPDDR5X)
        },
        'DRAM_PIM': {
            'name': 'DRAM-PIM (H2LLM In-die NMP)',
            'peak_performance': 0.8e12,  # FLOP/s (FP16)
            'memory_bandwidth': 0.8e12,  # Byte/s (Near-memory computing)
        },
        'NAND_PIM': {
            'name': 'NAND-PIM (Lincoln -w/o speculated decode)',
            'peak_performance': 0.2e12,  # FLOP/s 
            'memory_bandwidth': 0.2e12,  # Byte/s 
        },
        'Intel_Xeon': {
            'name': 'Intel Xeon Platinum 8380',
            'peak_performance': 2.8e12,  # FLOP/s (FP32, AVX-512)
            'memory_bandwidth': 204.8e9,  # Byte/s (DDR4-3200)
        },
        'Apple_M2_Ultra': {
            'name': 'Apple M2 Ultra',
            'peak_performance': 27.2e12,  # FLOP/s (FP32)
            'memory_bandwidth': 800e9,  # Byte/s (Unified Memory)
        },
        'AMD_MI250X': {
            'name': 'AMD Instinct MI250X',
            'peak_performance': 47.9e12,  # FLOP/s (FP32)
            'memory_bandwidth': 3.28e12,  # Byte/s (HBM2e)
        }
    }
    
    # Select hardware configuration to use
    selected_hardware = 'NVIDIA_A100'  # Change this to test different hardware
    hardware_config = hardware_configs[selected_hardware]
    
    # Usage example
    #csv_path = "llama-true_density_transformer_analysis.csv"
    csv_path = "Mixtral-8x7B_density_transformer_analysis.csv"
    
    analyze_from_csv(csv_path, hardware_config)
    
    # Optional: Generate roofline models for all hardware configurations
    # for hw_name, hw_config in hardware_configs.items():
    #     output_path = f"roofline_{hw_name}.png"
    #     analyze_from_csv(csv_path, hw_config, output_path)