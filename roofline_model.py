import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib import ticker

def plot_roofline_model(df, hardware_config, output_path="roofline_model.png"):
    """
    绘制单一硬件配置的Roofline模型图，使用折断坐标轴
    
    参数:
    df - 包含计算密度数据的DataFrame
    hardware_config - 单一硬件配置，包含名称、计算性能和内存带宽
    output_path - 输出图像路径
    """
    # 创建带有折断坐标轴的图
    fig = plt.figure(figsize=(12, 8))
    
    # 提取硬件参数
    name = hardware_config['name']
    peak_performance = hardware_config['peak_performance']  # FLOP/s
    memory_bandwidth = hardware_config['memory_bandwidth']  # Byte/s
    
    # 计算转折点
    ridge_point = peak_performance / memory_bandwidth
    
    # 设置x轴范围，确保覆盖所有数据点和转折点
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
    
    # 创建折断坐标轴
    # 第一部分：从0到转折点
    ax1 = fig.add_subplot(121)
    # 第二部分：从转折点到最大值
    ax2 = fig.add_subplot(122, sharey=ax1)
    
    # 隐藏右边图的y轴标签
    plt.setp(ax2.get_yticklabels(), visible=False)
    
    # 设置x轴范围
    ax1.set_xlim(0, ridge_point * 1.5)  # 第一部分显示到转折点
    ax2.set_xlim(ridge_point * 1, max_density)  # 第二部分从转折点开始
    
    # 设置y轴范围
    ax1.set_ylim(0, peak_performance * 1.2)
    ax2.set_ylim(0, peak_performance * 1.2)
    
    # 创建x轴数据点
    x_range1 = np.linspace(0, ridge_point * 1.5, 500)
    x_range2 = np.linspace(ridge_point * 1, max_density, 500)
    
    # 绘制内存带宽受限线和计算能力上限线
    memory_bound1 = [min(x * memory_bandwidth, peak_performance) for x in x_range1]
    memory_bound2 = [min(x * memory_bandwidth, peak_performance) for x in x_range2]
    
    # 绘制Roofline
    ax1.plot(x_range1, memory_bound1, 'r-', linewidth=2)
    ax2.plot(x_range2, memory_bound2, 'r-', linewidth=2)
    
    # 绘制计算能力上限线
    ax1.axhline(y=peak_performance, linestyle='--', color='g', linewidth=2)
    ax2.axhline(y=peak_performance, linestyle='--', color='g', linewidth=2)
    
    # 标记转折点
    # ax1.scatter([ridge_point], [peak_performance], marker='o', s=100, color='blue', zorder=10)
    # ax2.scatter([ridge_point], [peak_performance], marker='o', s=100, color='blue', zorder=10)
    
    # 添加区域标签
    ax1.text(ridge_point * 0.3, peak_performance * 0.5, "Memory\nBound", color='red', fontsize=12)
    ax2.text(max_density * 0.7, peak_performance * 0.9, "Compute\nBound", color='green', fontsize=12)
    
    # 标记转折点坐标
    ax1.annotate(f"I_max = {ridge_point:.2f}", 
                xy=(ridge_point, 0), 
                xytext=(0, -20),
                textcoords='offset points',
                ha='center',
                fontsize=10)
    
    # 标记最大性能
    ax1.annotate(f"π = {peak_performance:.2e}", 
                xy=(0, peak_performance), 
                xytext=(-40, 0),
                textcoords='offset points',
                va='center',
                fontsize=10)
    
    # 标记内存带宽
    ax1.annotate(f"β = {memory_bandwidth:.2e}", 
                xy=(0, 0), 
                xytext=(20, 20),
                textcoords='offset points',
                fontsize=10)
    
    # 定义操作类型和阶段的标记和颜色
    operation_markers = {
        'QKV': 's',  # 方形
        'RoPE': 'p',  # 五角形
        'Attention': 'o',  # 圆形
        'Output': 'v',  # 三角形
        'FFN-1': 'd',  # 菱形
        'FFN-2': 'h'   # 六边形
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
        'Prefill': '^',  # 上三角
        'Decode': 'o',   # 圆形
        'Decode_Last': 's'  # 方形
    }
    
    # 为图例创建空列表
    legend_elements = []
    
    # 创建操作类型和阶段的映射
    operation_map = {}
    
    # 绘制数据点
    for _, row in df.iterrows():
        if row['Density (Op/Byte)'] == '-':
            continue
            
        density = float(row['Density (Op/Byte)'])
        operation = row['Operation']
        phase = row['Phase']
        flops = row['FLOPs']
        
        # 确定操作类型
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
        
        # 获取标记和颜色
        marker = phase_markers.get(phase, 'x')  # 根据阶段选择标记
        color = operation_colors.get(op_type, 'gray')  # 根据操作类型选择颜色
        
        # 计算在roofline上的实际性能
        if density < ridge_point:
            # 内存带宽受限
            attainable_perf = density * memory_bandwidth
        else:
            # 计算能力受限
            attainable_perf = peak_performance
        
        # 选择正确的子图并确保点被绘制
        if density <= ridge_point * 1.5:
            ax = ax1
            ax.scatter(density, attainable_perf, marker=marker, s=80, 
                      color=color, edgecolors='black', alpha=0.7)
        else:
            ax = ax2
            ax.scatter(density, attainable_perf, marker=marker, s=80, 
                      color=color, edgecolors='black', alpha=0.7)
        
        # 记录操作和阶段的组合，用于图例
        key = f"{op_type}_{phase}"
        if key not in operation_map:
            operation_map[key] = {
                'op_type': op_type,
                'phase': phase,
                'color': color,
                'marker': marker,
                'operation': operation
            }
    
    # 添加折断标记
    d = .015  # 折断线的大小
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1-d, 1+d), (-d, +d), **kwargs)
    ax1.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
    
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    ax2.plot((-d, +d), (1-d, 1+d), **kwargs)
    
    # 创建图例
    # 1. 操作类型图例
    op_legend = []
    for op_type, color in operation_colors.items():
        op_legend.append(mpatches.Patch(color=color, label=op_type))
    
    # 2. 阶段图例
    phase_legend = []
    for phase, marker in phase_markers.items():
        if phase == 'Decode_Last':
            phase_display = 'Decode (Last Token)'
        else:
            phase_display = phase
        phase_legend.append(plt.Line2D([0], [0], marker=marker, color='black', linestyle='None', 
                                      markersize=8, label=phase_display))
    
    # 3. 具体操作图例
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
    
    # 设置图表属性
    ax1.set_xlabel('Operational Intensity (I) [FLOP/Byte]')
    ax2.set_xlabel('Operational Intensity (I) [FLOP/Byte]')
    ax1.set_ylabel('Attainable Performance (P) [FLOP/s]')
    fig.suptitle(f'Roofline Model - {name}', fontsize=16)
    
    # 添加网格
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # 添加图例，分组显示
    # 1. 操作类型图例
    legend1 = ax1.legend(handles=op_legend, loc='upper left', title='Operation Types')
    ax1.add_artist(legend1)
    
    # 2. 阶段图例
    legend2 = ax2.legend(handles=phase_legend, loc='upper left', title='Phases')
    ax2.add_artist(legend2)
    
    # 在plot_roofline_model函数中，修改保存图表的部分
    
    # 修改前:
    # 3. 创建单独的图例图
    fig_legend = plt.figure(figsize=(12, 3))
    fig_legend.legend(handles=op_phase_legend, loc='center', ncol=3, title='Operations by Phase')
    fig_legend.tight_layout()
    fig_legend.savefig(output_path.replace('.png', '_legend.png'))
    
    # 保存主图表
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    
    # 修改后:
    # 3. 创建单独的图例图
    legend_path = output_path.replace('.png', '_legend.png')
    fig_legend = plt.figure(figsize=(12, 3))
    fig_legend.legend(handles=op_phase_legend, loc='center', ncol=3, title='Operations by Phase')
    fig_legend.tight_layout()
    fig_legend.savefig(legend_path)
    plt.close(fig_legend)  # 关闭图例图
    
    # 保存主图表
    fig.tight_layout()
    fig.savefig(output_path)
    plt.show()
    
    print(f"\n✅ Roofline model saved to '{output_path}'")
    print(f"✅ Legend saved to '{legend_path}'")

def analyze_from_csv(csv_path, hardware_config, output_path="llama-roofline_model.png"):
    """
    从CSV文件读取数据并分析
    
    参数:
    csv_path - CSV文件路径
    hardware_config - 硬件配置
    output_path - 输出图像路径
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_path)
        
        # 绘制Roofline模型
        plot_roofline_model(df, hardware_config, output_path)
        
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

# 示例使用
if __name__ == "__main__":
    # 定义硬件配置
    hardware_config = {
        'name': 'NVIDIA A100',
        'peak_performance': 19.5e12,  # FLOP/s
        'memory_bandwidth': 1.555e12,  # Byte/s
    }
    
    # 使用示例
    csv_path = "llama-true_density_transformer_analysis.csv"
    
    analyze_from_csv(csv_path, hardware_config)