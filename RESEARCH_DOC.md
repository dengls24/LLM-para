# LLM-Para 研究文档

> **LLM Parallel-Architecture Roofline Analyzer**  
> 大语言模型推理性能、能耗与硬件架构的一阶建模分析工具

---

## 目录

1. [研究背景与动机](#1-研究背景与动机)
2. [系统总体架构](#2-系统总体架构)
3. [核心模块详解](#3-核心模块详解)
   - 3.1 [analyzer.py — 算子分析引擎](#31-analyzerpy--算子分析引擎)
   - 3.2 [metrics.py — 多维度性能指标](#32-metricspy--多维度性能指标)
   - 3.3 [hetero.py — 异构分层内存建模](#33-heteropy--异构分层内存建模)
   - 3.4 [dse.py — 设计空间探索引擎](#34-dsepy--设计空间探索引擎)
   - 3.5 [configs.py — 模型与硬件配置库](#35-configspy--模型与硬件配置库)
   - 3.6 [app.py — REST API 服务](#36-apppy--rest-api-服务)
4. [数学模型与公式推导](#4-数学模型与公式推导)
   - 4.1 [FLOPs 计算模型](#41-flops-计算模型)
   - 4.2 [内存访问量模型](#42-内存访问量模型)
   - 4.3 [屋顶线模型（Roofline）](#43-屋顶线模型roofline)
   - 4.4 [能量屋顶线模型](#44-能量屋顶线模型)
   - 4.5 [TCO 全生命周期成本模型](#45-tco-全生命周期成本模型)
   - 4.6 [碳排放模型（CO₂e）](#46-碳排放模型co₂e)
   - 4.7 [异构架构建模](#47-异构架构建模)
5. [支持的算子列表与计算方法](#5-支持的算子列表与计算方法)
6. [支持的模型与硬件](#6-支持的模型与硬件)
7. [API 接口说明](#7-api-接口说明)
8. [量化配置说明](#8-量化配置说明)
9. [实验设计建议](#9-实验设计建议)
10. [主要参考文献](#10-主要参考文献)

---

## 1. 研究背景与动机

大语言模型（LLM）的推理代价极高，主要瓶颈集中于两类资源：

| 瓶颈类型 | 原因 | 典型算子 |
|----------|------|----------|
| **内存带宽受限** | 权重数据量远大于计算量（decode 阶段） | Q/K/V 投影、O 投影、FFN 权重加载 |
| **计算受限** | 大 batch 或长序列时计算密度高（prefill 阶段） | 矩阵乘法（GEMM）、大批量 Attention |

传统分析工具（如 LLM-Viewer）只考虑了 FLOPs 和带宽两个维度。本项目目标是在此基础上扩展为**多维度一阶性能分析框架**，具体包括：

- **Performance Roofline**：算术强度 × 带宽上界 → 性能上界
- **Energy Roofline**：功率分解模型 → 每瓦算力（GFLOPS/W）
- **TCO 分析**：硬件折旧 + 电费 → 单位算力成本（$/EFLOP）
- **CO₂e 建模**：运营碳排 + 制造碳排
- **异构分层存储**：SRAM → DRAM → Flash 三级存储的 decode 吞吐建模
- **设计空间探索（DSE）**：多目标 Pareto 最优硬件配置搜索

### 核心研究问题

> Q1. 对于给定 LLM，哪些算子是真正的性能瓶颈？  
> Q2. 从能量效率视角，计算和内存访问各消耗多少功率？  
> Q3. 在 Flash 近存加速器（如 Cambricon-LLM）上，decode 吞吐如何受限于存储层次？  
> Q4. 什么样的硬件参数组合在性能-成本-碳排放上是 Pareto 最优的？

---

## 2. 系统总体架构

```
┌──────────────────────────────────────────────────────────────────┐
│                    LLM-Para 系统架构                              │
├─────────────────────────┬────────────────────────────────────────┤
│     前端 (Browser)       │         后端 (Flask/Python)            │
│                         │                                        │
│  ┌─────────────────┐    │   ┌──────────────────────────────┐    │
│  │  配置面板        │    │   │  analyzer.py                 │    │
│  │  - 模型架构参数  │◄──►│   │  每算子 FLOPs + 内存访问计算  │    │
│  │  - 硬件平台选择  │    │   └──────────────┬───────────────┘    │
│  │  - 量化配置      │    │                  │                    │
│  └─────────────────┘    │   ┌──────────────▼───────────────┐    │
│                         │   │  metrics.py                  │    │
│  ┌─────────────────┐    │   │  能量Roofline / TCO / CO₂e   │    │
│  │  可视化面板      │    │   └──────────────┬───────────────┘    │
│  │  Tab 1: 算子表  │    │                  │                    │
│  │  Tab 2: Roofline│◄──►│   ┌──────────────▼───────────────┐    │
│  │  Tab 3: 分析图  │    │   │  hetero.py                   │    │
│  │  Tab 4: 内存分析│    │   │  三级存储分层建模              │    │
│  │  Tab 5: 能量    │    │   └──────────────┬───────────────┘    │
│  │  Tab 6: TCO     │    │                  │                    │
│  │  Tab 7: 异构    │    │   ┌──────────────▼───────────────┐    │
│  │  Tab 8: DSE     │    │   │  dse.py                      │    │
│  └─────────────────┘    │   │  设计空间扫描 + Pareto 分析   │    │
│                         │   └──────────────────────────────┘    │
│                         │                                        │
│                         │   configs.py (19 模型 + 24 硬件)       │
└─────────────────────────┴────────────────────────────────────────┘
```

**数据流**：用户配置 → `POST /api/analyze` → `LLMAnalyzer.analyze()` → 每算子 `OpResult` 列表 → 各 Tab 独立调用扩展 API（`/api/metrics`、`/api/hetero`、`/api/dse/run`）

---

## 3. 核心模块详解

### 3.1 `analyzer.py` — 算子分析引擎

**核心类**：`LLMAnalyzer`

**输入**：一个 `config` 字典，包含模型架构参数和量化配置。  
**输出**：`List[dict]`，每个字典对应一个算子在单层的计算量，已乘以层数。

#### `OpResult` 数据结构

```python
@dataclass
class OpResult:
    phase: str            # 'Prefill' | 'Decode' | 'Output'
    operation: str        # 算子名称（如 'Q Projection'）
    category: str         # 'QKV' | 'Attention' | 'FFN' | 'Norm' | 'Embed' | 'RoPE'
    flops: float          # 单层单 batch 的 FLOPs
    param_count: float    # 参数量（用于计算模型大小）
    input_bytes: float    # 激活输入内存访问量（Bytes）
    weight_bytes: float   # 权重内存访问量（Bytes）
    output_bytes: float   # 激活输出内存访问量（Bytes）
    total_bytes: float    # 总内存访问量（Bytes）
    density: float        # 算术强度 = flops / total_bytes（FLOP/Byte）
    note: str             # 备注（GQA 比例、专家数量等）
```

#### 核心辅助函数 `_add()`

```python
def _add(self, phase, name, category, flops, param_count,
         input_shape, weight_shape, output_shape, act_bits, weight_bits, note="")
```

- 根据 shape + 比特宽度自动计算 `input_bytes`、`weight_bytes`、`output_bytes`
- **注意**：内存访问量 = 激活张量大小 + 权重大小（不含 weight 缓存假设）
- 密度 = `flops / total_bytes`，这是屋顶线模型的横轴坐标

#### 分析流程

```
analyze()
├── Token Embedding（仅 Prefill，不乘层数）
├── for phase in [Prefill, Decode]:
│   ├── Norm (pre-Attn)
│   ├── if use_mla: _analyze_mla()
│   │   else:       _analyze_standard_attention()
│   ├── Norm (pre-FFN)
│   └── if use_moe: _analyze_moe_ffn()
│       else:       _analyze_dense_ffn()
└── LM Head（Output 阶段，不乘层数）
```

所有层内算子结果最终 `× L`（层数）得到全模型总量。

---

### 3.2 `metrics.py` — 多维度性能指标

#### 3.2.1 能量屋顶线分析

**核心函数**：`energy_roofline_point(flops, total_bytes, density, hardware)`

基于 Ghane et al. ISPASS 2018 的功率分解模型：

```
P_active = TDP × [α × u_compute + (1−α) × u_memory]

其中：
  α         = compute_power_frac（计算单元占 TDP 的比例，约 0.5~0.6）
  u_compute = min(1, I / ridge)    （计算单元利用率）
  u_memory  = min(1, ridge / I)    （内存单元利用率）
  ridge     = peak_perf / bandwidth （屋顶线脊点）
  I         = density               （算术强度，FLOP/Byte）
```

输出的核心指标：
- `energy_efficiency_gflops_per_w`：能量效率（GFLOPS/W）
- `power_w`：预估有效功耗（W）
- `energy_j`：执行该算子消耗的能量（Joule）
- `bound`：`'Memory'` 或 `'Compute'`

**曲线生成**：`build_energy_roofline_curve()` 生成 300 个点的 GFLOPS/W vs 算术强度曲线，供前端绘图。

#### 3.2.2 TCO 模型

**函数**：`compute_tco(hardware, lifetime_years, utilization, electricity_price_per_kwh, pue)`

```
TCO = 硬件成本 + 电费
    = cost_usd + TDP × lifetime_hours × utilization × PUE × electricity_price

TCO per EFLOP = TCO / (peak_FLOPS × lifetime_seconds × utilization)
              = 单位算力的总拥有成本（美元/EFLOP）
```

参数说明：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lifetime_years` | 3 | 硬件使用寿命 |
| `utilization` | 0.5 | GPU 平均利用率 |
| `electricity_price_per_kwh` | $0.10 | 电价（数据中心通常 $0.06~$0.15） |
| `pue` | 1.3 | 电力使用效率（含冷却等开销） |

#### 3.2.3 CO₂e 碳排放模型

**函数**：`compute_co2e(hardware, flops_analyzed, region, lifetime_years, utilization)`

```
运营碳排 = energy_kwh × carbon_intensity(gCO₂/kWh)
         （取决于电网碳强度，冰岛28 vs 中国550 gCO₂/kWh）

制造碳排 = 芯片制造总排放 × (分析的计算量 / 全生命周期总计算量)
         （估算：约 1 吨 CO₂e / $1000 芯片成本）

总 CO₂e per EFLOP = (运营碳排 + 制造碳排) / FLOPs
```

内置电网碳强度数据库（`CARBON_INTENSITY_GRID`）覆盖20+国家/地区。

#### 3.2.4 内存容量分析

**函数**：`compute_memory_footprint(model_config, results)`

```
总内存需求 = 模型权重 + KV 缓存 + 激活值峰值

模型权重 = Σ(每层各投影矩阵) × 量化比特 / 8
         + embedding + lm_head

KV 缓存 = L × kv_n × d_head × 2 × (kv_bit/8) × (seq_len + gen_len) × batch

激活峰值 = max(Attention 矩阵, FFN 中间层)
         = max(b×n×seq×seq, b×seq×intermediate) × (act_bit/8)
```

---

### 3.3 `hetero.py` — 异构分层内存建模

**核心类**：`HeteroAnalyzer`

**研究背景**：Cambricon-LLM（Yu et al. 2024）提出了 Chiplet 异构架构，将存储分为三层：
- **SRAM**（~8 TB/s，~512 MB）：最快，存放热激活和最近层的权重
- **DRAM/HBM**（~512 GB/s，~16 GB）：中速，存放 KV 缓存和中间层权重
- **NAND Flash**（~14 GB/s，~256 GB）：大容量，存放完整模型权重（溢出部分）

#### 数据放置策略（贪心算法）

```python
def compute_data_placement():
    # 优先级：越热的数据放越快的层
    1. 激活值         → SRAM（必须快速读写）
    2. 当前层权重      → SRAM（尽量多塞层数）
    3. KV 缓存        → DRAM（温数据，每 token 读写一次）
    4. 剩余权重        → DRAM（填满后溢出到 Flash）
    5. Flash 溢出      → Flash（速度瓶颈）
```

#### Decode 吞吐量计算（关键公式）

```
tokens/s = decode_bottleneck_BW / bytes_per_token

bytes_per_token = L × (w_Q + w_KV + w_O + w_FFN) × batch_size
                × (weight_bits / 8)

其中 decode_bottleneck_BW 取决于权重主要所在的存储层：
  - 权重全在 SRAM → SRAM BW（最快，tok/s 最高）
  - 权重在 DRAM  → DRAM BW
  - 权重溢出 Flash → Flash BW（最慢，通常 < 2 tok/s for 70B 模型）
```

这解释了为什么 Flash 存储推理在大模型下生成速度极慢（受 Flash 带宽 ~14 GB/s 限制）。

#### 输出结果

```python
{
  'placement': {
    'summary': {
      'layers_in_sram': int,          # 多少层权重在 SRAM
      'layers_in_dram': int,          # 多少层权重在 DRAM
      'layers_in_flash': int,         # 多少层权重在 Flash
      'decode_bottleneck': str,       # 'SRAM' | 'DRAM' | 'Flash'
      'decode_bw': float,             # 实际 decode 使用的带宽（B/s）
      'sram_utilization_pct': float,
      'flash_spills': bool,           # 是否发生 Flash 溢出
    }
  },
  'hetero_ops': List[dict],           # 每算子有效带宽 vs 理想带宽
  'throughput': {
    'decode_tokens_per_sec': float,   # 关键性能指标
    'tokens_per_sec_SRAM': float,
    'tokens_per_sec_DRAM': float,
    'tokens_per_sec_Flash': float,
    'ttft_ms': float,                 # Time-to-First-Token（预填充延迟）
  }
}
```

---

### 3.4 `dse.py` — 设计空间探索引擎

**核心类**：`DSEEngine`

**设计目标**：系统性搜索硬件设计参数空间，找到在多目标下的 **Pareto 最优**配置。

#### 扫描参数空间（5 维）

| 参数 | 含义 | 单位 | 典型范围 |
|------|------|------|----------|
| `peak_performance_tflops` | 峰值算力 | TFLOPS | 0.1 ~ 1000 |
| `memory_bandwidth_gbs` | 内存带宽 | GB/s | 10 ~ 5000 |
| `memory_capacity_gb` | 内存容量 | GB | 2 ~ 512 |
| `tdp_w` | 热设计功耗 | W | 2 ~ 1000 |
| `cost_usd` | 系统成本 | USD | 50 ~ 50000 |

对每个硬件点计算：
1. **可达性能**（Roofline 上界）
2. **能量效率**（GFLOPS/W）
3. **TCO/EFLOP**
4. **CO₂e/EFLOP**
5. **内存是否能装下模型**

#### Pareto 前沿算法

```python
def _pareto_2d(points, obj1, obj2, max1=True, max2=True):
    """
    点 a 支配点 b 当且仅当：
      - a 在 obj1 上不劣于 b
      - a 在 obj2 上不劣于 b
      - a 在至少一个目标上严格优于 b
    """
    pareto = [p for p in points if not any(dominates(q, p) for q in points)]
    return sorted(pareto, key=lambda r: r[obj1], reverse=max1)
```

提供三组 Pareto 前沿：
- **性能 vs 成本**（Performance-Cost）
- **性能 vs 能效**（Performance-Energy）
- **性能 vs 碳排放**（Performance-Carbon）

#### 敏感度分析

```python
engine.sensitivity_analysis(base_hw, param='memory_bandwidth',
                             multipliers=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
```

回答问题：**"带宽翻倍，性能/能效/TCO 分别提升多少？"**

#### 内置 DSE 预设

| 预设名称 | 应用场景 | 点数 |
|----------|----------|------|
| Quick Scan (3×3 grid) | 快速粗扫，27个点 | 27 |
| Edge/Mobile DSE | 边端/手机 NPU | 600 |
| Datacenter GPU DSE | 数据中心 GPU | 384 |
| PIM / Near-Memory DSE | PIM/近存架构 | 400 |

---

### 3.5 `configs.py` — 模型与硬件配置库

#### 模型配置（19 个预设）

每个模型配置包含：

```python
{
  "hidden_size": int,              # 隐层维度 h
  "num_heads": int,                # 注意力头数 n
  "num_key_value_heads": int,      # GQA KV 头数（< num_heads 时启用 GQA）
  "num_layers": int,               # Transformer 层数 L
  "intermediate_size": int,        # FFN 中间层维度
  "vocab_size": int,               # 词表大小
  "seq_len": int,                  # 输入序列长度（prefill）
  "batch_size": int,               # 批大小
  "max_gen_len": int,              # 最大生成长度（decode 阶段 KV 缓存上界）
  "use_gate_ffn": bool,            # 是否使用 SwiGLU（带门控的 FFN）
  "use_rmsnorm": bool,             # True=RMSNorm, False=LayerNorm
  "num_experts_per_tok": int,      # MoE: 每 token 激活专家数（无 MoE 时为 1）
  "num_local_experts": int,        # MoE: 总专家数
  "use_mla": bool,                 # 是否使用 DeepSeek MLA
  "rope_theta": float,             # RoPE 基频 θ
  "quant_config": {
    "activation": int,             # 激活量化比特（8/16）
    "weight_attn": int,            # 注意力权重比特（2/4/8/16）
    "weight_ffn": int,             # FFN 权重比特
    "kv_cache": int,               # KV 缓存量化比特
    "rope_bit": int,               # RoPE 旋转位置编码精度
  }
}
```

#### 硬件配置（24 个预设）

每个硬件配置除基础参数外，还包含扩展参数：

```python
{
  # 基础参数
  "peak_performance": float,       # FP32 算力（FLOP/s）
  "peak_performance_fp16": float,  # FP16 算力（FLOP/s）
  "memory_bandwidth": float,       # 内存带宽（Byte/s）
  "memory_capacity": float,        # 内存容量（Bytes）

  # 扩展参数（用于 Energy/TCO/CO₂e 分析）
  "tdp_w": float,                  # 热设计功耗（W）
  "cost_usd": float,               # 系统采购成本（USD）
  "tech_node_nm": int,             # 工艺节点（nm）
  "carbon_mfg_kgco2e": float,      # 制造碳排放（kg CO₂e）
  "compute_power_frac": float,     # 计算单元占 TDP 的比例（0~1）

  # 异构架构专用参数（仅 Chiplet/PIM 类型）
  "is_heterogeneous": bool,
  "memory_tiers": {
    "SRAM":  {"bandwidth": float, "capacity": float,
              "energy_per_byte_pj": float, "latency_ns": float},
    "DRAM":  {...},
    "Flash": {...},
  }
}
```

**硬件类别分布**：

| 类别 | 数量 | 代表型号 |
|------|------|----------|
| GPU（NVIDIA） | 7 | H100 SXM, A100 80GB, RTX 4090 |
| GPU（AMD） | 2 | MI300X, MI250X |
| CPU/NPU（Apple） | 3 | M3 Ultra, M2 Ultra, M2 Max |
| AI 加速器（Intel） | 2 | Gaudi 3, Xeon Platinum |
| 移动 NPU | 3 | Snapdragon 8 Gen 3, Dimensity 9300 |
| PIM | 3 | HBM-PIM, NAND-PIM, SRAM-PIM |
| Chiplet 异构 | 3 | Cambricon-LLM, Flash-LLM, NAND-PIM Near-Storage |
| 自定义 | 1 | Custom Hardware |

---

### 3.6 `app.py` — REST API 服务

所有端点统一使用 JSON 通信，`POST` 请求体为 JSON，`GET` 返回静态配置。

```
GET  /                        → 返回前端 HTML
GET  /api/models              → 所有模型预设列表
GET  /api/hardware            → 所有硬件预设列表（含扩展参数）
GET  /api/constants           → 前端颜色/形状常量

POST /api/analyze             → 核心分析：FLOPs + 内存访问 + Roofline
POST /api/roofline            → 指定硬件的 Roofline 数据
POST /api/compare             → 多模型对比
POST /api/export/csv          → 导出 CSV
POST /api/export/json         → 导出 JSON

POST /api/metrics             → 能量 Roofline + TCO + CO₂e + 内存容量
GET  /api/metrics/regions     → 电网碳强度地区列表

POST /api/hetero              → 异构架构分析（分层内存建模）
GET  /api/hetero/hardware     → 仅返回异构硬件列表

GET  /api/dse/presets         → DSE 参数预设列表
POST /api/dse/run             → 运行 DSE 扫描
POST /api/dse/sensitivity     → 单参数敏感度分析
```

---

## 4. 数学模型与公式推导

### 4.1 FLOPs 计算模型

以下所有公式针对**单层**，最终结果乘以层数 `L`。  
符号定义：`B`=batch_size, `s`=seq_len（prefill）或`1`（decode）, `h`=hidden_size, `n`=num_heads, `kv_n`=kv_heads, `d`=h/n（head_dim）, `I`=intermediate_size

#### Q/K/V 投影（含 GQA）

```
kv_h = h × kv_n / n    （KV 隐层维度，GQA 下 < h）

FLOPs_Q = 2 × B × s × h × h
FLOPs_K = 2 × B × s × h × kv_h
FLOPs_V = 2 × B × s × h × kv_h

GQA 收益 = (h - kv_h) × 2 / h × 100%    （KV 权重参数节省比例）
```

#### Attention 计算

```
ctx = s（prefill）或 s + gen_steps（decode 历史上下文）

FLOPs(Q×Kᵀ) = 2 × B × n × s × ctx × d
FLOPs(Softmax) = B × n × s × ctx × 5     （exp+max+sum+div+normalize）
FLOPs(Attn×V)  = 2 × B × n × s × ctx × d

总注意力 FLOPs = 4 × B × n × s × ctx × d + B × n × s × ctx × 5
```

**FlashAttention 优化**：算子融合，内存访问量从 O(s²) 降至 O(s)，但 FLOPs 不变。

#### SwiGLU FFN

```
FLOPs(Up+Gate) = 2 × B × s × h × I × 2 + B × s × I    （GEMM + SiLU element-wise）
FLOPs(Down)    = 2 × B × s × I × h
总 FFN FLOPs   = 4 × B × s × h × I + B × s × I
             ≈ 4 × B × s × h × I    （I >> 1）
```

#### MoE FFN（混合专家）

```
n_tok = num_experts_per_tok     （每 token 激活的专家数）
N     = num_local_experts        （总专家数）

FLOPs(Router)   = 2 × B × s × h × N
FLOPs(Expert_FFN) = FLOPs(Dense_FFN) × n_tok    （只计算激活专家）

Prefill 加载参数量：全部 N 个专家
Decode  加载参数量：仅 n_tok 个专家（关键 MoE 推理优势）
```

#### DeepSeek MLA（多头潜在注意力）

```
r_kv = mla_kv_lora_rank   （KV 压缩秩，约 512）
r_q  = mla_q_lora_rank    （Q 压缩秩，约 1536）
d_nope = qk_nope_head_dim  （无位置编码的 head 维度，约 128）
d_rope = qk_rope_head_dim  （RoPE 部分的 head 维度，约 64）
d_v  = v_head_dim          （V head 维度，约 128）

FLOPs(Q_down) = 2 × B × s × h × r_q
FLOPs(Q_up)   = 2 × B × s × r_q × n × (d_nope + d_rope)
FLOPs(KV_compress) = 2 × B × s × h × r_kv          （可缓存！）
FLOPs(KV_expand)   = 2 × B × s × r_kv × n × (d_nope + d_v)

MLA 核心优势：KV 缓存从 O(n × d) 压缩到 O(r_kv)
             r_kv / (n × d) ≈ 512 / (128 × 128) ≈ 3%
```

#### RoPE 旋转位置编码

```
FLOPs(RoPE-Q) = B × s × n × (d/2) × 4    （sin/cos乘法 + 加法，每半维度 4 FLOPs）
FLOPs(RoPE-K) = B × s × kv_n × (d/2) × 4
```

#### LM Head（输出层）

```
FLOPs(LM_Head) = 2 × B × 1 × h × vocab_size    （decode 阶段每步一次）
```

---

### 4.2 内存访问量模型

**内存访问量 = 输入激活 + 权重 + 输出激活**（不考虑 L1/L2 缓存命中）

```python
B_total = B_input + B_weight + B_output
        = input_shape × act_bits/8 + weight_shape × weight_bits/8 + output_shape × act_bits/8
```

**关键区别**：
- `Prefill`：激活量 ∝ `seq_len`，权重相对"便宜"（计算密集）
- `Decode`：激活量极小（1 token），权重加载成为主导（内存受限）

**KV 缓存内存量**：
```
KV_bytes = L × kv_n × d × 2 × (kv_bit/8) × context_length × batch
```

---

### 4.3 屋顶线模型（Roofline）

经典双目标 Roofline：

```
可达性能（P_attain）= min(I × BW, P_peak)

其中：
  I     = 算术强度（FLOP/Byte）= flops / total_bytes
  BW    = 内存带宽（Byte/s）
  P_peak = 峰值算力（FLOP/s）
  ridge  = P_peak / BW（脊点，FLOP/Byte）

当 I < ridge：内存受限，P_attain = I × BW
当 I ≥ ridge：计算受限，P_attain = P_peak
```

**效率指标**：
```
η_roofline = P_attain / P_peak = min(1, I / ridge)
```

---

### 4.4 能量屋顶线模型

基于 Ghane et al. ISPASS 2018，将 TDP 分解为计算功率和内存功率：

```
P_active = TDP × [α × u_c + (1−α) × u_m]

u_c = min(1, I/ridge)    （计算单元利用率）
u_m = min(1, ridge/I)    （内存接口利用率）
α   = compute_power_frac  （计算功率占比，H100 约 0.55）

执行时间 T = max(FLOPs/P_peak, Bytes/BW)

能量 E = T × P_active    （Joules）

能量效率 η_E = FLOPs / E    （FLOP/J，即等价 FLOPS/W @ attainable perf）
```

**与传统 Roofline 的关系**：
- 传统 Roofline 只描述性能上界
- 能量 Roofline 描述**每焦耳可获得的计算量**，同时受到硬件功率分配的影响

---

### 4.5 TCO 全生命周期成本模型

参考 LLMCompass Section V 和 Sun et al. 2026：

```
TCO = C_hw + C_energy

C_hw    = 硬件采购成本（USD）

C_energy = TDP × T_lifetime × utilization × PUE × electricity_price
         = TDP(W) × (lifetime_years × 8766 hours) × util × PUE × $/kWh / 1000

TCO per EFLOP = TCO / (P_peak × T_lifetime × utilization)
              = 单位计算量的总成本（$/EFLOP）

GFLOPS/$ = P_peak / C_hw    （性能价格比，反映购买效率）
```

---

### 4.6 碳排放模型（CO₂e）

参考 MLPerf Power 方法论和 Hardware Co-Design Scaling Laws (Sun 2026)：

```
运营碳排 = E_kwh × CI_grid（gCO₂/kWh）
         其中 E_kwh = TDP × T_exec / 3600000，T_exec = FLOPs / P_peak

制造碳排（摊销）= carbon_mfg_total × (FLOPs_analyzed / FLOPs_lifetime)
               其中 carbon_mfg 估算：约 1 吨 CO₂e / $1000 半导体成本（lifecycle analysis 估值）

总 CO₂e per EFLOP = (运营碳排 + 制造碳排) / (FLOPs / 1e18)
```

**碳强度参考值**（2024）：
| 地区 | gCO₂/kWh |
|------|-----------|
| 冰岛（地热） | 28 |
| 法国（核电） | 58 |
| 美国加州 | 200 |
| 欧盟平均 | 255 |
| 美国平均 | 386 |
| 德国 | 350 |
| 全球平均 | 420 |
| 中国 | 550 |

---

### 4.7 异构架构建模

**decode 吞吐量推导**（Cambricon-LLM 分析的核心）：

```
每 token 需要加载的权重量：
  W_per_token = L × (W_QKV + W_O + W_FFN_active) × B × (w_bit/8)

  W_QKV = h×h + 2×h×kv_h
  W_O   = h×h
  W_FFN = h×I×3（SwiGLU）或 h×I×2（普通），MoE 时 × n_tok/N

decode 吞吐量（tokens/s）= BW_bottleneck / W_per_token

BW_bottleneck 由权重的主要存储层决定：
  - 权重全在 SRAM  → BW_SRAM  ≈ 8 TB/s（最快）
  - 权重在 DRAM   → BW_DRAM  ≈ 512 GB/s
  - 权重溢入 Flash → BW_Flash ≈ 14 GB/s（极慢）
```

**实例计算**（LLaMA-3 8B on Cambricon-LLM Chiplet）：
```
W_per_token = 32 × (3×4096² + 4096×14336×3) × 1 × (16/8) bytes
            ≈ 32 × (50M + 176M) × 2 ≈ 14.5 GB

SRAM 容量 0.5 GB < 14.5 GB → 权重溢入 Flash
tokens/s = 14 GB/s（Flash BW）/ 14.5 GB ≈ 1 tok/s
```

这说明对 8B 模型，Flash 存储方案 decode 速度约 1 tok/s（交互式体验差）。

---

## 5. 支持的算子列表与计算方法

| 算子名称 | 类别 | 计算量公式 | 内存瓶颈 |
|----------|------|------------|----------|
| Token Embedding | Embed | lookup（微小） | 权重读取 vocab×h |
| RMSNorm / LayerNorm | Norm | 4h（RMS）或 6h（LN）每 token | 小 |
| Q Projection | QKV | 2Bsh² | 权重 h×h |
| K/V Projection (GQA) | QKV | 2Bsh×kv_h | 权重 h×kv_h |
| RoPE-Q/K | RoPE | Bsn(d/2)×4 | sin/cos 表 |
| Q×Kᵀ (Score) | Attention | 2Bnsctx×d | KV 缓存读取 |
| Softmax | Attention | Bnsctx×5 | 激活值 |
| Attn×V | Attention | 2Bnsctx×d | KV 缓存读取 |
| FlashAttention (fused) | Attention | 4Bnsctx×d+5Bnsctx | 优化内存访问 |
| O Projection | QKV | 2Bsh² | 权重 h×h |
| FFN-Up+Gate (SwiGLU) | FFN | 4BsHI+BsI | 权重 2h×I |
| FFN-Down | FFN | 2BsIh | 权重 I×h |
| MoE Router | FFN | 2BshN | 权重 h×N |
| MoE FFN（激活专家） | FFN | 上述×n_tok | 仅 n_tok 专家权重 |
| Q Down/Up (MLA) | QKV | 2Bsh×r_q, 2Bsr_q×n×d' | LoRA 分解权重 |
| KV Compress/Expand (MLA) | QKV | 2Bsh×r_kv, 2Bsr_kv×n×d' | 压缩 KV（可缓存） |
| LM Head | Embed | 2Bh×vocab | 权重 h×vocab |

---

## 6. 支持的模型与硬件

### 模型列表

| 模型系列 | 规格 | 特性 |
|----------|------|------|
| LLaMA-2 | 7B, 13B, 70B | GQA（70B）, SwiGLU |
| LLaMA-3 | 8B, 70B | GQA, SwiGLU, 大词表 128K |
| Mixtral | 8×7B, 8×22B | MoE (2/8), SwiGLU |
| Qwen2 | 7B, 72B | GQA, 大词表 152K |
| Qwen2-MoE | 57B-A14B | MoE (4/64) |
| DeepSeek-V2 | 236B-A21B | MLA + MoE (6/160) |
| DeepSeek-R1 | 671B | MLA + MoE (8/256) |
| Gemma-2 | 9B, 27B | GQA |
| Phi-3 | Mini 3.8B | 标准架构 |
| BitNet b1.58 | 3B | 1.58-bit 权重量化 |

### 异构/PIM 硬件

| 架构名称 | 存储层次 | Decode BW | 适用场景 |
|----------|----------|-----------|----------|
| Cambricon-LLM (Chiplet) | SRAM 0.5GB + HBM 16GB + Flash 256GB | 14 GB/s (Flash 瓶颈) | 大模型离线推理 |
| Flash-LLM (NAND Storage) | SRAM 0.25GB + LPDDR5 4GB + Flash 512GB | 14 GB/s | 移动端大模型 |
| NAND-PIM (Near-Storage) | 128MB SRAM + 2GB DRAM + Flash PIM 256GB | 1.6 TB/s (PIM direct) | 近存计算 |

---

## 7. API 接口说明

### `/api/analyze`（核心分析）

**请求**：
```json
{
  "hidden_size": 4096,
  "num_heads": 32,
  "num_key_value_heads": 8,
  "num_layers": 32,
  "intermediate_size": 14336,
  "vocab_size": 128256,
  "seq_len": 2048,
  "batch_size": 1,
  "max_gen_len": 4096,
  "use_gate_ffn": true,
  "use_rmsnorm": true,
  "rope_theta": 500000.0,
  "quant_config": {
    "activation": 16,
    "weight_attn": 16,
    "weight_ffn": 16,
    "kv_cache": 16,
    "rope_bit": 32
  },
  "hardware_key": "NVIDIA H100 SXM"
}
```

**响应**：
```json
{
  "success": true,
  "results": [
    {
      "phase": "Prefill",
      "operation": "Q Projection",
      "category": "QKV",
      "flops": 68719476736,
      "flops_total": 2199023255552,
      "density": 7.999,
      "total_bytes_total": 274877906944,
      "is_memory_bound": false,
      "note": ""
    }
  ],
  "summary": { ... },
  "roofline": { ... }
}
```

### `/api/metrics`（扩展指标）

**请求**：
```json
{
  "config": { ... },
  "hardware_key": "NVIDIA H100 SXM",
  "tco_params": {
    "lifetime_years": 3,
    "utilization": 0.5,
    "electricity_price": 0.10,
    "pue": 1.3
  },
  "co2_region": "Global_Average"
}
```

**响应关键字段**：
```json
{
  "metrics": {
    "energy_points": [ { "energy_efficiency_gflops_per_w": 12.3, ... } ],
    "energy_curve": { "x": [...], "y": [...], "ridge_point": 20.0 },
    "tco": { "total_tco_usd": 31196, "tco_per_eflop_usd": 0.05 },
    "co2e": { "total_co2e_kg": 0.00004, "operational_co2e_kg": ... },
    "memory": { "weights_gb": 15.2, "kv_max_gb": 2.1, "model_fits": true }
  }
}
```

### `/api/dse/run`（设计空间探索）

**请求**：
```json
{
  "config": { ... },
  "dse_params": {
    "peak_performance_tflops": [1, 10, 100],
    "memory_bandwidth_gbs": [10, 100, 1000],
    "memory_capacity_gb": [4, 16, 64],
    "tdp_w": [20, 100, 400],
    "cost_usd": [500, 5000, 30000]
  },
  "co2_region": "US_Average",
  "max_points": 300
}
```

---

## 8. 量化配置说明

工具支持细粒度量化配置，可对不同组件设置独立精度：

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `activation` | 激活值精度 | 8（INT8）或 16（BF16/FP16） |
| `weight_attn` | 注意力层权重精度 | 4（INT4）、8（INT8）、16（FP16） |
| `weight_ffn` | FFN 层权重精度 | 同上；MoE 模型通常 4-bit |
| `kv_cache` | KV 缓存量化精度 | 8（节省约 50% KV 内存） |
| `rope_bit` | RoPE 旋转编码精度 | 32（FP32，通常不量化） |

**BitNet b1.58 配置示例**（三值量化 -1/0/+1）：
```json
{
  "weight_attn": 2,
  "weight_ffn": 2,
  "activation": 8,
  "kv_cache": 4,
  "rope_bit": 16
}
```

---

## 9. 实验设计建议

### 实验 1：算子级性能瓶颈分析

**目标**：识别 Prefill vs Decode 阶段各类算子的内存/计算瓶颈

**方法**：
```python
from analyzer import LLMAnalyzer
from configs import MODEL_CONFIGS, HARDWARE_CONFIGS

models = ['LLaMA-3 8B', 'Mixtral 8x7B', 'DeepSeek-V2 (MLA+MoE)']
hw     = HARDWARE_CONFIGS['NVIDIA H100 SXM']

for mname in models:
    a = LLMAnalyzer(MODEL_CONFIGS[mname])
    ops = a.analyze()
    # 统计各类别在 Prefill/Decode 的 FLOPs 占比
    # 统计内存受限 vs 计算受限算子比例
    # 绘制 Roofline 散点图
```

**期望发现**：
- Decode 阶段几乎所有算子都是内存受限（密度 < ridge）
- Prefill 的 FFN/Attention 在大 batch 下可能进入计算受限区域
- MLA 的 KV 压缩率约为标准 GQA 的 3%

### 实验 2：能量效率对比

**目标**：比较不同硬件平台在 LLM 推理上的能量效率

**方法**：
```python
from metrics import run_full_metrics
from configs import MODEL_CONFIGS, HARDWARE_CONFIGS

hw_list = ['NVIDIA H100 SXM', 'AMD MI300X', 'Apple M3 Ultra',
           'Snapdragon 8 Gen 3 NPU', 'DRAM-PIM (HBM-PIM)']

for hw_key in hw_list:
    metrics = run_full_metrics(ops, summary, HARDWARE_CONFIGS[hw_key], cfg)
    # 记录: peak_efficiency, avg_efficiency, total_energy_j
    # 绘制能量 Roofline 曲线对比
```

### 实验 3：异构架构 Decode 吞吐分析

**目标**：量化 Flash 存储近存推理的性能瓶颈

**方法**：
```python
from hetero import HeteroAnalyzer
from configs import MODEL_CONFIGS, HARDWARE_CONFIGS

models = ['LLaMA-3 8B', 'LLaMA-2 70B', 'Mixtral 8x7B']
hw     = HARDWARE_CONFIGS['Cambricon-LLM (Chiplet)']

for m in models:
    ha = HeteroAnalyzer(MODEL_CONFIGS[m], hw)
    r  = ha.run_full_analysis()
    print(f"{m}: {r['throughput']['decode_tokens_per_sec']:.2f} tok/s"
          f" ({r['placement']['summary']['decode_bottleneck']} bottleneck)")
```

**期望发现**：
- 7B/8B 模型在 Flash 受限时约 1 tok/s
- 量化到 4-bit 后约 2 tok/s（权重减半）
- 需要 SRAM 容量 > 模型大小才能达到 SRAM 速度

### 实验 4：多目标 DSE 硬件设计探索

**目标**：为特定 LLM 工作负载找到最优硬件设计点

**方法**：
```python
from dse import DSEEngine, DSE_PRESETS

engine = DSEEngine(MODEL_CONFIGS['LLaMA-3 8B'])
result = engine.run(DSE_PRESETS['PIM / Near-Memory DSE'],
                    co2_region='China')

# 分析 Pareto 前沿
for p in result['pareto_perf_cost']:
    print(f"BW={p['memory_bandwidth_gbs']:.0f} GB/s, "
          f"峰值={p['peak_performance_tflops']:.1f} TFLOPS, "
          f"TCO={p['tco_per_eflop_usd']:.2f} $/EFLOP")

# 敏感度：带宽对性能的影响
sa = engine.sensitivity_analysis(HARDWARE_CONFIGS['DRAM-PIM (HBM-PIM)'],
                                  param='memory_bandwidth',
                                  multipliers=[0.25, 0.5, 1, 2, 4, 8])
```

---

## 10. 主要参考文献

1. **Roofline 模型（原始）**  
   Williams, S., Waterman, A., & Patterson, D. (2009).  
   *Roofline: An insightful visual performance model for multicore architectures.*  
   Communications of the ACM, 52(4), 65-76.

2. **能量 Roofline**  
   Ghane, S., et al. (2018).  
   *Power and Energy-efficiency Roofline Model for GPUs.*  
   IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS).

3. **LLMCompass（硬件评估框架）**  
   Zhang, H., et al. (2024).  
   *LLMCompass: Enabling Efficient Hardware Design for Large Language Model Inference.*  
   ISCA 2024. [[arxiv]](https://arxiv.org/abs/2312.03134)

4. **Hardware Co-Design Scaling Laws**  
   Sun, Y., et al. (2026).  
   *Hardware Co-Design Scaling Laws via Roofline Modelling for On-Device LLMs.*

5. **Cambricon-LLM（异构 Chiplet 架构）**  
   Yu, Z., et al. (2024).  
   *Cambricon-LLM: A Chiplet-Based Hybrid Architecture for On-Device Inference of 70B LLM.*

6. **FlashAttention-2**  
   Dao, T. (2023).  
   *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.*  
   ICLR 2024.

7. **GQA（Grouped Query Attention）**  
   Ainslie, J., et al. (2023).  
   *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.*  
   EMNLP 2023.

8. **MoE（混合专家）**  
   Fedus, W., et al. (2022).  
   *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.*  
   JMLR 2022.

9. **DeepSeek-V2（MLA 多头潜在注意力）**  
   DeepSeek-AI. (2024).  
   *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model.*

10. **Flash-LLM（Flash 存储 LLM 推理）**  
    Xue, C., et al. (2024).  
    *FlashLLM: Enabling Cost-Effective and Highly-Efficient LLM Inference with Unstructured Sparsity.*

11. **BitNet b1.58（极限量化）**  
    Ma, S., et al. (2024).  
    *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits.*

---

> **工具主页**：[https://llm-para.onrender.com](https://llm-para.onrender.com)  
> **代码仓库**：[https://github.com/dengls24/LLM-para](https://github.com/dengls24/LLM-para)  
> **联系方式**：如有问题请提 GitHub Issue
