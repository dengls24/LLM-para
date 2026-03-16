# 🚀 部署指南 — llm-para.com

## 第一步：推送到 GitHub

### 1.1 关联远程仓库（替换现有 repo 或新建）

```bash
# 进入项目目录
cd 4-LLM-pare

# 关联到你的 GitHub repo（用你自己的 token 替换 YOUR_TOKEN）
git remote add origin https://dengls24:YOUR_TOKEN@github.com/dengls24/LLM-para.git

# 推送 main 分支（新 web 版本，作为默认分支）
git push -u origin main --force

# 推送 v1-cli 分支（保留原始 CLI 版本）
git push origin v1-cli
```

> 推送后 GitHub 上会有两个分支：
> - `main` → 新 Web 版本（默认分支，对外展示）
> - `v1-cli` → 原始 CLI 版本（存档保留）

### 1.2 在 GitHub 上设置 Release

1. 进入 https://github.com/dengls24/LLM-para
2. 点击 **Releases → Draft a new release**
3. 填写：
   - Tag: `v2.0.0`
   - Title: `v2.0 — Web Interface + Comprehensive Operator Analysis`
   - 描述（粘贴下面的内容）

```
## What's New in v2.0

### 🌐 Web Interface
Interactive visualization powered by Flask + Chart.js. Open your browser, configure model parameters, and instantly visualize FLOPs, memory, and Roofline performance.

### 🔬 New Operators (16 total)
Token Embedding, RMSNorm/LayerNorm, Q/K/V Projections, RoPE, 
Softmax, FlashAttention, O Projection, MoE Router, SwiGLU FFN, 
DeepSeek MLA, LM Head — every operator in the inference pipeline.

### 🤖 19 Model Presets
GPT-2 · LLaMA-2/3 (7B–405B) · Mixtral 8x7B/8x22B ·
Qwen2 7B/72B/MoE · DeepSeek-V2/R1 · Gemma-2 · Phi-3 · BitNet b1.58

### 💻 20+ Hardware Platforms
H100 · A100 · MI300X · RTX 4090 · Apple M3 Ultra ·
Snapdragon 8 Gen 3 · NAND-PIM (HILOS) · DRAM-PIM · Gaudi 3

### 📊 4 Analysis Views
Operations Table · Roofline Model · Analysis Charts · Memory Analysis

> The original CLI tool is preserved in the `v1-cli` branch.
```

---

## 第二步：部署到公网（让所有人都能访问）

### 方案 A：Render.com（推荐，免费）

1. 访问 https://render.com → 注册账号（用 GitHub 登录）
2. 点击 **New → Web Service**
3. 选择 GitHub 仓库 `dengls24/LLM-para`
4. 配置如下：

   | 字段 | 值 |
   |---|---|
   | Name | `llm-para` |
   | Runtime | `Python 3` |
   | Build Command | `pip install -r requirements.txt` |
   | Start Command | `gunicorn wsgi:app --workers 2 --bind 0.0.0.0:$PORT --timeout 120` |
   | Plan | Free |

5. 点击 **Create Web Service**
6. 等待约 2-3 分钟部署完成
7. 你会得到：`https://llm-para.onrender.com`（免费子域名，可直接访问）

### 方案 B：Railway.app（国内访问更稳定）

```bash
# 安装 Railway CLI
npm install -g @railway/cli

# 登录
railway login

# 部署
cd 4-LLM-pare
railway init
railway up
```

### 方案 C：Vercel（前后端分离，需改造）

不直接支持 Flask，需要配置 serverless，推荐用 Render。

---

## 第三步：绑定自定义域名 llm-para.com

### 3.1 注册域名
在以下任一平台购买 `llm-para.com`：
- [Namecheap](https://www.namecheap.com)（约 $10/年，国际常用）
- [阿里云万网](https://wanwang.aliyun.com)（国内访问快，需备案如在国内服务器）
- [腾讯云 DNSPod](https://dnspod.cloud.tencent.com)

### 3.2 在 Render.com 添加自定义域名

1. 进入你的 Render Web Service
2. Settings → **Custom Domains** → Add Custom Domain
3. 输入 `llm-para.com`（以及 `www.llm-para.com`）
4. Render 会给你一个 CNAME 记录，例如：
   ```
   CNAME  @    llm-para.onrender.com
   CNAME  www  llm-para.onrender.com
   ```

### 3.3 在域名注册商处配置 DNS

进入域名的 DNS 管理，添加以下记录：

| 类型 | 主机名 | 值 |
|---|---|---|
| CNAME | `@` 或 `llm-para.com` | `llm-para.onrender.com` |
| CNAME | `www` | `llm-para.onrender.com` |

DNS 生效需要 5 分钟 ～ 48 小时（通常 10 分钟内）。

### 3.4 HTTPS 证书

Render 会自动为你的自定义域名申请 Let's Encrypt SSL 证书，无需额外配置。

---

## 第四步：验证部署

```bash
# 检查网站是否可访问
curl -I https://llm-para.com

# 测试 API
curl -X POST https://llm-para.com/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"hidden_size":4096,"num_heads":32,"num_key_value_heads":8,"num_layers":32,
       "intermediate_size":14336,"vocab_size":128256,"seq_len":2048,"batch_size":1,
       "max_gen_len":4096,"use_gate_ffn":true,"use_rmsnorm":true,
       "rope_theta":500000,"rope_scaling_factor":1.0,
       "quant_config":{"activation":16,"weight_attn":16,"weight_ffn":16,"kv_cache":16,"rope_bit":32}}'
```

---

## 快速命令参考

```bash
# 查看分支
git branch -a

# 查看 v1-cli 原始代码
git checkout v1-cli
git checkout main  # 回到新版本

# 更新部署（修改代码后）
git add .
git commit -m "update: ..."
git push origin main  # Render 自动检测并重新部署
```

---

## 架构说明

```
llm-para.com  (Render.com 或其他 PaaS)
    │
    ├── GET  /              → static/index.html (Web UI)
    ├── GET  /api/models    → 19 个预设模型列表
    ├── GET  /api/hardware  → 20+ 硬件平台列表
    ├── POST /api/analyze   → 运行分析，返回完整结果
    ├── POST /api/roofline  → Roofline 数据
    ├── POST /api/export/*  → CSV / JSON 导出
    └── POST /api/compare   → 多模型对比
```
