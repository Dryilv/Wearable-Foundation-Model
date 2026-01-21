---
marp: true
theme: gaia
paginate: true
backgroundColor: #fff
header: 'CWT-MAE-RoPE 模型架构与微调指南'
footer: 'version preview 0.0.5'
---

<!-- _class: lead -->
<!-- _backgroundColor: #1E3A8A -->
<!-- _color: #fff -->

# **CWT-MAE-RoPE**
## 模型架构与微调指南
<br>
**主讲人：Your Name**
**日期：2026-01-21**

---

## 1. 项目概述 (Project Overview)

本项目是一个针对一维时间序列的 **"预训练-微调"** 深度学习框架。

**核心流程:**
1.  **信号转图像**: 利用 **CWT** 将一维信号转换为三通道时频图。
2.  **预训练**: 使用 **MAE** 范式（带 RoPE 和张量分解）学习通用特征。
3.  **微调**: 加载预训练 Encoder，连接高级分类头（如思维链）进行下游任务。

![bg right:40%](https://i.imgur.com/3j3b6qj.png) <!-- 这是一个示意图占位符，你可以换成自己的架构图 -->

---

<!-- _backgroundColor: #f0f0f0 -->

# **第一部分**
## 核心架构与预训练 (`model.py`)

---

### 1.1 信号预处理模块 (CWT Module)

模型不直接处理原始信号，而是处理其 **时频表征 (Spectrogram)**。

*   **多视图生成 (`cwt_wrap`)**:
    *   **输入**: 原始信号 $x$。
    *   **增强**: 计算一阶 ($d1$) 和二阶 ($d2$) 差分。
    *   **变换**: 对 $[x, d1, d2]$ 分别进行 Ricker 小波变换。
    *   **输出**: `(Batch, 3, Scales, Length)` 的 4D 张量，视为 3 通道图像。

---

### 1.2 基础组件 (Basic Components)

*   **DecomposedPatchEmbed**:
    *   使用标准 `nn.Conv2d` 将 CWT 图像切分为 Patch 并嵌入。
    *   内存安全，避免中间张量膨胀。

*   **RoPE (Rotary Positional Embedding)**:
    *   在 Attention 层引入旋转位置编码，增强相对位置感知。
    *   支持动态缓存，适应变长序列。

*   **TensorizedLinear**:
    *   在 Encoder 的 MLP 中，将全连接层分解为 $U \times V$。
    *   通过 `rank_ratio` 控制参数量，降低过拟合风险。

---

### 1.3 预训练模型: `CWT_MAE-RoPE` (非对称架构)

#### A. 编码器 (Encoder)
*   **输入**: 部分可见的 Patch（`mask_ratio` 随机采样）。
*   **位置编码**: 结合可学习的绝对位置编码和 RoPE。
*   **结构**: 堆叠多个 `TensorizedBlock`。

#### B. 解码器 (Decoder)
*   **输入**: 编码器输出 + 可学习的 `[MASK]` token。
*   **结构**: 轻量级 Transformer，仅用于预训练。

---

### 1.3 预训练模型 (续): 双重损失函数

模型同时优化频域和时域的重建质量：
$$ Loss_{total} = Loss_{spec} + \lambda \cdot Loss_{time} $$

1.  **谱重建 (`forward_loss_spec`)**:
    *   计算被 Mask 掉的时频图 Patch 的 MSE Loss。

2.  **时域重建 (`forward_loss_time`)**:
    *   通过 `time_reducer` 聚合特征，预测原始一维信号，计算 MSE Loss。

---

<!-- _backgroundColor: #f0f0f0 -->

# **第二部分**
## 下游任务微调 (`model_finetune.py`)

---

### 2.1 分类器封装: `TF_MAE_Classifier`

该类负责将预训练模型适配到具体的分类任务。

*   **自动瘦身**: 初始化时自动删除 Decoder，显著降低显存。
*   **位置编码插值**: 输入信号长度变化时，自动对位置编码进行双三次插值。
*   **特征提取**: 强制 `mask_ratio=0`，丢弃 CLS Token，仅使用 Patch Tokens。

---

### 2.2 高级分类头 (Heads)

#### A. 隐式思维链头 (Latent Reasoning Head / CoT)
*   **机制**:
    1.  初始化一组可学习的 **推理令牌 (Reasoning Tokens)**。
    2.  通过 **Cross-Attention** 查询 Encoder 输出。
    3.  通过 **Self-Attention** 整合推理结果。
    4.  最终通过分类器输出 Logits。

#### B. 残差 MLP 头 (Residual MLP)
*   **机制**: 全局平均池化 -> 多个残差块 -> 输出。

---

<!-- _backgroundColor: #f0f0f0 -->

# **第三部分**
## API 参数说明

---

### 3.1 预训练模型 (`CWT_MAE_RoPE`)

- **`signal_len`**: 输入信号长度 (默认 `3000`)
- **`cwt_scales`**: CWT 尺度 / 图像高度 (默认 `64`)
- **`embed_dim`**: Encoder 维度 (默认 `768`)
- **`depth`**: Encoder 层数 (默认 `12`)
- **`mask_ratio`**: 预训练掩码比例 (默认 `0.75`)
- **`mlp_rank_ratio`**: MLP 压缩比 (默认 `0.5`)
- **`time_loss_weight`**: 时域损失权重 (默认 `1.0`)

---

### 3.2 微调模型 (`TF_MAE_Classifier`)

- **`pretrained_path`**: **(必填)** 预训练权重路径
- **`num_classes`**: **(必填)** 类别数量
- **`use_cot`**: 是否使用推理头 (默认 `True`)
- **`num_reasoning_tokens`**: CoT 模式下的 Token 数量 (默认 `8`)
- **`num_res_blocks`**: MLP 模式下的残差块数量 (默认 `2`)

---

<!-- _backgroundColor: #f0f0f0 -->

# **第四部分**
## 使用示例

---

### 4.1 阶段一：预训练 (Pre-training)

```python
import torch
from model import CWT_MAE_RoPE

# 1. 初始化模型
model = CWT_MAE_RoPE(
    signal_len=3000,
    cwt_scales=64,
    mask_ratio=0.75
)

# 2. 输入数据
x = torch.randn(32, 3000)

# 3. 前向传播 (返回 Loss)
loss, _, _, _ = model(x)
print(f"Pre-training Loss: {loss.item()}")

