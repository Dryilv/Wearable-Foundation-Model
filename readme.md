# CWT-MAE-RoPE 模型文档

## 1. 模型概述 (Overview)

**CWT-MAE-RoPE** 是一个用于时间序列分析的深度学习模型。它不直接处理原始的一维信号，而是通过连续小波变换（CWT）将信号转换为时频图（Spectrogram），并将其视为图像进行处理。

**核心特点：**
*   **多视图输入**：利用原始信号及其一阶、二阶差分生成 3 通道时频图。
*   **RoPE (Rotary Positional Embedding)**：在 Transformer 的 Attention 层引入旋转位置编码，增强模型对序列相对位置的感知能力。
*   **参数高效 (Tensorized Linear)**：在 Encoder 的 MLP 层使用低秩分解（UV分解），减少参数量并降低过拟合风险。
*   **双重重建任务**：模型同时重建 **时频图（频域）** 和 **原始信号（时域）**，通过多任务学习提取更鲁棒的特征。

---

## 2. 核心组件详解

### 2.1 信号预处理模块 (CWT Module)
该模块负责将一维时间序列转换为二维时频图像。

*   **`create_ricker_wavelets` & `cwt_ricker`**:
    *   生成 Ricker 小波（墨西哥帽小波）核。
    *   对输入信号进行一维卷积，提取不同尺度（频率）下的特征。
*   **`cwt_wrap`**:
    *   **输入**: 原始信号 $x$。
    *   **差分增强**: 计算一阶差分 ($d1$) 和二阶差分 ($d2$)。
    *   **通道堆叠**: 将 $[x, d1, d2]$ 堆叠为 3 个通道。
    *   **输出**: 形状为 `(Batch, 3, Scales, Length)` 的 4D 张量（类似彩色图片）。

### 2.2 位置编码 (Positional Embeddings)
模型使用了两种位置编码策略：

1.  **绝对位置编码 (Learnable Absolute PE)**:
    *   在 Patch Embedding 后直接相加，用于保留全局位置信息。
2.  **旋转位置编码 (RoPE)**:
    *   **类**: `RotaryEmbedding`
    *   **作用**: 在 Attention 计算 $Q, K$ 时注入相对位置信息。
    *   **缓存机制**: 动态缓存 `cos` 和 `sin` 值，支持变长序列推理。

### 2.3 基础 Transformer 组件
*   **`DecomposedPatchEmbed` (优化版)**:
    *   使用标准的 `nn.Conv2d` 将 CWT 生成的图像切分为 Patch 并映射到嵌入维度。
    *   *注意*：代码中已针对内存安全进行了优化，避免了中间张量过大导致的显存溢出。
*   **`TensorizedLinear`**:
    *   将全连接层 $W$ 分解为 $U \times V$。
    *   **Rank Ratio**: 控制分解的秩（默认为 0.5），在保持性能的同时压缩模型。
*   **`TensorizedBlock` (Encoder)**:
    *   包含 RoPE Attention 和 Tensorized MLP。
*   **`Block` (Decoder)**:
    *   包含 RoPE Attention 和标准 MLP（Decoder 通常参数较少，未做分解）。

---

## 3. 主模型架构: `CWT_MAE_RoPE`

### 3.1 初始化参数
| 参数名 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `signal_len` | 3000 | 输入信号的时间长度 |
| `cwt_scales` | 64 | CWT 的尺度数量（对应图像高度） |
| `patch_size_time` | 50 | 时间维度的 Patch 大小 |
| `patch_size_freq` | 4 | 频率维度的 Patch 大小 |
| `embed_dim` | 768 | Encoder 嵌入维度 |
| `depth` | 12 | Encoder 层数 |
| `mask_ratio` | 0.75 | 预训练时的掩码比例 |
| `mlp_rank_ratio` | 0.5 | MLP 低秩分解比例 |
| `time_loss_weight` | 1.0 | 时域重建损失的权重 |

### 3.2 前向传播流程 (Forward Pass)

1.  **输入处理**:
    *   输入形状: `(B, L)` 或 `(B, 1, L)`。
    *   执行 CWT 得到 `(B, 3, 64, 3000)`。
    *   **归一化**: 对 CWT 图像进行 Instance Normalization (均值为0，方差为1)。

2.  **Encoder (编码器)**:
    *   **Patch Embed**: 将图像切分为 Patch。
    *   **Masking**:
        *   若 `mask_ratio > 0`: 随机遮蔽 75% 的 Patch，仅保留可见部分。
        *   若 `mask_ratio == 0`: 保留所有 Patch (用于微调或推理)。
    *   **RoPE**: 根据 Patch 的真实位置索引生成旋转编码。
    *   **Transformer**: 处理可见的 Patch。

3.  **Decoder (解码器)**:
    *   **重组**: 将 Encoder 输出的潜在向量与可学习的 `[MASK]` token 拼接，恢复原始序列顺序。
    *   **RoPE**: 对完整序列应用旋转位置编码。
    *   **Transformer**: 处理完整序列。

4.  **预测头 (Heads)**:
    *   **谱重建头 (`decoder_pred_spec`)**: 预测被遮蔽 Patch 的像素值。
    *   **时域重建头 (`time_pred`)**:
        *   通过 `time_reducer` (Conv2d) 聚合频率维度的特征。
        *   通过线性层映射回时间维度的原始信号片段。

### 3.3 损失函数 (Loss Function)

总损失由两部分组成：
$$ Loss_{total} = Loss_{spec} + \lambda \cdot Loss_{time} $$

1.  **谱重建损失 (`forward_loss_spec`)**:
    *   计算预测的时频图 Patch 与 原始 CWT 图像 Patch 之间的 **MSE Loss**。
    *   **仅计算被 Mask 掉的部分** (MAE 的核心特性)。

2.  **时域重建损失 (`forward_loss_time`)**:
    *   计算预测的时间序列与原始输入信号之间的 **MSE Loss**。
    *   原始信号在计算 Loss 前会进行标准化 (Standardization)。

---

## 4. 输入输出说明

### 输入 (Input)
*   **张量**: `x`
*   **形状**: `(Batch_Size, Sequence_Length)` 例如 `(32, 3000)`。
*   **类型**: `torch.Tensor` (Float)。

### 输出 (Output)
模型 `forward` 方法返回一个元组：
1.  **`total_loss`**: 标量，用于反向传播。
2.  **`pred_spec`**: `(B, N, P)`，重建的时频图 Patch 像素。
3.  **`pred_time`**: `(B, L)`，重建的一维时间序列信号。
4.  **`imgs`**: `(B, 3, H, W)`，CWT 变换后的真实时频图（用于可视化对比）。

---

## 5. 使用示例

```python
import torch
from model import CWT_MAE_RoPE

# 1. 实例化模型
model = CWT_MAE_RoPE(
    signal_len=3000,
    cwt_scales=64,
    embed_dim=768,
    depth=12,
    num_heads=12,
    mask_ratio=0.75,       # 预训练设为 0.75
    mlp_rank_ratio=0.5,    # 启用参数压缩
    time_loss_weight=1.0
)

# 2. 创建虚拟数据 (Batch=2, Length=3000)
x = torch.randn(2, 3000)

# 3. 前向传播
loss, pred_spec, pred_time, cwt_imgs = model(x)

print(f"Total Loss: {loss.item()}")
print(f"Reconstructed Signal Shape: {pred_time.shape}") # (2, 3000)
print(f"CWT Image Shape: {cwt_imgs.shape}")             # (2, 3, 64, 3000)

# 4. 微调/推理模式 (关闭 Mask)
model.mask_ratio = 0.0
loss_ft, _, _, _ = model(x)
print(f"Fine-tuning Loss: {loss_ft.item()}")
```

## 6. 关键修改记录 (Changelog)

*   **RoPE 集成**: 在 Encoder 和 Decoder 的 Attention 层中均集成了 RoPE，且正确处理了 Mask 后的非连续位置索引。
*   **微调优化**: 在 `forward_encoder` 中增加了 `if self.mask_ratio == 0.0` 分支，避免了在推理或微调时进行不必要的随机排序计算，提高了效率。