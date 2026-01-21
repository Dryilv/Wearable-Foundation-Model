# CWT-MAE-RoPE 模型架构与微调指南

## 1. 项目概述 (Project Overview)

本项目实现了一个针对一维时间序列信号的深度学习框架。它结合了信号处理（连续小波变换 CWT）与计算机视觉中的掩码自编码器（MAE）思想，通过 **"预训练-微调" (Pre-train then Fine-tune)** 的范式，实现对时间序列的高效表征学习和下游分类。

**核心流程：**
1.  **信号转图像**：利用 CWT 将一维信号转换为三通道时频图（Spectrogram）。
2.  **预训练 (Pre-training)**：使用带有 RoPE 和张量分解的 Transformer，通过重建被遮蔽的时频图和原始信号来学习通用特征。
3.  **微调 (Fine-tuning)**：丢弃解码器，加载编码器权重，并连接高级分类头（如思维链推理头）进行具体任务训练。

---

## 第一部分：核心架构与预训练 (`model.py`)

### 1.1 信号预处理模块 (CWT Module)
模型不直接处理原始信号，而是处理其时频表征。

*   **多视图生成 (`cwt_wrap`)**:
    *   **输入**: 原始信号 $x$。
    *   **增强**: 计算一阶差分 ($d1$) 和二阶差分 ($d2$)。
    *   **变换**: 对 $[x, d1, d2]$ 分别进行 Ricker 小波变换。
    *   **输出**: `(Batch, 3, Scales, Length)` 的 4D 张量，视为 3 通道图像。
*   **Ricker 小波**: 使用墨西哥帽小波核，通过卷积提取不同频率尺度的特征。

### 1.2 基础组件 (Basic Components)

*   **DecomposedPatchEmbed (内存安全版)**:
    *   使用标准的 `nn.Conv2d` 将 CWT 图像切分为 Patch 并映射到嵌入维度。
    *   *优化*: 避免了中间张量膨胀导致的显存溢出问题。
*   **RoPE (Rotary Positional Embedding)**:
    *   在 Attention 层引入旋转位置编码，增强模型对序列相对位置的感知。
    *   支持动态缓存，适应变长序列。
*   **TensorizedLinear (参数高效层)**:
    *   在 Encoder 的 MLP 中，将全连接层分解为 $U \times V$。
    *   通过 `rank_ratio` (默认 0.5) 控制参数量，降低过拟合风险。

### 1.3 预训练模型: `CWT_MAE_RoPE`

这是一个非对称的 Encoder-Decoder 架构。

#### A. 编码器 (Encoder)
*   **输入**: 部分可见的 Patch（根据 `mask_ratio` 随机采样）。
*   **位置编码**: 结合了可学习的绝对位置编码和 RoPE。
*   **结构**: 堆叠多个 `TensorizedBlock`。

#### B. 解码器 (Decoder)
*   **输入**: 编码器的潜在向量 + 可学习的 `[MASK]` token（恢复完整序列）。
*   **结构**: 轻量级 Transformer Blocks（标准 Linear，未分解）。
*   **任务**: 仅用于预训练阶段的重建任务。

#### C. 双重损失函数 (Dual Loss)
模型同时优化频域和时域的重建质量：
$$ Loss_{total} = Loss_{spec} + \lambda \cdot Loss_{time} $$

1.  **谱重建 (`forward_loss_spec`)**: 计算被 Mask 掉的时频图 Patch 的 MSE Loss。
2.  **时域重建 (`forward_loss_time`)**: 通过 `time_reducer` 聚合特征，预测原始一维信号，计算 MSE Loss。

---

## 第二部分：下游任务微调 (`model_finetune.py`)

### 2.1 分类器封装: `TF_MAE_Classifier`
该类负责将预训练模型适配到具体的分类任务。

*   **自动瘦身**: 初始化时自动删除预训练模型的 Decoder 部分（包括重建头、Mask Token 等），显著降低显存占用。
*   **位置编码插值**: 如果微调时的输入信号长度与预训练不一致，自动对 `pos_embed` 进行双三次插值（Bicubic Interpolation）。
*   **特征提取**: 强制关闭 Mask (`mask_ratio=0`)，并丢弃 CLS Token，仅使用 Patch Tokens 作为特征输入。

### 2.2 高级分类头 (Heads)

模型提供了两种分类头模式，通过 `use_cot` 参数切换：

#### A. 隐式思维链头 (Latent Reasoning Head / CoT)
*   **适用场景**: 复杂信号分类，需要捕捉局部关键特征。
*   **机制**:
    1.  初始化一组可学习的 **推理令牌 (Reasoning Tokens)**。
    2.  通过 **Cross-Attention** 查询 Encoder 输出的全局特征。
    3.  通过 **Self-Attention** 整合推理结果。
    4.  最终通过分类器输出 Logits。

#### B. 残差 MLP 头 (Residual MLP)
*   **适用场景**: 简单任务或作为基准对比。
*   **机制**: 全局平均池化 -> [LayerNorm -> Linear -> GELU -> Dropout -> Residual] x N -> Output。

---

## 第三部分：API 参数说明

### 3.1 预训练模型 (`CWT_MAE_RoPE`)
| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `signal_len` | 3000 | 输入信号长度 |
| `cwt_scales` | 64 | CWT 尺度（图像高度） |
| `embed_dim` | 768 | Encoder 维度 |
| `depth` | 12 | Encoder 层数 |
| `mask_ratio` | 0.75 | 预训练掩码比例 |
| `mlp_rank_ratio` | 0.5 | MLP 压缩比 |
| `time_loss_weight`| 1.0 | 时域损失权重 |

### 3.2 微调模型 (`TF_MAE_Classifier`)
| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `pretrained_path` | Required | 预训练权重路径 |
| `num_classes` | Required | 类别数量 |
| `use_cot` | True | 是否使用推理头 (False 则使用 MLP) |
| `num_reasoning_tokens`| 8 | CoT 模式下的 Token 数量 |
| `num_res_blocks` | 2 | MLP 模式下的残差块数量 |

---

## 第四部分：使用示例

### 4.1 阶段一：预训练 (Pre-training)

```python
import torch
from model import CWT_MAE_RoPE

# 1. 初始化模型
model = CWT_MAE_RoPE(
    signal_len=3000,
    cwt_scales=64,
    mask_ratio=0.75  # 高掩码率用于自监督学习
)

# 2. 输入数据 (Batch, Length)
x = torch.randn(32, 3000)

# 3. 前向传播 (返回 Loss 用于反向传播)
loss, pred_spec, pred_time, imgs = model(x)
print(f"Pre-training Loss: {loss.item()}")
```

### 4.2 阶段二：微调 (Fine-tuning)

```python
import torch
from model_finetune import TF_MAE_Classifier

# 1. 初始化分类器 (加载预训练权重)
classifier = TF_MAE_Classifier(
    pretrained_path="checkpoints/mae_epoch_100.pth",
    num_classes=10,
    embed_dim=768,       # 需与预训练一致
    use_cot=True,        # 启用思维链推理头
    num_reasoning_tokens=16
)

# 2. 输入数据 (可以是不同的 Batch Size)
x_val = torch.randn(16, 3000)

# 3. 推理/训练
logits = classifier(x_val) # Output: (16, 10)
print(f"Logits shape: {logits.shape}")
```

---

## 第五部分：关键技术细节与注意事项

1.  **显存优化**:
    *   `DecomposedPatchEmbed` 现已修正为标准卷积，解决了早期版本中间张量过大的问题。
    *   微调时，`TF_MAE_Classifier` 会主动删除 Encoder 中不需要的组件（如 Decoder），使得在较小显存的 GPU 上也能运行大 Batch 微调。

2.  **输入长度适配**:
    *   预训练和微调的信号长度可以不同（例如预训练 3000，微调 6000）。模型会自动对位置编码进行插值。但建议长度差异不要过大，以免频域特征分布发生剧烈变化。

3.  **特征选择策略**:
    *   在微调阶段，分类头**不使用** CLS Token (Index 0)。代码逻辑是 `latent[:, 1:, :]`。这是因为在 MAE 架构中，Patch Tokens 通常保留了更丰富的局部纹理和频率信息，更适合 CoT 头进行查询。

4.  **RoPE 缓存**:
    *   `RotaryEmbedding` 具有缓存机制。如果推理时的序列长度超过了缓存上限，它会自动重新计算并更新缓存，无需人工干预。