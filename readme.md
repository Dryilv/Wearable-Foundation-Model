# CWT-MAE 下游微调模型文档

## 1. 模型概述 (Overview)

**TF_MAE_Classifier** 是一个专为下游分类任务设计的封装模型。它加载预训练的 `CWT_MAE_RoPE` 权重作为特征提取器（Encoder），并在此基础上添加了高级分类头。

**核心特性：**
*   **轻量化部署**：自动移除预训练模型中的 Decoder 部分（包括重建头、掩码 Token 等），显著降低显存占用。
*   **灵活的输入适配**：内置位置编码插值算法（Interpolation），允许微调时的输入信号长度与预训练时不同。
*   **双模式分类头**：
    *   **CoT 模式 (默认)**：基于 Attention 的隐式思维链推理头，适合捕捉复杂特征。
    *   **MLP 模式**：基于残差连接的深层 MLP，适合简单任务或作为基准。

---

## 2. 核心组件详解

### 2.1 隐式思维链头 (`LatentReasoningHead`)
这是一个基于 Transformer Decoder 思想设计的分类头。它不使用简单的全局平均池化，而是通过可学习的“推理令牌”去主动查询 Encoder 的输出特征。

*   **工作原理**:
    1.  **推理令牌 (Reasoning Tokens)**: 初始化一组可学习的向量（Query），代表模型试图寻找的特定特征模式。
    2.  **Cross-Attention**: 推理令牌作为 Query，Encoder 的输出 Patch 作为 Key/Value。这一步从全局特征中聚合关键信息。
    3.  **Self-Attention**: 推理令牌之间进行交互，整合不同维度的特征信息。
    4.  **FFN & Pooling**: 经过前馈网络后，对推理令牌取平均，输入线性分类器。
*   **优势**: 相比简单的 Global Average Pooling，能更灵活地关注时频图中的局部显著区域。

### 2.2 残差 MLP 块 (`ResidualMLPBlock`)
用于构建深层 MLP 分类头的基本单元。
*   **结构**: `Norm -> Linear -> GELU -> Dropout -> Linear -> Dropout -> Residual Add`。
*   **作用**: 在不使用 Attention 的情况下，通过增加深度和非线性能力来处理特征。

---

### 2.3 主分类器 (`TF_MAE_Classifier`)

这是用户直接交互的主类，负责整合 Encoder 和 Head。

#### 初始化参数
| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `pretrained_path` | str | Required | 预训练 `.pth` 权重文件的路径 |
| `num_classes` | int | Required | 下游任务的类别数量 |
| `mlp_rank_ratio` | float | 0.5 | Encoder 中 Tensorized Linear 的压缩比 |
| `use_cot` | bool | True | **True**: 使用 LatentReasoningHead<br>**False**: 使用 Residual MLP |
| `num_reasoning_tokens`| int | 8 | CoT 模式下的推理令牌数量 |
| `num_res_blocks` | int | 2 | MLP 模式下的残差块数量 |
| `hidden_dim` | int | 512 | MLP 模式下的隐藏层维度 |
| `**kwargs` | dict | - | 传递给 `CWT_MAE_RoPE` 的其他参数 (如 `embed_dim`, `depth`) |

#### 关键机制

1.  **权重加载与清洗 (`_load_pretrained_weights`)**:
    *   自动识别并移除 `state_dict` 中的 Decoder 权重（如 `decoder_blocks`, `mask_token` 等）。
    *   处理 DDP 训练产生的 `module.` 前缀。
    *   **位置编码插值**: 如果微调时的输入长度导致 Patch 数量变化，自动对 `pos_embed` 进行双三次插值（Bicubic Interpolation）以适配新尺寸。

2.  **前向传播流程 (`forward`)**:
    1.  **信号变换**: 输入 `(B, L)` -> CWT 变换 -> 归一化 -> `(B, 3, H, W)`。
    2.  **编码 (Encoder)**:
        *   强制 `mask_ratio = 0.0` (关闭掩码)。
        *   获取 Latent Feature `(B, N+1, Dim)`。
    3.  **特征选择**:
        *   **丢弃 CLS Token**: 代码中选择 `latent[:, 1:, :]`，即只使用 Patch Tokens 作为特征。
    4.  **分类 (Head)**:
        *   **CoT 模式**: 输入所有 Patch Tokens `(B, N, Dim)` 进行 Attention 交互。
        *   **MLP 模式**: 对 Patch Tokens 进行全局平均池化 `(B, Dim)`，然后输入 MLP。

---

## 3. 使用示例

### 3.1 基础使用 (CoT 模式)

```python
import torch
from model_finetune import TF_MAE_Classifier

# 1. 定义模型配置 (需与预训练时的配置一致)
model = TF_MAE_Classifier(
    pretrained_path="checkpoints/mae_pretrain_epoch_100.pth",
    num_classes=10,          # 假设有 10 类
    embed_dim=768,           # 预训练模型的维度
    depth=12,                # 预训练模型的层数
    num_heads=12,
    use_cot=True,            # 启用推理头
    num_reasoning_tokens=16  # 使用 16 个推理令牌
)

# 2. 准备数据 (Batch=4, Length=3000)
x = torch.randn(4, 3000)

# 3. 前向传播
logits = model(x) # Output: (4, 10)
print(f"Logits shape: {logits.shape}")
```

### 3.2 使用 MLP 模式 (轻量化)

```python
model_mlp = TF_MAE_Classifier(
    pretrained_path="checkpoints/mae_pretrain_epoch_100.pth",
    num_classes=2,
    use_cot=False,           # 关闭 CoT
    num_res_blocks=3,        # 使用 3 层残差块
    hidden_dim=256
)

logits = model_mlp(x)
```

---

## 4. 常见问题与注意事项

1.  **显存优化**:
    *   模型初始化时会自动执行 `del self.encoder_model.decoder_blocks` 等操作。加载预训练模型后，显存占用应显著小于预训练阶段。
2.  **输入长度变化**:
    *   如果预训练是 3000 点，微调是 6000 点，模型会自动插值位置编码。但建议微调长度不要偏离预训练长度过大（如 >2倍），否则可能影响 CWT 的频域特征分布。
3.  **CLS Token**:
    *   当前实现中，分类头**不使用** Encoder 输出的 CLS Token (Index 0)，而是完全依赖 Patch Tokens (Index 1:)。这是因为在 MAE 架构中，Patch Tokens 通常包含更丰富的局部纹理信息。
4.  **初始化**:
    *   Encoder 部分加载预训练权重。
    *   Head 部分使用 Xavier Uniform 初始化（CoT 的 `reasoning_tokens` 使用正态分布初始化）。