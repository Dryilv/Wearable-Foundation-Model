### 修改后的 `README.md` (适配 Marp)

```markdown
---
marp: true
theme: gaia
paginate: true
header: 'CWT-MAE-RoPE 模型架构与微调指南'
footer: 'Version Preview 0.0.5'
backgroundColor: '#f8f9fa'
---

<!-- 
这是一个 Marp PPT 的封面页。
你可以使用 ![bg left:40%](path/to/your/logo.png) 语法来添加一个背景Logo。
-->

# **CWT-MAE-RoPE**
## 模型架构与微调指南
**Version Preview 0.0.5**

---

## 1. 项目概述 (Project Overview)

本项目实现了一个针对一维时间序列信号的深度学习框架，结合了**连续小波变换 (CWT)** 与**掩码自编码器 (MAE)**，通过 **"预训练-微调"** 的范式，实现对时间序列的高效表征学习。

**核心流程：**
1.  **信号转图像**：利用 CWT 将一维信号转换为三通道时频图。
2.  **预训练 (Pre-training)**：使用带有 RoPE 和张量分解的 Transformer，通过重建被遮蔽的时频图和原始信号来学习通用特征。
3.  **微调 (Fine-tuning)**：加载预训练编码器，连接高级分类头进行具体任务训练。

---

<!-- _class: lead -->
## **第一部分**
## 核心架构与预训练 (`model.py`)

---

### 1.1 信号预处理模块 (CWT Module)

模型不直接处理原始信号，而是处理其时频表征。

*   **多视图生成 (`cwt_wrap`)**:
    *   **输入**: 原始信号 $x$。
    *   **增强**: 计算一阶差分 ($d1$) 和二阶差分 ($d2$)。
    *   **变换**: 对 $[x, d1, d2]$ 分别进行 Ricker 小波变换。
    *   **输出**: `(Batch, 3, Scales, Length)` 的 4D 张量，视为 3 通道图像。

---

### 1.2 基础组件 (Basic Components)

*   **DecomposedPatchEmbed (内存安全版)**:
    *   使用标准的 `nn.Conv2d` 将 CWT 图像切分为 Patch 并映射到嵌入维度。
*   **RoPE (Rotary Positional Embedding)**:
    *   在 Attention 层引入旋转位置编码，增强模型对序列相对位置的感知。
*   **TensorizedLinear (参数高效层)**:
    *   在 Encoder 的 MLP 中，将全连接层分解为 $U \times V$。
    *   通过 `rank_ratio` 控制参数量，降低过拟合风险。

---

### 1.3 预训练模型: `CWT_MAE_RoPE`

这是一个非对称的 **Encoder-Decoder** 架构。

**A. 编码器 (Encoder)**
*   **输入**: 部分可见的 Patch（根据 `mask_ratio` 随机采样）。
*   **结构**: 堆叠多个 `TensorizedBlock`，结合可学习位置编码与 RoPE。

**B. 解码器 (Decoder)**
*   **输入**: 编码器的潜在向量 + 可学习的 `[MASK]` token。
*   **结构**: 轻量级 Transformer Blocks。
*   **任务**: 仅用于预训练阶段的重建任务。

---

### 1.3 预训练模型 (续): 双重损失函数

模型同时优化频域和时域的重建质量：
$$ Loss_{total} = Loss_{spec} + \lambda \cdot Loss_{time} $$

1.  **谱重建 (`forward_loss_spec`)**: 计算被 Mask 掉的时频图 Patch 的 MSE Loss。
2.  **时域重建 (`forward_loss_time`)**: 通过 `time_reducer` 聚合特征，预测原始一维信号，计算 MSE Loss。

---

<!-- _class: lead -->
## **第二部分**
## 下游任务微调 (`model_finetune.py`)

---

### 2.1 分类器封装: `TF_MAE_Classifier`

该类负责将预训练模型适配到具体的分类任务。

*   **自动瘦身**: 初始化时自动删除预训练模型的 Decoder 部分。
*   **位置编码插值**: 自动对 `pos_embed` 进行双三次插值以适应不同长度的信号。
*   **特征提取**: 强制关闭 Mask (`mask_ratio=0`)，并丢弃 CLS Token。

---

### 2.2 高级分类头 (Heads)

模型提供了两种分类头模式，通过 `use_cot` 参数切换：

#### A. 隐式思维链头 (Latent Reasoning Head / CoT)
*   **机制**:
    1.  初始化一组可学习的 **推理令牌 (Reasoning Tokens)**。
    2.  通过 **Cross-Attention** 查询 Encoder 输出的全局特征。
    3.  通过 **Self-Attention** 整合推理结果。
    4.  最终通过分类器输出 Logits。

#### B. 残差 MLP 头 (Residual MLP)
*   **机制**: 全局平均池化 -> N x [Residual Block] -> Output。

---

<!-- _class: lead -->
## **第三部分**
## API 参数说明

---

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

---

### 3.2 微调模型 (`TF_MAE_Classifier`)

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `pretrained_path` | Required | 预训练权重路径 |
| `num_classes` | Required | 类别数量 |
| `use_cot` | True | 是否使用推理头 (False 则使用 MLP) |
| `num_reasoning_tokens`| 8 | CoT 模式下的 Token 数量 |
| `num_res_blocks` | 2 | MLP 模式下的残差块数量 |

---

<!-- _class: lead -->
## **第四部分**
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

# 3. 前向传播
loss, _, _, _ = model(x)
print(f"Pre-training Loss: {loss.item()}")
```

---

### 4.2 阶段二：微调 (Fine-tuning)

```python
import torch
from model_finetune import TF_MAE_Classifier

# 1. 初始化分类器 (加载预训练权重)
classifier = TF_MAE_Classifier(
    pretrained_path="checkpoints/mae.pth",
    num_classes=10,
    embed_dim=768,       # 需与预训练一致
    use_cot=True,        # 启用思维链推理头
)

# 2. 输入数据
x_val = torch.randn(16, 3000)

# 3. 推理/训练
logits = classifier(x_val) # Output: (16, 10)
print(f"Logits shape: {logits.shape}")```

---

<!-- _class: lead -->
## **第五部分**
## 关键技术细节

---

### 5.1 关键技术细节 (1/2)

1.  **显存优化**:
    *   `DecomposedPatchEmbed` 修正为标准卷积，解决中间张量过大问题。
    *   微调时，`TF_MAE_Classifier` 会主动删除 Decoder，降低显存占用。

2.  **输入长度适配**:
    *   预训练和微调的信号长度可以不同，模型会自动对位置编码进行插值。

---

### 5.2 关键技术细节 (2/2)

3.  **特征选择策略**:
    *   在微调阶段，分类头**不使用** CLS Token，而是使用所有 Patch Tokens (`latent[:, 1:, :]`)，以保留更丰富的局部信息。

4.  **RoPE 缓存**:
    *   `RotaryEmbedding` 具有缓存机制，能自动适应变长序列推理，无需人工干预。

---

<!-- _class: lead -->
## **第六部分**
## 相关工作与参考文献

---

### 6.1 相关工作 (1/4): MAE & 时频分析

*   **MAE**: (He, et al. 2022) 证明了通过重建被遮蔽的图像 Patch 可以学习到强大的视觉表征。
*   **TFMAE**: (Zheng, et al. 2023) 提出了在时域和频域同时进行掩码重建的思想。
*   **SpectralMAE**: (Cao, et al. 2023) 探索了在光谱维度进行掩码重建的有效性。
*   **Ti-MAE**: (Li, et al. 2023) 证明了在时间序列上，生成式任务优于对比学习。

---

### 6.2 相关工作 (2/4): 旋转位置编码 (RoPE)

*   **RoFormer**: (Su, et al. 2021) 首次提出 RoPE，将绝对位置编码转化为相对位置依赖。
*   **RoPE in Vision**: (Heo, et al. 2024) 验证了 RoPE 在二维视觉任务中的有效性，特别是在处理不同分辨率输入时的外推能力。
*   **CyRoPE**: (Rao, et al. 2024) 针对多通道肌电信号提出的柱状 RoPE，证明了其在复杂时序几何结构中的适用性。

---

### 6.3 相关工作 (3/4): 参数高效微调

*   **Tensorized Transformer**: (Ma, et al. 2019) 利用张量分解来压缩 Transformer 的权重矩阵。
*   **Low-Rank Adaptation (LoRA)**: (Hu, et al. 2021) `TensorizedLinear` 的设计思想与 LoRA 异曲同工，即通过低秩矩阵来近似全秩权重更新。

---

### 6.4 相关工作 (4/4): 时间序列推理

*   **Time-Series Reasoning**: (Chow, et al. 2024) 探索如何让大模型理解时间序列的物理特性并生成可解释的推理步骤。
*   **Latent Reasoning Skills (LaRS)**: (Yang, et al. 2024) 提出在潜在空间中学习“推理技能”向量，与本项目的 `reasoning_tokens` 设计思路高度一致。
*   **Reinforcement Learning for CoT**: (Parker, et al. 2025) 探索利用强化学习激发模型在时间序列分析任务中的多步推理能力。
```

### 主要修改说明：

1.  **添加 Marp 配置头**：在文件最顶部添加了 `---` 包裹的 YAML 配置，定义了主题 (`gaia`)、页码 (`paginate: true`)、页眉页脚等。
2.  **明确分页**：在每个你希望另起一页的地方，都插入了一个单独的 `---`。我根据你的内容结构，将每个主要部分、次要部分、代码示例等都分成了独立的页面，避免单页内容过多。
3.  **优化标题页**：为封面页和每个主要部分的标题页添加了 `<!-- _class: lead -->` 指令，这会让这些页面的标题居中并放大，更具视觉冲击力。
4.  **内容拆分**：对于内容较长的部分（如“关键技术细节”和“相关工作”），我将其拆分成了多个页面，并在标题处用 (1/2) 这样的方式标注，方便阅读。