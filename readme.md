# CWT-MAE-RoPE: Continuous Wavelet Transform Masked Autoencoder

这是一个结合了连续小波变换 (CWT) 和掩码自编码器 (MAE) 的先进时间序列预训练模型。该模型专为处理多通道时间序列信号设计，通过结合时频分析与 Transformer 架构，实现了对信号的高效表征学习。

## 核心特性 (Key Features)

- **内置 CWT 变换**: 使用 Ricker 小波将 1D 信号转换为 2D 时频图 (Scalogram)，无需离线预处理。
- **因子化注意力机制 (Factorized Attention)**: 编码器采用 `TrueFactorizedBlock`，将注意力分解为 **Time Attention** (时间维) 和 **Channel Attention** (通道维)，有效降低计算复杂度并捕捉跨通道相关性。
- **旋转位置编码 (RoPE)**: 引入 Rotary Embedding 提升长序列建模能力。
- **双重重建任务**: 同时重建 **时频图 (Spectrogram)** 和 **原始时域信号 (Time-domain signal)**，确保模型兼顾频域特征和时域波形。
- **Tubelet Masking**: 采用多通道同步掩码策略，防止信息泄漏，特别适合学习通道间的相位关系（如 PTT）。

## 架构概览 (Architecture)

1.  **Input**: 原始时间序列信号 `(Batch, Channels, Length)`。
2.  **Preprocessing**: `cwt_wrap` 执行 CWT 变换及归一化 -> 输出 `(Batch, M, Scales, Length)`。
3.  **Patch Embed**: 将时频图切分为 Patch。
4.  **Encoder**: 
    - 叠加 `TrueFactorizedBlock`。
    - 使用 RoPE 和 Tubelet Masking。
5.  **Decoder**: 
    - 标准 Transformer Block。
    - 仅处理被 Mask 的区域（训练时）或全部区域。
6.  **Heads**:
    - `decoder_pred_spec`: 重建时频图 Patch。
    - `time_pred`: 通过 `time_reducer` 聚合特征后重建原始信号。

## 环境依赖 (Requirements)

- Python 3.8+
- PyTorch 1.10+ (推荐 2.0+ 以支持 `torch.compiler`)

## 快速开始 (Quick Start)

### 1. 初始化模型

```python
import torch
from model import CWT_MAE_RoPE

# 定义模型参数
model = CWT_MAE_RoPE(
    signal_len=3000,        # 输入信号长度
    cwt_scales=64,          # CWT 的尺度数量 (对应图像高度)
    patch_size_time=50,     # 时间维度的 Patch 大小
    patch_size_freq=4,      # 频率维度的 Patch 大小
    embed_dim=768,          # 编码器维度
    depth=12,               # 编码器层数
    num_heads=12,           # 编码器头数
    decoder_embed_dim=512,  # 解码器维度
    decoder_depth=8,        # 解码器层数
    mask_ratio=0.75,        # 掩码比例
    use_diff=False          # 是否使用差分通道增强
)

# 打印模型结构
print(model)
```

### 2. 前向传播 (Forward Pass)

```python
# 创建模拟输入数据 (Batch_Size=2, Channels=1, Length=3000)
# 注意：如果 use_diff=False，输入可以是 (B, L) 或 (B, 1, L)
x = torch.randn(2, 3000) 

# 前向传播
loss, loss_dict, pred_spec, pred_time, imgs, mask = model(x)

print(f"Total Loss: {loss.item()}")
print(f"Spec Loss: {loss_dict['loss_spec'].item()}")
print(f"Time Loss: {loss_dict['loss_time'].item()}")
print(f"Predicted Spectrogram Shape: {pred_spec.shape}")
print(f"Predicted Time Signal Shape: {pred_time.shape}")
```

## 参数说明 (Arguments)

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `signal_len` | int | 3000 | 输入信号的时间步长度 |
| `cwt_scales` | int | 64 | CWT 变换的尺度数（生成图像的高度） |
| `patch_size_time` | int | 50 | Patch 在时间轴上的长度 |
| `patch_size_freq` | int | 4 | Patch 在频率轴上的长度 |
| `embed_dim` | int | 768 | Encoder 的嵌入维度 |
| `depth` | int | 12 | Encoder 的 Transformer 层数 |
| `mask_ratio` | float | 0.75 | 预训练时的掩码比例 |
| `time_loss_weight` | float | 1.0 | 时域重建损失的权重 |
| `use_diff` | bool | False | 是否计算一阶/二阶差分作为额外通道 |

## 损失函数 (Loss Function)

总损失由两部分组成：
```python
Loss = Loss_Spec + time_loss_weight * Loss_Time
```
1.  **Loss_Spec**: 预测的时频图 Patch 与真实 CWT 结果之间的 MSE Loss（仅计算被 Mask 的部分）。
2.  **Loss_Time**: 预测的时域信号与归一化后的原始信号之间的 MSE Loss。


## 下游任务微调 (Downstream Fine-tuning)

本仓库提供了完善的下游分类任务微调支持，封装在 `TF_MAE_Classifier` 类中。支持 **隐式思维链 (Latent Reasoning/CoT)** 和 **ArcFace 度量学习** 等高级特性。

### 1. 微调模型架构

微调模型 (`model_finetune.py`) 基于预训练的 Encoder，并移除 Decoder，接上特定的任务头：

- **Encoder**: 加载预训练权重的 `CWT_MAE_RoPE` (冻结或全参数微调)。
- **Head Options**:
    - **Linear Head**: 标准的线性分类头 (Global Average Pooling -> Linear)。
    - **Latent Reasoning Head (CoT)**: 使用一组可学习的 "Reasoning Tokens" 通过 Cross-Attention 聚合特征，模拟隐式推理过程，提升复杂信号分类性能。
    - **ArcFace Head**: 用于度量学习，增大类间距离，减小类内距离，适合细粒度分类或开集识别。

### 2. 运行微调

使用 `finetune.py` 脚本启动单机多卡 (DDP) 微调。

#### 示例命令

```bash
torchrun --nproc_per_node=4 finetune.py \
    --data_root "./data/downstream_dataset" \
    --split_file "./data/split.json" \
    --pretrained_path "./checkpoints/mae_pretrain.pth" \
    --save_dir "./checkpoints_finetune" \
    --num_classes 2 \
    --batch_size 32 \
    --epochs 50 \
    --lr 1e-4 \
    --use_cot \
    --num_reasoning_tokens 16 \
    --channel_policy "default_5ch"
```

#### 关键参数

| 参数名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `--pretrained_path` | str | 预训练模型权重路径 (.pth) |
| `--use_cot` | bool | 启用 Latent Reasoning Head (Chain-of-Thought) |
| `--num_reasoning_tokens` | int | CoT 模式下的推理 Token 数量 (默认 16) |
| `--use_arcface` | bool | 启用 ArcFace Loss (需配合特定 Loss 使用) |
| `--channel_policy` | str | 数据通道策略: `default_5ch` (全量), `ppg_only`, `ecg_ppg` |
| `--clean_indices_path` | str | (可选) 包含清洗后样本索引的 .npy 文件路径 |

## 许可证 (License)

MIT License