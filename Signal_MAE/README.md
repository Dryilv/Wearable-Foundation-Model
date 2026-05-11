# Signal-MAE: 基于原始信号 Token 划分的 MAE 模型

## 项目概况

Signal-MAE 是一种基于原始信号直接划分 token 的 Masked Autoencoder 模型,用于 ECG/PPG 生理信号处理。

### 与 CWT-MAE v3 的区别

| 特性 | CWT-MAE v3 | Signal-MAE |
|------|------------|------------|
| **输入表示** | CWT 时频图 + 原始信号融合 | 仅原始 1D 信号 |
| **Token 划分** | 在 2D 时频图上按 (freq, time) 划分 | 在 1D 信号上按时间窗口划分 |
| **重建目标** | 双重域: CWT 谱图 + 时域信号 | 仅原始时域信号 |
| **计算复杂度** | 较高 (需 CWT 变换) | 较低 (直接处理原始信号) |
| **适用场景** | 需要频域信息的任务 | 端到端时域建模 |

### 核心设计理念
- **直接 Token 划分**: 跳过 CWT 变换,直接在原始信号上按固定时间窗口划分 token
- **简化架构**: 去除 CWT 相关代码,保留 RoPE、masking、decoder 等核心组件
- **保持兼容性**: 数据集、训练脚本、配置文件结构与 CWT_MAE_v3 完全一致

## 技术细节

### 模型架构
- **Patch Embedding**: 1D Convolution (kernel=patch_size, stride=patch_size)
- **位置编码**: RoPE (Rotary Position Embedding)
- **Masking Strategy**: 随机 token masking (默认 75%)
- **Encoder**: 12 层 Transformer, 768 维, 12 heads
- **Decoder**: 8 层 Transformer, 512 维, 16 heads
- **通道类型**: 通过 `channel_type_embed` 区分 ECG (id=0) 和 PPG (id=1)

### 参数配置
```yaml
signal_len: 3000
patch_size: 50          # 每个 token 覆盖 50 个时间点
num_patches: 60         # 3000 / 50
embed_dim: 768
depth: 12
num_heads: 12
mask_ratio: 0.75
```

## 文件结构

```
Signal_MAE/
├── model.py              # 核心模型定义
├── config.yaml           # 预训练配置
├── finetune_config.yaml  # 下游任务微调配置
├── train.py              # 训练脚本 (与 CWT_MAE_v3 一致)
├── dataset.py            # 数据集加载 (与 CWT_MAE_v3 一致)
├── finetune.py           # 微调脚本 (待实现)
├── utils.py              # 工具函数 (待实现)
└── README.md             # 本文档
```

## 快速开始

### 预训练
```bash
python train.py --config config.yaml
```

### 微调
```bash
python finetune.py --config finetune_config.yaml
```

## 与 CWT_MAE_v3 的兼容性

- ✅ 数据集格式完全兼容 (使用相同的 `dataset.py`)
- ✅ 训练流程一致 (使用相同的 `train.py`)
- ✅ 配置文件结构相同
- ⚠️ 模型权重不兼容 (架构不同)

## 注意事项

1. **信号长度**: 必须是 `patch_size` 的整数倍,否则会自动裁剪
2. **单通道模式**: 当前实现为单通道 (M=1),但架构支持未来多通道扩展
3. **归一化**: 模型内部自动进行 Z-score 归一化,无需外部预处理
