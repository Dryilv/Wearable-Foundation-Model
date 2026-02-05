# CWT-MAE v3 (Pixel-based / Point-based)

## 简介
`CWT_MAE_v3` 是 CWT-MAE 系列的最新演进版本，实现了从 **Patch-based (v1/v2)** 到 **Pixel/Point-based (v3)** 的核心架构迁移。

在此版本中，模型直接处理 **原始 1D 生理信号 (Raw Signal Points)**，不再将 CWT 时频图作为输入。这种设计显著降低了显存占用，并允许模型以极细的粒度（如 4个点甚至1个点）对信号进行建模，同时通过 **Dual-Objective Loss** 保持了对频域特征的敏感性。

## 核心变更

### 1. 架构迁移 (Architecture Shift)
*   **Input**: `(B, M, 3000)` Raw Signal -> `(B, M, 3, 64, 50)` CWT Spectrogram (v1/v2)
    *   **v3 Input**: 直接使用 `(B, M, 3000)` 原始信号。
*   **Embedding**:
    *   **v1/v2**: 2D Conv Patch Embedding (处理时频图)。
    *   **v3**: **1D Point Embedding** (使用 `Conv1d`，`patch_size` 可设为 4 甚至 1)。
*   **Reconstruction**:
    *   **v1/v2**: 重建 CWT 时频图的 Patch。
    *   **v3**: 直接重建原始信号的波形点。

### 2. 双重损失函数 (Dual-Objective Loss)
为了保留 CWT-MAE 的核心优势（时频分析），v3 引入了双重损失：
1.  **Time Domain Loss**: `MSE(Pred_Signal, Raw_Signal)` - 确保波形准确。
2.  **Frequency Domain Loss**: `MSE(CWT(Pred_Signal), CWT(Raw_Signal))` - 确保频域特征（如高频噪声、低频基线）一致。
    *   注意：CWT 变换仅在 Loss 计算阶段进行，不参与前向推理，大幅提升了训练速度。

### 3. 性能优势
*   **显存占用**: 降低约 60%。因为 Sequence Length 从 960 (v2) / 800 (v1) 变为 750 (v3 @ patch_size=4)，且无需存储巨大的 2D 特征图。
*   **Batch Size**: 在相同硬件下，Batch Size 可翻倍 (e.g., 128 -> 256)。
*   **精度**: Pixel 级建模通常能捕捉到更微小的形态学异常（如 ECG 的 J 点抬高）。

## 配置文件 (`config.yaml`)

关键参数说明：

```yaml
model:
  # Patch Size: 控制建模粒度
  # 1: 纯 Pixel 级别 (Sequence Length = 3000)，计算量大但最精细
  # 4: 推荐值 (Sequence Length = 750)，效率与精度的平衡
  patch_size: 4           
  
  # CWT Loss 权重
  cwt_loss_weight: 1.0    
```

## 使用指南

### 训练
```bash
python train.py --config config.yaml
```

### 推理
```bash
python inference.py --data_root /path/to/data --checkpoint checkpoint_pixel_mae_v3/checkpoint_last.pth --num_classes 2
```

## 迁移指南 (v1 -> v3)
如果您有 v1 的预训练权重，**无法直接加载** 到 v3，因为 Embedding 层维度完全不同（2D vs 1D）。建议重新预训练。
