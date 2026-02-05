# CWT-MAE v4: Contrastive Learning Version

## 简介
`CWT_MAE_v4` 是基于对比学习（Contrastive Learning）的预训练版本。与 v1/v2/v3 的生成式任务（MAE 重建）不同，v4 侧重于**判别式特征学习**，旨在解决 MAE 在特征区分性（Discriminative Power）上可能存在的不足。

## 核心理念
*   **Motivation**: MAE 虽然能很好地重建信号，但有时会过于关注低级细节（如噪声），而忽略了高层的语义特征（如波形类别、病理模式）。
*   **Method**: 采用 **SimCLR** 风格的对比学习框架。对同一段生理信号进行两次强增强（Two-View Augmentation），强制模型拉近同一信号的不同视图（Positive Pairs），推开不同信号的视图（Negative Pairs）。

## 核心变更

### 1. 模型架构 (`model.py`)
*   **Encoder Only**: 移除了 Decoder 和 Masking 机制。
*   **Projection Head**: 在 Encoder 后增加了一个 MLP 投影头（Linear-LN-GELU-Linear），将 768 维特征映射到 128 维空间进行对比 Loss 计算。
*   **NT-Xent Loss**: 实现了标准的对比损失函数。

### 2. 数据增强 (`dataset.py`)
实现了针对 1D 生理信号的 **Strong Augmentation** 策略：
*   **Random Resized Crop (Resize-Back)**: 模拟心率变化。
*   **Gaussian Noise**: 模拟传感器噪声。
*   **Channel Masking**: 模拟导联脱落。
*   **Amplitude Scaling**: 模拟信号增益变化。
*   **Time Flip**: 模拟时序反转。

### 3. 训练策略 (`config.yaml`)
*   **Batch Size**: 增加到 512（对比学习依赖大 Batch 提供足够的负样本）。
*   **Epochs**: 增加到 200（对比学习收敛较慢）。
*   **Learning Rate**: 适当调大。

## 使用指南

### 训练
```bash
cd c:\Users\Administrator\Desktop\model\CWT_MAE_v4
python train.py --config config.yaml
```

### 迁移到下游任务
预训练完成后，丢弃 `projection_head`，直接使用 Encoder 输出的特征（CLS Token 或 Average Pooling）训练线性分类器或微调整个网络。
