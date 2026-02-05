# 消融实验指南：输入表征与频域约束 (Ablation Study)

本目录下的 `CWT_MAE_v3` 模型非常适合作为 v1/v2 版本的消融实验对照组。核心探究问题是：**“显式的小波变换（CWT）作为输入是否是必须的？Transformer 能否直接从原始波形中学习到同等有效的特征？”**

## 实验设计

通过调整 `config.yaml` 中的参数，你可以构建三个层次的对比实验，形成一个完整的论文故事线：

### 实验 A: 纯时域基线 (Pure Time-Domain Baseline)
*   **配置**: v3 版本
*   **参数**: `cwt_loss_weight: 0.0`
*   **输入**: Raw Signal (1D)
*   **监督信号**: 仅重建原始波形 (MSE Loss)
*   **目的**: 验证如果不利用任何频域先验知识，仅靠 Transformer 自身的建模能力，效果如何。这通常作为 **Lower Bound (下界)**。

### 实验 B: 双重目标优化 (Proposed: Dual-Objective)
*   **配置**: v3 版本 (当前默认)
*   **参数**: `cwt_loss_weight: 1.0` (或 0.1 - 10.0)
*   **输入**: Raw Signal (1D)
*   **监督信号**: 重建波形 + 重建波形的CWT谱 (Time MSE + Freq MSE)
*   **目的**: 验证**“输入虽不变换，但Loss强制约束频域一致性”**是否能达到与 v1 相当的效果。
*   **优势论点**: 如果实验 B 效果接近或优于 v1，则证明**显式的 CWT 输入是不必要的**，从而节省了 60% 的显存和大量的预处理计算。

### 实验 C: 时频域输入 (Counterpart: Time-Frequency Input)
*   **配置**: v1 / v2 版本
*   **输入**: CWT Spectrogram (2D)
*   **监督信号**: 重建 CWT 谱
*   **目的**: 作为 **Strong Baseline**。验证显式引入归纳偏置（Inductive Bias）的收益。

## 预期结论与分析

| 实验组 | 显存占用 | 计算速度 | 预期精度 | 结论解释 |
| :--- | :--- | :--- | :--- | :--- |
| **A (Raw Only)** | 低 | 快 | 低 | Transformer 难以在无引导的情况下从嘈杂的生理信号中捕捉高频微结构。 |
| **B (Dual-Obj)** | **低** | **快** | **高** | **最佳权衡 (Trade-off)**。CWT Loss 充当了“软约束”，引导模型关注频域特征，同时保留了 Raw Input 的高效性。 |
| **C (CWT Input)**| 高 | 慢 | 高 | 显式特征提取确实有效，但代价昂贵（显存瓶颈、计算量大）。 |

## 如何运行消融实验

### 1. 运行实验 A (无 CWT)
修改 `config.yaml`:
```yaml
model:
  cwt_loss_weight: 0.0
```
运行训练: `python train.py --config config.yaml` (建议重命名 save_dir 为 `checkpoint_ablation_raw_only`)

### 2. 运行实验 B (有 CWT Loss)
修改 `config.yaml`:
```yaml
model:
  cwt_loss_weight: 1.0
```
运行训练: `python train.py --config config.yaml` (建议重命名 save_dir 为 `checkpoint_ablation_dual_obj`)

### 3. 对比评估
使用 `inference.py` 在相同的测试集上评估两个模型，记录准确率 (Accuracy) 或 F1-Score，并对比显存占用峰值。
