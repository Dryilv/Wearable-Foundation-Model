# CWT-MAE: 基于连续小波变换的掩码自编码器

基于连续小波变换（CWT）和掩码自编码器（MAE）架构的信号分类预训练框架，专为医疗信号（ECG、PPG 等）的多标签分类任务设计。

## 项目特性

- **CWT 特征提取**: 将时域信号转换为时频图，捕获信号的频率特征
- **MAE 预训练**: 采用掩码自编码器策略进行自监督预训练
- **RoPE 位置编码**: 使用旋转位置编码（Rotary Position Embedding）增强位置感知
- **CoT 分类头**: 集成隐式思维链（Chain-of-Thought）推理模块，提升分类性能
- **多通道支持**: 支持变长通道输入，适用于多模态信号融合
- **多标签分类**: 支持 7 类心血管疾病的多标签分类任务

## 模型架构

### 预训练阶段
```
输入信号 → CWT 变换 → Patch 分割 → Masking → ViT Encoder → Decoder → 重建
```

### 微调阶段
```
输入信号 → CWT 变换 → Patch 分割 → ViT Encoder → CoT Head → 分类输出
```

### 关键参数
| 参数 | 预训练 | 微调 |
|------|--------|------|
| Embed Dim | 768 | 768 |
| Depth | 12 | 12 |
| Num Heads | 12 | 12 |
| Patch Size (Time) | 25 | 25 |
| Patch Size (Freq) | 8 | 8 |
| CWT Scales | 64 | 64 |
| Mask Ratio | 0.75 | 0.0 |
| CoT Tokens | - | 16 |

## 安装

### 环境要求
- Python 3.8+
- CUDA 11.0+ (推荐)
- GPU 显存: 预训练建议 80GB，微调建议 16GB+

### 安装依赖
```bash
pip install -r requirements.txt
```

### 主要依赖
- PyTorch
- NumPy, SciPy (CWT 计算)
- scikit-learn (评估指标)
- PyYAML (配置管理)
- tqdm, matplotlib (可视化)

## 使用方法

### 1. 数据准备

数据格式要求：
- 数据文件: `.pkl` 格式，包含信号和标签
- 划分文件: `train_test_split.json`，包含训练/测试索引

数据结构示例：
```python
# .pkl 文件内容
{
    'signal': np.ndarray,  # shape: (M, L), M 为通道数，L 为信号长度
    'label': {
        '高血压': 0/1,
        '高血糖': 0/1,
        '高血脂': 0/1,
        '冠心病': 0/1,
        '心律失常（房颤、频发早搏等）': 0/1,
        '糖尿病': 0/1,
        '颈动脉斑块': 0/1
    }
}
```

### 2. 预训练

修改 `config.yaml` 配置文件：
```yaml
train:
  epochs: 200
  batch_size: 512
  base_lr: 1.5e-4
  
data:
  index_path: "train_index.json"
  signal_len: 1000
  
model:
  mask_ratio: 0.75
  embed_dim: 768
  depth: 12
```

启动预训练：
```bash
python train.py --config config.yaml
```

支持分布式训练：
```bash
torchrun --nproc_per_node=4 train.py --config config.yaml
```

### 3. 微调

修改 `finetune_config.yaml` 配置文件：
```yaml
train:
  epochs: 30
  batch_size: 128
  base_lr: 5.0e-5
  
data:
  data_root: "/path/to/data"
  split_file: "/path/to/split.json"
  num_classes: 7
  
model:
  pretrained_path: "./checkpoint/checkpoint_last.pth"
  use_cot: true
  num_reasoning_tokens: 16
```

启动微调：
```bash
python finetune.py --config finetune_config.yaml
```

### 4. 推理

使用训练好的模型进行推理：
```bash
python inference.py --model_path ./finetune/best_model.pth --input signal.pkl
```

## 配置说明

### 预训练配置 (`config.yaml`)

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `epochs` | 训练轮次 | 200 |
| `batch_size` | 批次大小 | 512 |
| `base_lr` | 基础学习率 | 1.5e-4 |
| `mask_ratio` | 掩码比例 | 0.75 |
| `use_diff` | 使用差分信号 | true |
| `time_loss_weight` | 时域损失权重 | 1.0 |

### 微调配置 (`finetune_config.yaml`)

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `epochs` | 训练轮次 | 30 |
| `batch_size` | 批次大小 | 128 |
| `base_lr` | 基础学习率 | 5.0e-5 |
| `use_cot` | 使用 CoT 模块 | true |
| `num_reasoning_tokens` | CoT token 数量 | 16 |
| `use_focal_loss` | 使用 Focal Loss | false |
| `pos_weight` | 正样本权重 | auto |

## 输出文件

### 预训练输出
- `checkpoint/checkpoint_last.pth`: 最新检查点
- `checkpoint/checkpoint_best.pth`: 最佳检查点
- `checkpoint/train.log`: 训练日志

### 微调输出
- `finetune/best_model.pth`: 最佳模型权重
- `finetune/best_model_cpu.onnx`: ONNX 导出模型
- `finetune/best_threshold.json`: 最佳阈值配置
- `finetune/finetune.log`: 微调日志

## 评估指标

- **Accuracy**: 精确匹配准确率
- **Macro AUC**: 多标签宏平均 AUC
- **Macro F0.5**: F-beta 分数 (beta=0.5)
- **Precision/Recall**: 宏平均精确率/召回率

## 可视化工具

- `visualize_cwt.py`: CWT 时频图可视化
- `visualize_reconstruction.py`: 重建效果可视化
- `visualize_inference.py`: 推理结果可视化

## 数据清洗工具

- `clean_pretrain_nan.py`: 预训练数据 NaN 清洗
- `clean_downstream_nan.py`: 下游任务数据清洗

## 项目结构

```
CWT_MAE_v3/
├── train.py              # 预训练脚本
├── finetune.py           # 微调脚本
├── inference.py          # 推理脚本
├── model.py              # MAE 模型定义
├── model_finetune.py     # 分类模型定义
├── dataset.py            # 预训练数据集
├── dataset_finetune.py   # 微调数据集
├── utils.py              # 工具函数
├── utils_metrics.py      # 评估指标
├── config.yaml           # 预训练配置
├── finetune_config.yaml  # 微调配置
├── requirements.txt      # 依赖列表
└── README.md             # 项目说明
```

## 注意事项

1. **显存优化**: 预训练使用大 batch size，建议使用梯度累积或降低 batch size
2. **学习率**: 微调阶段建议使用较小的学习率（5e-5）
3. **数据清洗**: 建议在训练前检查数据中的 NaN 值
4. **阈值调整**: 多标签分类建议根据验证集调整阈值

## License

MIT License