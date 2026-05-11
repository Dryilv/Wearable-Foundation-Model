# Signal-MAE 项目总结

## 已完成的工作

### 1. 核心模型实现 (`model.py`)

**Signal_MAE_RoPE** - 基于原始信号直接划分 token 的 MAE 模型

#### 关键组件:
- **SignalPatchEmbed**: 1D Convolution 将原始信号划分为 token
- **RotaryEmbedding**: RoPE 位置编码
- **Block**: 标准 Transformer Block
- **Signal_MAE_RoPE**: 完整的 MAE 架构

#### 与 CWT_MAE_v3 的主要区别:

| 组件 | CWT_MAE_v3 | Signal_MAE |
|------|------------|------------|
| **输入处理** | CWT 变换 → 2D 时频图 | 直接 1D 信号 |
| **Patch Embedding** | DecomposedPatchEmbed (2D Conv) | SignalPatchEmbed (1D Conv) |
| **Token 数量** | H_grid × W_grid (时频二维) | signal_len // patch_size (一维) |
| **重建目标** | CWT 谱图 + 时域信号 | 仅时域信号 |
| **损失函数** | forward_loss_spec + forward_loss_time | forward_loss (统一) |
| **预处理** | cwt_wrap (复杂) | prepare_tokens (简单归一化) |

### 2. 配置文件

#### `config.yaml` (预训练)
```yaml
signal_len: 1000
patch_size: 50          # 20 个 token (1000/50)
embed_dim: 768
depth: 12
num_heads: 12
mask_ratio: 0.75
```

#### `finetune_config.yaml` (下游任务)
- 支持 7 分类任务
- 预训练权重加载
- CoT (Chain-of-Thought) 模块

### 3. 训练脚本 (`train.py`)

已适配 Signal_MAE:
- ✅ 导入 `Signal_MAE_RoPE`
- ✅ 配置参数映射更新
- ✅ 日志信息更新
- ✅ 保持分布式训练支持
- ✅ 梯度累积、AMP 等优化

### 4. 支持文件

从 CWT_MAE_v3 复制 (保持不变):
- `dataset.py` - 数据集加载 (完全兼容)
- `finetune.py` - 微调脚本
- `model_finetune.py` - 微调模型定义
- `dataset_finetune.py` - 微调数据集
- `utils.py` - 工具函数
- `utils_metrics.py` - 实验跟踪

### 5. 测试脚本 (`test_model.py`)

验证模型功能:
- 前向传播
- Masking 逻辑
- 损失计算
- 形状验证

## 项目结构

```
Signal_MAE/
├── model.py                 # ✅ 核心模型 (新实现)
├── config.yaml              # ✅ 预训练配置 (新实现)
├── finetune_config.yaml     # ✅ 微调配置 (新实现)
├── train.py                 # ✅ 训练脚本 (已适配)
├── finetune.py              # ✅ 微调脚本 (复制)
├── dataset.py               # ✅ 数据集 (复制)
├── model_finetune.py        # ✅ 微调模型 (复制)
├── dataset_finetune.py      # ✅ 微调数据集 (复制)
├── utils.py                 # ✅ 工具函数 (复制)
├── utils_metrics.py         # ✅ 指标跟踪 (复制)
├── test_model.py            # ✅ 测试脚本 (新实现)
├── requirements.txt         # ✅ 依赖 (新实现)
└── README.md                # ✅ 文档 (新实现)
```

## 使用方式

### 预训练
```bash
cd Signal_MAE
python train.py --config config.yaml
```

### 微调
```bash
python finetune.py --config finetune_config.yaml
```

### 测试
```bash
python test_model.py
```

## 技术亮点

1. **简化架构**: 去除 CWT 变换,降低计算复杂度
2. **直接建模**: 在原始信号上学习表征,无需频域先验
3. **完全兼容**: 数据集、训练流程与 CWT_MAE_v3 一致
4. **RoPE 位置编码**: 保持相对位置信息
5. **通道类型感知**: 通过 embedding 区分 ECG/PPG

## 注意事项

1. **信号长度**: 必须是 `patch_size` 的整数倍
2. **本地环境**: 当前本地 PyTorch 环境可能有问题,建议在远程服务器运行
3. **远程路径**: 需同步到 `/home/bml/.storage/mnt/v-044d0fb740b04ad3/org/WFM/vit16trans/Signal_MAE`

## 下一步

- [ ] 在远程服务器运行预训练
- [ ] 对比 CWT_MAE_v3 和 Signal_MAE 的性能
- [ ] 实现可视化脚本 (参考 CWT_MAE_v3 的 visualize_*.py)
- [ ] 添加推理脚本 (inference.py)
