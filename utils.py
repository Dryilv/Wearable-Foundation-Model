import torch
import matplotlib.pyplot as plt
import os
import numpy as np
import logging
import torch.distributed as dist
from collections import deque
import datetime

# -------------------------------------------------------------------
# Logging & Metrics (保持不变)
# -------------------------------------------------------------------
class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a window or the global series average."""
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=max(self.deque),
            value=self.deque[-1]
        )

def format_time(seconds):
    time_delta = datetime.timedelta(seconds=int(seconds))
    return str(time_delta)

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def setup_logger(save_dir):
    logger = logging.getLogger("TF-MAE")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        return logger
    
    if is_main_process():
        handler = logging.FileHandler(os.path.join(save_dir, "train.log"))
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)
    else:
        logger.addHandler(logging.NullHandler())
    return logger

def init_distributed_mode():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(gpu)
        dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
        dist.barrier()
        print(f"| distributed init (rank {rank}): success")
        return gpu, rank, world_size
    else:
        print('Not using distributed mode')
        return 0, 0, 1
    
    
# In utils.py

import torch

def get_layer_wise_lr(model, base_lr, layer_decay):
    """
    【修正版】
    为双编码器模型 (DualEncoder_Classifier) 设置分层学习率。
    - 分别为 ppg_encoder 和 ecg_encoder 的 Transformer blocks 设置递减的学习率。
    - 为两个编码器的其他部分 (patch_embed, cls_token, pos_embed) 设置基础学习率 * decay^num_layers。
    - 为分类头 (head) 和其他参数设置基础学习率。
    """
    param_groups = {}

    # --- 辅助函数，用于处理单个编码器 ---
    def process_encoder(encoder, prefix, num_layers):
        # 遍历编码器的所有命名参数
        for name, param in encoder.named_parameters():
            if not param.requires_grad:
                continue

            # 为 Transformer blocks 设置递减学习率
            if name.startswith('blocks.'):
                # 从 name 中解析出层号, e.g., 'blocks.5.norm1.weight' -> 5
                try:
                    layer_id = int(name.split('.')[1])
                    lr_scale = layer_decay ** (num_layers - 1 - layer_id)
                    group_name = f'{prefix}_layer_{layer_id}'
                    
                    if group_name not in param_groups:
                        param_groups[group_name] = {'params': [], 'lr': base_lr * lr_scale}
                    param_groups[group_name]['params'].append(param)
                except (ValueError, IndexError):
                    # 如果解析失败，则使用默认学习率
                    if 'default' not in param_groups:
                        param_groups['default'] = {'params': [], 'lr': base_lr}
                    param_groups['default']['params'].append(param)

            # 为 patch_embed, cls_token, pos_embed 设置一个固定的、较低的学习率
            elif name.startswith(('patch_embed', 'cls_token', 'pos_embed')):
                lr_scale = layer_decay ** num_layers
                group_name = f'{prefix}_embed'
                
                if group_name not in param_groups:
                    param_groups[group_name] = {'params': [], 'lr': base_lr * lr_scale}
                param_groups[group_name]['params'].append(param)
            
            # 其他参数 (如 norm) 使用默认学习率
            else:
                if 'default' not in param_groups:
                    param_groups['default'] = {'params': [], 'lr': base_lr}
                param_groups['default']['params'].append(param)

    # --- 主逻辑 ---
    
    # 1. 处理 PPG Encoder
    if hasattr(model, 'ppg_encoder'):
        print("Applying layer-wise learning rate decay to ppg_encoder.")
        num_layers_ppg = len(model.ppg_encoder.blocks)
        process_encoder(model.ppg_encoder, 'ppg', num_layers_ppg)
    
    # 2. 处理 ECG Encoder
    if hasattr(model, 'ecg_encoder'):
        print("Applying layer-wise learning rate decay to ecg_encoder.")
        num_layers_ecg = len(model.ecg_encoder.blocks)
        process_encoder(model.ecg_encoder, 'ecg', num_layers_ecg)

    # 3. 处理分类头 (Head) - 通常使用基础学习率
    if hasattr(model, 'head'):
        print("Applying base learning rate to the classification head.")
        if 'head' not in param_groups:
            param_groups['head'] = {'params': [], 'lr': base_lr}
        param_groups['head']['params'].extend(model.head.parameters())

    # 4. 检查是否有任何参数被遗漏 (安全措施)
    all_params = set(model.parameters())
    grouped_params = set()
    for group in param_groups.values():
        grouped_params.update(group['params'])
    
    ungrouped_params = all_params - grouped_params
    if ungrouped_params:
        print(f"Warning: {len(ungrouped_params)} parameters were not assigned to any group. Adding them to default group.")
        if 'default' not in param_groups:
            param_groups['default'] = {'params': [], 'lr': base_lr}
        param_groups['default']['params'].extend(list(ungrouped_params))

    return list(param_groups.values())
# Visualization (已修改：支持反归一化)
# -------------------------------------------------------------------
def save_reconstruction_images(model, x_time, epoch, save_dir, patch_size):
    """
    Time-Only Visualization with Denormalization
    """
    model.eval()
    vis_dir = os.path.join(save_dir, "vis_results")
    os.makedirs(vis_dir, exist_ok=True)

    with torch.no_grad():
        # 1. 获取模型实例 (处理 DDP/Compile 包装)
        real_model = model.module if hasattr(model, 'module') else model
        if hasattr(real_model, '_orig_mod'):
            real_model = real_model._orig_mod
        
        # 2. 计算统计量并归一化 (用于模型输入)
        # x_time 是原始数据 [B, 1, L]
        mean = x_time.mean(dim=-1, keepdim=True)
        std = x_time.std(dim=-1, keepdim=True)
        x_norm = (x_time - mean) / (std + 1e-6)

        # 3. 模型推理 (输入归一化数据)
        latent, mask, ids_restore = real_model.forward_encoder(x_norm)
        pred_norm = real_model.forward_decoder(latent, ids_restore)
        
        # 4. 数据后处理 (取第一个样本 idx=0)
        idx = 0
        
        # 获取该样本的 Mean 和 Std (用于反归一化)
        sample_mean = mean[idx, 0, 0].cpu().numpy()
        sample_std = std[idx, 0, 0].cpu().numpy()
        
        # A. 获取原始信号 (Raw Signal)
        orig_signal_raw = x_time[idx, 0].cpu().numpy()
        
        # B. 获取重建信号 (Normalized -> Denormalized)
        rec_signal_norm = pred_norm[idx].flatten().cpu().numpy()
        # *** 关键步骤：反归一化 ***
        rec_signal_denorm = (rec_signal_norm * sample_std) + sample_mean
        
        # C. 获取 Mask
        mask_np = mask[idx].cpu().numpy()

        # 5. 对齐长度 (处理 Patch 整除问题)
        L = len(orig_signal_raw)
        
        # 扩展 Mask
        mask_expanded = np.repeat(mask_np, patch_size)
        mask_expanded = mask_expanded[:L]
        
        # 截断重建信号
        rec_signal_denorm = rec_signal_denorm[:L]

        # 6. 绘图
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        plt.suptitle(f"Epoch {epoch} Reconstruction (Original Scale)", fontsize=16)

        # --- Subplot 1: 原始信号 (Raw) ---
        axs[0].plot(orig_signal_raw, 'k', alpha=0.8, lw=1)
        axs[0].set_title(f"Original Raw Signal\n(Mean={sample_mean:.2f}, Std={sample_std:.2f})")
        axs[0].grid(True, alpha=0.3)

        # --- Subplot 2: Masked Input (Raw Scale) ---
        masked_view = orig_signal_raw.copy()
        masked_view[mask_expanded == 1] = np.nan # 被 Mask 的部分设为 NaN
        
        axs[1].plot(orig_signal_raw, 'lightgray', alpha=0.5) # 背景灰线
        axs[1].plot(masked_view, 'b', lw=1) # 蓝色实线 (可见部分)
        axs[1].set_title("Masked Input (Visible Parts)")
        axs[1].grid(True, alpha=0.3)

        # --- Subplot 3: 重建对比 (Raw Scale) ---
        axs[2].plot(orig_signal_raw, 'gray', alpha=0.5, label='Original')
        axs[2].plot(rec_signal_denorm, 'r', alpha=0.8, lw=1, label='Reconstructed')
        axs[2].set_title("Reconstruction Overlay")
        axs[2].legend()
        axs[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"epoch_{epoch}.png"))
        plt.close()
        
    model.train()