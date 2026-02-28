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
        if d.numel() == 0:
            return 0.0
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        if d.numel() == 0:
            return 0.0
        return d.mean().item()

    @property
    def global_avg(self):
        if self.count == 0:
            return 0.0
        return self.total / self.count

    def __str__(self):
        if self.count == 0:
            return "N/A"
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=max(self.deque) if len(self.deque) > 0 else 0,
            value=self.deque[-1] if len(self.deque) > 0 else 0
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
# Visualization (已修改：支持 5 通道多变量可视化)
# -------------------------------------------------------------------
def save_reconstruction_images(model, x_time, epoch, save_dir, modality_ids=None):
    """
    5-Channel Visualization for CWT-MAE-v3
    """
    model.eval()
    vis_dir = os.path.join(save_dir, "vis_results")
    os.makedirs(vis_dir, exist_ok=True)

    with torch.no_grad():
        # 1. 获取模型实例 (处理 DDP/Compile 包装)
        real_model = model.module if hasattr(model, 'module') else model
        if hasattr(real_model, '_orig_mod'):
            real_model = real_model._orig_mod
        
        # 2. 模型推理 (直接使用原始信号，模型内部处理 CWT 和归一化)
        # x_time shape: (B, 5, L)
        if modality_ids is None:
            output = real_model(x_time)
        else:
            output = real_model(x_time, modality_ids=modality_ids)
        
        # Unpack output (Handling potentially varying return values)
        # CWT_MAE_RoPE.forward returns: loss, loss_dict, pred_spec, pred_time, imgs, mask
        if len(output) == 6:
            loss, loss_dict, pred_spec, pred_time, imgs, mask = output
        elif len(output) == 5:
            # Fallback for old signature or other models
            loss, pred_spec, pred_time, imgs, mask = output
        else:
            raise ValueError(f"Unexpected model output length: {len(output)}")
        
        # 3. 数据后处理 (取第一个样本 idx=0)
        idx = 0
        orig_signal = x_time[idx].cpu().numpy()      # (M, L)
        
        # Log statistics for debugging
        print(f"[Vis Epoch {epoch}] Orig Signal Stats: Mean={orig_signal.mean():.4e}, Std={orig_signal.std():.4e}, Max={orig_signal.max():.4e}, Min={orig_signal.min():.4e}")
        
        x_f32 = x_time.float()
        mean = x_f32.mean(dim=-1, keepdim=True)
        std = torch.clamp(x_f32.std(dim=-1, keepdim=True), min=1e-5)
        pred_time_denorm = pred_time.float() * std + mean
        recon_signal = pred_time_denorm[idx].cpu().numpy()    # (M, L)
        mask_val = mask[idx].cpu().numpy()            # (M * N_patches,)
        
        M, L = orig_signal.shape
        N_freq, N_time = real_model.grid_size
        N_patches = N_freq * N_time
        patch_size = real_model.patch_size_time
        
        # 4. 绘图 (M 行, 3 列)
        fig, axs = plt.subplots(M, 3, figsize=(18, 3 * M), squeeze=False)
        plt.suptitle(f"Epoch {epoch} Reconstruction (5 Channels)", fontsize=16)
        
        # 通道名称示例 (根据实际情况调整)
        channel_names = ["ECG", "ACC1", "ACC2", "ACC3", "PPG"]
        if M != 5:
            channel_names = [f"Ch {i}" for i in range(M)]

        for m in range(M):
            # --- Column 1: Original Signal ---
            axs[m, 0].plot(orig_signal[m], 'k', lw=1)
            axs[m, 0].set_ylabel(channel_names[m] if m < len(channel_names) else f"Ch {m}")
            if m == 0: axs[m, 0].set_title("Original")
            axs[m, 0].grid(True, alpha=0.3)

            # --- Column 2: Masked Input ---
            # 提取该通道对应的 mask 片段
            # 注意：mask 的顺序是所有通道拼接在一起的
            m_mask = mask_val[m * N_patches : (m + 1) * N_patches]
            # 这里的 mask 真正代表的是时间步的 Mask
            # 经过修复，所有频率在同一时间点的 Mask 是一致的。
            # mask_val (M * N_patches,) 包含了每个通道每一块 Patch 的判定，
            # 由于 N_patches = N_freq * N_time, 我们只取时间轴：
            m_mask_2d = m_mask.reshape(N_freq, N_time)
            
            # 【修复】提取纯时间步 Mask (直接取第一行，因为整个列都是相同的值)
            m_mask_time = m_mask_2d[0, :] # shape: (N_time,)
            
            # 将粗粒度的时间 Patch 扩展到原始信号级
            m_mask_time_expanded = np.repeat(m_mask_time, patch_size)
            if m_mask_time_expanded.shape[0] < L:
                m_mask_time_expanded = np.pad(m_mask_time_expanded, (0, L - m_mask_time_expanded.shape[0]), constant_values=0)
            else:
                m_mask_time_expanded = m_mask_time_expanded[:L]
            masked_signal = orig_signal[m].copy()
            masked_signal[m_mask_time_expanded == 1] = np.nan
            
            axs[m, 1].plot(orig_signal[m], 'lightgray', alpha=0.5, label='Original')
            axs[m, 1].plot(masked_signal, 'b', lw=1, label='Visible')
            if m == 0: 
                axs[m, 1].set_title("Masked Input (Blue=Visible)")
                axs[m, 1].legend(loc='upper right', fontsize='small')
            axs[m, 1].grid(True, alpha=0.3)

            # --- Column 3: Reconstruction Overlay ---
            # 【修复】构建混合信号：Visible 部分使用原始信号，Masked 部分使用重构信号
            combined_signal = orig_signal[m].copy()
            # 仅在被 Mask 的区域使用模型预测值
            combined_signal[m_mask_time_expanded == 1] = recon_signal[m][m_mask_time_expanded == 1]

            axs[m, 2].plot(orig_signal[m], 'gray', alpha=0.5, label='Original')
            axs[m, 2].plot(combined_signal, 'r', alpha=0.8, lw=1, label='Reconstructed')
            if m == 0: 
                axs[m, 2].set_title("Reconstruction (Merged)")
                axs[m, 2].legend(loc='upper right', fontsize='small')
            axs[m, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"epoch_{epoch}.png"))
        plt.close(fig)
        
    model.train()
