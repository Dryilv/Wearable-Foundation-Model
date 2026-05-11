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
    param_groups = {}
    handled = set()

    def add_group(group_name, param, lr):
        if group_name not in param_groups:
            param_groups[group_name] = {'params': [], 'lr': lr}
        param_groups[group_name]['params'].append(param)
        handled.add(param)

    def process_encoder(encoder, prefix):
        num_layers = len(getattr(encoder, 'blocks', []))
        for name, param in encoder.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith('blocks.') and num_layers > 0:
                try:
                    layer_id = int(name.split('.')[1])
                    lr_scale = layer_decay ** max(num_layers - 1 - layer_id, 0)
                    add_group(f'{prefix}_layer_{layer_id}', param, base_lr * lr_scale)
                except (ValueError, IndexError):
                    add_group('default', param, base_lr)
            elif name.startswith(('patch_embed', 'cls_token', 'pos_embed')):
                lr_scale = layer_decay ** num_layers
                add_group(f'{prefix}_embed', param, base_lr * lr_scale)
            else:
                add_group('default', param, base_lr)

    if hasattr(model, 'encoder_model'):
        print("Applying layer-wise learning rate decay to encoder_model.")
        process_encoder(model.encoder_model, 'encoder')

    if hasattr(model, 'ppg_encoder'):
        print("Applying layer-wise learning rate decay to ppg_encoder.")
        process_encoder(model.ppg_encoder, 'ppg')

    if hasattr(model, 'ecg_encoder'):
        print("Applying layer-wise learning rate decay to ecg_encoder.")
        process_encoder(model.ecg_encoder, 'ecg')

    if hasattr(model, 'head'):
        print("Applying base learning rate to the classification head.")
        for param in model.head.parameters():
            if param.requires_grad:
                add_group('head', param, base_lr)

    all_params = {p for p in model.parameters() if p.requires_grad}
    ungrouped_params = all_params - handled
    if ungrouped_params:
        print(f"Warning: {len(ungrouped_params)} parameters were not assigned to any group. Adding them to default group.")
        for param in ungrouped_params:
            add_group('default', param, base_lr)

    return list(param_groups.values())
# -------------------------------------------------------------------
# Visualization (适配 Signal_MAE: 1D 原始信号重建可视化)
# -------------------------------------------------------------------
def save_reconstruction_images(model, x_time, channel_ids, epoch, save_dir):
    """
    Signal_MAE Visualization (1D 信号)
    """
    model.eval()
    vis_dir = os.path.join(save_dir, "vis_results")
    os.makedirs(vis_dir, exist_ok=True)

    with torch.no_grad():
        real_model = model.module if hasattr(model, 'module') else model
        if hasattr(real_model, '_orig_mod'):
            real_model = real_model._orig_mod

        x_time_input = x_time
        output = real_model(x_time_input, channel_ids)

        # Signal_MAE returns: loss, loss_dict, pred, x_target, mask, latent, _
        if len(output) >= 6:
            loss, loss_dict, pred, x_target, mask, latent = output[:6]
        else:
            raise ValueError(f"Unexpected model output length: {len(output)}")

        idx = 0
        orig_signal = x_time_input[idx].cpu().numpy()  # (M, L)
        print(f"[Vis Epoch {epoch}] Orig Signal Stats: Mean={orig_signal.mean():.4e}, Std={orig_signal.std():.4e}, Max={orig_signal.max():.4e}, Min={orig_signal.min():.4e}")

        # Signal_MAE: pred shape (B, M, N_patches, patch_size)
        # 将 pred reshape 回原始信号长度
        B, M, N_patches, patch_size = pred.shape
        L = N_patches * patch_size
        pred_signal = pred.reshape(B, M, L).cpu().numpy()
        recon_signal = pred_signal[idx]  # (M, L)

        mask_val = mask[idx].cpu().numpy()  # (M * N_patches,) or (N_patches,)

        M_ch, L_orig = orig_signal.shape

        # 获取 patch_size
        if hasattr(real_model, 'patch_size'):
            p_size = real_model.patch_size
        else:
            p_size = L // N_patches

        fig, axs = plt.subplots(M_ch, 3, figsize=(18, 3 * M_ch), squeeze=False)
        plt.suptitle(f"Epoch {epoch} Signal-MAE Reconstruction ({M_ch} Channels)", fontsize=16)

        channel_names = ["ECG", "PPG"]
        if M_ch != 2:
            channel_names = [f"Ch {i}" for i in range(M_ch)]

        for m in range(M_ch):
            # Mask 处理: Signal_MAE mask shape 可能是 (B, M*N) 或 (B, N)
            # 简化: 假设 mask 是 (B, N_patches) 统一应用于所有通道
            if mask_val.shape[0] == M_ch * N_patches:
                m_mask = mask_val[m * N_patches : (m + 1) * N_patches]
            else:
                m_mask = mask_val[:N_patches]

            m_mask_expanded = np.repeat(m_mask, p_size)
            if m_mask_expanded.shape[0] < L_orig:
                m_mask_expanded = np.pad(m_mask_expanded, (0, L_orig - m_mask_expanded.shape[0]), constant_values=0)
            else:
                m_mask_expanded = m_mask_expanded[:L_orig]

            # Column 1: Original Signal
            axs[m, 0].plot(orig_signal[m], 'k', lw=1)
            axs[m, 0].set_ylabel(channel_names[m] if m < len(channel_names) else f"Ch {m}")
            if m == 0: axs[m, 0].set_title("Original")
            axs[m, 0].grid(True, alpha=0.3)

            # Column 2: Masked Input
            masked_signal = orig_signal[m].copy()
            masked_signal[m_mask_expanded == 1] = np.nan
            axs[m, 1].plot(orig_signal[m], 'lightgray', alpha=0.5, label='Original')
            axs[m, 1].plot(masked_signal, 'b', lw=1, label='Visible')
            if m == 0:
                axs[m, 1].set_title("Masked Input (Blue=Visible)")
                axs[m, 1].legend(loc='upper right', fontsize='small')
            axs[m, 1].grid(True, alpha=0.3)

            # Column 3: Reconstruction Overlay
            combined_signal = orig_signal[m].copy()
            combined_signal[m_mask_expanded == 1] = recon_signal[m][m_mask_expanded == 1]

            axs[m, 2].plot(orig_signal[m], 'k', lw=1, label='Original', alpha=0.5)
            axs[m, 2].plot(recon_signal[m], 'r', lw=1, label='Reconstructed (Masked)')
            if m == 0:
                axs[m, 2].set_title("Reconstruction (Red=Masked Area)")
                axs[m, 2].legend(loc='upper right', fontsize='small')
            axs[m, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(vis_dir, f"recon_epoch_{epoch:03d}.png")
        plt.savefig(save_path, dpi=100)
        plt.close(fig)
        print(f"[Vis] Saved to {save_path}")

    model.train()
