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
    
    
def get_layer_wise_lr(model, base_lr, layer_decay=0.65):
    """
    为 Time-Only MAE 微调构建分层学习率参数组。
    
    Args:
        model: TF_MAE_Classifier 实例
        base_lr: Head 层的学习率 (通常较大, e.g. 1e-4)
        layer_decay: 每一层的衰减系数 (0 < decay < 1), 越到底层 LR 越小
    """
    param_groups = []
    
    # -------------------------------------------------------
    # 1. Head & Final Norm (最高学习率 = base_lr)
    # -------------------------------------------------------
    head_params = list(model.head.parameters())
    
    # Encoder 的最终 Norm 层也应该跟随较高的学习率
    if hasattr(model.encoder_model, 'norm'):
        head_params.extend(list(model.encoder_model.norm.parameters()))
        
    param_groups.append({
        'params': head_params,
        'lr': base_lr,
        'name': 'head_and_norm'
    })

    # -------------------------------------------------------
    # 2. Transformer Blocks (逐层衰减)
    # -------------------------------------------------------
    # blocks 位于 model.encoder_model.blocks
    if hasattr(model.encoder_model, 'blocks'):
        blocks = model.encoder_model.blocks
        num_layers = len(blocks)

        for i in range(num_layers - 1, -1, -1):
            # i: 最后一层索引 -> 0
            # distance_from_head: 1 -> num_layers
            distance_from_head = num_layers - i
            
            layer_lr = base_lr * (layer_decay ** distance_from_head)
            
            param_groups.append({
                'params': blocks[i].parameters(),
                'lr': layer_lr,
                'name': f'block_{i}'
            })

    # -------------------------------------------------------
    # 3. Embeddings (最低学习率)
    # -------------------------------------------------------
    # LR = base_lr * decay^(num_layers + 1)
    embed_lr = base_lr * (layer_decay ** (num_layers + 1))
    
    embed_params = []
    if hasattr(model.encoder_model, 'patch_embed'):
        embed_params.extend(list(model.encoder_model.patch_embed.parameters()))
    if hasattr(model.encoder_model, 'pos_embed'):
        embed_params.append(model.encoder_model.pos_embed)
    if hasattr(model.encoder_model, 'cls_token'):
        embed_params.append(model.encoder_model.cls_token)

    if embed_params:
        param_groups.append({
            'params': embed_params,
            'lr': embed_lr,
            'name': 'embeddings'
        })
    
    return param_groups
# -------------------------------------------------------------------
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