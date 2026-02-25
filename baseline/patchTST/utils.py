import torch
import matplotlib.pyplot as plt
import os
import numpy as np
import logging
import torch.distributed as dist
from collections import deque
import datetime
import math

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
    logger = logging.getLogger("PatchTST")
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

# Learning Rate Scheduler
# -------------------------------------------------------------------
def adjust_learning_rate_per_step(optimizer, current_step, total_steps, warmup_steps, base_lr, min_lr):
    if current_step < warmup_steps:
        lr = base_lr * current_step / warmup_steps
    else:
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        lr = min_lr + (base_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * progress))
            
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

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
    
# Visualization
# -------------------------------------------------------------------
def save_reconstruction_images(model, x, epoch, save_dir):
    """
    Visualization for PatchTST
    x: (B, C, L) - original signal from DataLoader
    """
    model.eval()
    vis_dir = os.path.join(save_dir, "vis_results")
    os.makedirs(vis_dir, exist_ok=True)

    with torch.no_grad():
        # 1. 获取模型实例
        real_model = model.module if hasattr(model, 'module') else model
        if hasattr(real_model, '_orig_mod'):
            real_model = real_model._orig_mod
        
        # 2. 准备输入 (PatchTST 期望 B, L, C)
        # x is (B, C, L), so permute
        x_input = x.permute(0, 2, 1).to(x.device) # (B, L, C)
        
        # 3. 模型推理
        # output: loss, pred_patches, target_patches, mask
        # pred_patches: (B*C, N_patches, Patch_Len)
        # target_patches: (B*C, N_patches, Patch_Len)
        # mask: (B*C, N_patches)
        ret = real_model(x_input)
        _, pred_patches, target_patches, mask = ret
        
        # 4. 数据后处理 (取第一个样本 idx=0)
        idx = 0
        B, C, L = x.shape
        # 我们只看第一个 Batch 的数据，但 PatchTST 内部把 Batch 和 Channel 混在一起了
        # 实际上 x_input 的第一个样本对应 B*C 中的前 C 个条目
        # (因为 view(B*C, ...) 是先排 Batch 再排 Channel，还是反过来？
        # x.permute(0, 2, 1) -> (B, L, C)
        # view(B*C, L, 1) -> 它是怎么 view 的？
        # PyTorch view 默认是 C 变化最快？不，view 是按内存顺序。
        # x.permute(0, 2, 1) 后的内存布局是 Batch 维度 stride 最大，Channel 维度 stride 最小 (1)
        # 也就是 (Batch 0, Time 0, Ch 0), (Batch 0, Time 0, Ch 1), ...
        # 等等，view(B*C, L, 1) 假设是把 Batch 和 Channel 展平。
        # 如果是 x.view(B*C, L, 1)，那么必须保证 x 在内存中是连续的。
        # PatchTST 代码中: x = x.permute(0, 2, 1).contiguous().view(B * M, L, 1)
        # 所以对于 (B, L, C)，contiguous() 后，最后一个维度 C 是最紧密的。
        # 所以 view(B*C, L, 1) 实际上是把 (Time, Channel) 混在一起了吗？
        # 不！如果 x 是 (B, L, C)， contiguous 后，数据顺序是:
        # B0L0C0, B0L0C1, B0L0C2... B0L1C0...
        # 这样 view 成 (B*C, L, 1) 是不对的！这样会把时间步混进去。
        
        # Wait, check PatchTST code again.
        # x = x.permute(0, 2, 1).contiguous().view(B * M, L, 1)
        # If x is (B, L, M).
        # We want to treat each channel as an independent sample.
        # So we want (B, M, L) -> (B*M, L, 1).
        # If input is (B, L, M), permute(0, 2, 1) gives (B, M, L).
        # contiguous() makes it (B, M, L) in memory.
        # Then view(B*M, L, 1) works perfectly!
        
        # BUT, wait.
        # PatchTST code says:
        # x input: [Batch, Seq_Len, Channels]
        # x = x.permute(0, 2, 1).contiguous().view(B * M, L, 1)
        # This implies it permutes (B, L, M) to (B, M, L) then flattens B and M.
        # Correct.
        
        # So, the output pred_patches has shape (B*M, N, P).
        # The first M entries correspond to the first Batch sample's M channels.
        
        # We want to visualize the first sample (all 5 channels).
        orig_signal = x[idx].cpu().numpy() # (M, L)
        
        # Extract reconstruction for the first sample (M channels)
        # indices 0 to M-1
        M = orig_signal.shape[0]
        
        # pred_patches: (B*M, N, P)
        sample_pred_patches = pred_patches[:M] # (M, N, P)
        sample_mask = mask[:M] # (M, N)
        
        # Reconstruct signal from patches
        # Since stride = patch_len, we can just reshape
        # (M, N, P) -> (M, N*P)
        # Note: N*P might be slightly larger or smaller than L depending on padding/truncation
        # PatchTST implementation:
        # patch_num = int((seq_len - patch_len) / stride + 1)
        # unfold(dimension=1, size=self.patch_len, step=self.stride)
        # If stride=patch_len, it covers coverage is (N-1)*stride + patch_len = N*patch_len
        # If seq_len is multiple of patch_len, N*P = seq_len.
        
        N, P = sample_pred_patches.shape[1], sample_pred_patches.shape[2]
        recon_signal = sample_pred_patches.view(M, -1).cpu().numpy() # (M, N*P)
        
        # 裁剪或填充到原始长度 L
        L_recon = recon_signal.shape[1]
        L_orig = orig_signal.shape[1]
        
        if L_recon > L_orig:
            recon_signal = recon_signal[:, :L_orig]
        elif L_recon < L_orig:
            # pad
            recon_signal = np.pad(recon_signal, ((0,0), (0, L_orig - L_recon)))
            
        # Mask visualization
        # mask is (M, N), 1 means masked (removed), 0 means kept
        # We want to show which parts were masked.
        # Expand mask to signal length
        # mask (M, N) -> (M, N, P) -> (M, N*P)
        mask_expanded = sample_mask.unsqueeze(-1).repeat(1, 1, P).view(M, -1).cpu().numpy()
        if mask_expanded.shape[1] > L_orig:
            mask_expanded = mask_expanded[:, :L_orig]
        elif mask_expanded.shape[1] < L_orig:
            mask_expanded = np.pad(mask_expanded, ((0,0), (0, L_orig - mask_expanded.shape[1])))
            
        # 4. 绘图 (M 行, 3 列)
        fig, axs = plt.subplots(M, 3, figsize=(18, 3 * M), squeeze=False)
        plt.suptitle(f"Epoch {epoch} Reconstruction (5 Channels)", fontsize=16)
        
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
            # mask=1 means masked (removed), mask=0 means visible
            # We want to show visible parts
            
            masked_signal = orig_signal[m].copy()
            masked_signal[mask_expanded[m] == 1] = np.nan # Hide masked parts
            
            axs[m, 1].plot(orig_signal[m], 'lightgray', alpha=0.5, label='Original')
            axs[m, 1].plot(masked_signal, 'b', lw=1, label='Visible')
            if m == 0: 
                axs[m, 1].set_title("Masked Input (Blue=Visible)")
            axs[m, 1].grid(True, alpha=0.3)

            # --- Column 3: Reconstruction Overlay ---
            # Combined: Visible parts + Reconstructed Masked parts
            combined_signal = orig_signal[m].copy()
            # Replace masked parts with reconstruction
            # Note: The model output `pred_patches` is the reconstruction of ALL patches (masked or not).
            # Usually we only care about masked parts for loss, but for vis we can show all or mixed.
            # Let's show mixed.
            
            # recon_signal[m] contains reconstruction
            combined_signal[mask_expanded[m] == 1] = recon_signal[m][mask_expanded[m] == 1]

            axs[m, 2].plot(orig_signal[m], 'gray', alpha=0.5, label='Original')
            axs[m, 2].plot(combined_signal, 'r', alpha=0.8, lw=1, label='Reconstructed')
            if m == 0: 
                axs[m, 2].set_title("Reconstruction (Merged)")
                axs[m, 2].legend(loc='upper right', fontsize='small')
            axs[m, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"epoch_{epoch}.png"))
        plt.close()
        
    model.train()
