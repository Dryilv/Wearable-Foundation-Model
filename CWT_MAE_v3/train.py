import os
import sys
import argparse
import yaml
import math
import time
import datetime
import logging
from pathlib import Path
from collections import deque, defaultdict
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler

# 允许编译失败时自动回退
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# 导入你的模型和数据集
from model import CWT_MAE_RoPE, cwt_wrap
from dataset import PhysioSignalDataset

# 启用 TensorFloat-32 (A100/3090/4090 必备加速)
torch.set_float32_matmul_precision('high') 

# -------------------------------------------------------------------
# 1. 辅助工具类
# -------------------------------------------------------------------
class SmoothedValue(object):
    """用于平滑记录 Loss 和 LR"""
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
    logger = logging.getLogger("CWT-MAE")
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

# -------------------------------------------------------------------
# 2. 关键：处理变长通道的 Collate Function
# -------------------------------------------------------------------
def variable_channel_collate_fn(batch):
    """
    处理 Batch 中不同样本通道数不一致的情况。
    Batch: list of (tensor, label), where tensor shape is (M_i, L)
    Output: (padded_signals, labels)
    """
    # 1. 解包信号和标签
    signals = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # 2. 找到当前 Batch 中最大的通道数
    max_m = max([s.shape[0] for s in signals])
    signal_len = signals[0].shape[1]
    batch_size = len(signals)
    
    # 3. 初始化全 0 张量并填充信号
    padded_signals = torch.zeros((batch_size, max_m, signal_len), dtype=signals[0].dtype)
    for i, s in enumerate(signals):
        m = s.shape[0]
        padded_signals[i, :m, :] = s
        
    # 4. 堆叠标签
    labels = torch.stack(labels)
        
    return padded_signals, labels

# -------------------------------------------------------------------
# 3. 可视化函数 (v3: 1D Signal + CWT Spec)
# -------------------------------------------------------------------
def save_reconstruction_images(model, batch, epoch, save_dir):
    model.eval()
    with torch.no_grad():
        # 优先使用 bfloat16
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            # forward 返回: total_loss, pred_signal, None, x_norm_signal
            loss, pred_signal, _, orig_signal = model(batch)
        
        # 取 Batch 中的第一个样本 (Index 0)
        # orig_signal: (B, M, L)
        # pred_signal: (B, M, L)
        
        sample_idx = 0
        channel_idx = 0 
        
        # --- 1. 时域波形 ---
        orig_wave = orig_signal[sample_idx, channel_idx].float().cpu().numpy()
        recon_wave = pred_signal[sample_idx, channel_idx].float().cpu().numpy()
        
        plt.figure(figsize=(15, 12))
        
        plt.subplot(3, 1, 1)
        plt.plot(orig_wave, label='Original (Norm)', color='black', alpha=0.6, linewidth=1)
        plt.plot(recon_wave, label='Reconstructed', color='red', alpha=0.6, linewidth=1)
        plt.title(f"Epoch {epoch} - Time Domain (Sample 0, Channel 0)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # --- 2. 频域图谱 (手动计算 CWT) ---
        # 计算 CWT 用于可视化对比
        # 需要把 tensor 转回 device 计算
        orig_t = torch.tensor(orig_wave).view(1, 1, -1).to(batch.device)
        recon_t = torch.tensor(recon_wave).view(1, 1, -1).to(batch.device)
        
        # 调用 model 中的 loss 计算用的 scales
        cwt_scales = 64
        orig_cwt = cwt_wrap(orig_t, num_scales=cwt_scales)[0, 0].float().cpu().numpy()
        recon_cwt = cwt_wrap(recon_t, num_scales=cwt_scales)[0, 0].float().cpu().numpy()
        
        plt.subplot(3, 1, 2)
        plt.imshow(orig_cwt, aspect='auto', origin='lower', cmap='jet')
        plt.title("Original CWT")
        plt.colorbar()

        plt.subplot(3, 1, 3)
        plt.imshow(recon_cwt, aspect='auto', origin='lower', cmap='jet')
        plt.title("Reconstructed CWT")
        plt.colorbar()

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"epoch_{epoch}_vis.png")
        plt.savefig(save_path)
        plt.close()
        
    model.train()

# -------------------------------------------------------------------
# 4. 学习率调度器
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

# -------------------------------------------------------------------
# 5. 训练逻辑
# -------------------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, scaler, epoch, logger, config, device, start_time_global, 
                    total_steps, warmup_steps, base_lr, min_lr):
    model.train()
    metric_logger = defaultdict(lambda: SmoothedValue(window_size=20))
    metric_logger['loss'] = SmoothedValue(window_size=20, fmt='{median:.4f} ({global_avg:.4f})')
    metric_logger['lr'] = SmoothedValue(window_size=1, fmt='{value:.6f}')
    
    header = f'Epoch: [{epoch}/{config["train"]["epochs"]}]'
    num_steps_per_epoch = len(dataloader)
    
    # 优先使用 bfloat16
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    for step, (batch, labels) in enumerate(dataloader):
        global_step = epoch * num_steps_per_epoch + step
        
        # 调整 LR
        adjust_learning_rate_per_step(
            optimizer, 
            current_step=global_step, 
            total_steps=total_steps, 
            warmup_steps=warmup_steps, 
            base_lr=base_lr, 
            min_lr=min_lr
        )

        # batch shape: (B, M, L)
        batch = batch.to(device, non_blocking=True)
        # labels = labels.to(device, non_blocking=True) # MAE 训练暂不需要标签

        # 混合精度前向传播
        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=config['train']['use_amp']):
            loss, _, _, _ = model(batch)
        
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        
        # 使用 Scaler 处理反向传播 (兼容 fp16)
        scaler.scale(loss).backward()
        
        # Unscale 之后才能 clip grad
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['train']['clip_grad'])
        
        scaler.step(optimizer)
        scaler.update()

        metric_logger['loss'].update(loss_value)
        metric_logger['lr'].update(optimizer.param_groups[0]["lr"])

        if step % 50 == 0 and is_main_process():
            elapsed = time.time() - start_time_global
            logger.info(
                f"{header} Step: [{step}/{num_steps_per_epoch}] "
                f"Loss: {metric_logger['loss']} "
                f"LR: {metric_logger['lr']} "
                f"Elapsed: {format_time(elapsed)}"
            )
            
    if is_main_process():
        logger.info(f"Epoch {epoch} done. Avg Loss: {metric_logger['loss'].global_avg:.4f}")
    
    return metric_logger['loss'].global_avg

# -------------------------------------------------------------------
# 6. 主函数
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', type=str)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    gpu_id, rank, world_size = init_distributed_mode()
    device = torch.device(f"cuda:{gpu_id}")
    
    seed = 42 + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True 

    if is_main_process():
        Path(config['train']['save_dir']).mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(config['train']['save_dir'])

    # Dataset
    dataset = PhysioSignalDataset(
        index_file=config['data']['index_path'],
        signal_len=config['data']['signal_len'],
        stride=config['data'].get('stride'),
        original_len=config['data'].get('original_len', 3000),
        mode='train',
        min_std_threshold=config['data'].get('min_std_threshold', 1e-4),
        max_std_threshold=config['data'].get('max_std_threshold', 5000.0),
        max_abs_value=config['data'].get('max_abs_value', 1e5)
    )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    # DataLoader (使用自定义 collate_fn)
    dataloader = DataLoader(
        dataset,
        batch_size=config['train']['batch_size'],
        sampler=sampler,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=True,
        collate_fn=variable_channel_collate_fn # 关键：处理变长通道
    )

    num_steps_per_epoch = len(dataloader)
    total_epochs = config['train']['epochs']
    warmup_epochs = config['train']['warmup_epochs']
    
    total_steps = num_steps_per_epoch * total_epochs
    warmup_steps = num_steps_per_epoch * warmup_epochs
    
    base_lr = float(config['train']['base_lr'])
    min_lr = float(config['train']['min_lr'])
    
    if is_main_process():
        logger.info(f"Total Steps: {total_steps}, Warmup Steps: {warmup_steps}")
        logger.info(f"Base LR: {base_lr}, Min LR: {min_lr}")

    # 初始化模型 (v3 Pixel-based 参数)
    model = CWT_MAE_RoPE(
        signal_len=config['data']['signal_len'],
        patch_size=config['model'].get('patch_size', 4), # 使用 patch_size
        embed_dim=config['model']['embed_dim'],
        depth=config['model']['depth'],
        num_heads=config['model']['num_heads'],
        decoder_embed_dim=config['model']['decoder_embed_dim'],
        decoder_depth=config['model']['decoder_depth'],
        decoder_num_heads=config['model']['decoder_num_heads'],
        mask_ratio=config['model'].get('mask_ratio', 0.75),
        mlp_rank_ratio=config['model'].get('mlp_rank_ratio', 0.5),
        cwt_scales=config['model'].get('cwt_scales', 64),
        cwt_loss_weight=config['model'].get('cwt_loss_weight', 1.0)
    )
    model.to(device)

    # 编译模型
    try:
        model = torch.compile(model)
        if is_main_process():
            logger.info("Model compiled with torch.compile()")
    except Exception as e:
        if is_main_process():
            logger.warning(f"Could not compile model: {e}")

    model = DDP(model, device_ids=[gpu_id], output_device=gpu_id, find_unused_parameters=False)

    param_groups = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(param_groups, lr=base_lr, weight_decay=float(config['train']['weight_decay']))
    
    # GradScaler 用于混合精度
    scaler = GradScaler(enabled=config['train']['use_amp'])
    
    start_epoch = 0
    if config['train']['resume'] and os.path.isfile(config['train']['resume']):
        checkpoint = torch.load(config['train']['resume'], map_location='cpu')
        model.module.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        if is_main_process():
            logger.info(f"Resumed from epoch {start_epoch}")

    # 获取一个 Batch 用于固定可视化
    vis_batch = None
    if is_main_process():
        try:
            vis_batch, _ = next(iter(dataloader))
            vis_batch = vis_batch.to(device)
        except StopIteration:
            pass

    start_time_global = time.time()
    
    for epoch in range(start_epoch, total_epochs):
        sampler.set_epoch(epoch)
        
        train_one_epoch(
            model, dataloader, optimizer, scaler, epoch, logger, config, device, start_time_global,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            base_lr=base_lr,
            min_lr=min_lr
        )
        
        if is_main_process():
            # 保存可视化
            if vis_batch is not None:
                save_reconstruction_images(
                    model, 
                    vis_batch, 
                    epoch, 
                    config['train']['save_dir']
                )
            
            # 保存 Checkpoint
            save_dict = {
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'epoch': epoch,
                'config': config
            }
            torch.save(save_dict, os.path.join(config['train']['save_dir'], "checkpoint_last.pth"))
            if epoch % config['train']['save_freq'] == 0:
                torch.save(save_dict, os.path.join(config['train']['save_dir'], f"checkpoint_epoch_{epoch}.pth"))

    dist.destroy_process_group()

if __name__ == '__main__':
    main()
