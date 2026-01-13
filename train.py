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
from torch.cuda.amp import autocast

# 导入新模型
from model import CWT_MAE 
from dataset import PhysioSignalDataset

torch.set_float32_matmul_precision('high') 

# -------------------------------------------------------------------
# 1. 辅助工具类
# -------------------------------------------------------------------
class SmoothedValue(object):
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
# 2. 可视化函数
# -------------------------------------------------------------------
def save_reconstruction_images(model, batch, epoch, save_dir):
    model.eval()
    with torch.no_grad():
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with autocast(dtype=amp_dtype):
            loss, pred_spec, pred_time, imgs = model(batch)
        
        idx = 0
        
        # --- 1. 时域波形 ---
        orig_signal = batch[idx, 0, :].float().cpu().numpy()
        recon_signal = pred_time[idx].float().cpu().numpy()
        
        orig_mean = orig_signal.mean()
        orig_std = orig_signal.std()
        recon_signal = recon_signal * (orig_std + 1e-6) + orig_mean

        plt.figure(figsize=(15, 10))
        
        plt.subplot(3, 1, 1)
        plt.plot(orig_signal, label='Original', color='black', alpha=0.6, linewidth=1)
        plt.plot(recon_signal, label='Reconstructed', color='red', alpha=0.6, linewidth=1)
        plt.title(f"Epoch {epoch} - Time Domain")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # --- 2. 频域图谱 ---
        if isinstance(model, DDP):
            patch_embed = model.module.patch_embed
        else:
            patch_embed = model.patch_embed
            
        p_h, p_w = patch_embed.patch_size
        B, C, H, W = imgs.shape
        
        orig_spec = imgs[idx, 0, :, :].float().cpu().numpy()
        
        pred_patches = pred_spec[idx].view(H // p_h, W // p_w, C, p_h, p_w)
        pred_patches = pred_patches.permute(2, 0, 3, 1, 4)
        recon_img = pred_patches.reshape(C, H, W)
        recon_spec = recon_img[0].float().cpu().numpy()

        plt.subplot(3, 1, 2)
        plt.imshow(orig_spec, aspect='auto', origin='lower', cmap='jet')
        plt.title("Original CWT (Ch 0)")
        plt.colorbar()

        plt.subplot(3, 1, 3)
        plt.imshow(recon_spec, aspect='auto', origin='lower', cmap='jet')
        plt.title("Reconstructed CWT (Ch 0)")
        plt.colorbar()

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"epoch_{epoch}_vis.png")
        plt.savefig(save_path)
        plt.close()
        
    model.train()

# -------------------------------------------------------------------
# 3. 学习率调度器 (基于 Step)
# -------------------------------------------------------------------
def adjust_learning_rate_per_step(optimizer, current_step, total_steps, warmup_steps, base_lr, min_lr):
    """
    基于 Step 的 Cosine Decay 调度器
    """
    if current_step < warmup_steps:
        # Linear Warmup
        lr = base_lr * current_step / warmup_steps
    else:
        # Cosine Decay
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        lr = min_lr + (base_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * progress))
            
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

# -------------------------------------------------------------------
# 4. 训练逻辑
# -------------------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, epoch, logger, config, device, start_time_global, 
                    total_steps, warmup_steps, base_lr, min_lr):
    model.train()
    metric_logger = defaultdict(lambda: SmoothedValue(window_size=20))
    metric_logger['loss'] = SmoothedValue(window_size=20, fmt='{median:.4f} ({global_avg:.4f})')
    metric_logger['lr'] = SmoothedValue(window_size=1, fmt='{value:.6f}')
    
    header = f'Epoch: [{epoch}/{config["train"]["epochs"]}]'
    num_steps_per_epoch = len(dataloader)
    
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    for step, batch in enumerate(dataloader):
        # --- [修改点] 每个 Step 更新学习率 ---
        # 计算全局步数
        global_step = epoch * num_steps_per_epoch + step
        
        # 更新 LR
        adjust_learning_rate_per_step(
            optimizer, 
            current_step=global_step, 
            total_steps=total_steps, 
            warmup_steps=warmup_steps, 
            base_lr=base_lr, 
            min_lr=min_lr
        )
        # ----------------------------------

        batch = batch.to(device, non_blocking=True)

        with autocast(dtype=amp_dtype, enabled=config['train']['use_amp']):
            loss, _, _, _ = model(batch)
        
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['train']['clip_grad'])
        optimizer.step()

        metric_logger['loss'].update(loss_value)
        metric_logger['lr'].update(optimizer.param_groups[0]["lr"])

        if step % 20 == 0 and is_main_process():
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
# 5. 主函数
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

    dataset = PhysioSignalDataset(
        index_file=config['data']['index_path'],
        signal_len=config['data']['signal_len'],
        mode='train'
    )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=config['train']['batch_size'],
        sampler=sampler,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=True
    )

    # --- [修改点] 计算总步数和 Warmup 步数 ---
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
    # ---------------------------------------

    model = CWT_MAE(
        signal_len=config['data']['signal_len'],
        cwt_scales=config['model'].get('cwt_scales', 64),
        patch_size_time=config['model'].get('patch_size_time', 50),
        patch_size_freq=config['model'].get('patch_size_freq', 4),
        embed_dim=config['model']['embed_dim'],
        depth=config['model']['depth'],
        num_heads=config['model']['num_heads'],
        decoder_embed_dim=config['model']['decoder_embed_dim'],
        decoder_depth=config['model']['decoder_depth'],
        decoder_num_heads=config['model']['decoder_num_heads'],
        mask_ratio=config['model']['mask_ratio'],
        time_loss_weight=config['model'].get('time_loss_weight', 1.0)
    )
    model.to(device)

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
    
    start_epoch = 0
    if config['train']['resume'] and os.path.isfile(config['train']['resume']):
        checkpoint = torch.load(config['train']['resume'], map_location='cpu')
        model.module.load_state_dict(checkpoint['model'])
        
        # 如果是调整 LR 后 Resume，建议注释掉下面这行
        # optimizer.load_state_dict(checkpoint['optimizer'])
        
        start_epoch = checkpoint['epoch'] + 1
        if is_main_process():
            logger.info(f"Resumed from epoch {start_epoch}")

    vis_batch = None
    if is_main_process():
        vis_batch = next(iter(dataloader)).to(device)

    start_time_global = time.time()
    
    for epoch in range(start_epoch, total_epochs):
        sampler.set_epoch(epoch)
        
        # 移除旧的 adjust_learning_rate 调用
        
        train_one_epoch(
            model, dataloader, optimizer, epoch, logger, config, device, start_time_global,
            # 传入 Step 调度所需的参数
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            base_lr=base_lr,
            min_lr=min_lr
        )
        
        if is_main_process():
            save_reconstruction_images(
                model, 
                vis_batch, 
                epoch, 
                config['train']['save_dir']
            )
            
            save_dict = {
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'config': config
            }
            torch.save(save_dict, os.path.join(config['train']['save_dir'], "checkpoint_last.pth"))
            if epoch % config['train']['save_freq'] == 0:
                torch.save(save_dict, os.path.join(config['train']['save_dir'], f"checkpoint_epoch_{epoch}.pth"))

    dist.destroy_process_group()

if __name__ == '__main__':
    main()