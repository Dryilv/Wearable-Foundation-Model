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

# 确保你的模型和数据集定义文件在同一目录下
from model_ppg2ecg import PPG2ECG_Translator
from dataset_paired import PairedPhysioDataset

# 启用 TensorFloat-32 以加速
torch.set_float32_matmul_precision('high')
# 允许 torch.compile 在遇到不支持的操作时自动回退，而不是报错
torch._dynamo.config.suppress_errors = True

# -------------------------------------------------------------------
# 1. 学习率调度器 (基于 Step)
# -------------------------------------------------------------------
def adjust_learning_rate(optimizer, current_step, total_steps, warmup_steps, base_lr, min_lr):
    """基于 Step 的余弦退火学习率调度器"""
    if current_step < warmup_steps:
        # Warmup 阶段: 学习率从 0 线性增加到 base_lr
        lr = base_lr * current_step / warmup_steps
    else:
        # Cosine Decay 阶段
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        lr = min_lr + (base_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * progress))
            
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

# -------------------------------------------------------------------
# 2. 辅助工具类和函数
# -------------------------------------------------------------------
# train_ppg2ecg.py

class SmoothedValue(object):
    """Tracks a series of values and provides smoothed summaries."""
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            # Default format string only uses median and global_avg
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
        if not self.deque:
            return 0.0
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        if not self.deque:
            return 0.0
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        if self.count == 0:
            return 0.0
        return self.total / self.count

    @property
    def value(self):
        if not self.deque:
            return 0.0
        return self.deque[-1]

    def __str__(self):
        # The format string `self.fmt` now dictates which properties are needed.
        # For example, if fmt="{value:.6f}", it will call self.value.
        # If fmt="{median:.4f}", it will call self.median.
        # This is more flexible and avoids the KeyError.
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            value=self.value
        )

def format_time(seconds): return str(datetime.timedelta(seconds=int(seconds)))
def is_main_process(): return not dist.is_initialized() or dist.get_rank() == 0

def setup_logger(save_dir):
    logger = logging.getLogger("PPG2ECG_Translator")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers(): return logger
    if is_main_process():
        handler = logging.FileHandler(os.path.join(save_dir, "train_translator.log"))
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
        return gpu, rank, world_size
    return 0, 0, 1

def save_translation_vis(model, ppg_batch, ecg_batch, epoch, save_dir):
    model.eval()
    with torch.no_grad():
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            pred_time, _ = model(ppg_batch, ecg_target=None)
        
        idx = 0 
        ppg_signal = ppg_batch[idx].squeeze().float().cpu().numpy()
        ecg_target = ecg_batch[idx].squeeze().float().cpu().numpy()
        ecg_pred = pred_time[idx].float().cpu().numpy()
        
        plt.figure(figsize=(15, 8))
        plt.subplot(2, 1, 1)
        plt.plot(ppg_signal, color='green', alpha=0.8, label='Input PPG (Normalized)')
        plt.title("Input PPG Signal")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.plot(ecg_target, label='Ground Truth ECG', color='black', alpha=0.6)
        plt.plot(ecg_pred, label='Generated ECG', color='red', alpha=0.8, linestyle='--')
        plt.title(f"Epoch {epoch} - ECG Reconstruction")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"translation_epoch_{epoch}.png"))
        plt.close()
    model.train()

# -------------------------------------------------------------------
# 3. 训练循环
# -------------------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, epoch, logger, config, device, 
                    total_steps, warmup_steps, base_lr, min_lr):
    model.train()
    
    metric_logger = defaultdict(lambda: SmoothedValue(window_size=50))
    metric_logger['loss'] = SmoothedValue(fmt='{median:.4f}')
    metric_logger['loss_spec'] = SmoothedValue(fmt='{median:.4f}')
    metric_logger['loss_mse'] = SmoothedValue(fmt='{median:.4f}')
    metric_logger['loss_corr'] = SmoothedValue(fmt='{median:.4f}')
    metric_logger['lr'] = SmoothedValue(window_size=1, fmt='{value:.6f}')
    metric_logger['data_time'] = SmoothedValue(fmt='{avg:.4f}s')
    metric_logger['batch_time'] = SmoothedValue(fmt='{avg:.4f}s')
    
    header = f'Epoch: [{epoch}/{config["train"]["epochs"]}]'
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    num_steps_per_epoch = len(dataloader)
    end = time.time()
    
    for step, (ppg, ecg) in enumerate(dataloader):
        global_step = epoch * num_steps_per_epoch + step
        adjust_learning_rate(optimizer, global_step, total_steps, warmup_steps, base_lr, min_lr)

        metric_logger['data_time'].update(time.time() - end)
        
        ppg = ppg.to(device, non_blocking=True)
        ecg = ecg.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=amp_dtype):
            loss, l_spec, l_mse, l_corr = model(ppg, ecg_target=ecg)
        
        if not math.isfinite(loss.item()):
            logger.error(f"Loss is {loss.item()}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['train'].get('clip_grad', 1.0))
        optimizer.step()

        metric_logger['loss'].update(loss.item())
        metric_logger['loss_spec'].update(l_spec)
        metric_logger['loss_mse'].update(l_mse)
        metric_logger['loss_corr'].update(l_corr)
        metric_logger['lr'].update(optimizer.param_groups[0]["lr"])
        
        metric_logger['batch_time'].update(time.time() - end)
        end = time.time()

        if step % config['train']['log_interval'] == 0 and is_main_process():
            steps_left = num_steps_per_epoch - step
            eta_seconds = metric_logger['batch_time'].avg * steps_left
            eta_string = format_time(eta_seconds)
            
            logger.info(
                f"{header}[{step:>{len(str(num_steps_per_epoch))}}/{num_steps_per_epoch}] "
                f"ETA: {eta_string} | "
                f"Loss: {metric_logger['loss']} "
                f"(Spec: {metric_logger['loss_spec']} | MSE: {metric_logger['loss_mse']} | Corr: {metric_logger['loss_corr']}) | "
                f"LR: {metric_logger['lr']} | "
                f"Data: {metric_logger['data_time']} | "
                f"Batch: {metric_logger['batch_time']}"
            )
            
    return metric_logger['loss'].global_avg

# train_ppg2ecg.py

def main():
    parser = argparse.ArgumentParser(description="PPG to ECG Translation Training")
    parser.add_argument('--config', default='config.yaml', type=str, help='Path to YAML config file')
    parser.add_argument('--pretrained', type=str, required=True, help='Path to pretrained CWT-MAE weights for initialization')
    parser.add_argument('--resume', type=str, default='', help='Path to a checkpoint to resume training from')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    gpu_id, rank, world_size = init_distributed_mode()
    device = torch.device(f"cuda:{gpu_id}")
    
    if is_main_process():
        Path(config['train']['save_dir']).mkdir(parents=True, exist_ok=True)
    logger = setup_logger(config['train']['save_dir'])

    dataset = PairedPhysioDataset(
        index_file=config['data']['index_path'],
        signal_len=config['data']['signal_len'],
        mode='train',
        row_ppg=4, # Input
        row_ecg=0  # Target
    )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=config['train']['batch_size'], sampler=sampler,
                            num_workers=config['data']['num_workers'], pin_memory=True, drop_last=True)

    # --- Model Initialization ---
    # NOTE: We pass pretrained_path=None here because we will handle loading manually
    model = PPG2ECG_Translator(
        pretrained_path=None, 
        corr_loss_weight=config['model'].get('corr_loss_weight', 1.0),
        signal_len=config['data']['signal_len'],
        cwt_scales=config['model'].get('cwt_scales', 64),
        embed_dim=config['model']['embed_dim'],
        depth=config['model']['depth'],
        num_heads=config['model']['num_heads'],
        decoder_embed_dim=config['model']['decoder_embed_dim'],
        decoder_depth=config['model']['decoder_depth'],
        decoder_num_heads=config['model']['decoder_num_heads'],
        time_loss_weight=config['model'].get('time_loss_weight', 2.0)
    )
    model.to(device)

    # --- Manual Weight Loading (The Fix) ---
    start_epoch = 0
    # Determine which checkpoint to load: resume takes precedence over pretrained
    load_path = args.resume if args.resume else args.pretrained
    
    if load_path:
        if is_main_process(): logger.info(f"Loading weights from: {load_path}")
        checkpoint = torch.load(load_path, map_location='cpu', weights_only=True)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        
        # Create a new state_dict with the correct prefixes for the nested `mae` model
        # We need to add the `mae.` prefix to the keys from the original MAE checkpoint
        new_state_dict = {}
        for k, v in state_dict.items():
            # Clean up any prefixes from the checkpoint file itself
            clean_k = k.replace('module.', '').replace('_orig_mod.', '')
            # Add the `mae.` prefix because we are loading into the nested `mae` attribute
            new_state_dict[f"mae.{clean_k}"] = v
            
        # Load the weights into the top-level model
        msg = model.load_state_dict(new_state_dict, strict=False)
        if is_main_process():
            logger.info(f"Weights loaded. Missing keys: {msg.missing_keys}, Unexpected keys: {msg.unexpected_keys}")

        # If resuming, also load the epoch
        if args.resume and 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            if is_main_process(): logger.info(f"Resumed from epoch {start_epoch}")

    # --- Compile and Wrap Model ---
    # if is_main_process(): logger.info("Compiling model with torch.compile() ...")
    # try:
    #     model = torch.compile(model)
    # except Exception as e:
    #     if is_main_process(): logger.warning(f"torch.compile failed: {e}")

    model = DDP(model, device_ids=[gpu_id], output_device=gpu_id, find_unused_parameters=False)
    
    base_lr = float(config['train']['base_lr'])
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=float(config['train']['weight_decay']))
    
    num_steps_per_epoch = len(dataloader)
    total_epochs = config['train']['epochs']
    total_steps = num_steps_per_epoch * total_epochs
    warmup_steps = num_steps_per_epoch * config['train']['warmup_epochs']
    min_lr = float(config['train']['min_lr'])

    if is_main_process():
        logger.info(f"Total Steps: {total_steps}, Warmup Steps: {warmup_steps}")
        vis_ppg, vis_ecg = next(iter(dataloader))
        vis_ppg = vis_ppg.to(device)
        vis_ecg = vis_ecg.to(device)

    total_start_time = time.time()
    
    for epoch in range(start_epoch, total_epochs):
        sampler.set_epoch(epoch)
        
        avg_loss = train_one_epoch(
            model, dataloader, optimizer, epoch, logger, config, device,
            total_steps, warmup_steps, base_lr, min_lr
        )
        
        if is_main_process():
            logger.info(f"Epoch {epoch} done. Average Loss: {avg_loss:.4f}")
            save_translation_vis(model, vis_ppg, vis_ecg, epoch, config['train']['save_dir'])
            
            # When saving, we get the state_dict from the DDP-wrapped model
            save_dict = {'model': model.state_dict(), 'epoch': epoch, 'config': config}
            torch.save(save_dict, os.path.join(config['train']['save_dir'], "checkpoint_translator_last.pth"))
            if (epoch + 1) % config['train']['save_freq'] == 0:
                torch.save(save_dict, os.path.join(config['train']['save_dir'], f"checkpoint_translator_{epoch}.pth"))

    total_time = time.time() - total_start_time
    if is_main_process(): logger.info(f"Total training time: {format_time(total_time)}")
    dist.destroy_process_group()

if __name__ == '__main__':
    main()