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

from model_ppg2ecg import PPG2ECG_Translator
from dataset_paired import PairedPhysioDataset

torch.set_float32_matmul_precision('high')
torch._dynamo.config.suppress_errors = True

# -------------------------------------------------------------------
# 1. 学习率调度器 (Step-based)
# -------------------------------------------------------------------
def adjust_learning_rate(optimizer, current_step, total_steps, warmup_steps, base_lr, min_lr):
    """
    基于 Step 的余弦退火调度
    """
    if current_step < warmup_steps:
        # Warmup 阶段：线性增加
        lr = base_lr * current_step / warmup_steps
    else:
        # Cosine Decay 阶段
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        lr = min_lr + (base_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * progress))
            
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

# -------------------------------------------------------------------
# 工具类
# -------------------------------------------------------------------
class SmoothedValue(object):
    def __init__(self, window_size=20, fmt=None):
        if fmt is None: fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt
    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n
    @property
    def median(self): return torch.tensor(list(self.deque)).median().item()
    @property
    def avg(self): return torch.tensor(list(self.deque)).mean().item()
    @property
    def global_avg(self): return self.total / self.count
    def __str__(self): return self.fmt.format(median=self.median, avg=self.avg, global_avg=self.global_avg, max=max(self.deque), value=self.deque[-1])

def format_time(seconds): return str(datetime.timedelta(seconds=int(seconds)))
def is_main_process(): return not dist.is_initialized() or dist.get_rank() == 0

def setup_logger(save_dir):
    logger = logging.getLogger("PPG2ECG")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers(): return logger
    if is_main_process():
        handler = logging.FileHandler(os.path.join(save_dir, "train_trans.log"))
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
            pred_time, pred_spec = model(ppg_batch, ecg_target=None)
        
        idx = 0 
        ppg_signal = ppg_batch[idx].squeeze().float().cpu().numpy()
        ecg_target = ecg_batch[idx].squeeze().float().cpu().numpy()
        ecg_pred = pred_time[idx].float().cpu().numpy()
        
        plt.figure(figsize=(15, 12))
        plt.subplot(4, 1, 1)
        plt.plot(ppg_signal, color='green', alpha=0.8)
        plt.title("Input: PPG Signal")
        plt.grid(True, alpha=0.3)
        plt.subplot(4, 1, 2)
        plt.plot(ecg_target, label='Ground Truth ECG', color='black', alpha=0.6)
        plt.plot(ecg_pred, label='Generated ECG', color='red', alpha=0.8, linestyle='--')
        plt.title(f"Epoch {epoch} - ECG Reconstruction")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"trans_epoch_{epoch}.png"))
        plt.close()
    model.train()

# -------------------------------------------------------------------
# 训练逻辑 (修改为接收调度参数)
# -------------------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, epoch, logger, config, device, 
                    total_steps, warmup_steps, base_lr, min_lr):
    model.train()
    
    metric_logger = defaultdict(lambda: SmoothedValue(window_size=20))
    metric_logger['loss'] = SmoothedValue(window_size=20, fmt='{median:.4f} ({global_avg:.4f})')
    metric_logger['data_time'] = SmoothedValue(window_size=20, fmt='{avg:.4f}')
    metric_logger['batch_time'] = SmoothedValue(window_size=20, fmt='{avg:.4f}')
    metric_logger['lr'] = SmoothedValue(window_size=1, fmt='{value:.6f}')
    
    header = f'Epoch: [{epoch}]'
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    num_steps_per_epoch = len(dataloader)
    end = time.time()
    
    for step, (ppg, ecg) in enumerate(dataloader):
        # 1. 计算全局 Step 并调整 LR
        global_step = epoch * num_steps_per_epoch + step
        adjust_learning_rate(optimizer, global_step, total_steps, warmup_steps, base_lr, min_lr)

        data_time_val = time.time() - end
        metric_logger['data_time'].update(data_time_val)
        
        ppg = ppg.to(device, non_blocking=True)
        ecg = ecg.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=amp_dtype):
            loss, _, _ = model(ppg, ecg_target=ecg)
        
        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['train'].get('clip_grad', 1.0))
        optimizer.step()

        metric_logger['loss'].update(loss.item())
        metric_logger['lr'].update(optimizer.param_groups[0]["lr"])
        
        batch_time_val = time.time() - end
        metric_logger['batch_time'].update(batch_time_val)
        end = time.time()

        if step % config['train']['log_interval'] == 0 and is_main_process():
            steps_left = num_steps_per_epoch - step
            eta_seconds = metric_logger['batch_time'].global_avg * steps_left
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            
            logger.info(
                f"{header}[{step:>{len(str(num_steps_per_epoch))}}/{num_steps_per_epoch}] "
                f"ETA: {eta_string}  "
                f"Loss: {metric_logger['loss']}  "
                f"LR: {metric_logger['lr']}  "
                f"Data: {metric_logger['data_time']}  "
                f"Batch: {metric_logger['batch_time']}"
            )
            
    return metric_logger['loss'].global_avg

# -------------------------------------------------------------------
# 主函数
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', type=str)
    # 这里的 pretrained 仅用于初始化，如果 resume 存在，会覆盖它
    parser.add_argument('--pretrained', type=str, required=True, help='Path to pretrained CWT-MAE weights')
    # 新增 resume 参数
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 【强制修改】设置新的学习率
    NEW_BASE_LR = 5e-5
    config['train']['base_lr'] = NEW_BASE_LR
    print(f"!!! Forcing Base LR to {NEW_BASE_LR} !!!")

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
    dataloader = DataLoader(
        dataset,
        batch_size=config['train']['batch_size'],
        sampler=sampler,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=True
    )

    # 初始化模型
    model = PPG2ECG_Translator(
        pretrained_path=args.pretrained,
        signal_len=config['data']['signal_len'],
        cwt_scales=config['model'].get('cwt_scales', 64),
        embed_dim=config['model']['embed_dim'],
        depth=config['model']['depth'],
        num_heads=config['model']['num_heads'],
        decoder_embed_dim=config['model']['decoder_embed_dim'],
        decoder_depth=config['model']['decoder_depth'],
        decoder_num_heads=config['model']['decoder_num_heads'],
        time_loss_weight=config['model'].get('time_loss_weight', 5.0)
    )
    model.to(device)

    # 编译模型
    if is_main_process():
        logger.info("Compiling model with torch.compile() ...")
    try:
        model = torch.compile(model)
    except Exception as e:
        logger.warning(f"torch.compile failed: {e}")

    model = DDP(model, device_ids=[gpu_id], output_device=gpu_id, find_unused_parameters=False)
    
    # 初始化优化器 (使用新的 LR)
    optimizer = optim.AdamW(model.parameters(), 
                            lr=NEW_BASE_LR, 
                            weight_decay=float(config['train']['weight_decay']))
    
    # -------------------------------------------------------------------
    # 断点续训逻辑
    # -------------------------------------------------------------------
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        if is_main_process():
            logger.info(f"Resuming from checkpoint: {args.resume}")
        
        checkpoint = torch.load(args.resume, map_location='cpu')
        
        # 1. 加载模型权重
        # 处理 DDP 和 Compile 前缀
        state_dict = checkpoint['model']
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('_orig_mod.', '') # 去除 compile 前缀
            # 注意：这里不需要去除 module. 因为当前 model 已经是 DDP 包装过的
            # 如果 checkpoint 也是 DDP 保存的，则保留 module.
            # 如果 checkpoint 不是 DDP，则需要处理。
            # 最稳妥的方式是加载到 model.module
            new_state_dict[name] = v
            
        # 加载到 DDP 内部的 module
        model.module.load_state_dict(new_state_dict, strict=False)
        
        # 2. 恢复 Epoch (可选)
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            
        # 3. 【重要】不加载优化器状态
        # 因为我们要改变 LR，所以让优化器从头开始初始化是最好的选择。
        if is_main_process():
            logger.info(f"Loaded weights. Restarting optimizer with new LR: {NEW_BASE_LR}. Start Epoch: {start_epoch}")

    # 计算总步数用于 Scheduler
    num_steps_per_epoch = len(dataloader)
    total_epochs = config['train']['epochs']
    total_steps = num_steps_per_epoch * total_epochs
    warmup_steps = num_steps_per_epoch * config['train']['warmup_epochs']
    min_lr = float(config['train']['min_lr'])

    vis_ppg, vis_ecg = next(iter(dataloader))
    vis_ppg = vis_ppg.to(device)
    vis_ecg = vis_ecg.to(device)

    total_start_time = time.time()
    
    for epoch in range(start_epoch, total_epochs):
        sampler.set_epoch(epoch)
        
        avg_loss = train_one_epoch(
            model, dataloader, optimizer, epoch, logger, config, device,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            base_lr=NEW_BASE_LR,
            min_lr=min_lr
        )
        
        if is_main_process():
            logger.info(f"Epoch {epoch} done. Avg Loss: {avg_loss:.4f}")
            save_translation_vis(model, vis_ppg, vis_ecg, epoch, config['train']['save_dir'])
            
            save_dict = {
                'model': model.module.state_dict(),
                'epoch': epoch,
                'config': config
            }
            torch.save(save_dict, os.path.join(config['train']['save_dir'], "checkpoint_trans_last.pth"))
            
            if epoch % config['train']['save_freq'] == 0:
                torch.save(save_dict, os.path.join(config['train']['save_dir'], f"checkpoint_trans_{epoch}.pth"))

    total_time = time.time() - total_start_time
    if is_main_process():
        logger.info(f"Training time: {format_time(total_time)}")
    dist.destroy_process_group()

if __name__ == '__main__':
    main()