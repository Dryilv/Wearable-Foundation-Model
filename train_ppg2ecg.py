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

# 导入你的模型和数据集
from model_ppg2ecg import PPG2ECG_Translator
from dataset_paired import PairedPhysioDataset

# 启用 TensorFloat-32
torch.set_float32_matmul_precision('high')

# -------------------------------------------------------------------
# 工具类 (保持不变)
# -------------------------------------------------------------------
class SmoothedValue(object):
    """用于平滑统计各项指标 (Loss, Time, etc.)"""
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

# -------------------------------------------------------------------
# 可视化函数
# -------------------------------------------------------------------
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
        
        # 简单的 Spec 可视化
        if isinstance(model, DDP): scales = model.module.mae.cwt_scales
        else: scales = model.mae.cwt_scales
        
        # 这里为了不引入 cwt_wrap 依赖，仅做时域展示，如果需要频域请自行取消注释并导入 cwt_wrap
        # ... (频域可视化代码略，保持简洁) ...

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"trans_epoch_{epoch}.png"))
        plt.close()
        
    model.train()

# -------------------------------------------------------------------
# 【核心修改】训练逻辑：增加时间监控和进度条
# -------------------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, epoch, logger, config, device):
    model.train()
    
    # 定义指标记录器
    metric_logger = defaultdict(lambda: SmoothedValue(window_size=20))
    metric_logger['loss'] = SmoothedValue(window_size=20, fmt='{median:.4f} ({global_avg:.4f})')
    metric_logger['data_time'] = SmoothedValue(window_size=20, fmt='{avg:.4f}') # 数据加载时间
    metric_logger['batch_time'] = SmoothedValue(window_size=20, fmt='{avg:.4f}') # 整体Batch时间
    
    header = f'Epoch: [{epoch}]'
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    num_steps = len(dataloader)
    start_time = time.time()
    end = time.time()
    
    for step, (ppg, ecg) in enumerate(dataloader):
        # 1. 记录数据加载时间 (当前时间 - 上一次循环结束时间)
        data_time_val = time.time() - end
        metric_logger['data_time'].update(data_time_val)
        
        ppg = ppg.to(device, non_blocking=True)
        ecg = ecg.to(device, non_blocking=True)

        # 2. 前向传播与反向传播
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            loss, _, _ = model(ppg, ecg_target=ecg)
        
        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['train'].get('clip_grad', 1.0))
        optimizer.step()

        # 3. 更新 Loss 记录
        metric_logger['loss'].update(loss.item())
        
        # 4. 记录 Batch 总时间
        batch_time_val = time.time() - end
        metric_logger['batch_time'].update(batch_time_val)
        end = time.time() # 重置计时器

        # 5. 日志打印
        if step % config['train']['log_interval'] == 0 and is_main_process():
            # 计算 ETA (预计剩余时间)
            steps_left = num_steps - step
            eta_seconds = metric_logger['batch_time'].global_avg * steps_left
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            
            # 打印格式：
            # Epoch: [0][  50/1000] ETA: 0:15:30  Loss: 0.5 (0.4)  Data: 0.005s  Batch: 0.100s
            logger.info(
                f"{header}[{step:>{len(str(num_steps))}}/{num_steps}] "
                f"ETA: {eta_string}  "
                f"Loss: {metric_logger['loss']}  "
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
    parser.add_argument('--pretrained', type=str, required=True, help='Path to pretrained CWT-MAE weights')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    gpu_id, rank, world_size = init_distributed_mode()
    device = torch.device(f"cuda:{gpu_id}")
    
    if is_main_process():
        Path(config['train']['save_dir']).mkdir(parents=True, exist_ok=True)
    logger = setup_logger(config['train']['save_dir'])

    # Dataset
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

    # Model
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
    model = DDP(model, device_ids=[gpu_id], output_device=gpu_id, find_unused_parameters=False)
    
    optimizer = optim.AdamW(model.parameters(), 
                            lr=float(config['train']['base_lr']), 
                            weight_decay=float(config['train']['weight_decay']))
    
    # 可视化 Batch
    vis_ppg, vis_ecg = next(iter(dataloader))
    vis_ppg = vis_ppg.to(device)
    vis_ecg = vis_ecg.to(device)

    total_start_time = time.time()
    
    for epoch in range(config['train']['epochs']):
        sampler.set_epoch(epoch)
        
        avg_loss = train_one_epoch(
            model, dataloader, optimizer, epoch, logger, config, device
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