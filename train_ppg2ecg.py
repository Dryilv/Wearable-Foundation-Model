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

# 导入你之前定义的模型 (假设保存在 model_ppg2ecg.py)
from model_ppg2ecg import PPG2ECG_Translator
# 导入新的成对数据集
from dataset_paired import PairedPhysioDataset

torch.set_float32_matmul_precision('high')

# -------------------------------------------------------------------
# 工具函数 (保持不变)
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

# -------------------------------------------------------------------
# 可视化函数 (专门针对翻译任务)
# -------------------------------------------------------------------
def save_translation_vis(model, ppg_batch, ecg_batch, epoch, save_dir):
    model.eval()
    with torch.no_grad():
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with autocast(dtype=amp_dtype):
            # 推理模式，不需要 target
            pred_time, pred_spec = model(ppg_batch, ecg_target=None)
        
        idx = 0 
        
        # 数据准备 (转 numpy)
        ppg_signal = ppg_batch[idx].squeeze().float().cpu().numpy()
        ecg_target = ecg_batch[idx].squeeze().float().cpu().numpy()
        ecg_pred = pred_time[idx].float().cpu().numpy()
        
        # 简单的反归一化用于显示 (可选，这里直接显示归一化后的波形对比形态)
        
        plt.figure(figsize=(15, 12))
        
        # 1. Input PPG
        plt.subplot(4, 1, 1)
        plt.plot(ppg_signal, color='green', alpha=0.8)
        plt.title("Input: PPG Signal")
        plt.grid(True, alpha=0.3)
        
        # 2. Target ECG vs Pred ECG (Time Domain)
        plt.subplot(4, 1, 2)
        plt.plot(ecg_target, label='Ground Truth ECG', color='black', alpha=0.6)
        plt.plot(ecg_pred, label='Generated ECG', color='red', alpha=0.8, linestyle='--')
        plt.title(f"Epoch {epoch} - ECG Reconstruction")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Target ECG Spec
        # 需要手动计算一下 Target 的 Spec 用于对比
        from model import cwt_wrap # 假设能引用到
        ecg_target_tensor = ecg_batch[idx:idx+1]
        # 注意：这里需要和模型内部参数一致
        if isinstance(model, DDP):
            scales = model.module.mae.cwt_scales
        else:
            scales = model.mae.cwt_scales
            
        target_cwt = cwt_wrap(ecg_target_tensor, num_scales=scales, lowest_scale=0.1, step=1.0)
        target_spec_img = target_cwt[0, 0].float().cpu().numpy() # Channel 0
        
        plt.subplot(4, 1, 3)
        plt.imshow(target_spec_img, aspect='auto', origin='lower', cmap='jet')
        plt.title("Target ECG Spectrogram")
        plt.colorbar()

        # 4. Pred ECG Spec
        # pred_spec 是 Patch 形式，需要还原。
        # 为了简单起见，我们直接对生成的时域信号再做一次 CWT 用于可视化
        pred_cwt = cwt_wrap(pred_time[idx:idx+1].unsqueeze(1), num_scales=scales, lowest_scale=0.1, step=1.0)
        pred_spec_img = pred_cwt[0, 0].float().cpu().numpy()
        
        plt.subplot(4, 1, 4)
        plt.imshow(pred_spec_img, aspect='auto', origin='lower', cmap='jet')
        plt.title("Generated ECG Spectrogram (Computed from Pred Signal)")
        plt.colorbar()

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"trans_epoch_{epoch}.png"))
        plt.close()
        
    model.train()

# -------------------------------------------------------------------
# 训练逻辑
# -------------------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, epoch, logger, config, device, start_time_global):
    model.train()
    metric_logger = defaultdict(lambda: SmoothedValue(window_size=20))
    metric_logger['loss'] = SmoothedValue(window_size=20, fmt='{median:.4f}')
    
    header = f'Epoch: [{epoch}]'
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    for step, (ppg, ecg) in enumerate(dataloader):
        ppg = ppg.to(device, non_blocking=True)
        ecg = ecg.to(device, non_blocking=True)

        with autocast(dtype=amp_dtype):
            # Forward: 传入 PPG 和 Target ECG
            loss, _, _ = model(ppg, ecg_target=ecg)
        
        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['train'].get('clip_grad', 1.0))
        optimizer.step()

        metric_logger['loss'].update(loss.item())

        if step % 50 == 0 and is_main_process():
            elapsed = time.time() - start_time_global
            logger.info(f"{header} Step: {step} Loss: {metric_logger['loss']} Elapsed: {format_time(elapsed)}")
            
    return metric_logger['loss'].global_avg

# -------------------------------------------------------------------
# 主函数
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', type=str)
    # 必须指定预训练权重路径
    parser.add_argument('--pretrained', type=str, required=True, help='Path to pretrained CWT-MAE weights')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    gpu_id, rank, world_size = init_distributed_mode()
    device = torch.device(f"cuda:{gpu_id}")
    
    if is_main_process():
        Path(config['train']['save_dir']).mkdir(parents=True, exist_ok=True)
    logger = setup_logger(config['train']['save_dir'])

    # 1. 初始化成对数据集
    dataset = PairedPhysioDataset(
        index_file=config['data']['index_path'],
        signal_len=config['data']['signal_len'],
        mode='train',
        row_ppg=4, # 你的数据第一行
        row_ecg=0  # 你的数据第五行
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

    # 2. 初始化翻译模型
    model = PPG2ECG_Translator(
        pretrained_path=args.pretrained, # 加载预训练权重
        signal_len=config['data']['signal_len'],
        cwt_scales=config['model'].get('cwt_scales', 64),
        embed_dim=config['model']['embed_dim'],
        depth=config['model']['depth'],
        num_heads=config['model']['num_heads'],
        decoder_embed_dim=config['model']['decoder_embed_dim'],
        decoder_depth=config['model']['decoder_depth'],
        decoder_num_heads=config['model']['decoder_num_heads'],
        time_loss_weight=5.0 # 建议调大时域 Loss 权重，保证波形准确
    )
    model.to(device)
    
    # 冻结 Encoder (可选，建议先冻结训练几个 Epoch，再解冻)
    # model._freeze_encoder() 

    model = DDP(model, device_ids=[gpu_id], output_device=gpu_id, find_unused_parameters=False)
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    
    # 固定一个 Batch 用于可视化
    vis_ppg, vis_ecg = next(iter(dataloader))
    vis_ppg = vis_ppg.to(device)
    vis_ecg = vis_ecg.to(device)

    start_time_global = time.time()
    
    for epoch in range(config['train']['epochs']):
        sampler.set_epoch(epoch)
        
        avg_loss = train_one_epoch(
            model, dataloader, optimizer, epoch, logger, config, device, start_time_global
        )
        
        if is_main_process():
            logger.info(f"Epoch {epoch} done. Avg Loss: {avg_loss:.4f}")
            
            # 保存可视化
            save_translation_vis(model, vis_ppg, vis_ecg, epoch, config['train']['save_dir'])
            
            # 保存权重
            save_dict = {
                'model': model.module.state_dict(),
                'epoch': epoch,
            }
            torch.save(save_dict, os.path.join(config['train']['save_dir'], "checkpoint_trans_last.pth"))

    dist.destroy_process_group()

if __name__ == '__main__':
    main()