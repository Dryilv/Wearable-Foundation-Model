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
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler

import torch._dynamo
torch._dynamo.config.suppress_errors = True

# 导入 v4 的对比学习模型
from model import CWT_Contrastive_RoPE
from dataset import PhysioSignalDataset

torch.set_float32_matmul_precision('high') 

# -------------------------------------------------------------------
# 1. 辅助工具类 (保持不变)
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
    logger = logging.getLogger("CWT-Contrastive")
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
        return gpu, rank, world_size
    else:
        return 0, 0, 1

# -------------------------------------------------------------------
# 2. 对比学习专用的 Collate Function
# -------------------------------------------------------------------
def contrastive_collate_fn(batch):
    """
    batch: List of ((view1, view2), label)
    view1/view2: Tensor (M_i, L)
    """
    # 1. 解包
    views_pairs = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # 2. 分离视图
    views1 = [p[0] for p in views_pairs]
    views2 = [p[1] for p in views_pairs]
    
    # 3. Padding (处理变长通道)
    # 假设一个 batch 内所有样本的通道数可能不同，需要 pad 到 max_m
    max_m = max([v.shape[0] for v in views1])
    signal_len = views1[0].shape[1]
    batch_size = len(batch)
    
    def pad_stack(views):
        padded = torch.zeros((batch_size, max_m, signal_len), dtype=views[0].dtype)
        for i, v in enumerate(views):
            m = v.shape[0]
            padded[i, :m, :] = v
        return padded
        
    padded_views1 = pad_stack(views1)
    padded_views2 = pad_stack(views2)
    
    # labels
    labels = torch.stack(labels)
    
    return (padded_views1, padded_views2), labels

# -------------------------------------------------------------------
# 3. 学习率调度器
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
# 4. 训练逻辑 (Contrastive)
# -------------------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, scaler, epoch, logger, config, device, start_time_global, 
                    total_steps, warmup_steps, base_lr, min_lr):
    model.train()
    metric_logger = defaultdict(lambda: SmoothedValue(window_size=20))
    metric_logger['loss'] = SmoothedValue(window_size=20, fmt='{median:.4f} ({global_avg:.4f})')
    metric_logger['lr'] = SmoothedValue(window_size=1, fmt='{value:.6f}')
    
    header = f'Epoch: [{epoch}/{config["train"]["epochs"]}]'
    num_steps_per_epoch = len(dataloader)
    
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    for step, ((view1, view2), _) in enumerate(dataloader):
        global_step = epoch * num_steps_per_epoch + step
        
        adjust_learning_rate_per_step(
            optimizer, 
            current_step=global_step, 
            total_steps=total_steps, 
            warmup_steps=warmup_steps, 
            base_lr=base_lr, 
            min_lr=min_lr
        )

        view1 = view1.to(device, non_blocking=True)
        view2 = view2.to(device, non_blocking=True)
        
        # 混合精度
        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=config['train']['use_amp']):
            # forward(x1, x2) 返回 loss, z1, z2
            loss, _, _ = model(view1, view2)
        
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
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

    # Dataset (Contrastive Mode)
    dataset = PhysioSignalDataset(
        index_file=config['data']['index_path'],
        signal_len=config['data']['signal_len'],
        stride=config['data'].get('stride'),
        original_len=config['data'].get('original_len', 3000),
        mode='train',
        contrastive_mode=True # 开启双视图增强
    )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['train']['batch_size'],
        sampler=sampler,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=True,
        collate_fn=contrastive_collate_fn 
    )

    num_steps_per_epoch = len(dataloader)
    total_epochs = config['train']['epochs']
    warmup_epochs = config['train']['warmup_epochs']
    total_steps = num_steps_per_epoch * total_epochs
    warmup_steps = num_steps_per_epoch * warmup_epochs
    
    base_lr = float(config['train']['base_lr'])
    min_lr = float(config['train']['min_lr'])

    # 初始化 v4 对比学习模型
    model = CWT_Contrastive_RoPE(
        signal_len=config['data']['signal_len'],
        patch_size=config['model'].get('patch_size', 4),
        embed_dim=config['model']['embed_dim'],
        depth=config['model']['depth'],
        num_heads=config['model']['num_heads'],
        mlp_rank_ratio=config['model'].get('mlp_rank_ratio', 0.5),
        projection_dim=config['model'].get('projection_dim', 128),
        temperature=config['model'].get('temperature', 0.1)
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
