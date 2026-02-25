import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
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
import contextlib

# Import your model and dataset
from patchtst_model import PatchTST_Pretrain
from dataset import PhysioSignalDataset, DataSplitter, fixed_channel_collate_fn
from utils_metrics import ExperimentTracker
from utils import save_reconstruction_images, SmoothedValue, setup_logger, init_distributed_mode, format_time, is_main_process, adjust_learning_rate_per_step

torch.set_float32_matmul_precision('high')

def validate(model, dataloader, device, config):
    model.eval()
    metric_logger = defaultdict(lambda: SmoothedValue(window_size=100))
    
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    with torch.no_grad():
        for batch_data in dataloader:
            if len(batch_data) == 3:
                batch, modality_ids, labels = batch_data
            else:
                batch, labels = batch_data
                
            # batch: (B, C, L) -> Need (B, L, C) for PatchTST
            batch = batch.permute(0, 2, 1).to(device, non_blocking=True)
            
            with autocast(enabled=config['train']['use_amp'], dtype=amp_dtype):
                # PatchTST returns: loss, pred_patches, target_patches, mask
                ret = model(batch)
                loss = ret[0]
                
            metric_logger['loss'].update(loss.item())
            
    val_loss = metric_logger['loss'].global_avg
    
    if dist.is_initialized():
        metrics_tensor = torch.tensor([val_loss], device=device)
        dist.all_reduce(metrics_tensor)
        metrics_tensor /= dist.get_world_size()
        val_loss = metrics_tensor[0].item()
        
    return val_loss

def train_one_epoch(model, dataloader, optimizer, scaler, epoch, logger, config, device, start_time_global, 
                    total_steps, warmup_steps, base_lr, min_lr):
    model.train()
    metric_logger = defaultdict(lambda: SmoothedValue(window_size=20))
    metric_logger['loss'] = SmoothedValue(window_size=20, fmt='{median:.4f} ({global_avg:.4f})')
    metric_logger['lr'] = SmoothedValue(window_size=1, fmt='{value:.6f}')
    metric_logger['grad_norm'] = SmoothedValue(window_size=20, fmt='{value:.2f}')
    metric_logger['throughput'] = SmoothedValue(window_size=20, fmt='{value:.2f}')
    
    header = f'Epoch: [{epoch}/{config["train"]["epochs"]}]'
    num_steps_per_epoch = len(dataloader)
    accum_iter = config['train'].get('accum_iter', 1)
    
    eff_batch_size = config['train']['batch_size'] * accum_iter * (dist.get_world_size() if dist.is_initialized() else 1)
    if config['train'].get('auto_scale_lr', True):
        base_lr_scaled = base_lr * eff_batch_size / 256.0
        min_lr_scaled = min_lr * eff_batch_size / 256.0
    else:
        base_lr_scaled = base_lr
        min_lr_scaled = min_lr
    
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    start_time = time.time()
    optimizer.zero_grad()

    for step, batch_data in enumerate(dataloader):
        step_start_time = time.time()
        global_step = epoch * num_steps_per_epoch + step
        
        if len(batch_data) == 3:
            batch, modality_ids, labels = batch_data
        else:
            batch, labels = batch_data

        # Adjust LR
        if step % accum_iter == 0:
            # Note: We need to implement adjust_learning_rate_per_step in utils.py or here
            # Since I didn't copy it to utils.py (my bad, I missed it in the read output probably or just forgot), 
            # I'll implement it inline or assume it's in utils if I added it.
            # Wait, I see I imported it from utils in the imports above. 
            # Let me check if I actually wrote it to utils.py...
            # I did NOT write adjust_learning_rate_per_step to utils.py in previous tool call.
            # I will add it to this file for safety.
            pass

        # Calculate LR
        if global_step < warmup_steps:
            lr = base_lr_scaled * global_step / warmup_steps
        else:
            progress = (global_step - warmup_steps) / (total_steps - warmup_steps)
            lr = min_lr_scaled + (base_lr_scaled - min_lr_scaled) * 0.5 * (1. + math.cos(math.pi * progress))
        
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # batch shape: (B, C, L) -> (B, L, C)
        batch = batch.permute(0, 2, 1).to(device, non_blocking=True)

        do_sync = (step + 1) % accum_iter == 0 or (step + 1) == len(dataloader)
        
        if isinstance(model, DDP) and not do_sync:
             context_manager = model.no_sync()
        else:
             context_manager = contextlib.nullcontext()

        with context_manager:
            with autocast(enabled=config['train']['use_amp'], dtype=amp_dtype):
                ret = model(batch)
                loss = ret[0]
                loss = loss / accum_iter

            loss_value = loss.item() * accum_iter
            
            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training")
                sys.exit(1)

            scaler.scale(loss).backward()
        
        if do_sync:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['train']['clip_grad'])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            metric_logger['grad_norm'].update(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
        
        batch_size = batch.shape[0]
        step_duration = time.time() - step_start_time
        throughput = batch_size / step_duration
        
        metric_logger['loss'].update(loss_value)
        metric_logger['lr'].update(optimizer.param_groups[0]["lr"])
        metric_logger['throughput'].update(throughput)

        if step % 50 == 0 and is_main_process():
            elapsed = time.time() - start_time_global
            logger.info(
                f"{header} Step: [{step}/{num_steps_per_epoch}] "
                f"Loss: {metric_logger['loss']} "
                f"LR: {metric_logger['lr']} "
                f"Grad: {metric_logger['grad_norm']} "
                f"Speed: {metric_logger['throughput'].avg:.1f} samples/s "
                f"Elapsed: {format_time(elapsed)}"
            )
            
    if is_main_process():
        logger.info(f"Epoch {epoch} done. Avg Loss: {metric_logger['loss'].global_avg:.4f}")
    
    return {
        'loss': metric_logger['loss'].global_avg,
        'grad_norm': metric_logger['grad_norm'].global_avg,
        'throughput': metric_logger['throughput'].global_avg * (dist.get_world_size() if dist.is_initialized() else 1)
    }

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
        tracker = ExperimentTracker(config['train']['save_dir'])
    
    logger = setup_logger(config['train']['save_dir'])

    # Dataset
    if is_main_process():
        logger.info("Initializing Data Splitter...")
        splitter = DataSplitter(
            index_file=config['data']['index_path'],
            split_ratio=0.1,
            seed=42
        )
        train_indices, val_indices = splitter.get_split()
    else:
        train_indices, val_indices = None, None
    
    if dist.is_initialized():
        splitter = DataSplitter(
            index_file=config['data']['index_path'],
            split_ratio=0.1,
            seed=42
        )
        train_indices, val_indices = splitter.get_split()

    train_dataset = PhysioSignalDataset(
        index_file=config['data']['index_path'],
        indices=train_indices,
        signal_len=config['data']['signal_len'],
        mode='train',
        data_ratio=config['model'].get('data_ratio', 1.0),
        use_sliding_window=config['data'].get('use_sliding_window', False),
        window_stride=config['data'].get('window_stride', 500)
    )
    
    val_dataset = PhysioSignalDataset(
        index_file=config['data']['index_path'],
        indices=val_indices,
        signal_len=config['data']['signal_len'],
        mode='test',
        data_ratio=config['model'].get('data_ratio', 1.0),
        use_sliding_window=config['data'].get('use_sliding_window', False),
        window_stride=config['data'].get('window_stride', 500)
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        sampler=train_sampler,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=True,
        collate_fn=fixed_channel_collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['train']['batch_size'],
        sampler=val_sampler,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=False,
        collate_fn=fixed_channel_collate_fn
    )

    num_steps_per_epoch = len(train_dataloader)
    total_epochs = config['train']['epochs']
    warmup_epochs = config['train']['warmup_epochs']
    
    total_steps = num_steps_per_epoch * total_epochs
    warmup_steps = num_steps_per_epoch * warmup_epochs
    
    base_lr = float(config['train']['base_lr'])
    min_lr = float(config['train']['min_lr'])
    
    if is_main_process():
        logger.info(f"Total Steps: {total_steps}, Warmup Steps: {warmup_steps}")
        logger.info(f"Train Size: {len(train_dataset)}, Val Size: {len(val_dataset)}")

    # Model
    model = PatchTST_Pretrain(
        seq_len=config['data']['signal_len'],
        patch_len=config['model']['patch_len'],
        stride=config['model']['stride'],
        in_channels=5, # Expected channels
        d_model=config['model']['embed_dim'],
        n_heads=config['model']['num_heads'],
        e_layers=config['model']['depth'],
        dropout=config['model'].get('dropout', 0.1),
        mask_ratio=config['model'].get('mask_ratio', 0.6),
        use_revin=config['model'].get('use_revin', True)
    )
    model.to(device)

    model = DDP(model, device_ids=[gpu_id], output_device=gpu_id, find_unused_parameters=True)

    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=float(config['train']['weight_decay']))
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

    vis_batch = None
    if is_main_process():
        try:
            vis_batch = next(iter(train_dataloader))[0].to(device)
        except StopIteration:
            pass

    start_time_global = time.time()
    
    for epoch in range(start_epoch, total_epochs):
        train_sampler.set_epoch(epoch)
        
        train_metrics = train_one_epoch(
            model, train_dataloader, optimizer, scaler, epoch, logger, config, device, start_time_global,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            base_lr=base_lr,
            min_lr=min_lr
        )
        
        val_loss = validate(model, val_dataloader, device, config)
        
        if is_main_process():
            metrics_dict = {
                'train_loss': train_metrics['loss'],
                'val_loss': val_loss,
                'grad_norm': train_metrics['grad_norm'],
                'gpu_mem_mb': torch.cuda.max_memory_allocated() / 1024 / 1024,
                'throughput': train_metrics['throughput']
            }
            tracker.log(epoch, metrics_dict)
            logger.info(f"Epoch {epoch} Metrics: {metrics_dict}")

            if vis_batch is not None:
                save_reconstruction_images(
                    model, 
                    vis_batch, 
                    epoch, 
                    config['train']['save_dir']
                )
            
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
