import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' # Fix OpenMP warning
os.environ['MKL_THREADING_LAYER'] = 'GNU'   # Prevent Intel/LLVM OpenMP conflict
import sys
import argparse
import yaml
import math
import time
from pathlib import Path
from collections import defaultdict
import numpy as np

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import autocast, GradScaler

# 允许编译失败时自动回退
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# 导入你的模型和数据集
from model import CWT_MAE_RoPE
from dataset import PhysioSignalDataset, fixed_channel_collate_fn
from utils_metrics import ExperimentTracker
from utils import save_reconstruction_images, SmoothedValue, format_time, is_main_process
from utils import setup_logger, init_distributed_mode

# 启用 TensorFloat-32 (A100/3090/4090 必备加速)
torch.set_float32_matmul_precision('high') 

# -------------------------------------------------------------------
# 1. 辅助工具类 (已迁移至 utils.py)
# -------------------------------------------------------------------

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
# 5. 训练与验证逻辑
# -------------------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, scaler, epoch, logger, config, device, start_time_global, 
                    total_steps, warmup_steps, base_lr, min_lr, mask_ratio=None, lr_start_step=0):
    model.train()
    metric_logger = defaultdict(lambda: SmoothedValue(window_size=20))
    metric_logger['loss'] = SmoothedValue(window_size=20, fmt='{median:.4f} ({global_avg:.4f})')
    metric_logger['loss_spec'] = SmoothedValue(window_size=20, fmt='{median:.4f} ({global_avg:.4f})')
    metric_logger['loss_time'] = SmoothedValue(window_size=20, fmt='{median:.4f} ({global_avg:.4f})')
    metric_logger['loss_stats'] = SmoothedValue(window_size=20, fmt='{median:.4f} ({global_avg:.4f})')
    metric_logger['lr'] = SmoothedValue(window_size=1, fmt='{value:.6f}')
    metric_logger['grad_norm'] = SmoothedValue(window_size=20, fmt='{value:.2f}')
    metric_logger['throughput'] = SmoothedValue(window_size=20, fmt='{value:.2f}')
    
    header = f'Epoch: [{epoch}/{config["train"]["epochs"]}]'
    num_steps_per_epoch = len(dataloader)
    accum_iter = config['train'].get('accum_iter', 1)
    
    # [Optimization] Linear Scaling Rule
    # Note: We apply this scaling to the base_lr passed in AFTER dynamic schedule adjustment
    eff_batch_size = config['train']['batch_size'] * accum_iter * (dist.get_world_size() if dist.is_initialized() else 1)
    if config['train'].get('auto_scale_lr', True):
        # Apply scaling based on effective batch size
        base_lr_scaled = base_lr * eff_batch_size / 256.0
        min_lr_scaled = min_lr * eff_batch_size / 256.0
    else:
        base_lr_scaled = base_lr
        min_lr_scaled = min_lr
    
    # 优先使用 bfloat16
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    start_time = time.time()
    
    optimizer.zero_grad(set_to_none=True) # Initialize gradients

    for step, batch_data in enumerate(dataloader):
        step_start_time = time.time()
        global_step = epoch * num_steps_per_epoch + step

        # 【修改】适配新的返回值格式: (batch, channel_ids, labels, stats)
        batch, channel_ids, labels, stats = batch_data

        # 调整 LR (按 step 调整，考虑 accum_iter)
        if step % accum_iter == 0:
            # When resuming at epoch 20 (or 45), lr_start_step is set to epoch * num_steps_per_epoch.
            # However, global_step also counts from 0 based on epoch * num_steps_per_epoch.
            # For phase restarting logic, current_step_for_lr resets to 0.
            current_step_for_lr = global_step - lr_start_step
            # Ensure non-negative
            if current_step_for_lr < 0: current_step_for_lr = 0
            
            adjust_learning_rate_per_step(
                optimizer, 
                current_step=current_step_for_lr // accum_iter, 
                total_steps=total_steps // accum_iter, 
                warmup_steps=warmup_steps // accum_iter, 
                base_lr=base_lr_scaled, 
                min_lr=min_lr_scaled
            )

        # batch shape: (B, M, L)
        batch = batch.to(device, non_blocking=True)
        # labels = labels.to(device, non_blocking=True) # MAE 训练暂不需要标签

        # 混合精度前向传播
        # 在 DDP 模式下，如果不是最后一次累积，使用 no_sync 上下文以减少通信
        do_sync = (step + 1) % accum_iter == 0 or (step + 1) == len(dataloader)
        
        # 处理 DDP no_sync
        my_context = model.no_sync if (isinstance(model, DDP) and not do_sync) else (lambda: contextlib.nullcontext())
        
        # 注意：python < 3.7 可能不支持这种 lambda 写法，但这里环境通常较高。
        # 为了保险，直接写逻辑：
        if isinstance(model, DDP) and not do_sync:
             context_manager = model.no_sync()
        else:
             import contextlib
             context_manager = contextlib.nullcontext()

        with context_manager:
            with autocast('cuda', dtype=amp_dtype, enabled=config['train']['use_amp']):
                # 【修改】传递 channel_ids 和 stats_target
                channel_ids = channel_ids.to(device, non_blocking=True)
                stats = stats.to(device, non_blocking=True)
                loss, loss_dict, _, _, _, _, _, _ = model(batch, channel_ids, stats_target=stats, mask_ratio=mask_ratio)
                loss = loss / accum_iter # Normalize loss for accumulation

            loss_value = loss.item() * accum_iter # Restore for logging
            loss_spec_val = loss_dict.get('loss_spec', torch.tensor(0.0)).item()
            loss_time_val = loss_dict.get('loss_time', torch.tensor(0.0)).item()
            loss_stats_val = loss_dict.get('loss_stats', torch.tensor(0.0)).item()
            
            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training")
                sys.exit(1)

            # 使用 Scaler 处理反向传播 (兼容 fp16)
            scaler.scale(loss).backward()
        
        if do_sync:
            # Unscale 之后才能 clip grad
            scaler.unscale_(optimizer)
            
            # Clip Grad
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['train']['clip_grad'])
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            metric_logger['grad_norm'].update(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
        else:
            # 如果没有 sync，grad_norm 暂不更新或保持上一次的值
            pass

        # Metrics Update
        batch_size = batch.shape[0]
        step_duration = time.time() - step_start_time
        throughput = batch_size / step_duration # samples/sec per GPU
        
        metric_logger['loss'].update(loss_value)
        metric_logger['loss_spec'].update(loss_spec_val)
        metric_logger['loss_time'].update(loss_time_val)
        metric_logger['loss_stats'].update(loss_stats_val)
        metric_logger['lr'].update(optimizer.param_groups[0]["lr"])
        metric_logger['throughput'].update(throughput)

        if step % 50 == 0 and is_main_process():
            elapsed = time.time() - start_time_global
            
            # [新增] 详细梯度监控日志
            # 避免直接在循环中调用 .item() 阻塞 GPU，将监控逻辑优化为先在 GPU 收集后统一处理
            # 为了减少开销，仅提取前几个关键层的梯度
            grad_stats = []
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if p.grad is not None:
                        if 'cls_token' in name or 'patch_embed' in name:
                            # 仍然会有一定的同步，但因为只挑选了少部分参数，开销可控
                            g_norm = p.grad.detach().norm(2).item()
                            if g_norm > 1.0 or 'cls_token' in name or 'patch_embed' in name:
                                w_std = p.detach().std().item()
                                grad_stats.append(f"{name}: g={g_norm:.2f}, w_std={w_std:.4f}")
            
            if grad_stats:
                logger.info(f"High Grads (>1.0) & Key Layers:\n" + "\n".join(grad_stats[:10])) # 限制输出行数
            
            logger.info(
                f"{header} Step: [{step}/{num_steps_per_epoch}] "
                f"Loss: {metric_logger['loss']} "
                f"Spec: {metric_logger['loss_spec']} "
                f"Time: {metric_logger['loss_time']} "
                f"Stats: {metric_logger['loss_stats']} "
                f"LR: {metric_logger['lr']} "
                f"Grad: {metric_logger['grad_norm']} "
                f"Speed: {metric_logger['throughput'].avg:.1f} samples/s "
                f"Elapsed: {format_time(elapsed)}"
            )
            
    if is_main_process():
        logger.info(f"Epoch {epoch} done. Avg Loss: {metric_logger['loss'].global_avg:.4f}")
    
    # Return metrics dict
    return {
        'loss': metric_logger['loss'].global_avg,
        'loss_spec': metric_logger['loss_spec'].global_avg,
        'loss_time': metric_logger['loss_time'].global_avg,
        'loss_stats': metric_logger['loss_stats'].global_avg,
        'grad_norm': metric_logger['grad_norm'].global_avg,
        'throughput': metric_logger['throughput'].global_avg * (dist.get_world_size() if dist.is_initialized() else 1)
    }

# -------------------------------------------------------------------
# 6. 主函数
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', type=str)
    args = parser.parse_args()
    
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    gpu_id, rank, world_size = init_distributed_mode()
    device = torch.device(f"cuda:{gpu_id}")
    
    seed = 42 + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True # 开启 cudnn.benchmark 以加速固定输入尺寸的卷积

    if is_main_process():
        Path(config['train']['save_dir']).mkdir(parents=True, exist_ok=True)
        # 初始化 Tracker
        tracker = ExperimentTracker(config['train']['save_dir'])
    
    logger = setup_logger(config['train']['save_dir'], name="CWT-MAE")

    # 1. Dataset - 使用全部数据作为训练集
    train_dataset = PhysioSignalDataset(
        index_file=config['data']['index_path'],
        indices=None,  # None 表示使用全部数据
        signal_len=config['data']['signal_len'],
        mode='train',
        data_ratio=config['model'].get('data_ratio', 1.0),
        use_sliding_window=config['data'].get('use_sliding_window', False),
        window_stride=config['data'].get('window_stride', 500)
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    # DataLoader (使用自定义 collate_fn)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        sampler=train_sampler,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=True,
        collate_fn=fixed_channel_collate_fn
    )
    
    # 创建可视化专用 DataLoader (batch_size=1, shuffle=True)
    vis_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        collate_fn=fixed_channel_collate_fn
    ) if is_main_process() else None

    num_steps_per_epoch = len(train_dataloader)
    total_epochs = config['train']['epochs']
    warmup_epochs = config['train']['warmup_epochs']
    
    total_steps = num_steps_per_epoch * total_epochs
    warmup_steps = num_steps_per_epoch * warmup_epochs
    
    base_lr = float(config['train']['base_lr'])
    min_lr = float(config['train']['min_lr'])
    
    if is_main_process():
        logger.info(f"Total Steps: {total_steps}, Warmup Steps: {warmup_steps}")
        logger.info(f"Base LR: {base_lr}, Min LR: {min_lr}")
        logger.info(f"Train Size: {len(train_dataset)}")

    # 初始化模型 (SingleTowerContrastiveMAE)
    # 构造 base_model_config
    base_model_config = {
        'signal_len': config['data']['signal_len'],
        'cwt_scales': config['model'].get('cwt_scales', 64),
        'patch_size_time': config['model'].get('patch_size_time', 50),
        'patch_size_freq': config['model'].get('patch_size_freq', 4),
        'embed_dim': config['model']['embed_dim'],
        'depth': config['model']['depth'],
        'num_heads': config['model']['num_heads'],
        'decoder_embed_dim': config['model']['decoder_embed_dim'],
        'decoder_depth': config['model']['decoder_depth'],
        'decoder_num_heads': config['model']['decoder_num_heads'],
        'mask_ratio': config['model'].get('mask_ratio', 0.75),
        'time_loss_weight': config['model'].get('time_loss_weight', 1.0),
        'use_diff': config['model'].get('use_diff', False),
        'diff_loss_weight': config['model'].get('diff_loss_weight', None),
        'stats_loss_weight': config['model'].get('stats_loss_weight', 1.0)
    }

    if is_main_process():
        logger.info("Initializing CWT_MAE_RoPE...")
    
    model = CWT_MAE_RoPE(**base_model_config)
    model.to(device)

    # 编译模型 (TITAN V 不完全支持编译特性，建议通过 config 控制)
    if config['train'].get('use_compile', False):
        try:
            model = torch.compile(model)
            if is_main_process():
                logger.info("Model compiled with torch.compile()")
        except Exception as e:
            if is_main_process():
                logger.warning(f"Could not compile model: {e}")
    else:
        if is_main_process():
            logger.info("torch.compile() is disabled via config.")

    model = DDP(model, device_ids=[gpu_id], output_device=gpu_id, find_unused_parameters=True) if dist.is_initialized() else model

    # 优化器参数分组
    base_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        base_params.append(param)
            
    if is_main_process():
        logger.info(f"Optimizer groups: Base Params={len(base_params)}")

    optimizer = optim.AdamW([
        {'params': base_params, 'lr': base_lr}
    ], weight_decay=float(config['train']['weight_decay']))
    
    # GradScaler 用于混合精度
    scaler = GradScaler(enabled=config['train']['use_amp'])
    
    # [新增] 梯度累积参数
    accum_iter = config['train'].get('accum_iter', 1)
    if is_main_process():
        logger.info(f"Gradient Accumulation Steps: {accum_iter}")
        logger.info(f"Effective Batch Size: {config['train']['batch_size'] * accum_iter * world_size}")
    
    start_epoch = 0
    if config['train']['resume'] and os.path.isfile(config['train']['resume']):
        checkpoint = torch.load(config['train']['resume'], map_location='cpu')
        
        # 处理 DDP state_dict 加载
        state_dict = checkpoint['model']
        if not dist.is_initialized():
            # 如果当前不是 DDP，但 checkpoint 是 DDP 保存的，去掉 'module.' 前缀
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace('module.', '')
                new_state_dict[name] = v
            msg = model.load_state_dict(new_state_dict, strict=False)
        else:
            msg = model.module.load_state_dict(state_dict, strict=False)
            
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        if is_main_process():
            logger.info(f"Resumed from epoch {start_epoch}")
            if msg.missing_keys:
                logger.warning(f"Missing keys when resuming (expected if adding new modules like stats_pred_head): {msg.missing_keys}")

    start_time_global = time.time()
    
    for epoch in range(start_epoch, total_epochs):
        train_sampler.set_epoch(epoch)
        
        # --- Fixed Mask Ratio (No Curriculum) ---
        current_mask_ratio = config['model'].get('mask_ratio', 0.75)

        # --- Standard Learning Rate Scheduling ---
        current_base_lr = base_lr
        current_total_steps = total_steps
        current_warmup_steps = warmup_steps
        current_lr_start_step = 0
        
        if is_main_process():
            logger.info(f"Epoch {epoch} Configuration: Mask Ratio = {current_mask_ratio:.4f}, Base LR Target = {current_base_lr}")
            
        # Train
        train_metrics = train_one_epoch(
            model, train_dataloader, optimizer, scaler, epoch, logger, config, device, start_time_global,
            total_steps=current_total_steps,
            warmup_steps=current_warmup_steps,
            base_lr=current_base_lr,
            min_lr=min_lr,
            mask_ratio=current_mask_ratio,
            lr_start_step=current_lr_start_step
        )

        if is_main_process():
            # Log Metrics
            metrics_dict = {
                'train_loss': train_metrics['loss'],
                'loss_spec': train_metrics['loss_spec'],
                'loss_time': train_metrics['loss_time'],
                'loss_stats': train_metrics['loss_stats'],
                'grad_norm': train_metrics['grad_norm'],
                'gpu_mem_mb': torch.cuda.max_memory_allocated() / 1024 / 1024,
                'throughput': train_metrics['throughput'],
            }
            tracker.log(epoch, metrics_dict)
            logger.info(f"Epoch {epoch} Metrics: {metrics_dict}")

            # Early Stopping Check
            if tracker.check_early_stopping(patience=3):
                logger.info("Early stopping triggered due to no improvement in feature quality.")
                # break # 取消注释以启用

            # 保存可视化 - 随机抽取一个样本
            if vis_dataloader is not None:
                try:
                    # 随机获取一个样本进行可视化
                    vis_batch_data = next(iter(vis_dataloader))
                    vis_batch = vis_batch_data[0].to(device)
                    vis_channel_ids = vis_batch_data[1].to(device)
                    
                    # 单通道可视化：需要 channel_ids
                    # 1. 提取 Encoder (CWT_MAE_RoPE)
                    real_model = model.module if hasattr(model, 'module') else model
                    if hasattr(real_model, 'encoder'):
                        encoder_model = real_model.encoder
                    else:
                        encoder_model = real_model

                    # 2. 传入单通道数据和 channel_ids 进行可视化
                    save_reconstruction_images(
                        encoder_model,
                        vis_batch,  # (B, 1, L)
                        vis_channel_ids,  # (B,)
                        epoch,
                        config['train']['save_dir']
                    )
                except Exception as e:
                    logger.warning(f"Failed to get batch for visualization at epoch {epoch}: {e}")
            
            # 保存 Checkpoint
            save_dict = {
                'model': model.module.state_dict() if dist.is_initialized() else model.state_dict(),
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
