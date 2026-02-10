import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' # Fix OpenMP warning
os.environ['MKL_THREADING_LAYER'] = 'GNU'   # Prevent Intel/LLVM OpenMP conflict
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
from torch.amp import autocast, GradScaler

# 允许编译失败时自动回退
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# 导入你的模型和数据集
from model import CWT_MAE_RoPE 
from dataset import PhysioSignalDataset, DataSplitter, fixed_channel_collate_fn
from utils_metrics import ExperimentTracker, train_linear_probe, evaluate_features_quality
from utils import save_reconstruction_images

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
        dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank, timeout=datetime.timedelta(minutes=120))
        dist.barrier()
        print(f"| distributed init (rank {rank}): success")
        return gpu, rank, world_size
    else:
        print('Not using distributed mode')
        return 0, 0, 1

# -------------------------------------------------------------------
# 2. 关键：处理变长通道的 Collate Function
# -------------------------------------------------------------------
# def variable_channel_collate_fn(batch):
#     """
#     [Deprecated] 处理 Batch 中不同样本通道数不一致的情况。
#     现在使用 dataset.py 中的 fixed_channel_collate_fn
#     """
#     pass



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
def validate(model, dataloader, device, config):
    model.eval()
    metric_logger = defaultdict(lambda: SmoothedValue(window_size=100))
    header = 'Test:'
    
    # 优先使用 bfloat16
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    with torch.no_grad():
        for batch, labels in dataloader:
            batch = batch.to(device, non_blocking=True)
            
            with autocast('cuda', dtype=amp_dtype, enabled=config['train']['use_amp']):
                loss, _, _, _, _ = model(batch)
            metric_logger['loss'].update(loss.item())
            
    # Gather metrics from all processes
    # 这里简单起见，只看主进程的，或者依赖 SmoothedValue 的 global_avg (如果它是跨进程同步的？当前实现不是)
    # 严格来说应该在这里做 all_reduce，但对于验证集 Loss，单进程近似通常足够，除非数据分布极不均匀
    # 为了准确，我们在 SmoothedValue 外面做一次 reduce
    
    val_loss = metric_logger['loss'].global_avg
    if dist.is_initialized():
        val_loss_tensor = torch.tensor(val_loss, device=device)
        dist.all_reduce(val_loss_tensor)
        val_loss = val_loss_tensor.item() / dist.get_world_size()
        
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
    
    # 优先使用 bfloat16
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    start_time = time.time()
    
    for step, (batch, labels) in enumerate(dataloader):
        step_start_time = time.time()
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
        with autocast('cuda', dtype=amp_dtype, enabled=config['train']['use_amp']):
            loss, _, _, _, _ = model(batch)
        
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        
        # 使用 Scaler 处理反向传播 (兼容 fp16)
        scaler.scale(loss).backward()
        
        # Unscale 之后才能 clip grad
        scaler.unscale_(optimizer)
        
        # 计算 Gradient Norm
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['train']['clip_grad'])
        
        scaler.step(optimizer)
        scaler.update()

        # Metrics Update
        batch_size = batch.shape[0]
        step_duration = time.time() - step_start_time
        throughput = batch_size / step_duration # samples/sec per GPU
        
        metric_logger['loss'].update(loss_value)
        metric_logger['lr'].update(optimizer.param_groups[0]["lr"])
        metric_logger['grad_norm'].update(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
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
    
    # Return metrics dict
    return {
        'loss': metric_logger['loss'].global_avg,
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
    
    with open(args.config, 'r') as f:
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
    
    logger = setup_logger(config['train']['save_dir'])

    # 1. Dataset Split Strategy
    if is_main_process():
        logger.info("Initializing Data Splitter...")
        splitter = DataSplitter(
            index_file=config['data']['index_path'],
            split_ratio=0.1, # 10% 验证集
            seed=42 # 固定 Seed
        )
        train_indices, val_indices = splitter.get_split()
    else:
        train_indices, val_indices = None, None
    
    # Broadcast indices to other ranks
    if dist.is_initialized():
        # 简单起见，这里让每个 rank 都重新计算 split (DataSplitter 是确定性的)
        # 或者 broadcast，这里选用重新计算方式，只要 seed 一致
        splitter = DataSplitter(
            index_file=config['data']['index_path'],
            split_ratio=0.1,
            seed=42
        )
        # 这里不需要再次保存 metadata，get_split 内部会处理
        train_indices, val_indices = splitter.get_split()

    # Dataset
    train_dataset = PhysioSignalDataset(
        index_file=config['data']['index_path'], # 实际上 Dataset 内部会再次加载 full index，优化空间：传入 index_data
        indices=train_indices,
        signal_len=config['data']['signal_len'],
        mode='train',
        data_ratio=config['model'].get('data_ratio', 1.0), # 传入 data_ratio
        use_sliding_window=config['data'].get('use_sliding_window', False),
        window_stride=config['data'].get('window_stride', 500)
    )
    
    val_dataset = PhysioSignalDataset(
        index_file=config['data']['index_path'],
        indices=val_indices,
        signal_len=config['data']['signal_len'],
        mode='test', # Val use test mode (deterministic crop)
        data_ratio=config['model'].get('data_ratio', 1.0), # 传入 data_ratio (通常 Val 也要采样吗？这里假设是)
        use_sliding_window=config['data'].get('use_sliding_window', False),
        window_stride=config['data'].get('window_stride', 500)
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
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
        logger.info(f"Base LR: {base_lr}, Min LR: {min_lr}")
        logger.info(f"Train Size: {len(train_dataset)}, Val Size: {len(val_dataset)}")

    # 初始化模型 (无 max_num_channels)
    model = CWT_MAE_RoPE(
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
        mask_ratio=config['model'].get('mask_ratio', 0.75),
        mlp_rank_ratio=config['model'].get('mlp_rank_ratio', 0.5),
        time_loss_weight=config['model'].get('time_loss_weight', 1.0),
        use_conv_stem=config['model'].get('use_conv_stem', False),
        use_factorized_attn=config['model'].get('use_factorized_attn', True)
    )
    model.to(device)

    # 编译模型
    try:
         # 尝试使用 JIT script 编译 Time Attention 部分 (如果可行)
         # 或者整个模型使用 torch.compile (PyTorch 2.0+)
         # 这里演示 torch.compile，它是比 JIT 更现代的方案
        model = torch.compile(model)
        if is_main_process():
            logger.info("Model compiled with torch.compile()")
    except Exception as e:
        if is_main_process():
            logger.warning(f"Could not compile model: {e}")

    model = DDP(model, device_ids=[gpu_id], output_device=gpu_id, find_unused_parameters=True)

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
            vis_batch, _ = next(iter(train_dataloader)) # Use train loader for vis
            vis_batch = vis_batch.to(device)
        except StopIteration:
            pass

    start_time_global = time.time()
    
    for epoch in range(start_epoch, total_epochs):
        train_sampler.set_epoch(epoch)
        
        # Train
        train_metrics = train_one_epoch(
            model, train_dataloader, optimizer, scaler, epoch, logger, config, device, start_time_global,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            base_lr=base_lr,
            min_lr=min_lr
        )
        
        # Validation
        val_loss = validate(model, val_dataloader, device, config)
        
        # Feature Evaluation
        linear_acc = -1.0
        sil_score = -1.0
        db_score = -1.0
        
        eval_freq = config['train'].get('eval_freq', 20)
        if epoch % eval_freq == 0 and epoch > 0:
            if is_main_process():
                logger.info("Running Feature Evaluation (Linear Probe & Clustering)...")
                # Linear Probe
                linear_acc = train_linear_probe(
                    model, 
                    train_dataloader, 
                    val_dataloader, 
                    device, 
                    num_classes=2,
                    limit_batches=config['train'].get('linear_probe_limit', 100)
                )
                # Clustering
                sil_score, db_score = evaluate_features_quality(
                    model, 
                    val_dataloader, 
                    device, 
                    config['train']['save_dir'], 
                    epoch
                )
                logger.info(f"Feature Eval: Acc={linear_acc:.4f}, Sil={sil_score:.4f}, DB={db_score:.4f}")
            
            # Barrier to wait for main process eval? 
            # 实际上 Linear Probe 比较慢，建议只在 rank 0 上跑，其他 rank 等待
            # 但 DDP model 在单进程跑可能需要处理，这里 evaluate_features_quality 内部处理了 model.module
            dist.barrier()

        if is_main_process():
            # Log Metrics
            metrics_dict = {
                'train_loss': train_metrics['loss'],
                'val_loss': val_loss,
                'grad_norm': train_metrics['grad_norm'],
                'gpu_mem_mb': torch.cuda.max_memory_allocated() / 1024 / 1024,
                'throughput': train_metrics['throughput'],
                'linear_acc': linear_acc,
                'sil_score': sil_score,
                'db_score': db_score
            }
            tracker.log(epoch, metrics_dict)
            logger.info(f"Epoch {epoch} Metrics: {metrics_dict}")

            # Early Stopping Check
            if tracker.check_early_stopping(patience=3):
                logger.info("Early stopping triggered due to no improvement in feature quality.")
                # break # 取消注释以启用

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