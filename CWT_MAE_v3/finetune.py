import os
import argparse
import yaml
import json
import random
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, classification_report, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from torch.amp import autocast, GradScaler

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

from dataset_finetune import DownstreamClassificationDataset
from model_finetune import TF_MAE_Classifier
from utils import get_layer_wise_lr

# -------------------------------------------------------------------
# 1. DDP 辅助函数
# -------------------------------------------------------------------
def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
        dist.barrier()
        return local_rank, rank, world_size
    else:
        print("Not using distributed mode")
        return 0, 0, 1

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model

def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

def setup_logger(save_dir):
    logger = logging.getLogger("finetune")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    if is_main_process():
        file_handler = logging.FileHandler(os.path.join(save_dir, "finetune.log"), encoding="utf-8")
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    else:
        logger.addHandler(logging.NullHandler())
    return logger

def save_checkpoint(path, model, optimizer, scheduler, epoch, best_metric, best_threshold, scaler):
    payload = {
        'epoch': epoch,
        'model': unwrap_model(model).state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_metric': best_metric,
        'best_threshold': best_threshold
    }
    if scaler is not None:
        payload['scaler'] = scaler.state_dict()
    torch.save(payload, path)

def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
    checkpoint = torch.load(path, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    unwrap_model(model).load_state_dict(state_dict, strict=True)
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    if scaler is not None and 'scaler' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler'])
    return checkpoint

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

def gather_tensors(tensor, device):
    if not dist.is_initialized():
        return tensor
    
    local_size = torch.tensor([tensor.shape[0]], device=device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
    dist.all_gather(all_sizes, local_size)
    max_size = max([x.item() for x in all_sizes])

    size_diff = max_size - local_size.item()
    if size_diff > 0:
        padding = torch.zeros((size_diff, *tensor.shape[1:]), device=device, dtype=tensor.dtype)
        tensor_padded = torch.cat((tensor, padding))
    else:
        tensor_padded = tensor

    gathered_tensors = [torch.zeros_like(tensor_padded) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_tensors, tensor_padded)

    output = []
    for i, gathered_tensor in enumerate(gathered_tensors):
        output.append(gathered_tensor[:all_sizes[i].item()])
    
    return torch.cat(output)

# -------------------------------------------------------------------
# 2. 关键：处理变长通道的 Collate Function (与 Pretrain 保持一致)
# -------------------------------------------------------------------
def variable_channel_collate_fn_cls(batch):
    """
    处理分类任务中不同样本通道数不一致的情况。
    Batch: list of tuples (signal_tensor, modality_ids, label)
    signal_tensor shape: (M_i, L)
    """
    # 分离信号和标签
    signals = [item[0] for item in batch]
    modality_ids = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    
    # 1. 找到当前 Batch 中最大的通道数
    max_m = max([s.shape[0] for s in signals])
    signal_len = signals[0].shape[1]
    batch_size = len(batch)
    
    # 2. 初始化全 0 张量 (B, Max_M, L)
    padded_signals = torch.zeros((batch_size, max_m, signal_len), dtype=signals[0].dtype)
    padded_modality_ids = torch.zeros((batch_size, max_m), dtype=torch.long)
    
    # 3. 填充数据
    for i, s in enumerate(signals):
        m = s.shape[0]
        padded_signals[i, :m, :] = s
        padded_modality_ids[i, :m] = modality_ids[i]
        
    return padded_signals, padded_modality_ids, torch.stack(labels)

# -------------------------------------------------------------------
# 3. 训练与验证逻辑
# -------------------------------------------------------------------
def mixup_data(x, y, alpha=1.0, device='cuda'):
    """
    x: (B, M, L)
    y: (B,)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    
    # Mixup 直接在 (B, M, L) 上进行是可行的
    # 注意：如果 M 不一致（通过 padding 补齐），mixup 会混合 真实信号 和 0
    # 这在生理信号中是可以接受的，相当于降低了信噪比
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        self.label_smoothing = float(label_smoothing)
        self.alpha_scalar = None
        if alpha is None:
            self.register_buffer("alpha_tensor", None)
        elif isinstance(alpha, (list, tuple)):
            self.register_buffer("alpha_tensor", torch.tensor(alpha, dtype=torch.float32))
        else:
            self.register_buffer("alpha_tensor", None)
            self.alpha_scalar = float(alpha)

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(
            logits,
            targets,
            reduction='none',
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_weight = (1.0 - pt) ** self.gamma
        loss = focal_weight * ce_loss

        if self.alpha_tensor is not None:
            alpha_t = self.alpha_tensor[targets]
            loss = alpha_t * loss
        elif self.alpha_scalar is not None:
            loss = self.alpha_scalar * loss

        if self.reduction == 'sum':
            return loss.sum()
        if self.reduction == 'none':
            return loss
        return loss.mean()

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, scaler=None, use_amp=True, mixup_alpha=0.2, grad_clip_norm=3.0, scheduler=None):
    model.train()
    if hasattr(loader.sampler, 'set_epoch'):
        loader.sampler.set_epoch(epoch)

    total_loss = 0
    count = 0
    
    # 优先使用 bfloat16
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    iterator = tqdm(loader, desc=f"Epoch {epoch + 1} Train") if is_main_process() else loader
    
    for batch in iterator:
        if len(batch) == 3:
            x, modality_ids, y = batch
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        else:
            x, y = batch
            modality_ids = None
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        
        # Mixup
        inputs, targets_a, targets_b, lam = mixup_data(x, y, alpha=mixup_alpha, device=device)
        
        optimizer.zero_grad()
        
        # Check if ArcFace is enabled (DDP wrapper -> module)
        real_model = unwrap_model(model)
        is_arcface = hasattr(real_model, 'use_arcface') and real_model.use_arcface

        with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
            if is_arcface:
                # ArcFace 需要硬标签 (Long Index)
                # 如果是软标签 (B, C)，取 argmax
                t_a = targets_a.argmax(dim=1) if targets_a.dim() > 1 else targets_a
                t_b = targets_b.argmax(dim=1) if targets_b.dim() > 1 else targets_b
                
                # ArcFace needs labels to calculate margin
                # Apply mixup logic manually: calc loss for target_a and target_b then mix
                logits_a = model(inputs, label=t_a)
                logits_b = model(inputs, label=t_b)
                
                # 计算 Loss
                # 注意：如果 targets_a 本身是软标签，这里转为硬标签传给 ArcFace 计算 Logits 是必须的
                # 但计算 Loss 时，我们依然可以用软标签 targets_a 吗？
                # ArcFace 输出的是 Logits，CrossEntropyLoss 支持软标签。
                # 所以：传给 Model 用硬标签，传给 Criterion 用原始标签 (软或硬)
                
                loss = lam * criterion(logits_a, targets_a) + (1 - lam) * criterion(logits_b, targets_b)
            else:
                logits = model(inputs)
                loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
        
        if use_amp and amp_dtype == torch.float16:
            if scaler is None:
                raise RuntimeError("GradScaler is required for float16 AMP training.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
        
        # Step-based scheduler
        if scheduler is not None:
            scheduler.step()

        if dist.is_initialized():
            reduced_loss = reduce_tensor(loss.detach())
            total_loss += reduced_loss.item()
        else:
            total_loss += loss.item()
            
        count += 1
        
        if is_main_process():
            current_lr = optimizer.param_groups[0]['lr']
            iterator.set_postfix({
                'loss': total_loss / count,
                'lr': f"{current_lr:.2e}"
            })

    return total_loss / count

def validate(model, loader, criterion, device, num_classes, total_len, use_amp=True, search_threshold=True, fixed_threshold=0.5, save_dir=None, epoch=None):
    model.eval()
    total_loss = 0
    count = 0
    
    local_labels = []
    local_probs = []
    
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    iterator = tqdm(loader, desc="Validating") if is_main_process() else loader

    with torch.no_grad():
        for batch in iterator:
            if len(batch) == 3:
                x, modality_ids, y = batch
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            else:
                x, y = batch
                modality_ids = None
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            
            if dist.is_initialized():
                reduced_loss = reduce_tensor(loss)
                total_loss += reduced_loss.item()
            else:
                total_loss += loss.item()
            count += 1
            
            probs = F.softmax(logits.float(), dim=1)
            
            local_labels.append(y)
            local_probs.append(probs) 

    local_labels = torch.cat(local_labels)
    local_probs = torch.cat(local_probs)

    if dist.is_initialized():
        all_labels = gather_tensors(local_labels, device)
        all_probs = gather_tensors(local_probs, device)
    else:
        all_labels = local_labels
        all_probs = local_probs

    if is_main_process():
        if len(all_labels) > total_len:
            all_labels = all_labels[:total_len]
            all_probs = all_probs[:total_len]

        all_labels_np = all_labels.cpu().numpy()
        all_probs_np = all_probs.cpu().numpy()

        row_sums = all_probs_np.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0 
        all_probs_np = all_probs_np / row_sums

        # 阈值搜索 (Binary Classification)
        best_threshold = fixed_threshold
        if num_classes == 2:
            y_scores = all_probs_np[:, 1]
            if search_threshold:
                thresholds = np.arange(0.01, 1.00, 0.01)
                best_f1_search = -1.0
                for th in thresholds:
                    preds_th = (y_scores >= th).astype(int)
                    f1_th = f1_score(all_labels_np, preds_th, average='macro')
                    if f1_th > best_f1_search:
                        best_f1_search = f1_th
                        best_threshold = th
                print(f"\n[Threshold Search] Best Threshold: {best_threshold:.2f} | Best Macro F1: {best_f1_search:.4f}")
            final_preds = (y_scores >= best_threshold).astype(int)
        else:
            final_preds = np.argmax(all_probs_np, axis=1)

        try:
            if num_classes == 2:
                auroc = roc_auc_score(all_labels_np, all_probs_np[:, 1])
            else:
                auroc = roc_auc_score(all_labels_np, all_probs_np, multi_class='ovr', average='macro')
        except Exception:
            auroc = 0.0

        final_acc = accuracy_score(all_labels_np, final_preds)
        final_f1 = f1_score(all_labels_np, final_preds, average='macro')
        precision = precision_score(all_labels_np, final_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels_np, final_preds, average='macro', zero_division=0)
        
        report_str = classification_report(all_labels_np, final_preds, digits=4)
        
        avg_loss = total_loss / count

        # Plot Precision-Recall Curve if applicable
        if num_classes == 2 and save_dir is not None:
            try:
                # Ensure using Agg backend to avoid GUI issues
                current_backend = plt.get_backend()
                if 'agg' not in current_backend.lower():
                    plt.switch_backend('agg')

                precisions, recalls, _ = precision_recall_curve(all_labels_np, all_probs_np[:, 1])
                avg_precision = average_precision_score(all_labels_np, all_probs_np[:, 1])
                
                plt.figure(figsize=(8, 6))
                plt.plot(recalls, precisions, label=f'AP={avg_precision:.4f}')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title(f'Precision-Recall Curve (Epoch {epoch})')
                plt.legend(loc='lower left')
                plt.grid(True)
                
                filename = f"pr_curve_epoch_{epoch}.png" if epoch is not None else "pr_curve_test.png"
                plot_path = os.path.join(save_dir, filename)
                plt.savefig(plot_path)
                plt.close()
                if is_main_process():
                    print(f"[Plot] Precision-Recall Curve saved to {plot_path}")
            except Exception as e:
                if is_main_process():
                    print(f"[Warning] Failed to plot PR curve: {e}")

        return avg_loss, final_acc, precision, recall, final_f1, auroc, report_str, best_threshold
    else:
        return 0, 0, 0, 0, 0, 0, None, 0

# -------------------------------------------------------------------
# 主函数
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='finetune_config.yaml', type=str)
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    train_cfg = config['train']
    data_cfg = config['data']
    model_cfg = config['model']
    seed = train_cfg.get('seed', 42)
    deterministic = train_cfg.get('deterministic', False)
    set_seed(seed, deterministic=deterministic)

    local_rank, rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    if is_main_process():
        os.makedirs(train_cfg['save_dir'], exist_ok=True)
        print(f"World Size: {world_size}, Master running on {device}")
    logger = setup_logger(train_cfg['save_dir'])
    torch.backends.cuda.matmul.allow_tf32 = train_cfg.get('allow_tf32', True)
    torch.backends.cudnn.allow_tf32 = train_cfg.get('allow_tf32', True)
    if not deterministic:
        torch.backends.cudnn.benchmark = train_cfg.get('cudnn_benchmark', True)

    with open(data_cfg['split_file'], 'r') as f:
        split_info = json.load(f)
    requested_val_mode = train_cfg.get('val_mode', 'val')
    requested_test_mode = train_cfg.get('test_mode', 'test')
    val_mode = requested_val_mode if requested_val_mode in split_info else ('test' if 'test' in split_info else requested_val_mode)
    test_mode = requested_test_mode if requested_test_mode in split_info else (val_mode if val_mode in split_info else requested_test_mode)
    threshold_calibration_only = (val_mode == test_mode)
    eval_split_name = "阈值校准集" if threshold_calibration_only else "验证集"
    if is_main_process():
        print(f"Validation split: {val_mode} | Test split: {test_mode}")
        if threshold_calibration_only:
            print("No separate validation split detected. The same split is used to tune inference threshold.")

    train_ds = DownstreamClassificationDataset(
        data_cfg['data_root'], data_cfg['split_file'], mode='train', 
        signal_len=data_cfg['signal_len'], num_classes=data_cfg['num_classes'],
        on_error=data_cfg.get('on_error', 'raise'),
        max_error_logs=data_cfg.get('max_error_logs', 20),
        refined_labels_path=data_cfg.get('refined_labels_path', None)
    )
    val_ds = DownstreamClassificationDataset(
        data_cfg['data_root'], data_cfg['split_file'], mode=val_mode, 
        signal_len=data_cfg['signal_len'], num_classes=data_cfg['num_classes'],
        on_error=data_cfg.get('on_error', 'raise'),
        max_error_logs=data_cfg.get('max_error_logs', 20)
    )
    test_ds = DownstreamClassificationDataset(
        data_cfg['data_root'], data_cfg['split_file'], mode=test_mode, 
        signal_len=data_cfg['signal_len'], num_classes=data_cfg['num_classes'],
        on_error=data_cfg.get('on_error', 'raise'),
        max_error_logs=data_cfg.get('max_error_logs', 20)
    )

    val_dataset_len = len(val_ds)
    test_dataset_len = len(test_ds)

    # 数据清洗逻辑 (可选)
    clean_indices_path = data_cfg.get('clean_indices_path')
    clean_val_indices_path = data_cfg.get('clean_val_indices_path')
    clean_test_indices_path = data_cfg.get('clean_test_indices_path')
    if clean_indices_path and os.path.exists(clean_indices_path):
        if is_main_process():
            print(f"\n[Data Cleaning] Loading clean indices from {clean_indices_path}...")
        clean_indices = np.load(clean_indices_path)
        clean_indices = clean_indices[clean_indices < len(train_ds)]
        train_ds = Subset(train_ds, clean_indices)
        
    if clean_val_indices_path and os.path.exists(clean_val_indices_path):
        if is_main_process():
            print(f"\n[Val Cleaning] Loading indices from {clean_val_indices_path}...")
        clean_val_indices = np.load(clean_val_indices_path)
        clean_val_indices = clean_val_indices[clean_val_indices < len(val_ds)]
        val_ds = Subset(val_ds, clean_val_indices)
        val_dataset_len = len(val_ds)

    if clean_test_indices_path and os.path.exists(clean_test_indices_path):
        if is_main_process():
            print(f"\n[Test Cleaning] Loading indices from {clean_test_indices_path}...")
        clean_test_indices = np.load(clean_test_indices_path)
        clean_test_indices = clean_test_indices[clean_test_indices < len(test_ds)]
        test_ds = Subset(test_ds, clean_test_indices)
        test_dataset_len = len(test_ds)

    if dist.is_initialized():
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False)
        shuffle_train = False
    else:
        # 单卡模式：使用 WeightedRandomSampler
        train_sampler = None
        val_sampler = None
        test_sampler = None
        shuffle_train = True
        
        # 尝试提取标签以计算权重
        try:
            if is_main_process():
                print("Single-GPU detected. Analyzing class distribution for WeightedRandomSampler...")
            
            # 1. 获取所有索引
            if isinstance(train_ds, Subset):
                indices = train_ds.indices
                base_ds = train_ds.dataset
            else:
                indices = list(range(len(train_ds)))
                base_ds = train_ds
            
            # 2. 快速读取标签 (使用简单的多线程)
            # 定义一个读取函数
            from concurrent.futures import ThreadPoolExecutor
            import pickle
            
            def load_label(idx):
                try:
                    # 直接访问 base_ds 的逻辑，避免 dataset.__getitem__ 的额外开销 (如 CWT, Norm)
                    filename = base_ds.file_list[idx]
                    file_path = os.path.join(base_ds.data_root, filename)
                    with open(file_path, 'rb') as f:
                        content = pickle.load(f)
                    
                    label = 0
                    if isinstance(content, dict) and 'label' in content:
                        target_label = content['label']
                        if isinstance(target_label, list):
                            if base_ds.task_index < len(target_label):
                                label_item = target_label[base_ds.task_index]
                                label = int(label_item['class']) if isinstance(label_item, dict) else int(label_item)
                        else:
                            label = int(target_label)
                    return label
                except:
                    return 0 # Default fallback

            # 3. 并行加载
            # 限制样本数以免卡太久，或者全量加载
            if len(indices) < 100000: # 限制一下规模
                with ThreadPoolExecutor(max_workers=16) as executor:
                    labels = list(tqdm(executor.map(load_label, indices), total=len(indices), desc="Loading labels"))
                
                # 4. 计算权重
                labels = np.array(labels)
                class_counts = np.bincount(labels, minlength=data_cfg['num_classes'])
                
                # 避免除以零
                class_counts = np.maximum(class_counts, 1)
                class_weights = 1.0 / class_counts
                
                # 每个样本的权重
                sample_weights = class_weights[labels]
                
                if is_main_process():
                    print(f"Class counts: {class_counts}")
                    print(f"Class weights: {class_weights}")
                
                # 5. 创建 Sampler
                train_sampler = WeightedRandomSampler(
                    weights=torch.from_numpy(sample_weights).double(),
                    num_samples=len(sample_weights),
                    replacement=True
                )
                shuffle_train = False # Sampler implies shuffle
                if is_main_process():
                    print("WeightedRandomSampler initialized successfully.")
            else:
                if is_main_process():
                    print("Dataset too large for quick label scanning. Skipping WeightedRandomSampler.")
                    
        except Exception as e:
            if is_main_process():
                print(f"Failed to init WeightedRandomSampler: {e}")
                print("Falling back to standard shuffling.")
            train_sampler = None
            shuffle_train = True

    # DataLoader: 必须使用 variable_channel_collate_fn_cls
    pin_memory = data_cfg.get('pin_memory', True)
    train_loader = DataLoader(
        train_ds, 
        batch_size=train_cfg['batch_size'], 
        sampler=train_sampler, 
        shuffle=shuffle_train,
        num_workers=data_cfg.get('num_workers', 4), 
        pin_memory=pin_memory,
        collate_fn=variable_channel_collate_fn_cls # 关键修改
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=train_cfg['batch_size'], 
        sampler=val_sampler, 
        shuffle=False,
        num_workers=data_cfg.get('num_workers', 4), 
        pin_memory=pin_memory,
        collate_fn=variable_channel_collate_fn_cls # 关键修改
    )
    test_loader = DataLoader(
        test_ds, 
        batch_size=train_cfg['batch_size'], 
        sampler=test_sampler, 
        shuffle=False,
        num_workers=data_cfg.get('num_workers', 4), 
        pin_memory=pin_memory,
        collate_fn=variable_channel_collate_fn_cls
    )

    if is_main_process():
        print(f"Initializing CWT-MAE Classifier (RoPE + Tensorized + CoT={model_cfg.get('use_cot', True)})...")
        
    model = TF_MAE_Classifier(
        pretrained_path=model_cfg.get('pretrained_path'),
        num_classes=data_cfg['num_classes'],
        # Encoder 参数
        signal_len=data_cfg['signal_len'],
        cwt_scales=model_cfg.get('cwt_scales', 64),
        patch_size_time=model_cfg.get('patch_size_time', 25),
        patch_size_freq=model_cfg.get('patch_size_freq', 8),
        embed_dim=model_cfg.get('embed_dim', 768),
        depth=model_cfg.get('depth', 12),
        num_heads=model_cfg.get('num_heads', 12),
        use_diff=model_cfg.get('use_diff', False),
        # Decoder 参数 (虽然会被删除，但初始化 Encoder 时需要)
        decoder_embed_dim=model_cfg.get('decoder_embed_dim', 512), 
        decoder_depth=model_cfg.get('decoder_depth', 8),
        decoder_num_heads=model_cfg.get('decoder_num_heads', 16),
        # CoT 参数
        use_cot=model_cfg.get('use_cot', True),
        num_reasoning_tokens=model_cfg.get('num_reasoning_tokens', 16),
        # ArcFace 参数
        use_arcface=model_cfg.get('use_arcface', False),
        arcface_s=model_cfg.get('arcface_s', 30.0),
        arcface_m=model_cfg.get('arcface_m', 0.50)
    )
    model.to(device)
    
    find_unused_parameters = train_cfg.get('find_unused_parameters', False)
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=find_unused_parameters)

    use_layer_wise_lr = train_cfg.get('use_layer_wise_lr', True)
    if use_layer_wise_lr:
        param_groups = get_layer_wise_lr(
            unwrap_model(model),
            base_lr=train_cfg['base_lr'],
            layer_decay=train_cfg.get('layer_decay', 0.65)
        )
        optimizer = optim.AdamW(param_groups, weight_decay=train_cfg['weight_decay'])
        if is_main_process():
            print("Optimizer: AdamW with layer-wise LR decay")
    else:
        optimizer = optim.AdamW(
            unwrap_model(model).parameters(),
            lr=train_cfg['base_lr'],
            weight_decay=train_cfg['weight_decay']
        )
        if is_main_process():
            print("Optimizer: AdamW with uniform learning rate")
    
    # LR Scheduler (Warmup + Cosine) - Step-based
    # 计算总步数
    steps_per_epoch = len(train_loader)
    total_steps = train_cfg['epochs'] * steps_per_epoch
    warmup_steps = int(train_cfg['warmup_epochs'] * steps_per_epoch)
    
    if warmup_steps > 0:
        scheduler_warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
        scheduler_cosine = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=train_cfg['min_lr']
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[scheduler_warmup, scheduler_cosine],
            milestones=[warmup_steps]
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=train_cfg['min_lr'])
    
    use_focal_loss = train_cfg.get('use_focal_loss', False)
    label_smoothing = train_cfg.get('label_smoothing', 0.1)
    if use_focal_loss:
        criterion = FocalLoss(
            gamma=train_cfg.get('focal_gamma', 2.0),
            alpha=train_cfg.get('focal_alpha', None),
            reduction=train_cfg.get('focal_reduction', 'mean'),
            label_smoothing=label_smoothing
        )
        if is_main_process():
            print(
                f"Loss: FocalLoss(gamma={train_cfg.get('focal_gamma', 2.0)}, "
                f"alpha={train_cfg.get('focal_alpha', None)}, "
                f"label_smoothing={label_smoothing})"
            )
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        if is_main_process():
            print(f"Loss: CrossEntropyLoss(label_smoothing={label_smoothing})")

    best_metric = float("-inf")
    best_threshold = 0.5
    best_epoch = -1
    start_epoch = 0
    no_improve_epochs = 0
    total_epochs = train_cfg['epochs']
    use_amp = train_cfg.get('use_amp', True)
    mixup_alpha = train_cfg.get('mixup_alpha', 0.2)
    grad_clip_norm = train_cfg.get('grad_clip_norm', 3.0)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = GradScaler(enabled=(use_amp and amp_dtype == torch.float16))
    early_stop_patience = train_cfg.get('early_stop_patience', 0)
    resume_path = train_cfg.get('resume_path')
    if (not resume_path) and train_cfg.get('auto_resume', True):
        candidate = os.path.join(train_cfg['save_dir'], "last_checkpoint.pth")
        if os.path.exists(candidate):
            resume_path = candidate
    if resume_path and os.path.exists(resume_path):
        resume_ckpt = load_checkpoint(resume_path, model, optimizer=optimizer, scheduler=scheduler, scaler=scaler)
        start_epoch = int(resume_ckpt.get('epoch', -1)) + 1
        best_metric = float(resume_ckpt.get('best_metric', best_metric))
        best_threshold = float(resume_ckpt.get('best_threshold', best_threshold))
        if is_main_process():
            logger.info(f"resume_from={resume_path} start_epoch={start_epoch}")

    for epoch in range(start_epoch, total_epochs):
        if is_main_process():
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch+1}/{total_epochs} | LR: {current_lr:.2e}")

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            scaler=scaler, use_amp=use_amp, mixup_alpha=mixup_alpha, grad_clip_norm=grad_clip_norm,
            scheduler=scheduler
        )
        
        # scheduler.step() # Moved to per-step inside train_one_epoch

        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc, val_report, best_th = validate(
            model, val_loader, criterion, device, data_cfg['num_classes'], 
            total_len=val_dataset_len, 
            use_amp=use_amp,
            search_threshold=(data_cfg['num_classes'] == 2),
            fixed_threshold=best_threshold,
            save_dir=train_cfg['save_dir'],
            epoch=epoch+1
        )

        if is_main_process():
            print(f"Train Loss: {train_loss:.4f}")
            print(f"{eval_split_name} Loss: {val_loss:.4f}")
            print("-" * 60)
            if data_cfg['num_classes'] == 2:
                print(f"Applied Threshold: {best_th:.2f}")
            print(f"{eval_split_name}准确率 (Accuracy): {val_acc:.4f}")
            print(f"AUC Score: {val_auc:.4f}")
            print("-" * 60)
            print(f"{eval_split_name}分类报告 (Classification Report):")
            print(val_report)
            print("-" * 60)

        metric_to_track = val_f1 if data_cfg['num_classes'] == 2 else (val_f1 if data_cfg['num_classes'] > 2 else val_auc)
        
        if metric_to_track > best_metric:
            best_metric = metric_to_track
            best_threshold = best_th
            best_epoch = epoch + 1
            torch.save(unwrap_model(model).state_dict(), os.path.join(train_cfg['save_dir'], "best_model.pth"))
            print(f">>> Best model saved! (Metric: {best_metric:.4f})")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        if is_main_process():
            torch.save(unwrap_model(model).state_dict(), os.path.join(train_cfg['save_dir'], "last_model.pth"))
            save_checkpoint(
                os.path.join(train_cfg['save_dir'], "last_checkpoint.pth"),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_metric,
                best_threshold=best_threshold,
                scaler=scaler
            )
        if is_main_process():
            logger.info(f"epoch={epoch+1} train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_acc={val_acc:.6f} val_f1={val_f1:.6f} val_auc={val_auc:.6f} lr={optimizer.param_groups[0]['lr']:.8e} th={best_th:.4f}")
        if early_stop_patience > 0 and no_improve_epochs >= early_stop_patience:
            if is_main_process():
                print(f"Early stopping triggered at epoch {epoch+1}")
                logger.info(f"early_stopping epoch={epoch+1}")
            break

    best_model_path = os.path.join(train_cfg['save_dir'], "best_model.pth")
    if os.path.exists(best_model_path):
        state_dict = torch.load(best_model_path, map_location=device)
        unwrap_model(model).load_state_dict(state_dict, strict=True)

    if is_main_process():
        threshold_payload = {
            "threshold": float(best_threshold),
            "epoch": int(best_epoch),
            "split_used": val_mode
        }
        threshold_path = os.path.join(train_cfg['save_dir'], "best_threshold.json")
        with open(threshold_path, "w", encoding="utf-8") as f:
            json.dump(threshold_payload, f, ensure_ascii=False, indent=2)
        print(f"Best threshold saved to: {threshold_path}")

    if not threshold_calibration_only:
        test_loss, test_acc, test_prec, test_rec, test_f1, test_auc, test_report, _ = validate(
            model, test_loader, criterion, device, data_cfg['num_classes'],
            total_len=test_dataset_len,
            use_amp=use_amp,
            search_threshold=False,
            fixed_threshold=best_threshold,
            save_dir=train_cfg['save_dir'],
            epoch="test"
        )

    if is_main_process():
        print(f"\nBest Epoch: {best_epoch}")
        if threshold_calibration_only:
            print(f"Inference Threshold: {best_threshold:.2f} (from split: {val_mode})")
        else:
            print(f"Test  Loss: {test_loss:.4f}")
            if data_cfg['num_classes'] == 2:
                print(f"Test Applied Threshold: {best_threshold:.2f}")
            print(f"最终测试集准确率 (Accuracy): {test_acc:.4f}")
            print(f"AUC Score: {test_auc:.4f}")
            print("-" * 60)
            print("最终测试集分类报告 (Classification Report):")
            print(test_report)
            print("-" * 60)
    
    cleanup_distributed()

if __name__ == "__main__":
    main()
