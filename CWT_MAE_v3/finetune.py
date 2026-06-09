import os
import argparse
import yaml
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report, precision_recall_curve, average_precision_score, fbeta_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from torch.amp import autocast, GradScaler
import copy
from muon import Muon

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

from dataset_finetune import DownstreamClassificationDataset
from model_finetune import TF_MAE_Classifier
from utils import setup_distributed, cleanup_distributed, is_main_process
from utils import unwrap_model, set_seed, setup_logger, reduce_tensor, gather_tensors

# -------------------------------------------------------------------
# 1. DDP 辅助函数 (已迁移至 utils.py)
# -------------------------------------------------------------------

def save_checkpoint(path, model, optimizer_muon, optimizer_adamw, scheduler_muon, scheduler_adamw, epoch, best_metric, best_threshold, scaler):
    payload = {
        'epoch': epoch,
        'model': unwrap_model(model).state_dict(),
        'optimizer_muon': optimizer_muon.state_dict(),
        'optimizer_adamw': optimizer_adamw.state_dict(),
        'scheduler_muon': scheduler_muon.state_dict(),
        'scheduler_adamw': scheduler_adamw.state_dict(),
        'best_metric': best_metric,
        'best_threshold': best_threshold
    }
    if scaler is not None:
        payload['scaler'] = scaler.state_dict()
    torch.save(payload, path)

def load_checkpoint(path, model, optimizer_muon=None, optimizer_adamw=None, scheduler_muon=None, scheduler_adamw=None, scaler=None):
    checkpoint = torch.load(path, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    unwrap_model(model).load_state_dict(state_dict, strict=True)
    if optimizer_muon is not None and 'optimizer_muon' in checkpoint:
        optimizer_muon.load_state_dict(checkpoint['optimizer_muon'])
    if optimizer_adamw is not None and 'optimizer_adamw' in checkpoint:
        optimizer_adamw.load_state_dict(checkpoint['optimizer_adamw'])
    if scheduler_muon is not None and 'scheduler_muon' in checkpoint:
        scheduler_muon.load_state_dict(checkpoint['scheduler_muon'])
    if scheduler_adamw is not None and 'scheduler_adamw' in checkpoint:
        scheduler_adamw.load_state_dict(checkpoint['scheduler_adamw'])
    if scaler is not None and 'scaler' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler'])
    return checkpoint

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
    channel_mask = torch.zeros((batch_size, max_m), dtype=torch.bool)
    
    # 3. 填充数据
    for i, s in enumerate(signals):
        m = s.shape[0]
        padded_signals[i, :m, :] = s
        padded_modality_ids[i, :m] = modality_ids[i]
        channel_mask[i, :m] = True
        
    return padded_signals, padded_modality_ids, torch.stack(labels), channel_mask

# -------------------------------------------------------------------
# 3. 训练与验证逻辑
# -------------------------------------------------------------------

def move_batch_to_device(batch, device):
    if len(batch) == 4:
        x, modality_ids, y, channel_mask = batch
    elif len(batch) == 3:
        x, modality_ids, y = batch
        channel_mask = None
    else:
        x, y = batch
        modality_ids = None
        channel_mask = None
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    if modality_ids is not None:
        modality_ids = modality_ids.to(device, non_blocking=True)
    if channel_mask is not None:
        channel_mask = channel_mask.to(device, non_blocking=True)
    return x, modality_ids, y, channel_mask

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
        # targets can be either hard labels (long) or soft labels (float)
        if targets.dtype == torch.long:
            # For hard labels, get standard cross entropy
            ce_loss = F.cross_entropy(
                logits,
                targets,
                reduction='none',
                label_smoothing=self.label_smoothing
            )
            # Calculate probabilities of the target class
            pt = torch.exp(-ce_loss)
            focal_weight = (1.0 - pt) ** self.gamma
            loss = focal_weight * ce_loss

            if self.alpha_tensor is not None:
                alpha_t = self.alpha_tensor[targets]
                loss = alpha_t * loss
            elif self.alpha_scalar is not None:
                loss = self.alpha_scalar * loss

        else:
            # For soft labels, use KL divergence or compute manually
            log_probs = F.log_softmax(logits, dim=-1)
            # Manually apply label smoothing if needed for soft labels
            if self.label_smoothing > 0:
                num_classes = logits.size(-1)
                targets = targets * (1.0 - self.label_smoothing) + self.label_smoothing / num_classes
                
            ce_loss = -(targets * log_probs).sum(dim=-1)
            probs = torch.exp(log_probs)
            # pt is the probability of the true distribution
            pt = (targets * probs).sum(dim=-1)
            
            focal_weight = (1.0 - pt) ** self.gamma
            loss = focal_weight * ce_loss
            
            if self.alpha_tensor is not None:
                # Approximate alpha for soft labels by taking expected alpha
                alpha_t = (targets * self.alpha_tensor).sum(dim=-1)
                loss = alpha_t * loss
            elif self.alpha_scalar is not None:
                loss = self.alpha_scalar * loss

        if self.reduction == 'sum':
            return loss.sum()
        if self.reduction == 'none':
            return loss
        return loss.mean()

class MultiLabelFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, pos_weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # alpha is for positive class, 1-alpha for negative
        self.reduction = reduction
        # pos_weight should be a tensor of shape [num_classes]
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits, targets):
        # Use BCEWithLogitsLoss with reduction='none' to get per-element BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none', pos_weight=self.pos_weight
        )
        
        # Calculate pt (probability of the true label)
        # Using pt = p if y=1 else 1-p, which can be computed efficiently:
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        
        # Calculate modulating factor (1 - pt) ^ gamma
        focal_weight = (1.0 - pt) ** self.gamma
        
        # Optional alpha weighting
        if self.alpha is not None:
            alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            focal_weight = alpha_weight * focal_weight
            
        loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for Multi-Label Classification
    From: "Asymmetric Loss For Multi-Label Classification" (Ben-Baruch et al., 2021)

    Key ideas:
      - Asymmetric focusing: gamma_neg > gamma_pos, so easy negatives are
        down-weighted much more aggressively than easy positives.
      - Probability shifting (margin m): for negative samples, max(p - m, 0)
        eliminates the loss contribution from very-low-probability negatives.
    """
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8,
                 disable_torch_grad_focal_loss=False, pos_weight=None):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        if pos_weight is not None:
            self.register_buffer('pos_weight', pos_weight)
        else:
            self.pos_weight = None

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)

        los_pos = targets * F.logsigmoid(logits)

        if self.gamma_pos > 0:
            pt_pos = (1 - p)
            if self.disable_torch_grad_focal_loss:
                pt_pos = pt_pos.detach()
            los_pos = los_pos * (pt_pos ** self.gamma_pos)

        p_m = torch.clamp(p - self.clip, min=0.0)

        los_neg = (1 - targets) * torch.log(torch.clamp(1 - p_m, min=self.eps))

        if self.gamma_neg > 0:
            pt_neg = p_m
            if self.disable_torch_grad_focal_loss:
                pt_neg = pt_neg.detach()
            los_neg = los_neg * (pt_neg ** self.gamma_neg)

        loss = los_pos + los_neg

        if self.pos_weight is not None:
            loss = loss * self.pos_weight

        return -loss.sum(dim=-1).mean()


def train_one_epoch(model, loader, criterion, optimizer_muon, optimizer_adamw, device, epoch, scaler=None, use_amp=True, grad_clip_norm=3.0, scheduler_muon=None, scheduler_adamw=None):
    model.train()
    if hasattr(loader.sampler, 'set_epoch'):
        loader.sampler.set_epoch(epoch)

    total_loss = 0
    count = 0

    amp_dtype = torch.bfloat16 if (device.type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16
    amp_enabled = use_amp and device.type == 'cuda'
    iterator = tqdm(loader, desc=f"Epoch {epoch + 1} Train") if is_main_process() else loader

    for batch in iterator:
        x, modality_ids, y, channel_mask = move_batch_to_device(batch, device)

        optimizer_muon.zero_grad(set_to_none=True)
        optimizer_adamw.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits = model(x, channel_mask=channel_mask)
            loss = criterion(logits, y)

        if use_amp and amp_dtype == torch.float16:
            if scaler is None:
                raise RuntimeError("GradScaler is required for float16 AMP training.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer_muon)
            scaler.unscale_(optimizer_adamw)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            # 全局 found_inf 检查：确保两个 optimizer 同步
            found_inf = sum(
                v.item()
                for state in scaler._per_optimizer_states.values()
                for v in state["found_inf_per_device"].values()
            )
            if found_inf == 0:
                scaler.step(optimizer_muon)
                scaler.step(optimizer_adamw)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer_muon.step()
            optimizer_adamw.step()

        # Step-based scheduler
        if scheduler_muon is not None:
            scheduler_muon.step()
        if scheduler_adamw is not None:
            scheduler_adamw.step()

        total_loss += loss.item()
        count += 1

        if is_main_process():
            current_lr = optimizer_adamw.param_groups[0]['lr']
            iterator.set_postfix({
                'loss': total_loss / count,
                'lr': f"{current_lr:.2e}"
            })

    if count == 0:
        return 0.0
    return total_loss / count

def validate(model, loader, criterion, device, num_classes, total_len, use_amp=True, search_threshold=True, fixed_threshold=0.5, save_dir=None, epoch=None):
    model.eval()
    total_loss = 0
    count = 0
    
    local_labels = []
    local_probs = []
    
    amp_dtype = torch.bfloat16 if (device.type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16
    amp_enabled = use_amp and device.type == 'cuda'

    iterator = tqdm(loader, desc="Validating") if is_main_process() else loader

    with torch.no_grad():
        for batch in iterator:
            x, modality_ids, y, channel_mask = move_batch_to_device(batch, device)
            
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                logits = model(x, channel_mask=channel_mask)
                loss = criterion(logits, y)
            
            if dist.is_initialized():
                reduced_loss = reduce_tensor(loss)
                total_loss += reduced_loss.item()
            else:
                total_loss += loss.item()
            count += 1
            
            probs = torch.sigmoid(logits.float())
            
            local_labels.append(y.cpu())
            local_probs.append(probs.cpu()) 

    if count == 0 or len(local_labels) == 0:
        if is_main_process():
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "Empty validation loader.", fixed_threshold, []
        return 0, 0, 0, 0, 0, 0, None, 0, []

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

        # Multi-label thresholds
        if search_threshold and num_classes > 2:
            best_thresholds = []
            for i in range(num_classes):
                disease_true = all_labels_np[:, i]
                disease_prob = all_probs_np[:, i]
                
                if len(np.unique(disease_true)) > 1:
                    best_th_i = 0.5
                    best_f1_i = 0.0
                    for th in np.arange(0.1, 0.9, 0.05):
                        preds_i = (disease_prob >= th).astype(int)
                        f1_i = fbeta_score(disease_true, preds_i, beta=0.5, zero_division=0)
                        if f1_i > best_f1_i:
                            best_f1_i = f1_i
                            best_th_i = th
                    best_thresholds.append(best_th_i)
                else:
                    best_thresholds.append(0.5)
            
            best_threshold = np.array(best_thresholds)
            final_preds = (all_probs_np >= best_threshold).astype(int)
        else:
            best_threshold = fixed_threshold
            final_preds = (all_probs_np >= best_threshold).astype(int)

        # 循环计算每种疾病的指标 (Macro AUC)
        auc_list = []
        for i in range(num_classes):
            disease_true = all_labels_np[:, i]
            disease_prob = all_probs_np[:, i]
            try:
                # 只有正负样本都存在时才能算 AUC
                if len(np.unique(disease_true)) > 1:
                    auc_i = roc_auc_score(disease_true, disease_prob)
                else:
                    auc_i = 0.5
            except Exception:
                auc_i = 0.5
            auc_list.append(auc_i)
            # 可以在此处打印每个类的AUC，或者仅由主进程打印
            # if is_main_process() and epoch is not None:
            #     print(f"Disease {i} AUC: {auc_i:.4f}")
                
        auroc = float(np.mean(auc_list))

        # 计算多标签下的其他宏平均指标
        final_acc = accuracy_score(all_labels_np, final_preds) # Exact match accuracy
        final_f1 = fbeta_score(all_labels_np, final_preds, beta=0.5, average='macro', zero_division=0)
        precision = precision_score(all_labels_np, final_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels_np, final_preds, average='macro', zero_division=0)
        
        report_str = classification_report(all_labels_np, final_preds, digits=4, zero_division=0)
        
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

        return avg_loss, final_acc, precision, recall, final_f1, auroc, report_str, best_threshold, auc_list
    else:
        return 0, 0, 0, 0, 0, 0, None, 0, []

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

    if torch.cuda.is_available():
        local_rank, rank, world_size = setup_distributed()
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank, rank, world_size = 0, 0, 1
        device = torch.device("cpu")
    
    if is_main_process():
        os.makedirs(train_cfg['save_dir'], exist_ok=True)
        print(f"World Size: {world_size}, Master running on {device}")
    logger = setup_logger(train_cfg['save_dir'], name="finetune")
    if device.type == 'cuda':
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

    # 单卡模式下使用 WeightedRandomSampler
    # --- 重要修复：如果在 DDP 模式下，也需要解决类别不平衡问题 ---
    # 但 DDP 下使用 WeightedRandomSampler 比较复杂，所以最稳妥的办法是：
    # 确保 FocalLoss 的 gamma 设置得足够大，或者传入 alpha 权重
    
    if dist.is_initialized():
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False)
        shuffle_train = False
    else:
        # 单卡模式
        train_sampler = None
        val_sampler = None
        test_sampler = None
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
        signal_len=data_cfg['signal_len'],
        cwt_scales=model_cfg.get('cwt_scales', 64),
        patch_size_time=model_cfg.get('patch_size_time', 25),
        patch_size_freq=model_cfg.get('patch_size_freq', 8),
        embed_dim=model_cfg.get('embed_dim', 768),
        depth=model_cfg.get('depth', 12),
        num_heads=model_cfg.get('num_heads', 12),
        use_diff=model_cfg.get('use_diff', False),
        decoder_embed_dim=model_cfg.get('decoder_embed_dim', 512), 
        decoder_depth=model_cfg.get('decoder_depth', 8),
        decoder_num_heads=model_cfg.get('decoder_num_heads', 16),
        use_cot=model_cfg.get('use_cot', True),
        num_reasoning_tokens=model_cfg.get('num_reasoning_tokens', 16),
        use_stats_features=model_cfg.get('use_stats_features', False),
        drop_path_rate=model_cfg.get('drop_path_rate', 0.0),
        cot_kv_layers=model_cfg.get('cot_kv_layers', None)
    )
    model.to(device)
    
    find_unused_parameters = train_cfg.get('find_unused_parameters', True)
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=find_unused_parameters)

    # --- Muon + AdamW 双优化器参数分组 ---
    muon_params = []
    adamw_params = []

    raw_model = unwrap_model(model)
    for module_name, module in raw_model.named_modules():
        if isinstance(module, nn.Linear):
            if module.weight.requires_grad:
                muon_params.append(module.weight)
            if module.bias is not None and module.bias.requires_grad:
                adamw_params.append(module.bias)
        else:
            for param_name, param in module.named_parameters(recurse=False):
                if param.requires_grad:
                    adamw_params.append(param)

    # 去重（named_modules 可能重复遍历）
    muon_params = list(dict.fromkeys(muon_params))
    adamw_params = list(dict.fromkeys(adamw_params))

    optimizer_muon = Muon(
        muon_params,
        lr=train_cfg['base_lr'],
        momentum=0.95,
        weight_decay=train_cfg['weight_decay']
    )
    optimizer_adamw = optim.AdamW(
        adamw_params,
        lr=train_cfg['base_lr'],
        weight_decay=train_cfg['weight_decay']
    )

    if is_main_process():
        print(f"Optimizer: Muon ({len(muon_params)} params) + AdamW ({len(adamw_params)} params)")
        total_muon = sum(p.numel() for p in muon_params)
        total_adamw = sum(p.numel() for p in adamw_params)
        print(f"  Muon parameters: {total_muon:,} | AdamW parameters: {total_adamw:,}")

    # LR Scheduler (Warmup + Cosine) - Step-based, 分别为两个 optimizer 创建
    steps_per_epoch = len(train_loader)
    total_steps = train_cfg['epochs'] * steps_per_epoch
    warmup_steps = int(train_cfg['warmup_epochs'] * steps_per_epoch)

    def build_scheduler(optimizer):
        if warmup_steps > 0:
            sched_warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
            sched_cosine = CosineAnnealingLR(
                optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=train_cfg['min_lr']
            )
            return SequentialLR(
                optimizer,
                schedulers=[sched_warmup, sched_cosine],
                milestones=[warmup_steps]
            )
        else:
            return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=train_cfg['min_lr'])

    scheduler_muon = build_scheduler(optimizer_muon)
    scheduler_adamw = build_scheduler(optimizer_adamw)
    
    # Resolve loss type: explicit 'loss_type' takes priority, else fall back to 'use_focal_loss' flag
    loss_type = train_cfg.get('loss_type', None)
    if loss_type is None:
        loss_type = 'focal' if train_cfg.get('use_focal_loss') else 'bce'

    loss_label = {'asl': 'AsymmetricLoss', 'focal': 'MultiLabelFocalLoss'}.get(loss_type, 'BCEWithLogitsLoss')
    if train_cfg.get('pos_weight') == 'auto':
        if is_main_process():
            print(f"Calculating pos_weight for {loss_label} automatically...")
            if isinstance(train_ds, Subset):
                indices = train_ds.indices
                base_ds = train_ds.dataset
            else:
                indices = list(range(len(train_ds)))
                base_ds = train_ds
            
            from concurrent.futures import ThreadPoolExecutor
            import pickle
            
            LABEL_NAMES = [
                '高血压', '高血糖', '高血脂', 
                '冠心病', '心律失常（房颤、频发早搏等）', '糖尿病', 
                '颈动脉斑块'
            ]
            
            def load_label_vector(idx):
                try:
                    filename = base_ds.file_list[idx]
                    file_path = os.path.join(base_ds.data_root, filename)
                    with open(file_path, 'rb') as f:
                        content = pickle.load(f)
                    if isinstance(content, dict) and 'label' in content:
                        target_label = content['label']
                        if isinstance(target_label, dict):
                            return np.array([float(target_label.get(name, 0)) for name in LABEL_NAMES])
                    return np.zeros(len(LABEL_NAMES))
                except:
                    return np.zeros(len(LABEL_NAMES))
                    
            with ThreadPoolExecutor(max_workers=16) as executor:
                labels_list = list(tqdm(executor.map(load_label_vector, indices), total=len(indices), desc="Loading labels for pos_weight"))
            
            labels_np = np.array(labels_list)
            pos_counts = labels_np.sum(axis=0)
            neg_counts = len(labels_np) - pos_counts
            
            pos_counts = np.maximum(pos_counts, 1)
            # 使用开根号平滑策略 (Square Root Smoothing) 避免权重过于极端
            calculated_weights = np.sqrt(neg_counts / pos_counts)
            # 限制最大权重避免极端不平衡导致的梯度爆炸
            calculated_weights = np.clip(calculated_weights, 1.0, 50.0) 
            
            weights_tensor = torch.tensor(calculated_weights, dtype=torch.float32, device=device)
        else:
            weights_tensor = torch.zeros(data_cfg['num_classes'], dtype=torch.float32, device=device)
            
        if dist.is_initialized():
            dist.broadcast(weights_tensor, src=0)
    elif train_cfg.get('pos_weight') is not None:
        weights_tensor = torch.tensor(train_cfg['pos_weight'], dtype=torch.float32).to(device)
    else:
        weights_tensor = None

    def _fmt_pos_weight(w):
        if w is None:
            return "None"
        return str([round(v, 2) for v in w.cpu().tolist()])

    if loss_type == 'asl':
        asl_cfg = train_cfg.get('asl', {})
        criterion = AsymmetricLoss(
            gamma_neg=asl_cfg.get('gamma_neg', 4),
            gamma_pos=asl_cfg.get('gamma_pos', 1),
            clip=asl_cfg.get('clip', 0.05),
            disable_torch_grad_focal_loss=asl_cfg.get('disable_torch_grad_focal_loss', False),
            pos_weight=weights_tensor
        )
        if is_main_process():
            print(f"Loss: AsymmetricLoss | gamma_neg={criterion.gamma_neg} gamma_pos={criterion.gamma_pos} clip={criterion.clip} | pos_weight={_fmt_pos_weight(weights_tensor)}")
    elif loss_type == 'focal':
        focal_cfg = train_cfg.get('focal', {})
        criterion = MultiLabelFocalLoss(
            gamma=focal_cfg.get('gamma', 2.0),
            alpha=focal_cfg.get('alpha', None),
            pos_weight=weights_tensor
        )
        if is_main_process():
            print(f"Loss: MultiLabelFocalLoss | gamma={criterion.gamma} alpha={criterion.alpha} | pos_weight={_fmt_pos_weight(weights_tensor)}")
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=weights_tensor) if weights_tensor is not None else nn.BCEWithLogitsLoss()
        if is_main_process():
            print(f"Loss: BCEWithLogitsLoss | pos_weight={_fmt_pos_weight(weights_tensor)}")

    best_metric = float("-inf")
    best_threshold = np.array([0.5] * data_cfg['num_classes']) if data_cfg['num_classes'] > 2 else 0.5
    best_epoch = -1
    start_epoch = 0
    no_improve_epochs = 0
    total_epochs = train_cfg['epochs']
    use_amp = train_cfg.get('use_amp', True)
    grad_clip_norm = train_cfg.get('grad_clip_norm', 3.0)
    amp_dtype = torch.bfloat16 if (device.type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = GradScaler(enabled=(use_amp and device.type == 'cuda' and amp_dtype == torch.float16))
    if is_main_process():
        amp_enabled = use_amp and device.type == 'cuda'
        print(f"AMP Enabled: {amp_enabled} | AMP DType: {amp_dtype} | GradScaler: {scaler.is_enabled()}")
        logger.info(f"amp_enabled={amp_enabled} amp_dtype={amp_dtype} grad_scaler={scaler.is_enabled()}")
    early_stop_patience = train_cfg.get('early_stop_patience', 0)
    resume_path = train_cfg.get('resume_path')
    if (not resume_path) and train_cfg.get('auto_resume', True):
        candidate = os.path.join(train_cfg['save_dir'], "last_checkpoint.pth")
        if os.path.exists(candidate):
            resume_path = candidate
    if resume_path and os.path.exists(resume_path):
        resume_ckpt = load_checkpoint(resume_path, model, optimizer_muon=optimizer_muon, optimizer_adamw=optimizer_adamw, scheduler_muon=scheduler_muon, scheduler_adamw=scheduler_adamw, scaler=scaler)
        start_epoch = int(resume_ckpt.get('epoch', -1)) + 1
        best_metric = float(resume_ckpt.get('best_metric', best_metric))
        saved_threshold = resume_ckpt.get('best_threshold', best_threshold)
        if isinstance(saved_threshold, (list, np.ndarray)):
            best_threshold = np.array(saved_threshold)
        else:
            best_threshold = float(saved_threshold)
        if is_main_process():
            logger.info(f"resume_from={resume_path} start_epoch={start_epoch}")

    for epoch in range(start_epoch, total_epochs):
        if is_main_process():
            current_lr = optimizer_adamw.param_groups[0]['lr']
            print(f"\nEpoch {epoch+1}/{total_epochs} | LR: {current_lr:.2e}")

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer_muon, optimizer_adamw, device, epoch,
            scaler=scaler, use_amp=use_amp, grad_clip_norm=grad_clip_norm,
            scheduler_muon=scheduler_muon, scheduler_adamw=scheduler_adamw
        )
        
        # scheduler.step() # Moved to per-step inside train_one_epoch

        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc, val_report, best_th, val_auc_list = validate(
            model, val_loader, criterion, device, data_cfg['num_classes'], 
            total_len=val_dataset_len, 
            use_amp=use_amp,
            search_threshold=True,
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
            elif data_cfg['num_classes'] > 2 and isinstance(best_th, np.ndarray):
                print(f"Applied Thresholds: {np.round(best_th, 2).tolist()}")
            print(f"{eval_split_name}准确率 (Accuracy): {val_acc:.4f}")
            print(f"AUC Score: {val_auc:.4f}")
            if len(val_auc_list) > 0:
                print(f"每类 AUC:")
                for i, auc_val in enumerate(val_auc_list):
                    print(f"  类别 {i}: {auc_val:.4f}")
            print("-" * 60)
            print(f"{eval_split_name}分类报告 (Classification Report):")
            print(val_report)
            print("-" * 60)

        # 使用 AUC 作为最佳权重的保留标准
        metric_to_track = val_auc
        
        if metric_to_track > best_metric:
            best_metric = metric_to_track
            best_threshold = best_th
            best_epoch = epoch + 1
            if is_main_process():
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
                optimizer_muon=optimizer_muon,
                optimizer_adamw=optimizer_adamw,
                scheduler_muon=scheduler_muon,
                scheduler_adamw=scheduler_adamw,
                epoch=epoch,
                best_metric=best_metric,
                best_threshold=best_threshold,
                scaler=scaler
            )
        if is_main_process():
            th_str = str(np.round(best_th, 4).tolist()) if isinstance(best_th, np.ndarray) else f"{best_th:.4f}"
            auc_str = str([round(a, 4) for a in val_auc_list]) if isinstance(val_auc_list, list) else "[]"
            logger.info(f"epoch={epoch+1} train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_acc={val_acc:.6f} val_f0.5={val_f1:.6f} val_auc={val_auc:.6f} val_auc_per_class={auc_str} lr={optimizer_adamw.param_groups[0]['lr']:.8e} th={th_str}")
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
            "threshold": float(best_threshold) if data_cfg['num_classes'] == 2 else best_threshold.tolist() if isinstance(best_threshold, np.ndarray) else best_threshold,
            "epoch": int(best_epoch),
            "split_used": val_mode
        }
        threshold_path = os.path.join(train_cfg['save_dir'], "best_threshold.json")
        with open(threshold_path, "w", encoding="utf-8") as f:
            json.dump(threshold_payload, f, ensure_ascii=False, indent=2)
        print(f"Best threshold saved to: {threshold_path}")

    if not threshold_calibration_only:
        test_loss, test_acc, test_prec, test_rec, test_f1, test_auc, test_report, _, test_auc_list = validate(
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
            if data_cfg['num_classes'] == 2:
                print(f"Inference Threshold: {best_threshold:.2f} (from split: {val_mode})")
            elif data_cfg['num_classes'] > 2 and isinstance(best_threshold, np.ndarray):
                print(f"Inference Thresholds: {np.round(best_threshold, 2).tolist()} (from split: {val_mode})")
        else:
            print(f"Test  Loss: {test_loss:.4f}")
            if data_cfg['num_classes'] == 2:
                print(f"Test Applied Threshold: {best_threshold:.2f}")
            elif data_cfg['num_classes'] > 2 and isinstance(best_threshold, np.ndarray):
                print(f"Test Applied Thresholds: {np.round(best_threshold, 2).tolist()}")
            print(f"最终测试集准确率 (Accuracy): {test_acc:.4f}")
            print(f"AUC Score: {test_auc:.4f}")
            if len(test_auc_list) > 0:
                print(f"每类 AUC:")
                for i, auc_val in enumerate(test_auc_list):
                    print(f"  类别 {i}: {auc_val:.4f}")
            print("-" * 60)
            print(f"最终测试集分类报告 (Classification Report):")
            print(test_report)
            print("-" * 60)
            test_auc_str = str([round(a, 4) for a in test_auc_list]) if isinstance(test_auc_list, list) else "[]"
            logger.info(f"final_test test_loss={test_loss:.6f} test_acc={test_acc:.6f} test_f0.5={test_f1:.6f} test_auc={test_auc:.6f} test_auc_per_class={test_auc_str} best_epoch={best_epoch}")
    
    cleanup_distributed()

if __name__ == "__main__":
    main()
