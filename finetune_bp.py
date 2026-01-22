import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
from torch.cuda.amp import GradScaler
from torch.amp import autocast 
from torch.optim.lr_scheduler import CosineAnnealingLR

# 引入自定义模块
from dataset_regression import BPDataset
from model_finetune_bp_prediction import TF_MAE_BloodPressure

# 尝试导入分层学习率工具，如果没有则使用默认
try:
    from utils import get_layer_wise_lr
except ImportError:
    def get_layer_wise_lr(model, base_lr, layer_decay):
        return filter(lambda p: p.requires_grad, model.parameters())

# -------------------------------------------------------------------
# DDP 辅助函数
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

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

def gather_tensors(tensor, device):
    if not dist.is_initialized(): return tensor
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

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -------------------------------------------------------------------
# 自定义损失函数：加权 Huber Loss
# -------------------------------------------------------------------
class WeightedHuberLoss(nn.Module):
    def __init__(self, delta=1.0, sbp_weight=2.0, dbp_weight=1.0):
        super().__init__()
        self.delta = delta
        self.sbp_weight = sbp_weight
        self.dbp_weight = dbp_weight
        self.huber = nn.HuberLoss(reduction='none', delta=delta)

    def forward(self, pred, target):
        # pred, target: [B, 2] (Column 0: SBP, Column 1: DBP)
        loss = self.huber(pred, target) # [B, 2]
        
        # 对 SBP 和 DBP 施加不同权重
        weighted_loss = loss[:, 0] * self.sbp_weight + loss[:, 1] * self.dbp_weight
        
        return weighted_loss.mean()

# -------------------------------------------------------------------
# 冻结/解冻 辅助函数
# -------------------------------------------------------------------
def freeze_encoder(model):
    """冻结 Encoder，只训练 Head 和 Projector"""
    model.train()
    # 这里的 model 是 DDP 包装过的，所以要用 model.module
    # 1. 冻结所有参数
    for param in model.module.parameters():
        param.requires_grad = False
    
    # 2. 解冻 Head
    for param in model.module.head.parameters():
        param.requires_grad = True
        
    # 3. 解冻 Tabular Projector (如果有)
    if hasattr(model.module, 'tabular_projector'):
        for param in model.module.tabular_projector.parameters():
            param.requires_grad = True
            
    if is_main_process():
        print(">>> [Training Strategy] Encoder Frozen. Training Head & Projector only.")

def unfreeze_all(model):
    """解冻所有参数"""
    for param in model.module.parameters():
        param.requires_grad = True
    if is_main_process():
        print(">>> [Training Strategy] Encoder Unfrozen. Fine-tuning entire model.")

# -------------------------------------------------------------------
# 训练逻辑
# -------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, epoch, use_amp=True):
    model.train()
    if hasattr(loader.sampler, 'set_epoch'):
        loader.sampler.set_epoch(epoch)

    total_loss = 0
    count = 0
    
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    iterator = tqdm(loader, desc=f"Epoch {epoch} Train") if is_main_process() else loader
    
    for batch in iterator:
        x = batch['signal'].to(device)
        x_tab = batch['tabular'].to(device)
        y = batch['target'].to(device)
        
        optimizer.zero_grad()
        
        with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
            preds = model(x, x_tab) 
            loss = criterion(preds, y)
        
        if use_amp and amp_dtype == torch.float16:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()

        if dist.is_initialized():
            reduced_loss = reduce_tensor(loss.detach())
            total_loss += reduced_loss.item()
        else:
            total_loss += loss.item()
            
        count += 1
        if is_main_process():
            iterator.set_postfix({'loss': total_loss / count})

    return total_loss / count

# -------------------------------------------------------------------
# 验证逻辑
# -------------------------------------------------------------------
def validate(model, loader, criterion, device, total_len, bp_mean, bp_std, use_amp=True):
    model.eval()
    total_loss = 0
    count = 0
    local_preds = []
    local_targets = []
    
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    iterator = tqdm(loader, desc="Validating") if is_main_process() else loader

    with torch.no_grad():
        for batch in iterator:
            x = batch['signal'].to(device)
            x_tab = batch['tabular'].to(device)
            y = batch['target'].to(device)
            
            with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                preds = model(x, x_tab)
                loss = criterion(preds, y)
            
            if dist.is_initialized():
                reduced_loss = reduce_tensor(loss)
                total_loss += reduced_loss.item()
            else:
                total_loss += loss.item()
            count += 1
            
            local_preds.append(preds.float())
            local_targets.append(y.float())

    local_preds = torch.cat(local_preds)
    local_targets = torch.cat(local_targets)

    if dist.is_initialized():
        all_preds = gather_tensors(local_preds, device)
        all_targets = gather_tensors(local_targets, device)
    else:
        all_preds = local_preds
        all_targets = local_targets

    if is_main_process():
        if len(all_preds) > total_len:
            all_preds = all_preds[:total_len]
            all_targets = all_targets[:total_len]

        # 反归一化
        bp_mean = bp_mean.to(device)
        bp_std = bp_std.to(device)
        pred_real = all_preds * bp_std + bp_mean
        target_real = all_targets * bp_std + bp_mean
        
        pred_np = pred_real.cpu().numpy()
        target_np = target_real.cpu().numpy()

        mae_sbp = mean_absolute_error(target_np[:, 0], pred_np[:, 0])
        mae_dbp = mean_absolute_error(target_np[:, 1], pred_np[:, 1])
        mae_avg = (mae_sbp + mae_dbp) / 2.0
        rmse = np.sqrt(mean_squared_error(target_np, pred_np))
        r2 = r2_score(target_np, pred_np)
        
        return total_loss / count, mae_sbp, mae_dbp, mae_avg, rmse, r2
    else:
        return 0, 0, 0, 0, 0, 0

# -------------------------------------------------------------------
# 主函数
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--signal_dir', type=str, required=True)
    parser.add_argument('--excel_path', type=str, required=True)
    parser.add_argument('--pretrained_path', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default="./checkpoints_bp")
    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--warmup_epochs', type=int, default=10, help="Epochs to freeze encoder")
    
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--head_lr', type=float, default=1e-3, help="LR for head during warmup")
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)

    # 数据参数
    parser.add_argument('--signal_len', type=int, default=1400) # 14s * 100Hz
    
    # 模型参数
    parser.add_argument('--embed_dim', type=int, default=768) 
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--cwt_scales', type=int, default=64)
    parser.add_argument('--patch_size_time', type=int, default=50)
    parser.add_argument('--patch_size_freq', type=int, default=4)
    parser.add_argument('--mlp_rank_ratio', type=float, default=0.5)
    parser.add_argument('--num_reasoning_tokens', type=int, default=8)

    args = parser.parse_args()

    local_rank, rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    set_seed(args.seed + rank)
    
    if is_main_process():
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"World Size: {world_size}, Master running on {device}")

    # 1. 初始化 Dataset
    full_dataset = BPDataset(
        signal_dir=args.signal_dir, 
        excel_path=args.excel_path, 
        seq_len=args.signal_len,
        normalize_labels=True
    )
    
    # 2. 按 Subject ID 划分数据集 (防止数据泄露)
    all_subject_ids = list(full_dataset.subject_info.keys())
    # 确保随机性一致
    rng = random.Random(args.seed)
    rng.shuffle(all_subject_ids)
    
    split_idx = int(len(all_subject_ids) * 0.8)
    train_ids = set(all_subject_ids[:split_idx])
    val_ids = set(all_subject_ids[split_idx:])
    
    train_indices = [i for i, s in enumerate(full_dataset.samples) if s['sid'] in train_ids]
    val_indices = [i for i, s in enumerate(full_dataset.samples) if s['sid'] in val_ids]
    
    train_ds = Subset(full_dataset, train_indices)
    val_ds = Subset(full_dataset, val_indices)
    
    bp_mean = full_dataset.bp_mean
    bp_std = full_dataset.bp_std
    
    if is_main_process():
        print(f"Data Split (Subject-wise): Train Subjs={len(train_ids)}, Val Subjs={len(val_ids)}")
        print(f"Samples: Train={len(train_ds)}, Val={len(val_ds)}")

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler, num_workers=4, pin_memory=True)

    # 3. 初始化模型
    if is_main_process(): print(f"Initializing Model...")
    model = TF_MAE_BloodPressure(
        pretrained_path=args.pretrained_path,
        output_dim=2,
        signal_len=args.signal_len,
        cwt_scales=args.cwt_scales,
        patch_size_time=args.patch_size_time,
        patch_size_freq=args.patch_size_freq,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_rank_ratio=args.mlp_rank_ratio,
        num_reasoning_tokens=args.num_reasoning_tokens,
        num_tabular_features=6
    )
    model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # 4. 损失函数 (加权 Huber)
    criterion = WeightedHuberLoss(delta=1.0, sbp_weight=2.0, dbp_weight=1.0).to(device)

    # 5. 优化器占位 (在循环中初始化)
    optimizer = None
    scheduler = None
    
    best_mae = float('inf')

    # 6. 训练循环
    for epoch in range(args.epochs):
        if is_main_process(): print(f"\nEpoch {epoch+1}/{args.epochs}")

        # --- 阶段 1: 冻结 Encoder (Warmup) ---
        if epoch == 0:
            freeze_encoder(model)
            # 只训练 Head，使用较大的学习率
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                    lr=args.head_lr, weight_decay=args.weight_decay)
            scheduler = None # Warmup 阶段通常不需要复杂调度，或者使用简单的线性

        # --- 阶段 2: 解冻 Encoder (Fine-tuning) ---
        if epoch == args.warmup_epochs:
            unfreeze_all(model)
            # 使用分层学习率
            param_groups = get_layer_wise_lr(model.module, base_lr=args.lr, layer_decay=0.75)
            optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)
            # 初始化 Cosine 调度器
            scheduler = CosineAnnealingLR(optimizer, T_max=(args.epochs - args.warmup_epochs), eta_min=1e-6)

        # 训练
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, use_amp=True)
        
        # 更新学习率
        if scheduler is not None:
            scheduler.step()

        # 验证
        val_loss, mae_sbp, mae_dbp, mae_avg, rmse, r2 = validate(
            model, val_loader, criterion, device, 
            total_len=len(val_ds), bp_mean=bp_mean, bp_std=bp_std, use_amp=True
        )

        if is_main_process():
            current_lr = optimizer.param_groups[0]['lr']
            print(f"LR: {current_lr:.2e} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"--------------------------------------------------")
            print(f"Val MAE SBP : {mae_sbp:.2f} mmHg")
            print(f"Val MAE DBP : {mae_dbp:.2f} mmHg")
            print(f"Val MAE Avg : {mae_avg:.2f} mmHg")
            print(f"Val R2 Score: {r2:.4f}")
            print(f"--------------------------------------------------")

            if mae_avg < best_mae:
                best_mae = mae_avg
                torch.save(model.module.state_dict(), os.path.join(args.save_dir, "best_model_bp.pth"))
                print(f">>> Best model saved! (MAE: {best_mae:.2f})")
    
    cleanup_distributed()

if __name__ == "__main__":
    main()