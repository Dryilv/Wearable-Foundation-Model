import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
from torch.cuda.amp import GradScaler
from torch.amp import autocast 

# 引入自定义模块
from dataset_regression import BPDataset
from model_finetune_bp_prediction import TF_MAE_BloodPressure
try:
    from utils import get_layer_wise_lr
except ImportError:
    # 如果没有 utils.py，使用一个简单的 fallback
    def get_layer_wise_lr(model, base_lr, layer_decay):
        return model.parameters()

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
    """收集所有 GPU 上的 tensor 到当前 GPU"""
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

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -------------------------------------------------------------------
# 训练逻辑 (回归)
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
        # 1. 获取数据
        x = batch['signal'].to(device)       # PPG 信号 [B, L]
        x_tab = batch['tabular'].to(device)  # 表格特征 [B, 6]
        y = batch['target'].to(device)       # 归一化后的血压 [B, 2]
        
        optimizer.zero_grad()
        
        # 2. 前向传播 (混合精度)
        with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
            # 传入信号和表格数据
            preds = model(x, x_tab) 
            loss = criterion(preds, y)
        
        # 3. 反向传播
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

        # 4. 记录损失
        if dist.is_initialized():
            reduced_loss = reduce_tensor(loss.detach())
            total_loss += reduced_loss.item()
        else:
            total_loss += loss.item()
            
        count += 1
        
        if is_main_process():
            iterator.set_postfix({'mse': total_loss / count})

    return total_loss / count

# -------------------------------------------------------------------
# 验证逻辑 (回归 + 反归一化)
# -------------------------------------------------------------------
def validate(model, loader, criterion, device, total_len, bp_mean, bp_std, use_amp=True):
    """
    Args:
        bp_mean: 训练集血压均值 (Tensor [2])，用于反归一化
        bp_std: 训练集血压标准差 (Tensor [2])
    """
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
            
            # 收集预测值和真实值 (保持归一化状态)
            local_preds.append(preds.float())
            local_targets.append(y.float())

    # 拼接当前 GPU 的结果
    local_preds = torch.cat(local_preds)
    local_targets = torch.cat(local_targets)

    # 收集所有 GPU 的结果
    if dist.is_initialized():
        all_preds = gather_tensors(local_preds, device)
        all_targets = gather_tensors(local_targets, device)
    else:
        all_preds = local_preds
        all_targets = local_targets

    if is_main_process():
        # 截断 Padding (DDP 可能会为了对齐 batch size 填充数据)
        if len(all_preds) > total_len:
            all_preds = all_preds[:total_len]
            all_targets = all_targets[:total_len]

        # --- 核心：反归一化 (Denormalization) ---
        # 将模型输出 (Z-Score) 转换回真实的 mmHg
        bp_mean = bp_mean.to(device)
        bp_std = bp_std.to(device)
        
        pred_real = all_preds * bp_std + bp_mean
        target_real = all_targets * bp_std + bp_mean
        
        pred_np = pred_real.cpu().numpy()
        target_np = target_real.cpu().numpy()

        # 计算指标 (分别计算 SBP 和 DBP)
        # index 0: SBP (收缩压), index 1: DBP (舒张压)
        mae_sbp = mean_absolute_error(target_np[:, 0], pred_np[:, 0])
        mae_dbp = mean_absolute_error(target_np[:, 1], pred_np[:, 1])
        mae_avg = (mae_sbp + mae_dbp) / 2.0
        
        rmse = np.sqrt(mean_squared_error(target_np, pred_np))
        r2 = r2_score(target_np, pred_np)
        
        avg_loss = total_loss / count
        return avg_loss, mae_sbp, mae_dbp, mae_avg, rmse, r2
    else:
        return 0, 0, 0, 0, 0, 0

# -------------------------------------------------------------------
# 主函数
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    # 数据路径
    parser.add_argument('--signal_dir', type=str, required=True, help="Path to folder containing .txt files")
    parser.add_argument('--excel_path', type=str, required=True, help="Path to .xlsx file with labels")
    
    # 训练配置
    parser.add_argument('--pretrained_path', type=str, default=None, help="Path to pretrained CWT-MAE weights")
    parser.add_argument('--save_dir', type=str, default="./checkpoints_bp")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)

    # 数据参数
    parser.add_argument('--signal_len', type=int, default=3000)
    
    # 模型参数 (需与预训练模型一致)
    parser.add_argument('--embed_dim', type=int, default=768) 
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--cwt_scales', type=int, default=64)
    parser.add_argument('--patch_size_time', type=int, default=50)
    parser.add_argument('--patch_size_freq', type=int, default=4)
    parser.add_argument('--mlp_rank_ratio', type=float, default=0.5)
    parser.add_argument('--num_reasoning_tokens', type=int, default=8)

    args = parser.parse_args()

    # 1. 初始化分布式环境
    local_rank, rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    set_seed(args.seed + rank)
    
    if is_main_process():
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"World Size: {world_size}, Master running on {device}")
        print(f"Arguments: {args}")

    # 2. 初始化 Dataset
    # normalize_labels=True 会对血压进行 Z-Score 归一化，这对回归任务收敛至关重要
    full_dataset = BPDataset(
        signal_dir=args.signal_dir, 
        excel_path=args.excel_path, 
        seq_len=args.signal_len,
        normalize_labels=True
    )
    
    # 划分训练集和验证集 (80% / 20%)
    total_len = len(full_dataset)
    train_len = int(0.8 * total_len)
    val_len = total_len - train_len
    
    train_ds, val_ds = random_split(
        full_dataset, [train_len, val_len], 
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    # 提取归一化参数 (用于验证时反归一化)
    # 注意：这里使用全集统计量。更严谨的做法是只用 train set 的统计量，
    # 但由于 BPDataset 在 init 时计算了全集，这里直接复用以简化代码。
    bp_mean = full_dataset.bp_mean
    bp_std = full_dataset.bp_std
    
    if is_main_process():
        print(f"Dataset loaded. Total: {total_len} | Train: {len(train_ds)} | Val: {len(val_ds)}")
        print(f"BP Stats (mmHg) - Mean: {bp_mean}, Std: {bp_std}")
        print(f"Tabular Features: {full_dataset.num_tabular_features} dims (Sex, Age, Height, Weight, HR, BMI)")

    # 3. DataLoader
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler, num_workers=4, pin_memory=True)

    # 4. 初始化模型
    if is_main_process():
        print(f"Initializing BP Regression Model (Multi-modal)...")
        
    model = TF_MAE_BloodPressure(
        pretrained_path=args.pretrained_path,
        output_dim=2, # 预测 SBP 和 DBP
        signal_len=args.signal_len,
        cwt_scales=args.cwt_scales,
        patch_size_time=args.patch_size_time,
        patch_size_freq=args.patch_size_freq,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_rank_ratio=args.mlp_rank_ratio,
        num_reasoning_tokens=args.num_reasoning_tokens,
        num_tabular_features=6 # 对应 dataset 中的 6 个特征
    )
    model.to(device)
    
    # 包装 DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # 5. 优化器与损失
    # 使用分层学习率衰减，让预训练的底层特征保持稳定，主要训练上层和 Head
    if isinstance(get_layer_wise_lr, type(lambda:0)): # 如果是 fallback 函数
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        param_groups = get_layer_wise_lr(model.module, base_lr=args.lr, layer_decay=0.75)
        optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)
    
    # 损失函数：MSE (均方误差)
    # 也可以尝试 nn.HuberLoss(delta=1.0) 如果数据中有较多离群点
    criterion = nn.MSELoss() 

    best_mae = float('inf')

    # 6. 训练循环
    for epoch in range(args.epochs):
        if is_main_process():
            print(f"\nEpoch {epoch+1}/{args.epochs}")

        # 训练
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, use_amp=True)

        # 验证
        val_loss, mae_sbp, mae_dbp, mae_avg, rmse, r2 = validate(
            model, val_loader, criterion, device, 
            total_len=len(val_ds), 
            bp_mean=bp_mean, 
            bp_std=bp_std,
            use_amp=True
        )

        if is_main_process():
            print(f"Train Loss (MSE): {train_loss:.4f}")
            print(f"Val   Loss (MSE): {val_loss:.4f}")
            print(f"--------------------------------------------------")
            print(f"Val   MAE SBP   : {mae_sbp:.2f} mmHg")
            print(f"Val   MAE DBP   : {mae_dbp:.2f} mmHg")
            print(f"Val   MAE Avg   : {mae_avg:.2f} mmHg  <-- Key Metric")
            print(f"Val   RMSE      : {rmse:.4f}")
            print(f"Val   R2 Score  : {r2:.4f}")
            print(f"--------------------------------------------------")

            # 保存最佳模型 (基于平均 MAE)
            if mae_avg < best_mae:
                best_mae = mae_avg
                save_path = os.path.join(args.save_dir, "best_model_bp.pth")
                torch.save(model.module.state_dict(), save_path)
                print(f">>> Best model saved to {save_path} (MAE: {best_mae:.2f} mmHg)")
            
            # 定期保存 Checkpoint
            if (epoch + 1) % 10 == 0:
                torch.save(model.module.state_dict(), os.path.join(args.save_dir, f"checkpoint_ep{epoch+1}.pth"))
    
    cleanup_distributed()

if __name__ == "__main__":
    main()