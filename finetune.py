import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import GradScaler
from torch.amp import autocast 

from dataset_cls import DownstreamClassificationDataset
from model_finetune import TF_MAE_Classifier
from utils import get_layer_wise_lr

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
# 训练与验证逻辑
# -------------------------------------------------------------------
def mixup_data(x, y, alpha=1.0, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, use_amp=True):
    model.train()
    if hasattr(loader.sampler, 'set_epoch'):
        loader.sampler.set_epoch(epoch)

    total_loss = 0
    count = 0
    
    # 【关键】优先使用 bfloat16，与预训练保持一致
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    iterator = tqdm(loader, desc=f"Epoch {epoch} Train") if is_main_process() else loader
    
    for x, y in iterator:
        x, y = x.to(device), y.to(device)
        
        inputs, targets_a, targets_b, lam = mixup_data(x, y, alpha=0.2, device=device)
        
        optimizer.zero_grad()
        
        with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
            logits = model(inputs)
            loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
        
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

def validate(model, loader, criterion, device, num_classes, use_amp=True):
    model.eval()
    total_loss = 0
    count = 0
    
    local_labels = []
    local_probs = []
    
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    iterator = tqdm(loader, desc="Validating") if is_main_process() else loader

    with torch.no_grad():
        for x, y in iterator:
            x, y = x.to(device), y.to(device)
            
            with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            
            if dist.is_initialized():
                reduced_loss = reduce_tensor(loss)
                total_loss += reduced_loss.item()
            else:
                total_loss += loss.item()
            count += 1
            
            probs = F.softmax(logits, dim=1)
            local_labels.append(y)
            local_probs.append(probs.float()) 

    local_labels = torch.cat(local_labels)
    local_probs = torch.cat(local_probs)

    if dist.is_initialized():
        all_labels = gather_tensors(local_labels, device)
        all_probs = gather_tensors(local_probs, device)
    else:
        all_labels = local_labels
        all_probs = local_probs

    if is_main_process():
        all_labels_np = all_labels.cpu().numpy()
        all_probs_np = all_probs.cpu().numpy()

        try:
            if num_classes == 2:
                auroc = roc_auc_score(all_labels_np, all_probs_np[:, 1])
            else:
                auroc = roc_auc_score(all_labels_np, all_probs_np, multi_class='ovr', average='macro')
        except:
            auroc = 0.0

        final_preds = np.argmax(all_probs_np, axis=1)
        final_acc = accuracy_score(all_labels_np, final_preds)
        final_f1 = f1_score(all_labels_np, final_preds, average='macro')
        precision = precision_score(all_labels_np, final_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels_np, final_preds, average='macro', zero_division=0)
        
        avg_loss = total_loss / count
        return avg_loss, final_acc, precision, recall, final_f1, auroc
    else:
        return 0, 0, 0, 0, 0, 0

# -------------------------------------------------------------------
# 主函数
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--split_file', type=str, required=True)
    parser.add_argument('--pretrained_path', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default="./checkpoints_cls768")
    
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--signal_len', type=int, default=1000)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)

    # CWT-MAE 模型参数
    parser.add_argument('--embed_dim', type=int, default=768) 
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--cwt_scales', type=int, default=64)
    parser.add_argument('--patch_size_time', type=int, default=50)
    parser.add_argument('--patch_size_freq', type=int, default=4)
    
    # 【新增】张量分解参数，必须与预训练一致
    parser.add_argument('--mlp_rank_ratio', type=float, default=0.5, help="MLP Rank Ratio (must match pretraining)")

    parser.add_argument('--clean_indices_path', type=str, default=None)
    parser.add_argument('--clean_test_indices_path', type=str, default=None)
    args = parser.parse_args()

    local_rank, rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    if is_main_process():
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"World Size: {world_size}, Master running on {device}")

    # Dataset
    train_ds = DownstreamClassificationDataset(
        args.data_root, args.split_file, mode='train', signal_len=args.signal_len
    )
    val_ds = DownstreamClassificationDataset(
        args.data_root, args.split_file, mode='test', signal_len=args.signal_len
    )

    if args.clean_indices_path and os.path.exists(args.clean_indices_path):
        if is_main_process():
            print(f"\n[Data Cleaning] Loading clean indices from {args.clean_indices_path}...")
        clean_indices = np.load(args.clean_indices_path)
        clean_indices = clean_indices[clean_indices < len(train_ds)]
        train_ds = Subset(train_ds, clean_indices)
        
    if args.clean_test_indices_path and os.path.exists(args.clean_test_indices_path):
        if is_main_process():
            print(f"\n[Test Cleaning] Loading indices from {args.clean_test_indices_path}...")
        clean_test_indices = np.load(args.clean_test_indices_path)
        clean_test_indices = clean_test_indices[clean_test_indices < len(val_ds)]
        val_ds = Subset(val_ds, clean_test_indices)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler, num_workers=4, pin_memory=True)

    if is_main_process():
        print(f"Initializing CWT-MAE Classifier (RoPE + Tensorized)...")
        
    model = TF_MAE_Classifier(
        pretrained_path=args.pretrained_path,
        num_classes=args.num_classes,
        signal_len=args.signal_len,
        cwt_scales=args.cwt_scales,
        patch_size_time=args.patch_size_time,
        patch_size_freq=args.patch_size_freq,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        # 传入新增参数
        mlp_rank_ratio=args.mlp_rank_ratio
    )
    model.to(device)
    
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # Optimizer
    print("Freezing Encoder for Linear Probing...")
    for param in model.encoder_model.parameters():
        param.requires_grad = False
    
    # 确保 Head 是可训练的
    for param in model.head.parameters():
        param.requires_grad = True
        
    # 此时 param_groups 只包含 head 的参数
    param_groups = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(param_groups, lr=1e-3, weight_decay=0.05) # LR 可以大一点
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_metric = 0.0

    for epoch in range(args.epochs):
        if is_main_process():
            print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, use_amp=True)

        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = validate(
            model, val_loader, criterion, device, args.num_classes, use_amp=True
        )

        if is_main_process():
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val   Loss: {val_loss:.4f}   | Acc: {val_acc:.4f}")
            print(f"      F1  : {val_f1:.4f}     | AUC: {val_auc:.4f}")

            metric_to_track = val_f1 if args.num_classes > 2 else val_auc
            
            if metric_to_track > best_metric:
                best_metric = metric_to_track
                torch.save(model.module.state_dict(), os.path.join(args.save_dir, "best_model.pth"))
                print(f">>> Best model saved! (Metric: {best_metric:.4f})")
    
    cleanup_distributed()

if __name__ == "__main__":
    main()