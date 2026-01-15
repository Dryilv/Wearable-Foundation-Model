import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler
from torch.amp import autocast
# 【关键】引入支持梯度的 all_gather
from torch.distributed.nn import all_gather 
from tqdm import tqdm
import numpy as np
import pickle
import json

# 导入自定义模块
from model_supcon import SupCon_CWT_MAE
from losses import SupConLoss
from augmentations import SignalAugmentation

# 开启矩阵乘法加速
torch.set_float32_matmul_precision('high')

# -------------------------------------------------------------------
# DDP 工具函数
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

# -------------------------------------------------------------------
# Dataset Wrapper
# -------------------------------------------------------------------
class SupConDataset(Dataset):
    def __init__(self, data_root, split_file, mode='train', signal_len=3000):
        self.data_root = data_root
        self.signal_len = signal_len
        
        with open(split_file, 'r') as f:
            self.file_list = json.load(f)[mode]
            
        # 初始化数据增强器
        self.augment = SignalAugmentation(signal_len=signal_len, mode=mode)
        
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"[{mode}] Loaded {len(self.file_list)} samples for SupCon.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        file_path = os.path.join(self.data_root, filename)
        
        try:
            with open(file_path, 'rb') as f:
                content = pickle.load(f)
            
            raw_data = content['data']
            if raw_data.ndim == 2: 
                raw_signal = raw_data[0, :]
            else:
                raw_signal = raw_data
            
            # 获取标签
            if 'label' in content and isinstance(content['label'], list):
                label = int(content['label'][0]['class'])
            else:
                label = 0 
            
            # [核心] 生成两个 View
            views = self.augment(raw_signal) # 返回 [view1, view2]
            
            return views[0], views[1], torch.tensor(label, dtype=torch.long)
            
        except Exception as e:
            # print(f"Error loading {file_path}: {e}")
            dummy = torch.zeros(1, self.signal_len)
            return dummy, dummy, torch.tensor(0, dtype=torch.long)

# -------------------------------------------------------------------
# 训练逻辑
# -------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, epoch, use_amp=True):
    model.train()
    if hasattr(loader.sampler, 'set_epoch'):
        loader.sampler.set_epoch(epoch)
        
    total_loss = 0
    count = 0
    
    # 优先使用 bfloat16
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = GradScaler(enabled=(use_amp and amp_dtype == torch.float16))
    
    iterator = tqdm(loader, desc=f"Epoch {epoch} SupCon") if dist.get_rank() == 0 else loader
    
    for images1, images2, labels in iterator:
        images1 = images1.to(device, non_blocking=True)
        images2 = images2.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        bsz = labels.shape[0]
        
        # 1. 拼接两个 View: [2*B, 1, L]
        images = torch.cat([images1, images2], dim=0)
        
        optimizer.zero_grad()
        
        with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
            # 2. Forward
            # features: [2*B, feat_dim] (已经归一化)
            features = model(images)
            
            # 3. Split views
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            
            # 4. DDP Gather (构建全局 Batch)
            if dist.is_initialized():
                # 【关键修复】使用 torch.distributed.nn.all_gather
                # 这个函数会自动处理 Autograd Graph，确保梯度能回传到所有 GPU
                # all_gather 返回的是 list of tensors，需要 cat 起来
                all_f1 = torch.cat(all_gather(f1), dim=0)
                all_f2 = torch.cat(all_gather(f2), dim=0)
                all_labels = torch.cat(all_gather(labels), dim=0)
                
                # 拼接成 SupConLoss 需要的格式: [Global_Batch, 2, Dim]
                features_global = torch.cat([all_f1.unsqueeze(1), all_f2.unsqueeze(1)], dim=1)
                labels_global = all_labels
            else:
                features_global = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                labels_global = labels

            # 5. Compute Loss
            loss = criterion(features_global, labels_global)
        
        # 6. Backward
        if use_amp and amp_dtype == torch.float16:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        count += 1
        
        if dist.get_rank() == 0:
            iterator.set_postfix({'loss': total_loss / count})
            
    return total_loss / count

# -------------------------------------------------------------------
# 主函数
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--split_file', type=str, required=True)
    parser.add_argument('--pretrained_path', type=str, default=None, help="CWT-MAE 预训练权重")
    parser.add_argument('--save_dir', type=str, default="./checkpoints_supcon")
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=64, help="Per-GPU batch size")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--base_lr', type=float, default=1e-4, help="Encoder LR")
    parser.add_argument('--head_lr', type=float, default=1e-3, help="Projection Head LR (should be larger)")
    parser.add_argument('--temp', type=float, default=0.1, help="SupCon temperature (try 0.1 or 0.2)")
    parser.add_argument('--feat_dim', type=int, default=128, help="Projection head dimension")
    
    # 模型参数 (必须与预训练一致)
    parser.add_argument('--signal_len', type=int, default=3000)
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--cwt_scales', type=int, default=64)
    parser.add_argument('--patch_size_time', type=int, default=50)
    parser.add_argument('--patch_size_freq', type=int, default=4)
    parser.add_argument('--mlp_rank_ratio', type=float, default=0.5)
    
    args = parser.parse_args()
    
    local_rank, rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"Running SupCon on {world_size} GPUs. Global Batch Size: {args.batch_size * world_size}")
    
    # 1. Dataset
    train_ds = SupConDataset(args.data_root, args.split_file, mode='train', signal_len=args.signal_len)
    
    sampler = DistributedSampler(train_ds, shuffle=True)
    # drop_last=True 很重要，保证所有 GPU 的 batch size 一致，否则 all_gather 会卡死
    loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, 
                        num_workers=4, pin_memory=True, drop_last=True)
    
    # 2. Model
    model = SupCon_CWT_MAE(
        pretrained_path=args.pretrained_path,
        head='mlp',
        feat_dim=args.feat_dim,
        # Encoder Args
        signal_len=args.signal_len,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        cwt_scales=args.cwt_scales,
        patch_size_time=args.patch_size_time,
        patch_size_freq=args.patch_size_freq,
        mlp_rank_ratio=args.mlp_rank_ratio
    ).to(device)
    
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    
    # 3. Optimizer (分层学习率)
    # Projection Head 是随机初始化的，需要更大的 LR 才能动起来
    head_params = list(map(id, model.module.head.parameters()))
    base_params = filter(lambda p: id(p) not in head_params, model.parameters())
    
    optimizer = optim.AdamW([
        {'params': base_params, 'lr': args.base_lr},       # Encoder: 1e-4
        {'params': model.module.head.parameters(), 'lr': args.head_lr} # Head: 1e-3
    ], weight_decay=0.05)
    
    criterion = SupConLoss(temperature=args.temp).to(device)
    
    # 4. Loop
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, loader, criterion, optimizer, device, epoch)
        
        if rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | Loss: {loss:.4f}")
            
            # 保存权重
            if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
                save_path = os.path.join(args.save_dir, f"supcon_encoder_epoch_{epoch+1}.pth")
                # 只保存 Encoder 部分，方便 Phase 2 加载
                torch.save(model.module.encoder_model.state_dict(), save_path)
                print(f"Saved encoder weights to {save_path}")

    cleanup_distributed()

if __name__ == "__main__":
    main()