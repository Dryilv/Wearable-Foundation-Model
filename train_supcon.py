import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import numpy as np
import os
import argparse
import random
from collections import defaultdict
import builtins

# 导入你的自定义模块
from model import CWT_MAE_RoPE, cwt_wrap
from dataset_cls import DownstreamClassificationDataset

# ===================================================================
# 0. DDP 基础设置工具
# ===================================================================
def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
        dist.barrier()
        return rank, local_rank, world_size
    else:
        print("Not using distributed mode")
        return 0, 0, 1

def cleanup():
    dist.destroy_process_group()

# 抑制非主进程的 print
def setup_for_distributed(is_master):
    builtin_print = builtins.print
    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    builtins.print = print

# ===================================================================
# 1. 分布式 Balanced Sampler (关键！)
# ===================================================================
class DistributedBalancedBatchSampler(Sampler):
    """
    每个 GPU 独立进行 P-K 采样。
    为了保证每个 GPU 上的 Batch 都有正样本对，我们不进行全局切分，
    而是让每个 GPU 都在全量数据上进行 P-K 采样，但通过 Seed 错开。
    """
    def __init__(self, dataset, n_classes, n_samples, num_replicas, rank):
        self.dataset = dataset
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.batch_size = n_classes * n_samples
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        
        # 构建索引 (只在初始化时做一次)
        # 注意：为了速度，这里假设 dataset 比较快，或者你可以缓存 labels
        self.label_to_indices = defaultdict(list)
        # 优化：如果 dataset 有 labels 属性直接读取，否则遍历
        if hasattr(dataset, 'labels'):
             labels = dataset.labels
             for idx, label in enumerate(labels):
                 self.label_to_indices[int(label)].append(idx)
        else:
            # 慢速遍历
            for idx in range(len(dataset)):
                _, label = dataset[idx]
                self.label_to_indices[int(label)].append(idx)
                
        self.classes = list(self.label_to_indices.keys())
        
        # 计算每个 GPU 应该跑多少个 Batch
        # 这里简单处理：每个 GPU 跑的总量近似等于 数据总量 / GPU数
        self.num_samples = int(len(dataset) // self.num_replicas)
        self.num_batches = self.num_samples // self.batch_size

    def __iter__(self):
        # 关键：设置随机种子，确保不同 GPU 选到的类不同，且随 Epoch 变化
        g = torch.Generator()
        g.manual_seed(self.epoch * 1000 + self.rank)
        
        for _ in range(self.num_batches):
            batch_indices = []
            
            # 1. 随机选 P 个类
            indices = torch.randperm(len(self.classes), generator=g).tolist()
            selected_class_indices = indices[:self.n_classes]
            # 如果类别不够，允许重复
            while len(selected_class_indices) < self.n_classes:
                selected_class_indices.extend(torch.randint(0, len(self.classes), (self.n_classes - len(selected_class_indices),), generator=g).tolist())
            
            selected_classes = [self.classes[i] for i in selected_class_indices]
            
            for cls in selected_classes:
                cls_indices = self.label_to_indices[cls]
                # 2. 每个类随机选 K 个样本
                if len(cls_indices) >= self.n_samples:
                    # 随机选
                    perm = torch.randperm(len(cls_indices), generator=g).tolist()
                    sel = [cls_indices[i] for i in perm[:self.n_samples]]
                    batch_indices.extend(sel)
                else:
                    # 样本不够，重复采样
                    sel = [cls_indices[torch.randint(0, len(cls_indices), (1,), generator=g).item()] for _ in range(self.n_samples)]
                    batch_indices.extend(sel)
            
            yield batch_indices

    def __len__(self):
        return self.num_batches
    
    def set_epoch(self, epoch):
        self.epoch = epoch

# ===================================================================
# 2. 模型定义 (保持不变)
# ===================================================================
class CWT_MAE_Encoder(CWT_MAE_RoPE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        del self.decoder_blocks, self.decoder_embed, self.decoder_pred_spec, self.time_pred, self.decoder_norm, self.decoder_pos_embed, self.mask_token

    def forward_features(self, x):
        if x.dim() == 3: x = x.squeeze(1)
        imgs = cwt_wrap(x, num_scales=self.cwt_scales, lowest_scale=0.1, step=1.0)
        dtype_orig = imgs.dtype
        imgs_f32 = imgs.float() 
        mean = imgs_f32.mean(dim=(2, 3), keepdim=True)
        std = imgs_f32.std(dim=(2, 3), keepdim=True)
        std = torch.clamp(std, min=1e-5)
        imgs = (imgs_f32 - mean) / std
        imgs = imgs.to(dtype=dtype_orig)

        x = self.patch_embed(imgs)
        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        B, SeqLen, _ = x.shape
        pos_ids = torch.arange(SeqLen, device=x.device).unsqueeze(0).expand(B, -1)
        rope_cos, rope_sin = self.rope_encoder(x, pos_ids)
        
        for blk in self.blocks:
            x = blk(x, rope_cos=rope_cos, rope_sin=rope_sin)
        x = self.norm(x)
        global_feat = torch.mean(x[:, 1:, :], dim=1) 
        return global_feat

class SupConMAE(nn.Module):
    def __init__(self, encoder, head_dim=128, feat_dim=768):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, head_dim)
        )

    def forward(self, x):
        feat = self.encoder.forward_features(x)
        feat = F.normalize(feat, dim=1)
        proj = self.head(feat)
        proj = F.normalize(proj, dim=1)
        return proj

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        mask = mask * logits_mask
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        mask_sum = mask.sum(1)
        mask_sum[mask_sum == 0] = 1.0
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        loss = - mean_log_prob_pos
        loss = loss.mean()
        return loss

# ===================================================================
# 3. 主流程
# ===================================================================
def main():
    # 1. 初始化 DDP
    rank, local_rank, world_size = setup_distributed()
    setup_for_distributed(rank == 0) # 只有 rank 0 打印日志

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--split_file', type=str, default='./split.json')
    parser.add_argument('--pretrained_path', type=str, default='./mae_pretrained.pth')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_supcon_ddp')
    # 注意：这里的 batch_size 是单卡的 batch size
    parser.add_argument('--n_classes', type=int, default=8, help='每个GPU采样多少类')
    parser.add_argument('--n_samples', type=int, default=8, help='每类采样多少个')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--temp', type=float, default=0.2) # 保持你调试好的温度
    args = parser.parse_args()

    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"Running DDP with {world_size} GPUs.")
        print(f"Per-GPU Batch Size: {args.n_classes * args.n_samples}")
        print(f"Global Batch Size: {args.n_classes * args.n_samples * world_size}")

    # 2. 准备数据
    train_dataset = DownstreamClassificationDataset(
        data_root=args.data_root,
        split_file=args.split_file,
        mode='train',
        signal_len=3000
    )

    # 使用自定义的分布式 Balanced Sampler
    sampler = DistributedBalancedBatchSampler(
        train_dataset, 
        n_classes=args.n_classes, 
        n_samples=args.n_samples,
        num_replicas=world_size,
        rank=rank
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_sampler=sampler, # 使用 batch_sampler
        num_workers=4, 
        pin_memory=True
    )

    # 3. 初始化模型
    base_encoder = CWT_MAE_Encoder(
        signal_len=3000, cwt_scales=64, embed_dim=768, depth=12, num_heads=12
    )

    # 加载权重 (Rank 0 加载即可，DDP 会广播，但为了保险通常每个进程都加载)
    if os.path.exists(args.pretrained_path):
        checkpoint = torch.load(args.pretrained_path, map_location='cpu')
        raw_state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        new_state_dict = {}
        for k, v in raw_state_dict.items():
            if k.startswith('_orig_mod.'): new_key = k[10:]
            elif k.startswith('module.'): new_key = k[7:]
            else: new_key = k
            new_state_dict[new_key] = v
        msg = base_encoder.load_state_dict(new_state_dict, strict=False)
        if rank == 0: print(f"Weights loaded: {msg}")

    model = SupConMAE(base_encoder, head_dim=128, feat_dim=768).cuda()
    
    # 转换为 SyncBatchNorm (如果有 BN 层的话，虽然这里主要是 LN)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    # 包装 DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # 4. 优化器 (分层学习率)
    optimizer = torch.optim.AdamW([
        {'params': model.module.encoder.parameters(), 'lr': 1e-5}, # 注意 model.module
        {'params': model.module.head.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-4)
    
    criterion = SupConLoss(temperature=args.temp)

    # 5. 训练循环
    for epoch in range(args.epochs):
        # 必须设置 epoch，否则 shuffle 不生效
        sampler.set_epoch(epoch)
        
        model.train()
        total_loss = torch.zeros(1).cuda()
        
        for batch_idx, (waveforms, labels) in enumerate(train_loader):
            waveforms, labels = waveforms.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            
            # Forward
            projections = model(waveforms)
            loss = criterion(projections, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.detach()
            
            if rank == 0 and batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}] Step [{batch_idx}] Loss: {loss.item():.4f}")

        # 汇总所有 GPU 的 Loss 用于打印
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        avg_loss = total_loss.item() / (len(train_loader) * world_size)
        
        if rank == 0:
            print(f"=== Epoch {epoch+1} Finished. Global Avg Loss: {avg_loss:.4f} ===")
            
            # 保存权重 (只保存 model.module)
            if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
                save_path = os.path.join(args.save_dir, f'supcon_ddp_epoch_{epoch+1}.pth')
                torch.save({
                    'encoder': model.module.encoder.state_dict(),
                    'head': model.module.head.state_dict(),
                    'epoch': epoch
                }, save_path)
                print(f"Checkpoint saved to {save_path}")
        
        dist.barrier() # 等待所有进程结束本轮

    cleanup()

if __name__ == "__main__":
    main()