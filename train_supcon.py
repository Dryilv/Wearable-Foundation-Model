import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse

# 导入你的自定义模块
from model import CWT_MAE_RoPE, cwt_wrap
from dataset_cls import DownstreamClassificationDataset

# ===================================================================
# 1. 模型改造: 提取特征用于对比学习
# ===================================================================
class CWT_MAE_Encoder(CWT_MAE_RoPE):
    """
    继承原模型，专门用于提取下游任务特征。
    改动点：
    1. 移除 Masking。
    2. 生成顺序的 RoPE 位置编码。
    3. 输出 Global Average Pooling 特征。
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 移除 Decoder 以节省显存 (对比学习不需要重建)
        del self.decoder_blocks
        del self.decoder_embed
        del self.decoder_pred_spec
        del self.time_pred
        del self.decoder_norm
        del self.decoder_pos_embed
        del self.mask_token

    def forward_features(self, x):
        # x shape: (B, 1, 3000) or (B, 3000)
        if x.dim() == 3: x = x.squeeze(1)
        
        # 1. CWT 变换
        # 注意：dataset 中已经做了一次 Instance Norm (1D)，
        # 这里 cwt_wrap 后模型内部通常还会做一次 2D 的 Norm，保持双重 Norm 有助于稳定性
        imgs = cwt_wrap(x, num_scales=self.cwt_scales, lowest_scale=0.1, step=1.0)
        
        # 2. 内部标准化 (保持与预训练一致)
        dtype_orig = imgs.dtype
        imgs_f32 = imgs.float() 
        mean = imgs_f32.mean(dim=(2, 3), keepdim=True)
        std = imgs_f32.std(dim=(2, 3), keepdim=True)
        std = torch.clamp(std, min=1e-5)
        imgs = (imgs_f32 - mean) / std
        imgs = imgs.to(dtype=dtype_orig)

        # 3. Patch Embedding
        x = self.patch_embed(imgs) # (B, N, D)
        x = x + self.pos_embed[:, 1:, :]

        # 4. CLS Token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (B, N+1, D)

        # 5. 生成顺序的 RoPE (关键修正：不再是乱序)
        B, SeqLen, _ = x.shape
        pos_ids = torch.arange(SeqLen, device=x.device).unsqueeze(0).expand(B, -1)
        rope_cos, rope_sin = self.rope_encoder(x, pos_ids)
        
        # 6. Encoder Forward
        for blk in self.blocks:
            x = blk(x, rope_cos=rope_cos, rope_sin=rope_sin)
            
        x = self.norm(x)

        # 7. 特征聚合: Global Average Pooling (GAP)
        # 忽略 CLS token (index 0)，对所有 Patch tokens 求平均
        global_feat = torch.mean(x[:, 1:, :], dim=1) 
        
        return global_feat

class SupConMAE(nn.Module):
    def __init__(self, encoder, head_dim=128, feat_dim=768):
        super().__init__()
        self.encoder = encoder
        # Projection Head: (Dim -> Dim -> 128)
        self.head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, head_dim)
        )

    def forward(self, x):
        feat = self.encoder.forward_features(x)
        feat = F.normalize(feat, dim=1) # 特征层归一化
        proj = self.head(feat)
        proj = F.normalize(proj, dim=1) # 投影层归一化 (SupCon 必须)
        return proj

# ===================================================================
# 2. Loss Function
# ===================================================================
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 构建 Mask (同类为1，异类为0)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # 移除对角线 (自己不与自己对比)
        logits_mask = torch.scatter(
            torch.ones_like(mask), 
            1, 
            torch.arange(batch_size).view(-1, 1).to(device), 
            0
        )
        mask = mask * logits_mask
        
        # 数值稳定性
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # 计算分母
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        
        # 计算分子 (只取正样本对)
        mask_sum = mask.sum(1)
        mask_sum[mask_sum == 0] = 1.0 # 避免除零
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        
        loss = - mean_log_prob_pos
        loss = loss.mean()
        return loss

# ===================================================================
# 3. 训练主流程
# ===================================================================
def main():
    # --- 配置参数 ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data', help='数据根目录')
    parser.add_argument('--split_file', type=str, default='./split.json', help='数据集划分文件')
    parser.add_argument('--pretrained_path', type=str, default='./mae_pretrained.pth', help='预训练权重路径')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_supcon', help='保存路径')
    parser.add_argument('--batch_size', type=int, default=64, help='SupCon 需要较大的 Batch Size')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--temp', type=float, default=0.07, help='SupCon Temperature')
    parser.add_argument('--task_index', type=int, default=0, help='Dataset task index')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. 准备数据 ---
    print(f"Loading dataset from {args.data_root}...")
    train_dataset = DownstreamClassificationDataset(
        data_root=args.data_root,
        split_file=args.split_file,
        mode='train',
        signal_len=3000,
        task_index=args.task_index
    )
    
    # drop_last=True 很重要，避免最后一个 batch 只有 1 个样本导致 BN 或 Loss 报错
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        drop_last=True 
    )

    # --- 2. 初始化模型 ---
    print("Initializing model...")
    # 必须与预训练时的参数保持一致
    base_encoder = CWT_MAE_Encoder(
        signal_len=3000,
        cwt_scales=64,
        embed_dim=768,
        depth=12,
        num_heads=12,
        patch_size_time=50,
        patch_size_freq=4
    )

    # --- 3. 加载预训练权重 ---
    if os.path.exists(args.pretrained_path):
        print(f"Loading pretrained weights from {args.pretrained_path}")
        checkpoint = torch.load(args.pretrained_path, map_location='cpu')
        
        # 获取原始 state_dict
        raw_state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        
        # --- 关键修复步骤：去除 _orig_mod. 前缀 ---
        new_state_dict = {}
        for k, v in raw_state_dict.items():
            # 去除 torch.compile 产生的 _orig_mod. 前缀
            if k.startswith('_orig_mod.'):
                new_key = k[10:] # 去掉前10个字符
            # 去除 DDP 可能产生的 module. 前缀 (以防万一)
            elif k.startswith('module.'):
                new_key = k[7:]
            else:
                new_key = k
            
            new_state_dict[new_key] = v
            
        # 加载修复后的权重
        msg = base_encoder.load_state_dict(new_state_dict, strict=False)
        
        # --- 验证加载结果 ---
        # 我们只允许 decoder 相关的层缺失，Encoder 的层必须全部加载成功
        missing_encoder_keys = [k for k in msg.missing_keys if not k.startswith('decoder') and not k.startswith('mask_token') and not k.startswith('time_')]
        
        if len(missing_encoder_keys) > 0:
            print("\n[ERROR] 严重警告：以下 Encoder 核心权重未加载成功！")
            print(missing_encoder_keys[:10]) # 打印前10个看看
            raise RuntimeError("预训练权重加载失败，请检查 Key 匹配情况。")
        else:
            print("\n[SUCCESS] Encoder 权重加载成功！(忽略 Decoder 缺失是正常的)")
            
    else:
        print("WARNING: Pretrained path not found! Training from scratch.")

    # 包装 SupCon
    model = SupConMAE(base_encoder, head_dim=128, feat_dim=768).to(device)

    # --- 4. 优化器与 Loss ---
    criterion = SupConLoss(temperature=args.temp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # 可选: 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # --- 5. 训练循环 ---
    print("Start SupCon Training...")
    model.train()
    
    for epoch in range(args.epochs):
        total_loss = 0
        step_count = 0
        
        for batch_idx, (waveforms, labels) in enumerate(train_loader):
            waveforms = waveforms.to(device) # (B, 1, 3000)
            labels = labels.to(device)       # (B,)

            # Forward
            projections = model(waveforms) # (B, 128)
            
            # Loss
            loss = criterion(projections, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            step_count += 1
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] Step [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

        avg_loss = total_loss / step_count
        scheduler.step()
        
        print(f"=== Epoch {epoch+1} Finished. Avg Loss: {avg_loss:.4f} ===")
        
        # 保存权重
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
            save_path = os.path.join(args.save_dir, f'supcon_epoch_{epoch+1}.pth')
            # 保存 encoder 的权重，方便后续做 Linear Probing
            torch.save({
                'encoder': model.encoder.state_dict(),
                'head': model.head.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }, save_path)
            print(f"Checkpoint saved to {save_path}")

    print("Training Complete.")
    print("Next Step: Load 'encoder' weights into a classifier and train only the Linear Layer (Linear Probing).")

if __name__ == "__main__":
    main()