import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# 假设你的原模型代码保存在 model.py 中
from model import CWT_MAE_RoPE, cwt_wrap

# ===================================================================
# 1. 改造模型：特征提取器 (Feature Extractor)
# ===================================================================
class CWT_MAE_Encoder(CWT_MAE_RoPE):
    """
    继承原模型，添加专门用于下游任务的特征提取方法。
    核心改动：
    1. 移除 Masking 过程
    2. 生成顺序的 RoPE 位置编码
    3. 使用 Global Average Pooling (GAP) 替代 CLS Token
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 如果显存紧张，可以删除 Decoder 部分，因为对比学习不需要它
        # del self.decoder_blocks
        # del self.decoder_embed
        # del self.decoder_pred_spec
        # del self.time_pred

    def forward_features(self, x):
        # -------------------------------------------------------
        # 1. 预处理 (CWT + Normalization) - 保持与预训练完全一致
        # -------------------------------------------------------
        if x.dim() == 3: x = x.squeeze(1)
        # CWT 变换
        imgs = cwt_wrap(x, num_scales=self.cwt_scales, lowest_scale=0.1, step=1.0)
        
        # 标准化 (Instance Normalization logic from original code)
        dtype_orig = imgs.dtype
        imgs_f32 = imgs.float() 
        mean = imgs_f32.mean(dim=(2, 3), keepdim=True)
        std = imgs_f32.std(dim=(2, 3), keepdim=True)
        std = torch.clamp(std, min=1e-5)
        imgs = (imgs_f32 - mean) / std
        imgs = imgs.to(dtype=dtype_orig)

        # -------------------------------------------------------
        # 2. Patch Embedding & Positional Embedding
        # -------------------------------------------------------
        x = self.patch_embed(imgs) # (B, N, D)
        x = x + self.pos_embed[:, 1:, :]

        # 拼接 CLS Token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (B, N+1, D)

        # -------------------------------------------------------
        # 3. 构建顺序的 RoPE (关键修正)
        # -------------------------------------------------------
        # 原代码是乱序 Mask 的，这里我们需要完整的顺序
        B, SeqLen, _ = x.shape
        # 生成 [0, 1, 2, ..., SeqLen-1]
        pos_ids = torch.arange(SeqLen, device=x.device).unsqueeze(0).expand(B, -1)
        
        # 获取旋转位置编码
        rope_cos, rope_sin = self.rope_encoder(x, pos_ids)
        
        # -------------------------------------------------------
        # 4. Transformer Encoder Forward
        # -------------------------------------------------------
        for blk in self.blocks:
            x = blk(x, rope_cos=rope_cos, rope_sin=rope_sin)
            
        x = self.norm(x)

        # -------------------------------------------------------
        # 5. 特征聚合 (Pooling)
        # -------------------------------------------------------
        # MAE 论文建议：对于分类任务，Global Average Pooling (GAP) 优于 CLS Token
        # x[:, 0] 是 CLS, x[:, 1:] 是所有 Patch
        global_feat = torch.mean(x[:, 1:, :], dim=1) 
        
        return global_feat

# ===================================================================
# 2. SupCon 整体架构 (Backbone + Projection Head)
# ===================================================================
class SupConMAE(nn.Module):
    def __init__(self, pretrained_model, head_dim=128, feat_dim=768):
        super().__init__()
        self.encoder = pretrained_model
        
        # Projection Head: 将特征映射到对比空间
        # 结构通常为: Linear -> ReLU -> Linear
        self.head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, head_dim)
        )

    def forward(self, x):
        # 1. 提取特征 (B, 768)
        feat = self.encoder.forward_features(x)
        
        # 2. 特征归一化 (可选，但在对比学习中通常有益)
        feat = F.normalize(feat, dim=1)
        
        # 3. 投影 (B, 128)
        proj = self.head(feat)
        
        # 4. 投影向量必须做 L2 归一化
        proj = F.normalize(proj, dim=1)
        
        return proj

# ===================================================================
# 3. Supervised Contrastive Loss (核心 Loss)
# ===================================================================
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: (batch_size, head_dim) 归一化后的投影特征
            labels: (batch_size) 真实标签
        """
        device = features.device
        batch_size = features.shape[0]
        
        # 计算相似度矩阵 (B, B)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 构建 Label Mask
        # mask[i, j] = 1 表示 i 和 j 是同一类
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # 移除对角线 (自己与自己的对比不计算在内)
        logits_mask = torch.scatter(
            torch.ones_like(mask), 
            1, 
            torch.arange(batch_size).view(-1, 1).to(device), 
            0
        )
        mask = mask * logits_mask
        
        # 数值稳定性处理 (减去每行最大值)
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # 计算分母: sum(exp(logits)) over all negatives and positives (except self)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        
        # 计算分子: 只取正样本对 (Same Class) 的 log_prob
        # mean_log_prob_pos: 每个样本与其所有正样本对的平均对数概率
        # 如果一个样本没有正样本对（batch里只有它自己这一类），mask.sum(1) 为 0，需要避免除零
        mask_sum = mask.sum(1)
        mask_sum[mask_sum == 0] = 1.0 # 避免除零，此时分子也是0，结果为0
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        
        # Loss
        loss = - mean_log_prob_pos
        loss = loss.mean()
        
        return loss

# ===================================================================
# 4. 训练流程示例
# ===================================================================
if __name__ == "__main__":
    # 配置参数
    SIGNAL_LEN = 3000
    EMBED_DIM = 768
    BATCH_SIZE = 32
    LR = 1e-4
    EPOCHS = 50
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PRETRAINED_PATH = "/home/bml/storage/mnt/v-044d0fb740b04ad3/org/WFM/vit16trans/checkpoint_rope_tensor_768/checkpoint_last.pth" # 你的预训练权重路径

    # 1. 初始化模型
    # 注意：参数必须与预训练时完全一致
    base_model = CWT_MAE_Encoder(
        signal_len=SIGNAL_LEN, 
        embed_dim=EMBED_DIM, 
        depth=12, 
        num_heads=12,
        cwt_scales=64
    )

    # 2. 加载预训练权重
    try:
        checkpoint = torch.load(PRETRAINED_PATH, map_location='cpu')
        # 处理权重键值不匹配问题 (例如预训练保存时带了 'module.' 前缀)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        msg = base_model.load_state_dict(state_dict, strict=False)
        print(f"Pretrained weights loaded: {msg}")
    except FileNotFoundError:
        print("Warning: Pretrained weights not found, initializing randomly.")

    # 3. 包装 SupCon 模型
    model = SupConMAE(base_model, head_dim=128, feat_dim=EMBED_DIM).to(DEVICE)

    # 4. 准备数据 (示例)
    # 假设你有 waveforms (N, 3000) 和 labels (N,)
    dummy_data = torch.randn(100, SIGNAL_LEN)
    dummy_labels = torch.randint(0, 5, (100,)) # 5类
    dataset = TensorDataset(dummy_data, dummy_labels)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # 5. 优化器 & Loss
    criterion = SupConLoss(temperature=0.07)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # 6. 训练循环
    print("Start SupCon Training...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, (waveforms, labels) in enumerate(dataloader):
            waveforms, labels = waveforms.to(DEVICE), labels.to(DEVICE)
            
            # Forward
            projections = model(waveforms)
            
            # Calculate Loss
            loss = criterion(projections, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}")

    # ===================================================================
    # 7. 训练结束后的使用方式 (Linear Probing)
    # ===================================================================
    print("\nTraining finished. Extracting encoder for classification...")
    
    # 提取训练好的 Encoder
    trained_encoder = model.encoder
    
    # 构建最终分类器
    class LinearClassifier(nn.Module):
        def __init__(self, encoder, num_classes):
            super().__init__()
            self.encoder = encoder
            self.fc = nn.Linear(EMBED_DIM, num_classes)
            
        def forward(self, x):
            # 此时只用 encoder 提取特征，不需要 projection head
            with torch.no_grad():
                feat = self.encoder.forward_features(x)
            return self.fc(feat)

    classifier = LinearClassifier(trained_encoder, num_classes=5).to(DEVICE)
    
    # 此时通常冻结 Encoder，只训练 FC
    for param in classifier.encoder.parameters():
        param.requires_grad = False
        
    print("Classifier ready for Linear Probing.")