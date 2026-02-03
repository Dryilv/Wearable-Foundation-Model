import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 确保 model.py (包含 CWT_MAE_RoPE 和 cwt_wrap) 在同一目录下
from model_1  import CWT_MAE_RoPE, cwt_wrap

# ===================================================================
# 1. 隐式思维链模块 (Latent Reasoning / Chain-of-Thought Head)
# ===================================================================
class LatentReasoningHead(nn.Module):
    def __init__(self, embed_dim, num_heads, num_classes, num_reasoning_tokens=32, dropout=0.1):
        super().__init__()
        self.num_reasoning_tokens = num_reasoning_tokens
        self.embed_dim = embed_dim
        
        # 推理令牌 (Query)
        self.reasoning_tokens = nn.Parameter(torch.zeros(1, num_reasoning_tokens, embed_dim))
        
        # Cross-Attention: Query=Reasoning, Key/Value=Signal Features
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Self-Attention: Reasoning tokens 内部交互
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm3 = nn.LayerNorm(embed_dim)
        
        # Classifier
        self.classifier = nn.Linear(embed_dim, num_classes)
        
        self._init_weights()
        nn.init.normal_(self.reasoning_tokens, std=0.02)

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x_encoder):
        # x_encoder: (B, Total_Tokens, D)
        # Total_Tokens = M * N_patches (变长)
        
        B = x_encoder.shape[0]
        queries = self.reasoning_tokens.expand(B, -1, -1) # (B, N_reason, D)
        
        # Cross Attention
        # 这里的 Key/Value 长度是 M*N，对于 MultiheadAttention 来说没问题
        attn_out, _ = self.cross_attn(query=queries, key=x_encoder, value=x_encoder)
        queries = self.norm1(queries + attn_out)
        
        # Self Attention
        attn_out2, _ = self.self_attn(query=queries, key=queries, value=queries)
        queries = self.norm2(queries + attn_out2)
        
        # FFN
        queries = self.norm3(queries + self.ffn(queries))
        
        # Global Pooling & Classify
        decision_token = queries.mean(dim=1) 
        logits = self.classifier(decision_token)
        return logits

# ===================================================================
# 2. 主分类器模型封装
# ===================================================================
class TF_MAE_Classifier(nn.Module):
    def __init__(self, pretrained_path, num_classes, 
                 mlp_rank_ratio=0.5, 
                 use_cot=True, 
                 num_reasoning_tokens=16, 
                 **kwargs):
        super().__init__()
        
        # 1. 初始化 Encoder (CWT-MAE-RoPE)
        # 注意：不再需要 max_num_channels
        self.encoder_model = CWT_MAE_RoPE(
            mlp_rank_ratio=mlp_rank_ratio,
            mask_ratio=0.0, # 微调时关闭 Mask
            **kwargs
        )
        self.embed_dim = kwargs.get('embed_dim', 768)
        
        # 2. 加载预训练权重
        if pretrained_path:
            self._load_pretrained_weights(pretrained_path)
        
        # 3. 清理 Decoder (节省显存)
        self._delete_decoder_components()

        # 4. 初始化分类头
        if use_cot:
            print(f">>> Initializing Latent Reasoning Head (CoT) with {num_reasoning_tokens} tokens.")
            self.head = LatentReasoningHead(
                embed_dim=self.embed_dim,
                num_heads=kwargs.get('num_heads', 12),
                num_classes=num_classes,
                num_reasoning_tokens=num_reasoning_tokens,
                dropout=0.2
            )
        else:
            # 简单的 Linear Head
            self.head = nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, num_classes)
            )

    def _delete_decoder_components(self):
        """删除预训练模型中的 Decoder 部分"""
        del self.encoder_model.decoder_blocks
        del self.encoder_model.decoder_embed
        del self.encoder_model.decoder_pred_spec
        del self.encoder_model.time_reducer
        del self.encoder_model.time_pred
        del self.encoder_model.mask_token
        del self.encoder_model.decoder_pos_embed
        # del self.encoder_model.decoder_channel_embed # 新模型已移除此属性
        del self.encoder_model.rope_decoder
        if hasattr(self.encoder_model, 'decoder_norm'):
            del self.encoder_model.decoder_norm

    def _load_pretrained_weights(self, path):
        print(f"Loading weights from {path}...")
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

        # 清洗 Key 名称
        new_state_dict = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        
        # 过滤 Decoder 权重
        encoder_dict = {}
        for k, v in new_state_dict.items():
            if any(x in k for x in ["decoder", "mask_token", "time_reducer", "time_pred", "rope_decoder"]):
                continue
            encoder_dict[k] = v
            
        # 位置编码插值 (处理不同长度输入)
        if hasattr(self.encoder_model, 'pos_embed'):
            self._interpolate_pos_embed(encoder_dict, 'pos_embed', self.encoder_model.pos_embed)

        msg = self.encoder_model.load_state_dict(encoder_dict, strict=False)
        print(f"Weights loaded. Missing keys (expected decoder keys): {msg.missing_keys}")

    def _interpolate_pos_embed(self, state_dict, key, new_pos_embed):
        if key not in state_dict: return
        old_pos_embed = state_dict[key] 
        if old_pos_embed.shape[1] == new_pos_embed.shape[1]: return

        print(f"Interpolating {key}: {old_pos_embed.shape[1]} -> {new_pos_embed.shape[1]}")
        cls_token = old_pos_embed[:, :1, :]
        patch_tokens = old_pos_embed[:, 1:, :] 
        
        grid_h, grid_w_new = self.encoder_model.grid_size
        n_old = patch_tokens.shape[1]
        grid_w_old = n_old // grid_h
        dim = patch_tokens.shape[-1]
        
        patch_tokens = patch_tokens.transpose(1, 2).reshape(1, dim, grid_h, grid_w_old)
        patch_tokens = F.interpolate(patch_tokens, size=(grid_h, grid_w_new), mode='bicubic', align_corners=False)
        patch_tokens = patch_tokens.flatten(2).transpose(1, 2)
        
        new_pos_embed_interpolated = torch.cat((cls_token, patch_tokens), dim=1)
        state_dict[key] = new_pos_embed_interpolated

    def forward(self, x):
        """
        x: (B, M, L) 多通道输入
        """
        # 兼容单通道输入 (B, L) -> (B, 1, L)
        if x.dim() == 2: x = x.unsqueeze(1)
        
        # 1. CWT 变换 (B, M, L) -> (B, M, 3, Scales, L)
        # 注意：cwt_wrap 现在支持多通道
        imgs = cwt_wrap(x, num_scales=self.encoder_model.cwt_scales, lowest_scale=0.1, step=1.0)
        
        # 2. Instance Norm (独立对每个通道、每个样本归一化)
        imgs_f32 = imgs.float()
        # mean/std over (H, W) -> (B, M, 3, 1, 1)
        mean = imgs_f32.mean(dim=(3, 4), keepdim=True)
        std = imgs_f32.std(dim=(3, 4), keepdim=True)
        std = torch.clamp(std, min=1e-5)
        imgs = (imgs_f32 - mean) / std
        
        # 数值鲁棒性增强：防止 CWT 产生的极端值引发梯度爆炸
        imgs = torch.nan_to_num(imgs, nan=0.0, posinf=100.0, neginf=-100.0)
        imgs = torch.clamp(imgs, min=-100.0, max=100.0)

        # 确保输入数据类型与模型权重一致
        target_dtype = next(self.encoder_model.parameters()).dtype
        imgs = imgs.to(dtype=target_dtype)

        # 3. Forward Encoder
        # 新版 forward_encoder 返回: x, mask, ids, M
        # x 的形状是 (B, M*N_patches + 1, D)
        self.encoder_model.mask_ratio = 0.0
        latent, _, _, _ = self.encoder_model.forward_encoder(imgs)
        
        # 4. 提取特征
        # 丢弃 CLS token (index 0)，保留 Patch tokens
        # patch_tokens: (B, M*N_patches, D)
        patch_tokens = latent[:, 1:, :] 
        
        # 5. Forward Head
        if isinstance(self.head, LatentReasoningHead):
            # CoT 模式: 输入所有 Patch Tokens (混合了所有通道的信息)
            # Cross-Attention 会自动处理 M*N 的长度
            logits = self.head(patch_tokens)
        else:
            # MLP 模式: Global Average Pooling
            global_feat = patch_tokens.mean(dim=1)
            logits = self.head(global_feat)
        
        return logits