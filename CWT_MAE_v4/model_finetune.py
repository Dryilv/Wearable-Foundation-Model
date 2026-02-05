import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 确保 model.py (包含 CWT_MAE_RoPE 和 cwt_wrap) 在同一目录下
from model import CWT_MAE_RoPE, cwt_wrap

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
# 2. 主分类器模型封装 (v3 Pixel-based)
# ===================================================================
class TF_MAE_Classifier(nn.Module):
    def __init__(self, pretrained_path, num_classes, 
                 mlp_rank_ratio=0.5, 
                 use_cot=True, 
                 num_reasoning_tokens=16, 
                 **kwargs):
        super().__init__()
        
        # 1. 初始化 Encoder (Pixel-MAE-RoPE)
        # 注意：不再需要 CWT 相关参数 (除非想保留 CWT Loss 结构用于其他用途，但在分类中不需要)
        # 我们只传递影响 Encoder 结构的参数
        self.encoder_model = CWT_MAE_RoPE(
            mlp_rank_ratio=mlp_rank_ratio,
            mask_ratio=0.0, # 微调时关闭 Mask
            patch_size=kwargs.get('patch_size', 4), # 使用 patch_size
            signal_len=kwargs.get('signal_len', 3000),
            embed_dim=kwargs.get('embed_dim', 768),
            depth=kwargs.get('depth', 12),
            num_heads=kwargs.get('num_heads', 12),
            **{k:v for k,v in kwargs.items() if k not in ['patch_size', 'signal_len', 'embed_dim', 'depth', 'num_heads']}
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
        components_to_delete = [
            'decoder_blocks', 'decoder_embed', 'decoder_pred_spec', 'pred_head',
            'mask_token', 'decoder_pos_embed', 'rope_decoder', 'decoder_norm'
        ]
        
        for component in components_to_delete:
            if hasattr(self.encoder_model, component):
                delattr(self.encoder_model, component)

    def _load_pretrained_weights(self, path):
        print(f"Loading weights from {path}...")
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

        # 清洗 Key 名称
        new_state_dict = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        
        # 过滤 Decoder 权重和不匹配的权重
        encoder_dict = {}
        for k, v in new_state_dict.items():
            # 过滤 decoder 相关
            if any(x in k for x in ["decoder", "mask_token", "pred_head", "rope_decoder"]):
                continue
            encoder_dict[k] = v
            
        # 位置编码插值 (处理不同长度输入)
        if hasattr(self.encoder_model, 'pos_embed'):
            self._interpolate_pos_embed(encoder_dict, 'pos_embed', self.encoder_model.pos_embed)

        msg = self.encoder_model.load_state_dict(encoder_dict, strict=False)
        print(f"Weights loaded. Missing keys (expected decoder keys): {msg.missing_keys}")
        print(f"Unexpected keys: {msg.unexpected_keys}")

    def _interpolate_pos_embed(self, state_dict, key, new_pos_embed):
        if key not in state_dict: return
        old_pos_embed = state_dict[key] 
        if old_pos_embed.shape[1] == new_pos_embed.shape[1]: return

        print(f"Interpolating {key}: {old_pos_embed.shape[1]} -> {new_pos_embed.shape[1]}")
        cls_token = old_pos_embed[:, :1, :]
        patch_tokens = old_pos_embed[:, 1:, :] 
        
        # 1D 插值
        # patch_tokens: (1, N_old, D) -> (1, D, N_old)
        patch_tokens = patch_tokens.transpose(1, 2)
        N_new = new_pos_embed.shape[1] - 1
        
        patch_tokens = F.interpolate(patch_tokens, size=(N_new,), mode='linear', align_corners=False)
        # -> (1, D, N_new) -> (1, N_new, D)
        patch_tokens = patch_tokens.transpose(1, 2)
        
        new_pos_embed_interpolated = torch.cat((cls_token, patch_tokens), dim=1)
        state_dict[key] = new_pos_embed_interpolated

    def forward(self, x):
        """
        x: (B, M, L) 多通道输入
        """
        # 兼容单通道输入 (B, L) -> (B, 1, L)
        if x.dim() == 2: x = x.unsqueeze(1)
        B, M, L = x.shape
        
        # 1. Normalize (Instance Norm per channel)
        # v3 模型直接接收 1D 信号
        x_flat = x.view(B * M, 1, L)
        mean = x_flat.mean(dim=2, keepdim=True)
        std = x_flat.std(dim=2, keepdim=True) + 1e-5
        x_norm = (x_flat - mean) / std
        
        # 确保输入数据类型与模型权重一致
        target_dtype = next(self.encoder_model.parameters()).dtype
        x_norm = x_norm.to(dtype=target_dtype)

        # 2. Forward Encoder
        self.encoder_model.mask_ratio = 0.0
        # forward_encoder 返回: x, mask, ids_restore
        # x 的形状是 (B*M, N_patches + 1, D)
        latent, _, _ = self.encoder_model.forward_encoder(x_norm)
        
        # 3. 提取特征
        # 丢弃 CLS token (index 0)，保留 Patch tokens
        # patch_tokens: (B*M, N_patches, D)
        # 但我们需要 reshape 成 (B, M*N_patches, D) 以保留 M 的结构供 CoT 使用
        N_patches = latent.shape[1] - 1
        D = latent.shape[2]
        
        patch_tokens = latent[:, 1:, :] 
        patch_tokens = patch_tokens.view(B, M * N_patches, D)
        
        # 4. Forward Head
        if isinstance(self.head, LatentReasoningHead):
            # CoT 模式: 输入所有 Patch Tokens (混合了所有通道的信息)
            logits = self.head(patch_tokens)
        else:
            # MLP 模式: Global Average Pooling
            global_feat = patch_tokens.mean(dim=1)
            logits = self.head(global_feat)
        
        return logits
