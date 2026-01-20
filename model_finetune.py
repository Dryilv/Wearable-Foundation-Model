import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 确保 model_rope_complete.py 在同一目录下
from model import CWT_MAE_RoPE, cwt_wrap

# ===================================================================
# 1. 隐式思维链模块 (Latent Reasoning / Chain-of-Thought Head)
#    核心思想：使用可学习的 Query 去"查询"信号特征，模拟分步推理
# ===================================================================
class LatentReasoningHead(nn.Module):
    def __init__(self, embed_dim, num_heads, num_classes, num_reasoning_tokens=8, dropout=0.1):
        """
        Args:
            embed_dim: 输入特征维度 (768)
            num_heads: 注意力头数
            num_classes: 分类类别数
            num_reasoning_tokens: "思维步骤"的数量 (建议 8-16)
        """
        super().__init__()
        self.num_reasoning_tokens = num_reasoning_tokens
        self.embed_dim = embed_dim
        
        # 1. 定义推理令牌 (Learnable Queries)
        # 形状: [1, N_reason, Dim]
        # 这些 Token 相当于医生的"检查清单" (Checklist)
        self.reasoning_tokens = nn.Parameter(torch.zeros(1, num_reasoning_tokens, embed_dim))
        nn.init.normal_(self.reasoning_tokens, std=0.02)
        
        # 2. Cross-Attention: Reasoning (Q) <-> Signal Features (K, V)
        # 作用: 让推理令牌去信号中寻找证据 (例如: Token 1 找 P波, Token 2 找噪声)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # 3. Self-Attention: Reasoning (Q) <-> Reasoning (K, V)
        # 作用: 整合不同证据 (例如: 结合"节律不齐"和"P波消失"得出结论)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # 4. FFN (前馈网络)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm3 = nn.LayerNorm(embed_dim)
        
        # 5. 最终分类器
        self.classifier = nn.Linear(embed_dim, num_classes)
        
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x_encoder):
        # x_encoder: [Batch, N_patches, Dim] (来自 Encoder 的 Patch Tokens)
        B = x_encoder.shape[0]
        
        # 1. 扩展推理令牌到 Batch 维度
        # queries: [B, N_reason, Dim]
        queries = self.reasoning_tokens.expand(B, -1, -1)
        
        # 2. Cross Attention: "带着问题看数据"
        # Query = Reasoning Tokens, Key/Value = Encoder Features
        attn_out, _ = self.cross_attn(query=queries, key=x_encoder, value=x_encoder)
        queries = self.norm1(queries + attn_out)
        
        # 3. Self Attention: "综合思考"
        attn_out2, _ = self.self_attn(query=queries, key=queries, value=queries)
        queries = self.norm2(queries + attn_out2)
        
        # 4. FFN: "逻辑处理"
        queries = self.norm3(queries + self.ffn(queries))
        
        # 5. 聚合决策
        # 将所有推理步骤的结果取平均，形成最终的诊断向量
        decision_token = queries.mean(dim=1) 
        
        logits = self.classifier(decision_token)
        return logits

# ===================================================================
# 2. 残差 MLP 块 (备用模块，用于非 CoT 模式)
# ===================================================================
class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.5):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return x + self.net(self.norm(x))

# ===================================================================
# 3. 主分类器模型封装
# ===================================================================
class TF_MAE_Classifier(nn.Module):
    def __init__(self, pretrained_path, num_classes, 
                 mlp_rank_ratio=0.5, 
                 # CoT 配置
                 use_cot=True, 
                 num_reasoning_tokens=8, 
                 # 备用 MLP 配置
                 hidden_dim=512, 
                 dropout_rate=0.0,
                 num_res_blocks=2,
                 **kwargs):
        super().__init__()
        
        # 1. 初始化 Encoder (CWT-MAE-RoPE)
        self.encoder_model = CWT_MAE_RoPE(
            mlp_rank_ratio=mlp_rank_ratio,
            mask_ratio=0.0, # 微调时关闭 Mask
            **kwargs
        )
        self.embed_dim = kwargs.get('embed_dim', 768)
        
        # 2. 加载预训练权重
        if pretrained_path:
            self._load_pretrained_weights(pretrained_path)
        
        # 3. 清理 Decoder 以节省显存
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
            print(f">>> Initializing Deep Residual MLP Head ({num_res_blocks} blocks).")
            layers = []
            # Projection
            layers.append(nn.LayerNorm(self.embed_dim))
            layers.append(nn.Linear(self.embed_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))
            # Residual Blocks
            for _ in range(num_res_blocks):
                layers.append(ResidualMLPBlock(hidden_dim, dropout_rate))
            # Output
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Linear(hidden_dim, num_classes))
            
            self.head = nn.Sequential(*layers)
            self._init_head_weights()

    def _delete_decoder_components(self):
        """删除预训练模型中的 Decoder 部分"""
        del self.encoder_model.decoder_blocks
        del self.encoder_model.decoder_embed
        del self.encoder_model.decoder_pred_spec
        del self.encoder_model.time_reducer
        del self.encoder_model.time_pred
        del self.encoder_model.mask_token
        del self.encoder_model.decoder_pos_embed
        del self.encoder_model.rope_decoder
        if hasattr(self.encoder_model, 'decoder_norm'):
            del self.encoder_model.decoder_norm

    def _init_head_weights(self):
        """仅用于 MLP Head 的初始化"""
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def _load_pretrained_weights(self, path):
        print(f"Loading weights from {path}...")
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

        # 清洗 Key 名称 (去除 DDP 前缀等)
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
        
        # Reshape & Interpolate
        patch_tokens = patch_tokens.transpose(1, 2).reshape(1, dim, grid_h, grid_w_old)
        patch_tokens = F.interpolate(patch_tokens, size=(grid_h, grid_w_new), mode='bicubic', align_corners=False)
        patch_tokens = patch_tokens.flatten(2).transpose(1, 2)
        
        new_pos_embed_interpolated = torch.cat((cls_token, patch_tokens), dim=1)
        state_dict[key] = new_pos_embed_interpolated

    def forward(self, x):
        # x: [B, 1, L]
        if x.dim() == 3: x = x.squeeze(1)
        x = x.float() 
        # 1. CWT 变换 (确保 float32 精度计算)
        imgs = cwt_wrap(x, num_scales=self.encoder_model.cwt_scales, lowest_scale=0.1, step=1.0)
        
        # 2. Instance Norm
        dtype_orig = imgs.dtype
        imgs_f32 = imgs.float()
        mean = imgs_f32.mean(dim=(2, 3), keepdim=True)
        std = imgs_f32.std(dim=(2, 3), keepdim=True)
        std = torch.clamp(std, min=1e-5)
        imgs = (imgs_f32 - mean) / std
        imgs = imgs.to(dtype=dtype_orig)

        # 3. Forward Encoder
        self.encoder_model.mask_ratio = 0.0
        # latent: [Batch, N_patches + 1, Dim]
        latent, _, _ = self.encoder_model.forward_encoder(imgs)
        
        # 4. 提取特征
        # 丢弃 CLS token (index 0)，保留 Patch tokens
        patch_tokens = latent[:, 1:, :] 
        
        # 5. Forward Head
        if isinstance(self.head, LatentReasoningHead):
            # CoT 模式: 输入所有 Patch Tokens 进行 Cross-Attention
            logits = self.head(patch_tokens)
        else:
            # MLP 模式: 使用 Global Average Pooling
            global_feat = patch_tokens.mean(dim=1)
            logits = self.head(global_feat)
        
        return logits