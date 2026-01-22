import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 确保 model.py 在同一目录下，且包含 CWT_MAE_RoPE 和 cwt_wrap
from model import CWT_MAE_RoPE, cwt_wrap

# ===================================================================
# 1. 改进版回归头 (支持 Late Fusion)
# ===================================================================
class LatentReasoningRegressionHead(nn.Module):
    def __init__(self, embed_dim, num_heads, output_dim=2, 
                 num_reasoning_tokens=8, dropout=0.1, 
                 num_tabular_features=6):
        """
        Args:
            embed_dim: Transformer 输出维度 (如 768)
            output_dim: 预测目标维度 (2: SBP, DBP)
            num_tabular_features: 表格特征数量 (如 6: Sex, Age, Height, Weight, HR, BMI)
        """
        super().__init__()
        self.num_reasoning_tokens = num_reasoning_tokens
        self.embed_dim = embed_dim
        
        # 1. 推理令牌 (Learnable Queries)
        self.reasoning_tokens = nn.Parameter(torch.zeros(1, num_reasoning_tokens, embed_dim))
        
        # 2. Attention 模块 (用于从波形中提取特征)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm3 = nn.LayerNorm(embed_dim)
        
        # 3. Late Fusion 模块 (处理表格特征)
        # 将 6 维特征映射到 64 维，增强表达能力
        self.tab_mlp = nn.Sequential(
            nn.Linear(num_tabular_features, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.Dropout(dropout)
        )
        
        # 4. 最终回归层
        # 输入维度 = Transformer特征(768) + 表格特征(64)
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim + 64, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, output_dim) 
        )
        
        self._init_weights()
        # 重新初始化 reasoning_tokens
        nn.init.normal_(self.reasoning_tokens, std=0.02)

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x_encoder, x_tabular):
        """
        Args:
            x_encoder: 来自 Transformer 的输出 [Batch, N_patches + 1, Dim]
            x_tabular: 原始表格特征 [Batch, num_tabular_features]
        """
        B = x_encoder.shape[0]
        queries = self.reasoning_tokens.expand(B, -1, -1)
        
        # --- Step 1: Attention 推理 (处理波形) ---
        # Cross Attention: Queries 关注 Encoder 输出
        attn_out, _ = self.cross_attn(query=queries, key=x_encoder, value=x_encoder)
        queries = self.norm1(queries + attn_out)
        
        # Self Attention: Queries 之间交互
        attn_out2, _ = self.self_attn(query=queries, key=queries, value=queries)
        queries = self.norm2(queries + attn_out2)
        
        # FFN
        queries = self.norm3(queries + self.ffn(queries))
        
        # Global Pooling: 聚合所有推理 Token
        decision_token = queries.mean(dim=1) # [Batch, embed_dim]
        
        # --- Step 2: Late Fusion (处理表格) ---
        tab_feat = self.tab_mlp(x_tabular)   # [Batch, 64]
        
        # --- Step 3: 拼接与回归 ---
        # 将波形特征与表格特征拼接
        combined_features = torch.cat([decision_token, tab_feat], dim=1) # [Batch, embed_dim + 64]
        
        # 输出预测值
        values = self.regressor(combined_features)
        return values

# ===================================================================
# 2. 血压预测主模型封装
# ===================================================================
class TF_MAE_BloodPressure(nn.Module):
    def __init__(self, pretrained_path, 
                 output_dim=2,  # [SBP, DBP]
                 mlp_rank_ratio=0.5, 
                 num_reasoning_tokens=8, 
                 num_tabular_features=6, # Sex, Age, Height, Weight, HR, BMI
                 **kwargs):
        super().__init__()
        
        # 1. 初始化 Encoder (CWT-MAE-RoPE)
        self.encoder_model = CWT_MAE_RoPE(
            mlp_rank_ratio=mlp_rank_ratio,
            mask_ratio=0.0, # 回归微调时关闭 Mask
            **kwargs
        )
        self.embed_dim = kwargs.get('embed_dim', 768)
        
        # 2. 加载预训练权重
        if pretrained_path:
            self._load_pretrained_weights(pretrained_path)
        
        # 3. 清理 Decoder 以节省显存
        self._delete_decoder_components()

        # 4. Early Fusion 投影层
        # 将表格特征映射为 Transformer 的维度，作为额外的 Token 输入
        self.tabular_projector = nn.Sequential(
            nn.Linear(num_tabular_features, self.embed_dim // 2),
            nn.GELU(),
            nn.Linear(self.embed_dim // 2, self.embed_dim),
            nn.LayerNorm(self.embed_dim)
        )

        # 5. 初始化回归头 (包含 Late Fusion)
        print(f">>> Initializing BP Regression Head with Dual Fusion (Early + Late)")
        self.head = LatentReasoningRegressionHead(
            embed_dim=self.embed_dim,
            num_heads=kwargs.get('num_heads', 12),
            output_dim=output_dim,
            num_reasoning_tokens=num_reasoning_tokens,
            dropout=0.1,
            num_tabular_features=num_tabular_features
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
            
        # 位置编码插值
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

    def forward(self, x, x_tabular):
        """
        Args:
            x: PPG 信号 [Batch, Length]
            x_tabular: 表格特征 [Batch, 6]
        """
        # 维度调整
        if x.dim() == 3: x = x.squeeze(1)
        x = x.float() 
        
        # 1. CWT 变换
        imgs = cwt_wrap(x, num_scales=self.encoder_model.cwt_scales, lowest_scale=0.1, step=1.0)
        
        # 2. Instance Norm (标准化 CWT 图谱)
        dtype_orig = imgs.dtype
        imgs_f32 = imgs.float()
        mean = imgs_f32.mean(dim=(2, 3), keepdim=True)
        std = imgs_f32.std(dim=(2, 3), keepdim=True)
        std = torch.clamp(std, min=1e-5)
        imgs = (imgs_f32 - mean) / std
        imgs = imgs.to(dtype=dtype_orig)

        # 3. Forward Encoder
        self.encoder_model.mask_ratio = 0.0
        latent, _, _ = self.encoder_model.forward_encoder(imgs)
        
        # 提取 Patch Tokens (丢弃原始 CLS token)
        patch_tokens = latent[:, 1:, :] 
        
        # 4. Early Fusion: 将表格特征作为 Token 注入
        # [B, 6] -> [B, 1, 768]
        tab_token_early = self.tabular_projector(x_tabular).unsqueeze(1)
        
        # 拼接: [Patch Tokens, Tabular Token]
        # 这样 Attention 机制可以同时看到波形特征和病人信息
        combined_tokens = torch.cat([patch_tokens, tab_token_early], dim=1)
        
        # 5. Forward Head (包含 Late Fusion)
        # 传入 combined_tokens 进行 Attention
        # 同时传入 x_tabular 进行最后的拼接
        bp_pred = self.head(combined_tokens, x_tabular)
        
        return bp_pred