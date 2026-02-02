import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 确保 model.py 在同一目录下
from model import CWT_MAE_RoPE, cwt_wrap

# ===================================================================
# 1. 隐式思维链模块 (Latent Reasoning Head)
#    【改动】支持序列输入 [B, N, D]，保留时间维度信息
# ===================================================================
class LatentReasoningHead(nn.Module):
    def __init__(self, embed_dim, num_heads, num_classes, num_reasoning_tokens=32, dropout=0.1):
        super().__init__()
        self.num_reasoning_tokens = num_reasoning_tokens
        self.embed_dim = embed_dim
        
        # 推理 Token (Query)
        self.reasoning_tokens = nn.Parameter(torch.zeros(1, num_reasoning_tokens, embed_dim))
        
        # Cross-Attention: Query=Reasoning, Key/Value=Fused_Sequence
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Self-Attention: 推理 Token 之间的交互
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
        """
        Args:
            x_encoder: 融合后的序列特征 [Batch, N_patches, Fused_Dim]
        """
        B = x_encoder.shape[0]
        
        # 扩展推理 Token 到当前 Batch
        queries = self.reasoning_tokens.expand(B, -1, -1) # [B, num_reasoning, D]
        
        # --- Cross Attention (核心步骤) ---
        # Query 是推理 Token，Key/Value 是包含时间信息的 PPG+ECG 序列
        # 模型会自动学习关注序列中的哪些 Patch (时间点) 对诊断最重要
        attn_out, _ = self.cross_attn(query=queries, key=x_encoder, value=x_encoder)
        queries = self.norm1(queries + attn_out)
        
        # --- Self Attention ---
        attn_out2, _ = self.self_attn(query=queries, key=queries, value=queries)
        queries = self.norm2(queries + attn_out2)
        
        # --- FFN ---
        queries = self.norm3(queries + self.ffn(queries))
        
        # --- Classification ---
        # 对所有推理 Token 取平均，得到最终决策向量
        decision_token = queries.mean(dim=1) 
        logits = self.classifier(decision_token)
        
        return logits

# ===================================================================
# 2. 主分类器模型封装 (Dual Encoder Version)
# ===================================================================
class DualEncoder_Classifier(nn.Module):
    def __init__(self, pretrained_path, num_classes, 
                 mlp_rank_ratio=0.5, 
                 use_cot=True, # 【建议】开启 CoT 以充分利用时间序列信息
                 num_reasoning_tokens=16, 
                 hidden_dim=1024, 
                 dropout_rate=0.2,
                 **kwargs):
        super().__init__()
        
        self.embed_dim = kwargs.get('embed_dim', 768)
        
        # 1. 初始化 PPG Encoder
        print(">>> Initializing PPG Encoder...")
        self.ppg_encoder = CWT_MAE_RoPE(
            mlp_rank_ratio=mlp_rank_ratio,
            mask_ratio=0.0, 
            **kwargs
        )
        
        # 2. 初始化 ECG Encoder
        print(">>> Initializing ECG Encoder...")
        self.ecg_encoder = CWT_MAE_RoPE(
            mlp_rank_ratio=mlp_rank_ratio,
            mask_ratio=0.0, 
            **kwargs
        )
        
        # 3. 加载预训练权重
        if pretrained_path:
            print("\n--- Loading weights for PPG Encoder ---")
            self._load_and_prepare_encoder(self.ppg_encoder, pretrained_path)
            print("\n--- Loading weights for ECG Encoder ---")
            self._load_and_prepare_encoder(self.ecg_encoder, pretrained_path)
        
        # 4. 初始化分类头
        fused_dim = self.embed_dim * 2 # 拼接后的维度
        self.use_cot = use_cot

        if use_cot:
            print(f">>> Initializing Latent Reasoning Head (CoT) with input dim {fused_dim}.")
            self.head = LatentReasoningHead(
                embed_dim=fused_dim,
                num_heads=kwargs.get('num_heads', 12),
                num_classes=num_classes,
                num_reasoning_tokens=num_reasoning_tokens,
                dropout=0.2
            )
        else:
            print(f">>> Initializing Deep Residual MLP Head with input dim {fused_dim}.")
            self.head = nn.Sequential(
                nn.LayerNorm(fused_dim),
                nn.Linear(fused_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim // 2, num_classes)
            )
            self._init_head_weights()

    def _delete_decoder_components(self, model):
        """删除 Decoder 以节省显存"""
        del model.decoder_blocks
        del model.decoder_embed
        del model.decoder_pred_spec
        del model.time_reducer
        del model.time_pred
        del model.mask_token
        del model.decoder_pos_embed
        del model.rope_decoder
        if hasattr(model, 'decoder_norm'):
            del model.decoder_norm

    def _init_head_weights(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def _load_and_prepare_encoder(self, model, path):
        print(f"Loading weights from {path}...")
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        new_state_dict = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        
        encoder_dict = {k: v for k, v in new_state_dict.items() if not any(x in k for x in ["decoder", "mask_token", "time_reducer", "time_pred", "rope_decoder"])}
        
        if hasattr(model, 'pos_embed'):
            self._interpolate_pos_embed(encoder_dict, 'pos_embed', model.pos_embed, model.grid_size)

        msg = model.load_state_dict(encoder_dict, strict=False)
        print(f"Weights loaded. Missing keys: {msg.missing_keys}")
        
        self._delete_decoder_components(model)
        print("Decoder components removed.")

    def _interpolate_pos_embed(self, state_dict, key, new_pos_embed, new_grid_size):
        if key not in state_dict: return
        old_pos_embed = state_dict[key] 
        if old_pos_embed.shape[1] == new_pos_embed.shape[1]: return

        print(f"Interpolating {key}: {old_pos_embed.shape[1]} -> {new_pos_embed.shape[1]}")
        cls_token = old_pos_embed[:, :1, :]
        patch_tokens = old_pos_embed[:, 1:, :] 
        
        grid_h, grid_w_new = new_grid_size
        n_old = patch_tokens.shape[1]
        grid_w_old = n_old // grid_h
        dim = patch_tokens.shape[-1]
        
        patch_tokens = patch_tokens.transpose(1, 2).reshape(1, dim, grid_h, grid_w_old)
        patch_tokens = F.interpolate(patch_tokens, size=(grid_h, grid_w_new), mode='bicubic', align_corners=False)
        patch_tokens = patch_tokens.flatten(2).transpose(1, 2)
        
        new_pos_embed_interpolated = torch.cat((cls_token, patch_tokens), dim=1)
        state_dict[key] = new_pos_embed_interpolated

    def forward(self, x):
        # x: [B, 2, L] (channel 0: PPG, channel 1: ECG)
        x_ppg = x[:, 0, :].unsqueeze(1)
        x_ecg = x[:, 1, :].unsqueeze(1)
        
        # --- 1. CWT & Instance Norm ---
        imgs_ppg = cwt_wrap(x_ppg.float(), num_scales=self.ppg_encoder.cwt_scales)
        mean_ppg = imgs_ppg.mean(dim=(2, 3), keepdim=True)
        std_ppg = torch.clamp(imgs_ppg.std(dim=(2, 3), keepdim=True), min=1e-5)
        imgs_ppg = (imgs_ppg - mean_ppg) / std_ppg
        
        imgs_ecg = cwt_wrap(x_ecg.float(), num_scales=self.ecg_encoder.cwt_scales)
        mean_ecg = imgs_ecg.mean(dim=(2, 3), keepdim=True)
        std_ecg = torch.clamp(imgs_ecg.std(dim=(2, 3), keepdim=True), min=1e-5)
        imgs_ecg = (imgs_ecg - mean_ecg) / std_ecg

        # --- 2. Forward Encoders ---
        # latent shape: [B, N_patches + 1, D]
        latent_ppg, _, _ = self.ppg_encoder.forward_encoder(imgs_ppg)
        latent_ecg, _, _ = self.ecg_encoder.forward_encoder(imgs_ecg)
        
        # --- 3. Feature Fusion (Token-level) ---
        # 【关键改动】丢弃 CLS token (index 0)，保留所有 Patch tokens
        # shape: [B, N_patches, D]
        patches_ppg = latent_ppg[:, 1:, :]
        patches_ecg = latent_ecg[:, 1:, :]
        
        # 沿特征维度拼接，保留序列长度 N_patches
        # fused_feat shape: [B, N_patches, D * 2]
        fused_feat = torch.cat([patches_ppg, patches_ecg], dim=-1) 
        
        # --- 4. Forward Head ---
        if self.use_cot:
            # CoT Head 接受序列输入，自动进行 Cross-Attention
            logits = self.head(fused_feat)
        else:
            # MLP Head 只能接受向量，所以需要先做 Global Average Pooling
            # 这会丢失时间信息，所以推荐使用 CoT
            global_feat = fused_feat.mean(dim=1) # [B, D*2]
            logits = self.head(global_feat)
        
        return logits