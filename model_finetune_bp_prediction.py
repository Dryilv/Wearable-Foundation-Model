import torch
import torch.nn as nn
import torch.nn.functional as F
from model import CWT_MAE_RoPE, cwt_wrap

class LatentReasoningRegressionHead(nn.Module):
    def __init__(self, embed_dim, num_heads, output_dim=2, num_reasoning_tokens=8, dropout=0.1):
        super().__init__()
        self.reasoning_tokens = nn.Parameter(torch.zeros(1, num_reasoning_tokens, embed_dim))
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
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, output_dim) 
        )
        self._init_weights()
        nn.init.normal_(self.reasoning_tokens, std=0.02)

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)

    def forward(self, x_encoder):
        # x_encoder: [Batch, N_patches + 1 (Tabular), Dim]
        B = x_encoder.shape[0]
        queries = self.reasoning_tokens.expand(B, -1, -1)
        
        # Cross Attention: Queries 关注 (信号 + 表格信息)
        attn_out, _ = self.cross_attn(query=queries, key=x_encoder, value=x_encoder)
        queries = self.norm1(queries + attn_out)
        
        attn_out2, _ = self.self_attn(query=queries, key=queries, value=queries)
        queries = self.norm2(queries + attn_out2)
        
        queries = self.norm3(queries + self.ffn(queries))
        decision_token = queries.mean(dim=1) 
        return self.regressor(decision_token)

class TF_MAE_BloodPressure(nn.Module):
    def __init__(self, pretrained_path, 
                 output_dim=2, 
                 mlp_rank_ratio=0.5, 
                 num_reasoning_tokens=8, 
                 num_tabular_features=6, # 新增参数
                 **kwargs):
        super().__init__()
        
        self.encoder_model = CWT_MAE_RoPE(
            mlp_rank_ratio=mlp_rank_ratio,
            mask_ratio=0.0, 
            **kwargs
        )
        self.embed_dim = kwargs.get('embed_dim', 768)
        
        if pretrained_path:
            self._load_pretrained_weights(pretrained_path)
        self._delete_decoder_components()

        # --- 新增：表格特征编码器 ---
        # 将 [Batch, 6] -> [Batch, 1, 768]
        self.tabular_projector = nn.Sequential(
            nn.Linear(num_tabular_features, self.embed_dim // 2),
            nn.GELU(),
            nn.Linear(self.embed_dim // 2, self.embed_dim),
            nn.LayerNorm(self.embed_dim)
        )

        print(f">>> Initializing BP Regression Head with Tabular Fusion")
        self.head = LatentReasoningRegressionHead(
            embed_dim=self.embed_dim,
            num_heads=kwargs.get('num_heads', 12),
            output_dim=output_dim,
            num_reasoning_tokens=num_reasoning_tokens,
            dropout=0.1
        )

    def _delete_decoder_components(self):
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
        # (保持原有的加载逻辑不变)
        print(f"Loading weights from {path}...")
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        new_state_dict = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        encoder_dict = {}
        for k, v in new_state_dict.items():
            if any(x in k for x in ["decoder", "mask_token", "time_reducer", "time_pred", "rope_decoder"]): continue
            encoder_dict[k] = v
        if hasattr(self.encoder_model, 'pos_embed'):
            self._interpolate_pos_embed(encoder_dict, 'pos_embed', self.encoder_model.pos_embed)
        self.encoder_model.load_state_dict(encoder_dict, strict=False)

    def _interpolate_pos_embed(self, state_dict, key, new_pos_embed):
        # (保持原有的插值逻辑不变)
        if key not in state_dict: return
        old_pos_embed = state_dict[key] 
        if old_pos_embed.shape[1] == new_pos_embed.shape[1]: return
        cls_token = old_pos_embed[:, :1, :]
        patch_tokens = old_pos_embed[:, 1:, :] 
        grid_h, grid_w_new = self.encoder_model.grid_size
        n_old = patch_tokens.shape[1]
        grid_w_old = n_old // grid_h
        dim = patch_tokens.shape[-1]
        patch_tokens = patch_tokens.transpose(1, 2).reshape(1, dim, grid_h, grid_w_old)
        patch_tokens = F.interpolate(patch_tokens, size=(grid_h, grid_w_new), mode='bicubic', align_corners=False)
        patch_tokens = patch_tokens.flatten(2).transpose(1, 2)
        state_dict[key] = torch.cat((cls_token, patch_tokens), dim=1)

    def forward(self, x, x_tabular):
        """
        Args:
            x: PPG Signal [B, L]
            x_tabular: Tabular features [B, 6]
        """
        if x.dim() == 3: x = x.squeeze(1)
        x = x.float() 
        
        # 1. 处理信号
        imgs = cwt_wrap(x, num_scales=self.encoder_model.cwt_scales, lowest_scale=0.1, step=1.0)
        dtype_orig = imgs.dtype
        imgs_f32 = imgs.float()
        mean = imgs_f32.mean(dim=(2, 3), keepdim=True)
        std = imgs_f32.std(dim=(2, 3), keepdim=True)
        imgs = (imgs_f32 - mean) / torch.clamp(std, min=1e-5)
        imgs = imgs.to(dtype=dtype_orig)

        self.encoder_model.mask_ratio = 0.0
        latent, _, _ = self.encoder_model.forward_encoder(imgs)
        
        # 提取信号特征 (Patch Tokens) [B, N_patches, Dim]
        patch_tokens = latent[:, 1:, :] 
        
        # 2. 处理表格特征
        # [B, 6] -> [B, Dim] -> [B, 1, Dim]
        tab_token = self.tabular_projector(x_tabular).unsqueeze(1)
        
        # 3. 特征融合
        # 将表格 Token 拼接到信号 Tokens 后面
        # 这样 CoT Head 的 Attention 就能同时看到波形和病人信息
        combined_features = torch.cat([patch_tokens, tab_token], dim=1)
        
        # 4. 预测
        bp_pred = self.head(combined_features)
        
        return bp_pred