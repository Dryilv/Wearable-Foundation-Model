import torch
import torch.nn as nn
import torch.nn.functional as F
from model import CWT_MAE_RoPE, cwt_wrap

class TF_MAE_Classifier(nn.Module):
    def __init__(self, pretrained_path, num_classes, hidden_dim=512, dropout_rate=0.5, 
                 mlp_rank_ratio=0.5, **kwargs):
        super().__init__()
        
        # 1. 初始化 Encoder
        self.encoder_model = CWT_MAE_RoPE(
            mlp_rank_ratio=mlp_rank_ratio,
            mask_ratio=0.0,
            **kwargs
        )
        self.embed_dim = kwargs.get('embed_dim', 768)
        
        # 2. 加载预训练权重
        if pretrained_path:
            self._load_pretrained_weights(pretrained_path)
        
        # 3. 删除 Decoder (保持不变)
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

        # ============================================================
        # 【关键修改 1】计算 Flatten 后的总维度
        # ============================================================
        # 我们需要知道 Patch 的数量 (N)。
        # 通常 encoder_model.grid_size 存储了 (H, W) 的网格大小
        # 如果报错，请确保你的 encoder_model 在初始化时正确计算了 grid_size
        grid_h, grid_w = self.encoder_model.grid_size
        self.num_patches = grid_h * grid_w
        
        # 展平后的维度 = Patch数量 * 每个Patch的维度
        self.flatten_dim = self.num_patches * self.embed_dim
        
        print(f"Model Mode: Flatten (Option B)")
        print(f" - Num Patches: {self.num_patches}")
        print(f" - Embed Dim: {self.embed_dim}")
        print(f" - Flattened Head Input Dim: {self.flatten_dim} (Check your GPU memory!)")

        # ============================================================
        # 【关键修改 2】分类头
        # ============================================================
        # 输入维度不再是 embed_dim，而是 flatten_dim
        self.head = nn.Sequential(
            nn.LayerNorm(self.flatten_dim), # 对展平后的长向量做归一化
            nn.Linear(self.flatten_dim, hidden_dim), # 巨大的压缩层
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )
        self._init_head_weights()

    def _init_head_weights(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _load_pretrained_weights(self, path):
        # ... (保持原样，不需要修改) ...
        print(f"Loading weights from {path}...")
        checkpoint = torch.load(path, map_location='cpu')
        if 'model' in checkpoint: state_dict = checkpoint['model']
        else: state_dict = checkpoint
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k
            if name.startswith('module.'): name = name[7:]
            if name.startswith('_orig_mod.'): name = name[10:]
            new_state_dict[name] = v
        encoder_dict = {}
        for k, v in new_state_dict.items():
            if any(x in k for x in ["decoder", "mask_token", "time_reducer", "time_pred", "rope_decoder"]):
                continue
            encoder_dict[k] = v
        if hasattr(self.encoder_model, 'pos_embed'):
            self._interpolate_pos_embed(encoder_dict, 'pos_embed', self.encoder_model.pos_embed)
        msg = self.encoder_model.load_state_dict(encoder_dict, strict=False)
        print(f"Weights loaded. Missing keys: {msg.missing_keys}")

    def _interpolate_pos_embed(self, state_dict, key, new_pos_embed):
        # ... (保持原样，不需要修改) ...
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
        # x: [B, 1, L]
        if x.dim() == 3: x = x.squeeze(1)
        
        # 1. CWT 变换
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
        
        # ============================================================
        # 【关键修改 3】Flatten 策略
        # ============================================================
        # 1. 去掉 CLS token (index 0)
        patch_tokens = latent[:, 1:, :]  # [B, N, D]
        
        # 2. 展平 (Flatten)
        # 将 [Batch, N, D] 变成 [Batch, N * D]
        flat_feat = patch_tokens.flatten(1) 
        
        # 3. 进入分类头
        logits = self.head(flat_feat)
        
        return logits