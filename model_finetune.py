import torch
import torch.nn as nn
import torch.nn.functional as F
# 导入 RoPE 版本的主模型
from model import CWT_MAE_RoPE, cwt_wrap

class TF_MAE_Classifier(nn.Module):
    def __init__(self, pretrained_path, num_classes, hidden_dim=512, dropout_rate=0.5, 
                 mlp_rank_ratio=0.5, **kwargs):
        """
        kwargs: 包含 embed_dim, depth, num_heads, cwt_scales, patch_size_time 等
        """
        super().__init__()
        
        # 1. 初始化 Encoder
        self.encoder_model = CWT_MAE_RoPE(
            mlp_rank_ratio=mlp_rank_ratio,
            mask_ratio=0.0, # 微调时不掩码
            **kwargs
        )
        self.embed_dim = kwargs.get('embed_dim', 768)
        
        # 2. 加载预训练权重
        if pretrained_path:
            self._load_pretrained_weights(pretrained_path)
        
        # 3. 删除 Decoder 和 Heads 以节省显存
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
        # 【关键改进】序列特征降维模块
        # ============================================================
        # 目的：保留时序结构信息，但避免直接 Flatten 导致的参数爆炸 (解决 loss=0.693)
        # 策略：先用 1x1 卷积将特征维度压缩 (768 -> 64)，再展平
        
        self.reduced_dim = 64  # 压缩后的特征维度，可调整 (32, 64, 128)
        
        self.seq_reducer = nn.Sequential(
            # 输入: [Batch, Embed_Dim, Num_Patches]
            nn.Conv1d(in_channels=self.embed_dim, out_channels=self.reduced_dim, kernel_size=1),
            nn.BatchNorm1d(self.reduced_dim),
            nn.GELU()
        )

        # 计算 Flatten 后的总维度
        # 获取 Patch 数量 (Grid Size)
        grid_h, grid_w = self.encoder_model.grid_size
        self.num_patches = grid_h * grid_w
        
        # 最终进入全连接层的维度 = Patch数量 * 压缩后的特征维度
        self.flatten_dim = self.num_patches * self.reduced_dim
        
        print(f"--- Classifier Setup ---")
        print(f"Mode: Conv1d Reduction + Flatten (Preserves Sequence Info)")
        print(f"Num Patches: {self.num_patches}")
        print(f"Feature Dim Reduction: {self.embed_dim} -> {self.reduced_dim}")
        print(f"Final Head Input Dim: {self.flatten_dim}")

        # 4. 分类头
        self.head = nn.Sequential(
            nn.LayerNorm(self.flatten_dim),
            nn.Linear(self.flatten_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )
        self._init_head_weights()

    def _init_head_weights(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                # 使用 Kaiming 初始化有助于深层网络收敛
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def _load_pretrained_weights(self, path):
        print(f"Loading weights from {path}...")
        checkpoint = torch.load(path, map_location='cpu')
        
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

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
        
        # 4. 特征处理
        # 丢弃 CLS token (index 0)，保留 Patch tokens
        patch_tokens = latent[:, 1:, :]  # [B, N, D]
        
        # 变换维度以适应 Conv1d: [B, N, D] -> [B, D, N]
        x = patch_tokens.transpose(1, 2)
        
        # 降维: [B, 768, N] -> [B, 64, N]
        x = self.seq_reducer(x)
        
        # 展平: [B, 64, N] -> [B, 64 * N]
        # 这里保留了所有 Patch 的位置信息，没有做平均
        flat_feat = x.flatten(1)
        
        # 5. 分类头
        logits = self.head(flat_feat)
        
        return logits