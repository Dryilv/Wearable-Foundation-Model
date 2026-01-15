import torch
import torch.nn as nn
import torch.nn.functional as F
from model_rope_complete import CWT_MAE_RoPE, cwt_wrap

class SupCon_CWT_MAE(nn.Module):
    def __init__(self, pretrained_path=None, head='mlp', feat_dim=128, 
                 mlp_rank_ratio=0.5, **kwargs):
        super().__init__()
        
        # 1. Encoder (Backbone)
        self.encoder_model = CWT_MAE_RoPE(
            mlp_rank_ratio=mlp_rank_ratio,
            mask_ratio=0.0, # SupCon 不需要 Mask
            **kwargs
        )
        self.embed_dim = kwargs.get('embed_dim', 768)
        
        # 加载预训练权重 (如果有)
        if pretrained_path:
            self._load_pretrained_weights(pretrained_path)
            
        # 删除 Decoder
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

        # 2. Projection Head
        # 将特征映射到低维超球面 (Hypersphere)
        if head == 'linear':
            self.head = nn.Linear(self.embed_dim, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dim, feat_dim)
            )
        else:
            raise NotImplementedError(f'head not supported: {head}')

    def _load_pretrained_weights(self, path):
        # ... (与之前 model_finetune.py 中的加载逻辑一致，略) ...
        # 确保正确加载 Encoder 权重
        print(f"Loading encoder weights from {path}...")
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '').replace('_orig_mod.', '')
            if any(x in name for x in ["decoder", "mask_token", "time_reducer", "time_pred", "rope_decoder"]):
                continue
            new_state_dict[name] = v
            
        if hasattr(self.encoder_model, 'pos_embed'):
            # 这里省略插值代码，假设您已有实现，或者直接复制之前的 _interpolate_pos_embed
            pass 

        self.encoder_model.load_state_dict(new_state_dict, strict=False)

    def forward(self, x):
        # x: [B, 1, L]
        if x.dim() == 3: x = x.squeeze(1)
        
        # CWT + Norm
        imgs = cwt_wrap(x, num_scales=self.encoder_model.cwt_scales, lowest_scale=0.1, step=1.0)
        dtype_orig = imgs.dtype
        imgs_f32 = imgs.float()
        mean = imgs_f32.mean(dim=(2, 3), keepdim=True)
        std = imgs_f32.std(dim=(2, 3), keepdim=True)
        std = torch.clamp(std, min=1e-5)
        imgs = (imgs_f32 - mean) / std
        imgs = imgs.to(dtype=dtype_orig)

        # Encoder
        self.encoder_model.mask_ratio = 0.0
        latent, _, _ = self.encoder_model.forward_encoder(imgs)
        
        # GAP (Global Average Pooling)
        patch_tokens = latent[:, 1:, :]
        feat = patch_tokens.mean(dim=1) # [B, Embed_Dim]
        
        # Projection
        feat = self.head(feat)
        
        # Normalize (关键：SupCon 必须在单位球面上计算)
        feat = F.normalize(feat, dim=1)
        
        return feat