
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import CWT_MAE_RoPE, cwt_wrap

# ===================================================================
# Latent Reasoning Head (保持不变，但维度会自动适配)
# ===================================================================
class LatentReasoningHead(nn.Module):
    def __init__(self, embed_dim, num_heads, num_classes, num_reasoning_tokens=32, dropout=0.1):
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
        self.classifier = nn.Linear(embed_dim, num_classes)
        self._init_weights()
        nn.init.normal_(self.reasoning_tokens, std=0.02)

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x_encoder):
        B = x_encoder.shape[0]
        queries = self.reasoning_tokens.expand(B, -1, -1)
        attn_out, _ = self.cross_attn(query=queries, key=x_encoder, value=x_encoder)
        queries = self.norm1(queries + attn_out)
        attn_out2, _ = self.self_attn(query=queries, key=queries, value=queries)
        queries = self.norm2(queries + attn_out2)
        queries = self.norm3(queries + self.ffn(queries))
        decision_token = queries.mean(dim=1) 
        logits = self.classifier(decision_token)
        return logits

# ===================================================================
# 单通道 ECG 分类器
# ===================================================================
class ECG_Classifier(nn.Module):
    def __init__(self, pretrained_path, num_classes, 
                 mlp_rank_ratio=0.5, 
                 use_cot=True, 
                 num_reasoning_tokens=16, 
                 hidden_dim=1024, 
                 dropout_rate=0.2,
                 **kwargs):
        super().__init__()
        
        self.embed_dim = kwargs.get('embed_dim', 768)
        
        # 1. 初始化单个 Encoder
        print(">>> Initializing ECG Encoder...")
        self.encoder = CWT_MAE_RoPE(
            mlp_rank_ratio=mlp_rank_ratio,
            mask_ratio=0.0, 
            **kwargs
        )
        
        # 2. 加载预训练权重
        if pretrained_path:
            print(f"\n--- Loading weights from {pretrained_path} ---")
            self._load_and_prepare_encoder(self.encoder, pretrained_path)
        
        # 3. 初始化分类头
        # 输入维度就是 embed_dim，因为没有拼接
        head_input_dim = self.embed_dim 
        self.use_cot = use_cot

        if use_cot:
            print(f">>> Initializing Latent Reasoning Head (CoT) with input dim {head_input_dim}.")
            self.head = LatentReasoningHead(
                embed_dim=head_input_dim,
                num_heads=kwargs.get('num_heads', 12),
                num_classes=num_classes,
                num_reasoning_tokens=num_reasoning_tokens,
                dropout=0.2
            )
        else:
            print(f">>> Initializing MLP Head with input dim {head_input_dim}.")
            self.head = nn.Sequential(
                nn.LayerNorm(head_input_dim),
                nn.Linear(head_input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim // 2, num_classes)
            )
            self._init_head_weights()

    def _delete_decoder_components(self, model):
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
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        new_state_dict = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        
        encoder_dict = {k: v for k, v in new_state_dict.items() if not any(x in k for x in ["decoder", "mask_token", "time_reducer", "time_pred", "rope_decoder"])}
        
        if hasattr(model, 'pos_embed'):
            self._interpolate_pos_embed(encoder_dict, 'pos_embed', model.pos_embed, model.grid_size)

        msg = model.load_state_dict(encoder_dict, strict=False)
        print(f"Weights loaded. Missing keys: {msg.missing_keys}")
        self._delete_decoder_components(model)

    def _interpolate_pos_embed(self, state_dict, key, new_pos_embed, new_grid_size):
        if key not in state_dict: return
        old_pos_embed = state_dict[key] 
        if old_pos_embed.shape[1] == new_pos_embed.shape[1]: return
        
        cls_token = old_pos_embed[:, :1, :]
        patch_tokens = old_pos_embed[:, 1:, :] 
        grid_h, grid_w_new = new_grid_size
        n_old = patch_tokens.shape[1]
        grid_w_old = n_old // grid_h
        dim = patch_tokens.shape[-1]
        
        patch_tokens = patch_tokens.transpose(1, 2).reshape(1, dim, grid_h, grid_w_old)
        patch_tokens = F.interpolate(patch_tokens, size=(grid_h, grid_w_new), mode='bicubic', align_corners=False)
        patch_tokens = patch_tokens.flatten(2).transpose(1, 2)
        state_dict[key] = torch.cat((cls_token, patch_tokens), dim=1)

    def forward(self, x):
        # x: [B, L] (Single channel ECG)
        
        # 1. CWT & Instance Norm
        imgs = cwt_wrap(x.float(), num_scales=self.encoder.cwt_scales) # [B, 3, Scales, L]
        
        mean = imgs.mean(dim=(2, 3), keepdim=True)
        std = torch.clamp(imgs.std(dim=(2, 3), keepdim=True), min=1e-5)
        imgs = (imgs - mean) / std

        # 2. Forward Encoder
        # latent shape: [B, N_patches + 1, D]
        latent, _, _ = self.encoder.forward_encoder(imgs)
        
        # 3. Feature Selection
        # 丢弃 CLS token (index 0)，保留所有 Patch tokens
        patches = latent[:, 1:, :] # [B, N_patches, D]
        
        # 4. Forward Head
        if self.use_cot:
            logits = self.head(patches)
        else:
            global_feat = patches.mean(dim=1) # [B, D]
            logits = self.head(global_feat)
        
        return logits