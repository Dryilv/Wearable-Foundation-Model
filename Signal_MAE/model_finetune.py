import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from model import Signal_MAE_RoPE

def is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

# ===================================================================
# 1. 隐式思维链模块 (Latent Reasoning / Chain-of-Thought Head)
# ===================================================================
class LatentReasoningHead(nn.Module):
    def __init__(self, embed_dim, num_heads, num_classes, num_reasoning_tokens=32, dropout=0.1):
        super().__init__()
        self.num_reasoning_tokens = num_reasoning_tokens
        self.embed_dim = embed_dim
        
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

    def forward(self, x_encoder, token_padding_mask=None):
        B = x_encoder.shape[0]
        queries = self.reasoning_tokens.expand(B, -1, -1) 
        
        attn_out, _ = self.cross_attn(query=queries, key=x_encoder, value=x_encoder, key_padding_mask=token_padding_mask)
        queries = self.norm1(queries + attn_out)
        
        attn_out2, _ = self.self_attn(query=queries, key=queries, value=queries)
        queries = self.norm2(queries + attn_out2)
        
        queries = self.norm3(queries + self.ffn(queries))
        
        decision_token = queries.mean(dim=1) 
        logits = self.classifier(decision_token)
        return logits

# ===================================================================
# 2. 主分类器模型封装
# ===================================================================
class TF_MAE_Classifier(nn.Module):
    def __init__(self, pretrained_path, num_classes, 
                 use_cot=True, 
                 num_reasoning_tokens=16, 
                 **kwargs):
        super().__init__()
        
        self.encoder_model = Signal_MAE_RoPE(
            mask_ratio=0.0, 
            **kwargs
        )
        self.embed_dim = kwargs.get('embed_dim', 768)
        
        if pretrained_path:
            self._load_pretrained_weights(pretrained_path)
        
        self._delete_decoder_components()

        if use_cot:
            if is_main_process():
                print(f">>> Initializing Latent Reasoning Head (CoT) with {num_reasoning_tokens} tokens.")
            self.head = LatentReasoningHead(
                embed_dim=self.embed_dim,
                num_heads=kwargs.get('num_heads', 12),
                num_classes=num_classes,
                num_reasoning_tokens=num_reasoning_tokens,
                dropout=0.2
            )
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, num_classes)
            )


    def _delete_decoder_components(self):
        components_to_delete =[
            'decoder_blocks', 'decoder_embed', 'decoder_pred_spec', 'decoder_pred',
            'time_reducer', 'time_pred', 'mask_token',
            'decoder_pos_embed', 'rope_decoder', 'decoder_norm',
            'decoder_channel_embed', 'channel_embed' 
        ]
        
        for component in components_to_delete:
            if hasattr(self.encoder_model, component):
                delattr(self.encoder_model, component)

    def _load_pretrained_weights(self, path):
        if is_main_process():
            print(f"Loading weights from {path}...")
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '').replace('_orig_mod.', '')
            if name.startswith('encoder.'):
                name = name[len('encoder.'):]
            new_state_dict[name] = v
        
        encoder_dict = {}
        for k, v in new_state_dict.items():
            if any(x in k for x in["decoder", "mask_token", "time_reducer", "time_pred", "rope_decoder"]):
                continue
            if "channel_embed" in k:
                continue
            encoder_dict[k] = v
            
        if hasattr(self.encoder_model, 'pos_embed'):
            self._interpolate_pos_embed(encoder_dict, 'pos_embed', self.encoder_model.pos_embed)

        msg = self.encoder_model.load_state_dict(encoder_dict, strict=False)
        
        actual_missing =[k for k in msg.missing_keys if not any(x in k for x in["decoder", "mask_token", "time_reducer", "time_pred", "rope_decoder"])]
        
        if is_main_process():
            print(f"Weights loaded.")
            if actual_missing:
                print(f"WARNING: Missing encoder keys: {actual_missing}")
            else:
                print("Encoder weights loaded successfully.")
            
        if msg.unexpected_keys:
             actual_unexpected =[k for k in msg.unexpected_keys if "proj_head" not in k]
             if actual_unexpected and is_main_process():
                 print(f"Unexpected keys: {actual_unexpected}")

    def _interpolate_pos_embed(self, state_dict, key, new_pos_embed):
        if key not in state_dict: return
        old_pos_embed = state_dict[key] 
        if old_pos_embed.shape[1] == new_pos_embed.shape[1]: return

        if is_main_process():
            print(f"Interpolating {key}: {old_pos_embed.shape[1]} -> {new_pos_embed.shape[1]}")
        
        patch_tokens = old_pos_embed 
        
        n_new = new_pos_embed.shape[1]
        dim = patch_tokens.shape[-1]
        
        patch_tokens = patch_tokens.transpose(1, 2) # (1, dim, N_old)
        patch_tokens = F.interpolate(patch_tokens, size=n_new, mode='linear', align_corners=False)
        patch_tokens = patch_tokens.transpose(1, 2) # (1, n_new, dim)
        
        state_dict[key] = patch_tokens

    def forward(self, x, channel_mask=None, channel_ids=None):
        if x.dim() == 2: x = x.unsqueeze(1)

        # 确保输入数据类型与模型参数类型一致
        param_dtype = next(self.encoder_model.parameters()).dtype
        if x.dtype != param_dtype:
            x = x.to(param_dtype)

        x_norm = self.encoder_model.prepare_tokens(x)

        if torch.onnx.is_in_onnx_export():
            if x.device != next(self.encoder_model.parameters()).device:
                x = x.to(next(self.encoder_model.parameters()).device)
            if x_norm.device != x.device:
                x_norm = x_norm.to(x.device)

        self.encoder_model.mask_ratio = 0.0
        if channel_ids is None:
            channel_ids = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        latent, _, _, _ = self.encoder_model.forward_encoder(x_norm, channel_ids)
        
        patch_tokens = latent 
        token_padding_mask = None
        if channel_mask is not None:
            channel_mask = channel_mask.to(patch_tokens.device, dtype=torch.bool)
            B_mask, total_tokens, _ = patch_tokens.shape
            M_mask = x.shape[1]
            if channel_mask.shape[0] == B_mask and channel_mask.shape[1] == M_mask and M_mask > 0 and total_tokens % M_mask == 0:
                n_patches = total_tokens // M_mask
                token_padding_mask = (~channel_mask).unsqueeze(-1).expand(B_mask, M_mask, n_patches).reshape(B_mask, total_tokens)
        
        if isinstance(self.head, LatentReasoningHead):
            logits = self.head(patch_tokens, token_padding_mask=token_padding_mask)
        else:
            global_feat = patch_tokens.mean(dim=1)
            logits = self.head(global_feat)
        
        return logits