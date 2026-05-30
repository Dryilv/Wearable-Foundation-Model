import torch
import torch.nn as nn
import torch.nn.functional as F

from model import CWT_MAE_RoPE, cwt_wrap, DropPath
from utils import is_main_process

# ===================================================================
# 1. 隐式思维链模块 (Latent Reasoning / Chain-of-Thought Head)
# ===================================================================
class LatentReasoningHead(nn.Module):
    def __init__(self, embed_dim, num_heads, num_classes,
                 num_reasoning_tokens=32,
                 num_kv_layers=1,
                 dropout=0.1,
                 drop_path=0.0):
        super().__init__()
        self.num_reasoning_tokens = num_reasoning_tokens
        self.embed_dim = embed_dim
        self.num_kv_layers = num_kv_layers

        effective_dim = embed_dim * num_kv_layers

        self.reasoning_tokens = nn.Parameter(torch.zeros(1, num_reasoning_tokens, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.cross_attn_q_proj = nn.Linear(embed_dim, embed_dim)
        self.cross_attn_kv_proj = nn.Linear(effective_dim, embed_dim * 2)
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

        self.cls_cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm_cls1 = nn.LayerNorm(embed_dim)
        self.norm_cls2 = nn.LayerNorm(embed_dim)
        self.cls_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        self.classifier = nn.Linear(embed_dim, num_classes)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self._init_weights()
        nn.init.normal_(self.reasoning_tokens, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x_encoder, token_padding_mask=None, extra_features=None,
                multi_layer_features=None):
        B = x_encoder.shape[0]
        queries = self.reasoning_tokens.expand(B, -1, -1)

        if multi_layer_features is not None:
            kv_input = torch.cat(multi_layer_features, dim=-1)
        else:
            kv_input = x_encoder

        q = self.cross_attn_q_proj(queries)
        kv = self.cross_attn_kv_proj(kv_input)
        k, v = kv.chunk(2, dim=-1)

        attn_out, _ = self.cross_attn(
            query=q, key=k, value=v,
            key_padding_mask=token_padding_mask
        )
        queries = self.norm1(queries + self.drop_path(attn_out))

        attn_out2, _ = self.self_attn(query=queries, key=queries, value=queries)
        queries = self.norm2(queries + self.drop_path(attn_out2))

        queries = self.norm3(queries + self.drop_path(self.ffn(queries)))

        cls = self.cls_token.expand(B, -1, -1)
        cls_attn, _ = self.cls_cross_attn(
            query=cls, key=queries, value=queries
        )
        cls = self.norm_cls1(cls + self.drop_path(cls_attn))
        cls = self.norm_cls2(cls + self.drop_path(self.cls_ffn(cls)))

        decision_token = cls.squeeze(1)

        if extra_features is not None:
            decision_token = torch.cat([decision_token, extra_features], dim=-1)
        logits = self.classifier(decision_token)
        return logits

# ===================================================================
# 2. 主分类器模型封装
# ===================================================================
class TF_MAE_Classifier(nn.Module):
    def __init__(self, pretrained_path, num_classes, 
                 use_cot=True, 
                 num_reasoning_tokens=16, 
                 cot_kv_layers=None,
                 **kwargs):
        super().__init__()
        
        self.use_stats_features = kwargs.get('use_stats_features', False)
        self.embed_dim = kwargs.get('embed_dim', 768)
        self.cot_kv_layers = cot_kv_layers
        depth = kwargs.get('depth', 12)
        
        encoder_kwargs = {k: v for k, v in kwargs.items() if k != 'use_stats_features'}
        
        self.encoder_model = CWT_MAE_RoPE(
            mask_ratio=0.0, 
            **encoder_kwargs
        )
        
        if pretrained_path:
            self._load_pretrained_weights(pretrained_path)
        
        self._delete_decoder_components()
        
        classifier_in_dim = self.embed_dim
        if self.use_stats_features:
            classifier_in_dim += 16

        if use_cot:
            num_kv_layers = len(cot_kv_layers) if cot_kv_layers else 1
            if is_main_process():
                print(f">>> Initializing Latent Reasoning Head (CoT) with {num_reasoning_tokens} tokens, {num_kv_layers} KV layers.")
            self.head = LatentReasoningHead(
                embed_dim=self.embed_dim,
                num_heads=kwargs.get('num_heads', 12),
                num_classes=num_classes,
                num_reasoning_tokens=num_reasoning_tokens,
                num_kv_layers=num_kv_layers,
                dropout=0.2,
                drop_path=kwargs.get('drop_path_rate', 0.0)
            )
            if self.use_stats_features:
                self.head.classifier = nn.Linear(self.embed_dim + 16, num_classes)
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(classifier_in_dim),
                nn.Linear(classifier_in_dim, num_classes)
            )


    def _delete_decoder_components(self):
        components_to_delete =[
            'decoder_blocks', 'decoder_embed', 'decoder_pred_spec',
            'time_reducer', 'time_pred', 'mask_token',
            'decoder_pos_embed', 'rope_decoder', 'decoder_norm',
            'decoder_channel_embed', 'channel_embed' 
        ]
        
        for component in components_to_delete:
            if hasattr(self.encoder_model, component):
                delattr(self.encoder_model, component)

    def _load_pretrained_weights(self, path):
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
        
        grid_h, grid_w_new = self.encoder_model.grid_size
        n_old = patch_tokens.shape[1]
        
        grid_w_old = n_old // grid_h
        dim = patch_tokens.shape[-1]
        
        patch_tokens = patch_tokens.transpose(1, 2).reshape(1, dim, grid_h, grid_w_old)
        patch_tokens = F.interpolate(patch_tokens, size=(grid_h, grid_w_new), mode='bicubic', align_corners=False)
        patch_tokens = patch_tokens.flatten(2).transpose(1, 2)
        
        state_dict[key] = patch_tokens

    def forward(self, x, channel_mask=None, channel_ids=None):
        if x.dim() == 2: x = x.unsqueeze(1)

        imgs = self.encoder_model.prepare_tokens(x)

        if torch.onnx.is_in_onnx_export():
            if x.device != next(self.encoder_model.parameters()).device:
                x = x.to(next(self.encoder_model.parameters()).device)
            if imgs.device != x.device:
                imgs = imgs.to(x.device)

        self.encoder_model.mask_ratio = 0.0
        if channel_ids is None:
            channel_ids = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        if self.cot_kv_layers and isinstance(self.head, LatentReasoningHead):
            latent, _, _, _, intermediate_features = self.encoder_model.forward_encoder(
                x, imgs, channel_ids, return_layer_indices=self.cot_kv_layers
            )
            multi_layer = [intermediate_features[i] for i in self.cot_kv_layers]
        else:
            latent, _, _, _ = self.encoder_model.forward_encoder(x, imgs, channel_ids)
            multi_layer = None
        
        latent_pooled = latent.mean(dim=1)
        pred_stats = self.encoder_model.stats_pred_head(latent_pooled)
        
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
            logits = self.head(
                patch_tokens,
                token_padding_mask=token_padding_mask,
                extra_features=pred_stats if self.use_stats_features else None,
                multi_layer_features=multi_layer
            )
        else:
            global_feat = patch_tokens.mean(dim=1)
            if self.use_stats_features:
                global_feat = torch.cat([global_feat, pred_stats], dim=-1)
            logits = self.head(global_feat)
        
        return logits
