import torch
import torch.nn as nn
import torch.nn.functional as F
# 【关键修改】导入 RoPE 版本的主模型
from model import CWT_MAE_RoPE, cwt_wrap

class TF_MAE_Classifier(nn.Module):
    def __init__(self, pretrained_path, num_classes, hidden_dim=512, dropout_rate=0.5, 
                 mlp_rank_ratio=0.5, **kwargs):
        """
        kwargs: 包含 embed_dim, depth, num_heads, cwt_scales, patch_size_time 等
        """
        super().__init__()
        
        # 1. 初始化 Encoder (使用下游任务的 signal_len)
        # 必须传入 mlp_rank_ratio 以匹配预训练模型的结构
        self.encoder_model = CWT_MAE_RoPE(
            mlp_rank_ratio=mlp_rank_ratio,
            mask_ratio=0.0, # 微调时默认不掩码
            **kwargs
        )
        self.embed_dim = kwargs.get('embed_dim', 768)
        
        # 2. 加载预训练权重
        if pretrained_path:
            self._load_pretrained_weights(pretrained_path)
        
        # 3. 删除 Decoder 和 Heads 以节省显存
        # RoPE 版本中 Decoder 相关的组件名称可能略有不同，确保删除干净
        del self.encoder_model.decoder_blocks
        del self.encoder_model.decoder_embed
        del self.encoder_model.decoder_pred_spec
        del self.encoder_model.time_reducer
        del self.encoder_model.time_pred
        del self.encoder_model.mask_token
        del self.encoder_model.decoder_pos_embed
        del self.encoder_model.rope_decoder # 删除 Decoder 的 RoPE 生成器
        
        if hasattr(self.encoder_model, 'decoder_norm'):
            del self.encoder_model.decoder_norm

        # 4. 分类头 (使用 CLS Token)
        self.head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, hidden_dim),
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
        print(f"Loading weights from {path}...")
        checkpoint = torch.load(path, map_location='cpu')
        
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # 1. 清洗 Key 名称
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k
            if name.startswith('module.'): name = name[7:]
            if name.startswith('_orig_mod.'): name = name[10:]
            new_state_dict[name] = v

        # 2. 过滤掉 Decoder 相关权重
        encoder_dict = {}
        for k, v in new_state_dict.items():
            # 过滤掉 decoder, mask_token, prediction heads, rope_decoder
            if any(x in k for x in ["decoder", "mask_token", "time_reducer", "time_pred", "rope_decoder"]):
                continue
            encoder_dict[k] = v
            
        # 3. 位置编码插值 (针对 Absolute Pos Embed)
        # 虽然有 RoPE，但 Absolute Pos Embed 依然作为 Global Anchor 存在，需要插值
        if hasattr(self.encoder_model, 'pos_embed'):
            self._interpolate_pos_embed(encoder_dict, 'pos_embed', self.encoder_model.pos_embed)

        # 4. 加载
        msg = self.encoder_model.load_state_dict(encoder_dict, strict=False)
        print(f"Weights loaded. Missing keys (expected decoder keys): {msg.missing_keys}")
        # 检查是否有非预期的 missing keys (例如 encoder 内部权重没对上)
        unexpected_missing = [k for k in msg.missing_keys if "decoder" not in k and "pred" not in k]
        if len(unexpected_missing) > 0:
            print(f"[WARNING] Unexpected missing keys in Encoder: {unexpected_missing}")

    def _interpolate_pos_embed(self, state_dict, key, new_pos_embed):
        """
        针对 CWT Patch 的 2D 结构进行插值
        pos_embed: [1, Num_Patches + 1, Dim]
        """
        if key not in state_dict:
            return

        old_pos_embed = state_dict[key] 
        
        if old_pos_embed.shape[1] == new_pos_embed.shape[1]:
            return

        print(f"Interpolating {key}: {old_pos_embed.shape[1]} -> {new_pos_embed.shape[1]}")

        # 1. 分离 CLS
        cls_token = old_pos_embed[:, :1, :]
        patch_tokens = old_pos_embed[:, 1:, :] 
        
        # 2. 恢复 2D 结构
        # 获取当前模型的 Grid Size
        grid_h, grid_w_new = self.encoder_model.grid_size
        
        n_old = patch_tokens.shape[1]
        grid_w_old = n_old // grid_h # 假设频率轴 (H) 不变
        dim = patch_tokens.shape[-1]
        
        # Reshape to [1, Dim, Grid_H, Grid_W_old]
        patch_tokens = patch_tokens.transpose(1, 2).reshape(1, dim, grid_h, grid_w_old)
        
        # 3. 插值
        patch_tokens = F.interpolate(
            patch_tokens, 
            size=(grid_h, grid_w_new), 
            mode='bicubic', 
            align_corners=False
        )
        
        # 4. 展平回 [1, N_new, D]
        patch_tokens = patch_tokens.flatten(2).transpose(1, 2)
        
        # 5. 拼回 CLS
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
        # 确保 mask_ratio 为 0，让 RoPE 看到完整序列
        self.encoder_model.mask_ratio = 0.0
        
        # forward_encoder 返回 (latent, mask, ids)
        latent, _, _ = self.encoder_model.forward_encoder(imgs)
        
        # 4. Classification Head (使用 CLS token)
        cls_token = latent[:, 0, :]
        logits = self.head(cls_token)
        
        return logits