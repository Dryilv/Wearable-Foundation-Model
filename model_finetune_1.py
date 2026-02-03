import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 确保 model_1.py (包含 CWT_MAE_RoPE 和 cwt_wrap) 在同一目录下
from model_1 import CWT_MAE_RoPE, cwt_wrap

# ===================================================================
# 1. 隐式思维链模块 (Latent Reasoning / Chain-of-Thought Head)
# ===================================================================
class ReasoningBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        # Cross-Attention: Query=Reasoning, Key/Value=Signal Features
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Self-Attention: Reasoning tokens 内部交互
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

    def forward(self, queries, x_encoder):
        # Cross Attention
        attn_out, _ = self.cross_attn(query=queries, key=x_encoder, value=x_encoder)
        queries = self.norm1(queries + attn_out)
        
        # Self Attention
        attn_out2, _ = self.self_attn(query=queries, key=queries, value=queries)
        queries = self.norm2(queries + attn_out2)
        
        # FFN
        queries = self.norm3(queries + self.ffn(queries))
        return queries

class LatentReasoningHead(nn.Module):
    def __init__(self, embed_dim, num_heads, num_classes, num_reasoning_tokens=32, depth=3, dropout=0.1):
        super().__init__()
        self.num_reasoning_tokens = num_reasoning_tokens
        self.embed_dim = embed_dim
        
        # 推理令牌 (Query)
        self.reasoning_tokens = nn.Parameter(torch.zeros(1, num_reasoning_tokens, embed_dim))
        
        # Stacking Reasoning Blocks
        self.blocks = nn.ModuleList([
            ReasoningBlock(embed_dim, num_heads, dropout) 
            for _ in range(depth)
        ])
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        
        self._init_weights()
        nn.init.normal_(self.reasoning_tokens, std=0.02)

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x_encoder):
        # x_encoder: (B, Total_Tokens, D)
        B = x_encoder.shape[0]
        queries = self.reasoning_tokens.expand(B, -1, -1) # (B, N_reason, D)
        
        for blk in self.blocks:
            queries = blk(queries, x_encoder)
        
        # Global Pooling & Classify
        decision_token = queries.mean(dim=1) 
        logits = self.classifier(decision_token)
        return logits

# ===================================================================
# 2. 主分类器模型封装
# ===================================================================
class TF_MAE_Classifier(nn.Module):
    def __init__(self, pretrained_path, num_classes, 
                 mlp_rank_ratio=0.5, 
                 use_cot=True, 
                 num_reasoning_tokens=16,
                 reasoning_depth=3, # 【新增】深度推理
                 train_signal_len=500, # 【新增】记录训练时的信号长度
                 **kwargs):
        super().__init__()
        
        self.train_signal_len = train_signal_len
        
        # Ensure signal_len is not passed twice
        if 'signal_len' in kwargs:
            kwargs.pop('signal_len')

        # 1. 初始化 Encoder (CWT-MAE-RoPE)
        self.encoder_model = CWT_MAE_RoPE(
            mlp_rank_ratio=mlp_rank_ratio,
            mask_ratio=0.0, # 微调时关闭 Mask
            signal_len=train_signal_len, # 确保 Encoder 也是按训练长度初始化
            **kwargs
        )
        self.embed_dim = kwargs.get('embed_dim', 768)
        
        # 2. 加载预训练权重
        if pretrained_path:
            self._load_pretrained_weights(pretrained_path)
        
        # 3. 清理 Decoder (节省显存)
        self._delete_decoder_components()

        # 4. 初始化分类头
        if use_cot:
            print(f">>> Initializing Latent Reasoning Head (CoT) with {num_reasoning_tokens} tokens, depth={reasoning_depth}.")
            self.head = LatentReasoningHead(
                embed_dim=self.embed_dim,
                num_heads=kwargs.get('num_heads', 12),
                num_classes=num_classes,
                num_reasoning_tokens=num_reasoning_tokens,
                depth=reasoning_depth,
                dropout=0.2
            )
        else:
            # 简单的 Linear Head
            self.head = nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, num_classes)
            )

    def _delete_decoder_components(self):
        """删除预训练模型中的 Decoder 部分"""
        # 使用 hasattr 检查，避免属性不存在时报错
        components_to_delete = [
            'decoder_blocks', 'decoder_embed', 'decoder_pred_spec',
            'time_reducer', 'time_pred', 'mask_token',
            'decoder_pos_embed', 'rope_decoder', 'decoder_norm',
            'decoder_channel_embed', 'channel_embed' # 确保清理旧版本可能存在的组件
        ]
        
        for component in components_to_delete:
            if hasattr(self.encoder_model, component):
                delattr(self.encoder_model, component)

    def _load_pretrained_weights(self, path):
        print(f"Loading weights from {path}...")
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

        # 清洗 Key 名称
        new_state_dict = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        
        # 过滤 Decoder 权重和不匹配的权重
        encoder_dict = {}
        for k, v in new_state_dict.items():
            # 过滤 decoder 相关
            if any(x in k for x in ["decoder", "mask_token", "time_reducer", "time_pred", "rope_decoder"]):
                continue
            # 过滤 channel_embed 相关 (如果预训练模型是旧版)
            if "channel_embed" in k:
                continue
            encoder_dict[k] = v
            
        # 位置编码插值 (处理不同长度输入)
        if hasattr(self.encoder_model, 'pos_embed'):
            self._interpolate_pos_embed(encoder_dict, 'pos_embed', self.encoder_model.pos_embed)

        msg = self.encoder_model.load_state_dict(encoder_dict, strict=False)
        print(f"Weights loaded. Missing keys (expected decoder keys): {msg.missing_keys}")
        print(f"Unexpected keys: {msg.unexpected_keys}")

    def _interpolate_pos_embed(self, state_dict, key, new_pos_embed):
        if key not in state_dict: return
        old_pos_embed = state_dict[key] 
        if old_pos_embed.shape[1] == new_pos_embed.shape[1]: return

        print(f"Interpolating {key}: {old_pos_embed.shape[1]} -> {new_pos_embed.shape[1]}")
        cls_token = old_pos_embed[:, :1, :]
        patch_tokens = old_pos_embed[:, 1:, :] 
        
        grid_h, grid_w_new = self.encoder_model.grid_size
        n_old = patch_tokens.shape[1]
        
        # 假设 grid_h (频率维度) 不变，只改变 grid_w (时间维度)
        # 如果 n_old 不能被 grid_h 整除，说明预训练时的 grid_h 可能不同，这里简单假设 grid_h 兼容
        grid_w_old = n_old // grid_h
        dim = patch_tokens.shape[-1]
        
        patch_tokens = patch_tokens.transpose(1, 2).reshape(1, dim, grid_h, grid_w_old)
        patch_tokens = F.interpolate(patch_tokens, size=(grid_h, grid_w_new), mode='bicubic', align_corners=False)
        patch_tokens = patch_tokens.flatten(2).transpose(1, 2)
        
        new_pos_embed_interpolated = torch.cat((cls_token, patch_tokens), dim=1)
        state_dict[key] = new_pos_embed_interpolated

    def forward(self, x):
        """
        x: (B, M, L) 多通道输入
        """
        # 兼容单通道输入 (B, L) -> (B, 1, L)
        if x.dim() == 2: x = x.unsqueeze(1)
        
        B, M, L = x.shape
        
        # 【新增】滑动窗口推理逻辑
        # 如果是评估模式，且输入长度超过训练长度，则切片预测并投票
        if not self.training and L > self.train_signal_len:
            return self.forward_sliding_window(x)

        return self._forward_impl(x)

    def forward_sliding_window(self, x):
        """
        对长序列进行滑动窗口预测，然后聚合结果
        """
        B, M, L = x.shape
        window_size = self.train_signal_len
        stride = window_size // 2 # 50% 重叠
        
        # unfold: (B, M, N_wins, W)
        # 注意：unfold 作用在最后一维
        windows = x.unfold(dimension=-1, size=window_size, step=stride)
        
        # (B, M, N_wins, W) -> (B, N_wins, M, W)
        windows = windows.permute(0, 2, 1, 3).contiguous()
        N_wins = windows.shape[1]
        
        # Flatten -> (B * N_wins, M, W)
        windows_flat = windows.view(B * N_wins, M, window_size)
        
        # Forward pass
        logits_flat = self._forward_impl(windows_flat) # (B * N_wins, Num_Classes)
        
        # Reshape & Aggregate
        logits = logits_flat.view(B, N_wins, -1)
        
        # 这里使用 Softmax 平均，比 Logits 平均更稳健（类似于 Voting）
        probs = F.softmax(logits, dim=-1)
        avg_probs = probs.mean(dim=1)
        
        # 注意：这里返回的是概率分布，或者我们可以返回 log 概率以配合 CrossEntropy
        # 为了兼容 validate 函数 (expect logits), 我们返回 log(avg_probs)
        return torch.log(avg_probs + 1e-6)

    def _forward_impl(self, x):
        # 1. CWT 变换 (B, M, L) -> (B, M, 3, Scales, L)
        # 注意：cwt_wrap 现在支持多通道
        imgs = cwt_wrap(x, num_scales=self.encoder_model.cwt_scales, lowest_scale=0.1, step=1.0)
        
        # 2. Instance Norm (独立对每个通道、每个样本归一化)
        imgs_f32 = imgs.float()
        # mean/std over (H, W) -> (B, M, 3, 1, 1)
        mean = imgs_f32.mean(dim=(3, 4), keepdim=True)
        std = imgs_f32.std(dim=(3, 4), keepdim=True)
        std = torch.clamp(std, min=1e-5)
        imgs = (imgs_f32 - mean) / std
        
        # 数值鲁棒性增强：防止 CWT 产生的极端值引发梯度爆炸
        imgs = torch.nan_to_num(imgs, nan=0.0, posinf=100.0, neginf=-100.0)
        imgs = torch.clamp(imgs, min=-100.0, max=100.0)

        # 确保输入数据类型与模型权重一致
        target_dtype = next(self.encoder_model.parameters()).dtype
        imgs = imgs.to(dtype=target_dtype)

        # 3. Forward Encoder
        # 新版 forward_encoder 返回: x, mask, ids, M
        # x 的形状是 (B, M*N_patches + 1, D)
        self.encoder_model.mask_ratio = 0.0
        latent, _, _, _ = self.encoder_model.forward_encoder(imgs)
        
        # 4. 提取特征
        # 丢弃 CLS token (index 0)，保留 Patch tokens
        # patch_tokens: (B, M*N_patches, D)
        patch_tokens = latent[:, 1:, :] 
        
        # 5. Forward Head
        if isinstance(self.head, LatentReasoningHead):
            # CoT 模式: 输入所有 Patch Tokens (混合了所有通道的信息)
            # Cross-Attention 会自动处理 M*N 的长度
            logits = self.head(patch_tokens)
        else:
            # MLP 模式: Global Average Pooling
            global_feat = patch_tokens.mean(dim=1)
            logits = self.head(global_feat)
        
        return logits
