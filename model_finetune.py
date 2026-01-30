# --- START OF FILE model_finetune.py ---

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 确保 model.py 在同一目录下，并且包含 CWT_MAE_RoPE 和 cwt_wrap
try:
    from model import CWT_MAE_RoPE, cwt_wrap
except ImportError:
    print("Error: 'model.py' not found or missing classes. Please ensure CWT_MAE_RoPE and cwt_wrap are available.")

# ===================================================================
# 1. 隐式思维链模块 (Latent Reasoning / Chain-of-Thought Head)
#    核心思想：使用可学习的 Query 去"查询"信号特征，模拟分步推理
# ===================================================================
class LatentReasoningHead(nn.Module):
    def __init__(self, embed_dim, num_heads, num_classes, num_reasoning_tokens=32, dropout=0.1):
        super().__init__()
        self.num_reasoning_tokens = num_reasoning_tokens
        self.embed_dim = embed_dim
        
        # 1. 定义推理令牌 (Learnable Queries)
        self.reasoning_tokens = nn.Parameter(torch.zeros(1, num_reasoning_tokens, embed_dim))
        
        # 2. Cross-Attention: Queries 关注 Input Features
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # 3. Self-Attention: Queries 之间交互 (推理步骤之间的关联)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # 4. FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm3 = nn.LayerNorm(embed_dim)
        
        # 5. Classifier
        self.classifier = nn.Linear(embed_dim, num_classes)
        
        # 6. 初始化权重
        self._init_weights()
        # 重新初始化 reasoning_tokens 为正态分布
        nn.init.normal_(self.reasoning_tokens, std=0.02)

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x_encoder):
        # x_encoder: [Batch, Seq_Len, Dim] (包含频域和时域的所有 Token)
        B = x_encoder.shape[0]
        queries = self.reasoning_tokens.expand(B, -1, -1)
        
        # Cross Attention: Reasoning Tokens 查询 Encoder Features
        attn_out, _ = self.cross_attn(query=queries, key=x_encoder, value=x_encoder)
        queries = self.norm1(queries + attn_out)
        
        # Self Attention: Reasoning Tokens 内部交互
        attn_out2, _ = self.self_attn(query=queries, key=queries, value=queries)
        queries = self.norm2(queries + attn_out2)
        
        # FFN
        queries = self.norm3(queries + self.ffn(queries))
        
        # Global Pooling & Classify
        # 将所有推理步骤的结果平均，作为最终决策依据
        decision_token = queries.mean(dim=1) 
        logits = self.classifier(decision_token)
        return logits

# ===================================================================
# 2. 时域形态特征提取器 (Time Domain Encoder) - NEW!
#    专门用于提取 PPG 原始波形、一阶导数(VPG)、二阶导数(APG) 的形态特征
# ===================================================================
class TimeDomainEncoder(nn.Module):
    def __init__(self, input_channels=3, embed_dim=768, kernel_size=7):
        super().__init__()
        # 输入通道=3 (Raw, 1st Deriv, 2nd Deriv)
        # 使用 1D 卷积提取波形细节
        
        self.proj = nn.Sequential(
            # Layer 1: 下采样
            nn.Conv1d(input_channels, embed_dim // 4, kernel_size=kernel_size, padding=kernel_size//2, stride=2),
            nn.BatchNorm1d(embed_dim // 4),
            nn.GELU(),
            # Layer 2: 下采样
            nn.Conv1d(embed_dim // 4, embed_dim // 2, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm1d(embed_dim // 2),
            nn.GELU(),
            # Layer 3: 下采样 -> 映射到 embed_dim
            nn.Conv1d(embed_dim // 2, embed_dim, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(embed_dim),
            nn.GELU()
        )
        
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x shape: [B, 3, L]
        feat = self.proj(x) # [B, Dim, L_reduced]
        feat = feat.transpose(1, 2) # [B, L_reduced, Dim]
        return self.out_norm(feat)

# ===================================================================
# 3. 残差 MLP 块 (备用模块)
# ===================================================================
class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.5):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return x + self.net(self.norm(x))

# ===================================================================
# 4. 主分类器模型封装 (Dual-Stream Architecture)
# ===================================================================
class TF_MAE_Classifier(nn.Module):
    def __init__(self, pretrained_path, num_classes, 
                 mlp_rank_ratio=0.5, 
                 # CoT 配置
                 use_cot=True, 
                 num_reasoning_tokens=8, 
                 # 备用 MLP 配置
                 hidden_dim=512, 
                 dropout_rate=0.0,
                 num_res_blocks=2,
                 **kwargs):
        super().__init__()
        
        self.embed_dim = kwargs.get('embed_dim', 768)
        
        # -----------------------------------------------------------
        # Stream 1: Frequency Domain (CWT + MAE Encoder)
        # -----------------------------------------------------------
        self.encoder_model = CWT_MAE_RoPE(
            mlp_rank_ratio=mlp_rank_ratio,
            mask_ratio=0.0, # 微调时关闭 Mask
            **kwargs
        )
        
        # -----------------------------------------------------------
        # Stream 2: Time Domain (Raw + Derivatives) -> NEW!
        # -----------------------------------------------------------
        # 这是一个轻量级的旁路，专门提取血管硬化特征
        self.time_encoder = TimeDomainEncoder(input_channels=3, embed_dim=self.embed_dim)
        
        # 2. 加载预训练权重
        if pretrained_path:
            self._load_pretrained_weights(pretrained_path)
        
        # 3. 清理 Decoder 以节省显存
        self._delete_decoder_components()

        # 4. 初始化分类头
        if use_cot:
            print(f">>> Initializing Latent Reasoning Head (CoT) with {num_reasoning_tokens} tokens.")
            self.head = LatentReasoningHead(
                embed_dim=self.embed_dim,
                num_heads=kwargs.get('num_heads', 12),
                num_classes=num_classes,
                num_reasoning_tokens=num_reasoning_tokens,
                dropout=0.2
            )
        else:
            print(f">>> Initializing Deep Residual MLP Head ({num_res_blocks} blocks).")
            layers = []
            layers.append(nn.LayerNorm(self.embed_dim))
            layers.append(nn.Linear(self.embed_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))
            for _ in range(num_res_blocks):
                layers.append(ResidualMLPBlock(hidden_dim, dropout_rate))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Linear(hidden_dim, num_classes))
            
            self.head = nn.Sequential(*layers)
            self._init_head_weights()

    def _delete_decoder_components(self):
        """删除预训练模型中的 Decoder 部分"""
        components_to_delete = [
            'decoder_blocks', 'decoder_embed', 'decoder_pred_spec', 
            'time_reducer', 'time_pred', 'mask_token', 
            'decoder_pos_embed', 'rope_decoder', 'decoder_norm'
        ]
        for comp in components_to_delete:
            if hasattr(self.encoder_model, comp):
                delattr(self.encoder_model, comp)

    def _init_head_weights(self):
        """仅用于 MLP Head 的初始化"""
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def _load_pretrained_weights(self, path):
        print(f"Loading weights from {path}...")
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

        # 清洗 Key 名称
        new_state_dict = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        
        # 过滤 Decoder 权重
        encoder_dict = {}
        for k, v in new_state_dict.items():
            if any(x in k for x in ["decoder", "mask_token", "time_reducer", "time_pred", "rope_decoder"]):
                continue
            encoder_dict[k] = v
            
        # 位置编码插值
        if hasattr(self.encoder_model, 'pos_embed'):
            self._interpolate_pos_embed(encoder_dict, 'pos_embed', self.encoder_model.pos_embed)

        msg = self.encoder_model.load_state_dict(encoder_dict, strict=False)
        print(f"Weights loaded. Missing keys (expected decoder keys): {msg.missing_keys}")

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

    def compute_derivatives(self, x):
        """
        计算一阶和二阶导数，并进行标准化
        x: [B, L]
        """
        # 1. 一阶导数 (Velocity)
        # 使用 torch.diff，并在最后补零以保持长度一致
        vpg = torch.diff(x, dim=-1, prepend=x[:, :1])
        
        # 2. 二阶导数 (Acceleration - APG)
        apg = torch.diff(vpg, dim=-1, prepend=vpg[:, :1])
        
        # 3. 归一化 (Instance Normalization)
        # 导数数值通常很小，必须归一化才能被网络有效学习
        def norm(t):
            mean = t.mean(dim=-1, keepdim=True)
            std = t.std(dim=-1, keepdim=True) + 1e-6
            return (t - mean) / std

        x_norm = norm(x)
        vpg_norm = norm(vpg)
        apg_norm = norm(apg)
        
        # Stack: [B, 3, L]
        return torch.stack([x_norm, vpg_norm, apg_norm], dim=1)

    def forward(self, x):
        # x: [B, 1, L] or [B, L]
        if x.dim() == 3: x = x.squeeze(1)
        x = x.float() 
        
        # ===========================
        # Stream 1: CWT + MAE (频域)
        # ===========================
        # 1. CWT 变换
        imgs = cwt_wrap(x, num_scales=self.encoder_model.cwt_scales, lowest_scale=0.1, step=1.0)
        
        # 2. Instance Norm for CWT
        dtype_orig = imgs.dtype
        imgs_f32 = imgs.float()
        mean = imgs_f32.mean(dim=(2, 3), keepdim=True)
        std = imgs_f32.std(dim=(2, 3), keepdim=True) + 1e-5
        imgs = (imgs_f32 - mean) / std
        imgs = imgs.to(dtype=dtype_orig)

        # 3. Forward Encoder
        self.encoder_model.mask_ratio = 0.0
        # latent_freq: [B, N_patches + 1, Dim]
        latent_freq, _, _ = self.encoder_model.forward_encoder(imgs)
        freq_tokens = latent_freq[:, 1:, :] # 丢弃 CLS token
        
        # ===========================
        # Stream 2: Time Domain (时域 + 导数)
        # ===========================
        # 1. 计算导数特征 [B, 3, L]
        time_inputs = self.compute_derivatives(x)
        
        # 2. 编码为 Tokens [B, L_reduced, Dim]
        time_tokens = self.time_encoder(time_inputs)
        
        # ===========================
        # Feature Fusion (特征融合)
        # ===========================
        # 将频域 Tokens 和 时域 Tokens 拼接
        combined_tokens = torch.cat([freq_tokens, time_tokens], dim=1)
        
        # ===========================
        # Classification (CoT)
        # ===========================
        if isinstance(self.head, LatentReasoningHead):
            # CoT Head 自动在所有 Token 中寻找关键特征
            logits = self.head(combined_tokens)
        else:
            # MLP 模式下做全局平均池化
            global_feat = combined_tokens.mean(dim=1)
            logits = self.head(global_feat)
        
        return logits

# ===================================================================
# 测试代码 (Sanity Check)
# ===================================================================
if __name__ == "__main__":
    # 模拟参数
    batch_size = 2
    seq_len = 512  # 假设 PPG 信号长度
    num_classes = 2
    embed_dim = 128 # 为了测试方便设小一点
    
    # 模拟输入
    x = torch.randn(batch_size, 1, seq_len)
    
    # 初始化模型 (不加载预训练权重)
    model = TF_MAE_Classifier(
        pretrained_path=None, 
        num_classes=num_classes,
        embed_dim=embed_dim,
        img_size=(32, 128), # 假设 CWT 输出尺寸
        use_cot=True,
        num_reasoning_tokens=4
    )
    
    print("Model initialized.")
    
    # 前向传播
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}") # 应该是 [2, 2]
    
    # 检查导数计算
    derivs = model.compute_derivatives(x.squeeze(1))
    print(f"Derivatives shape: {derivs.shape}") # 应该是 [2, 3, 512]