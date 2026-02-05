import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# ===================================================================
# 1. CWT 模块 (用于 Loss 计算)
# ===================================================================
@torch.compiler.disable
def create_ricker_wavelets(points, scales):
    scales = scales.float()
    t = torch.arange(0, points, device=scales.device).float() - (points - 1.0) / 2
    t = t.view(1, 1, -1) 
    scales = scales.view(-1, 1, 1)
    pi_factor = math.pi ** 0.25
    A = 2 / (torch.sqrt(3 * scales) * pi_factor + 1e-6)
    wsq = scales ** 2
    xsq = t ** 2
    mod = (1 - xsq / wsq)
    gauss = torch.exp(-xsq / (2 * wsq))
    wavelets = A * mod * gauss
    return wavelets

@torch.compiler.disable
def cwt_ricker(x, scales):
    batch_size, sequence_length = x.shape
    x = x.unsqueeze(1)
    num_scales = scales.shape[0]
    largest_scale = scales[-1].item()
    wavelet_len = min(10 * largest_scale, sequence_length)
    if wavelet_len % 2 == 0: wavelet_len += 1 
    wavelet_len = int(wavelet_len)
    wavelets = create_ricker_wavelets(wavelet_len, scales)
    wavelets = wavelets.to(dtype=x.dtype)
    padding = wavelet_len // 2
    cwt_output = F.conv1d(x, wavelets, padding=padding)
    if cwt_output.shape[-1] > sequence_length:
        cwt_output = cwt_output[..., :sequence_length]
    return cwt_output

@torch.compiler.disable
def cwt_wrap(x, num_scales=64, lowest_scale=0.1, step=1.0):
    # x: (B, M, L) or (B*M, L)
    if x.dim() == 2:
        x = x.unsqueeze(1) # (B, 1, L)
    
    # Flatten M into B
    B_in, M_in, L = x.shape
    x_flat = x.view(B_in * M_in, L)
    
    # CWT Calculation
    scales = torch.arange(num_scales, device=x.device) * step + lowest_scale
    cwt_out = cwt_ricker(x_flat, scales) # (BM, Scales, L)
    
    # Reshape back
    n_scales = cwt_out.shape[1]
    cwt_out = cwt_out.view(B_in, M_in, n_scales, L)
    return cwt_out

# ===================================================================
# 2. RoPE & 基础组件
# ===================================================================

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=6000):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self._update_cache(max_seq_len)

    def _update_cache(self, seq_len):
        self.max_seq_len = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype=torch.bfloat16), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype=torch.bfloat16), persistent=False)

    def forward(self, x, pos_ids):
        seq_len = torch.max(pos_ids) + 1
        if seq_len > self.max_seq_len:
            self._update_cache(int(seq_len * 1.5))
        cos = F.embedding(pos_ids, self.cos_cached).to(x.dtype)
        sin = F.embedding(pos_ids, self.sin_cached).to(x.dtype)
        return cos.unsqueeze(2), sin.unsqueeze(2)

def apply_rotary_pos_emb(q, k, cos, sin):
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class TensorizedLinear(nn.Module):
    def __init__(self, in_features, out_features, rank_ratio=0.5, bias=True):
        super().__init__()
        self.rank = int(min(in_features, out_features) * rank_ratio)
        self.rank = max(32, self.rank)
        self.v = nn.Linear(in_features, self.rank, bias=False)
        self.u = nn.Linear(self.rank, out_features, bias=bias)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.v.weight)
        nn.init.xavier_uniform_(self.u.weight)
        if self.u.bias is not None: nn.init.constant_(self.u.bias, 0)

    def forward(self, x):
        return self.u(self.v(x))

class PointPatchEmbed(nn.Module):
    """
    1D Signal Embedding:
    Maps 1D signal chunks (Patch Size) to Embedding Dimension.
    If patch_size=1, this is a Point-wise Linear Projection.
    """
    def __init__(self, signal_len=3000, patch_size=4, in_chans=1, embed_dim=768, norm_layer=None):
        super().__init__()
        self.signal_len = signal_len
        self.patch_size = patch_size
        self.num_patches = signal_len // patch_size
        
        # 使用 Conv1d 实现 Patch Embedding
        # kernel_size=patch_size, stride=patch_size 实现非重叠切片
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # x: (B, 1, L)
        x = self.proj(x) # (B, D, N)
        x = x.transpose(1, 2) # (B, N, D)
        x = self.norm(x)
        return x

class RoPEAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope_cos=None, rope_sin=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if rope_cos is not None and rope_sin is not None:
            q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TensorizedBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., 
                 rank_ratio=0.5, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = RoPEAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            TensorizedLinear(dim, hidden_dim, rank_ratio=rank_ratio), 
            nn.GELU(),
            nn.Dropout(drop),
            TensorizedLinear(hidden_dim, dim, rank_ratio=rank_ratio),
            nn.Dropout(drop)
        )

    def forward(self, x, rope_cos=None, rope_sin=None):
        x = x + self.attn(self.norm1(x), rope_cos, rope_sin)
        x = x + self.mlp(self.norm2(x))
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = RoPEAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop)
        )

    def forward(self, x, rope_cos=None, rope_sin=None):
        x = x + self.attn(self.norm1(x), rope_cos, rope_sin)
        x = x + self.mlp(self.norm2(x))
        return x

# ===================================================================
# 4. 主模型: Pixel/Point-based CWT-MAE (v3)
# ===================================================================

class CWT_MAE_RoPE(nn.Module):
    def __init__(
        self, 
        signal_len=3000, 
        patch_size=4,          # v3 关键参数: 如果设为1则为纯Pixel级
        embed_dim=768, 
        depth=12, 
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mask_ratio=0.75,       
        mlp_rank_ratio=0.5,    
        norm_layer=nn.LayerNorm,
        cwt_scales=64,         # 仅用于 CWT Loss
        cwt_loss_weight=1.0    # CWT Loss 权重
    ):
        super().__init__()
        
        self.mask_ratio = mask_ratio
        self.signal_len = signal_len
        self.patch_size = patch_size
        self.cwt_scales = cwt_scales
        self.cwt_loss_weight = cwt_loss_weight
        
        # 1. 1D Point/Patch Embed
        self.patch_embed = PointPatchEmbed(
            signal_len=signal_len,
            patch_size=patch_size,
            in_chans=1, # 处理单通道 (多通道在Batch维)
            embed_dim=embed_dim,
            norm_layer=norm_layer
        )
        self.num_patches = self.patch_embed.num_patches

        # 2. Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # RoPE
        self.rope_encoder = RotaryEmbedding(dim=embed_dim // num_heads)
        self.rope_decoder = RotaryEmbedding(dim=decoder_embed_dim // decoder_num_heads)

        # 3. Encoder
        self.blocks = nn.ModuleList([
            TensorizedBlock(
                embed_dim, num_heads, 
                rank_ratio=mlp_rank_ratio, 
                norm_layer=norm_layer
            ) 
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # 4. Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim))
        
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, num_heads=decoder_num_heads, norm_layer=norm_layer) 
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        
        # 5. Prediction Head
        # 预测每个 Patch 内的 Pixel 值 (长度为 patch_size)
        self.pred_head = nn.Linear(decoder_embed_dim, patch_size, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.pos_embed, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform_(m.weight)

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore, ids_keep

    def forward_encoder(self, x):
        # x: (B*M, 1, L) -> (B*M, N, D)
        x = self.patch_embed(x)
        
        # Add Pos Embed
        x = x + self.pos_embed[:, 1:, :]
        
        # Masking
        if self.mask_ratio == 0.0:
            x_masked = x
            mask = torch.zeros(x.shape[0], x.shape[1], device=x.device)
            ids_restore = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], -1)
            ids_keep = ids_restore
        else:
            x_masked, mask, ids_restore, ids_keep = self.random_masking(x, self.mask_ratio)

        # Append CLS Token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x_masked), dim=1)
        
        # RoPE IDs
        B, N_curr, _ = x.shape
        N_patches = self.num_patches
        cls_pos = torch.zeros(B, 1, device=x.device, dtype=torch.long)
        patch_pos = ids_keep + 1
        pos_ids = torch.cat((cls_pos, patch_pos), dim=1)
        
        # Transformer
        rope_cos, rope_sin = self.rope_encoder(x, pos_ids)
        for blk in self.blocks:
            x = blk(x, rope_cos=rope_cos, rope_sin=rope_sin)
        x = self.norm(x)
        
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], self.num_patches + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.decoder_pos_embed
        
        B, N, _ = x.shape
        pos_ids = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        rope_cos, rope_sin = self.rope_decoder(x, pos_ids)
        
        for blk in self.decoder_blocks:
            x = blk(x, rope_cos=rope_cos, rope_sin=rope_sin)
        x = self.decoder_norm(x)
        return x[:, 1:, :] # Drop CLS

    def forward_loss_raw(self, x_raw, pred_raw, mask):
        # x_raw: (B, 1, L)
        # pred_raw: (B, L)
        # mask: (B, N_patches) -> expand to pixels
        
        B, _, L = x_raw.shape
        P = self.patch_size
        
        # 重塑 target 以匹配 patch
        target = x_raw.view(B, -1, P) # (B, N, P)
        pred = pred_raw.view(B, -1, P) # (B, N, P)
        
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1) # (B, N)
        loss = (loss * mask).sum() / (mask.sum() + 1e-6)
        return loss

    def forward_loss_cwt(self, x_raw, pred_raw):
        # 计算频域 Loss
        # x_raw, pred_raw: (B, 1, L)
        # CWT 是不可学习的，只是一个变换
        
        # 归一化输入以确保 Loss 尺度一致
        x_cwt = cwt_wrap(x_raw, num_scales=self.cwt_scales)
        pred_cwt = cwt_wrap(pred_raw.unsqueeze(1), num_scales=self.cwt_scales)
        
        # 简单的 MSE Loss (Magnitude)
        # 可以只看幅度
        loss = F.mse_loss(pred_cwt, x_cwt)
        return loss

    def forward(self, x):
        """
        x: (B, M, L) or (B, L)
        """
        if x.dim() == 2: x = x.unsqueeze(1)
        B, M, L = x.shape
        
        # 1. Flatten Channels (Treat each channel as independent sample)
        x_flat = x.view(B * M, 1, L)
        
        # 2. Normalize (Instance Norm per channel)
        mean = x_flat.mean(dim=2, keepdim=True)
        std = x_flat.std(dim=2, keepdim=True) + 1e-5
        x_norm = (x_flat - mean) / std
        
        # 3. Encoder & Decoder
        latent, mask, ids = self.forward_encoder(x_norm)
        decoder_features = self.forward_decoder(latent, ids)
        
        # 4. Predict
        # decoder_features: (BM, N, D)
        # pred_head: (D) -> (P)
        pred_patches = self.pred_head(decoder_features) # (BM, N, P)
        pred_raw = pred_patches.view(B * M, L) # Reconstruct 1D signal
        
        # 5. Loss
        # 时域 Loss (只计算 Mask 部分)
        loss_raw = self.forward_loss_raw(x_norm, pred_raw, mask)
        
        # 频域 Loss (计算整体，或者也可以只计算 Mask 部分，这里计算整体简单点)
        # 注意：这里我们强制模型生成的信号不仅时域像，频域也要像
        loss_cwt = self.forward_loss_cwt(x_norm, pred_raw)
        
        total_loss = loss_raw + self.cwt_loss_weight * loss_cwt
        
        # 还原形状用于返回
        pred_signal = pred_raw.view(B, M, L)
        x_norm_signal = x_norm.view(B, M, L)
        
        return total_loss, pred_signal, None, x_norm_signal
