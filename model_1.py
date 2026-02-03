import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# ===================================================================
# 1. CWT 模块 (保持不变)
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
    # 支持 (B, M, L) 输入
    if x.dim() == 2:
        x = x.unsqueeze(1) # (B, 1, L)
    
    B, M, L = x.shape
    x_flat = x.view(B * M, L)
    
    x_pad = F.pad(x_flat, (1, 1), mode='replicate') 
    d1 = x_pad[:, 1:] - x_pad[:, :-1]
    d2 = d1[:, 1:] - d1[:, :-1]
    
    base = x_flat
    d1_cut = d1[:, :L]
    d2_cut = d2[:, :L]
    
    signals = torch.stack([base, d1_cut, d2_cut], dim=1) 
    BM, C, _ = signals.shape
    signals_flat = signals.view(BM * C, L)
    
    scales = torch.arange(num_scales, device=x.device) * step + lowest_scale
    cwt_out = cwt_ricker(signals_flat, scales)
    _, n_scales, _ = cwt_out.shape
    
    cwt_out = cwt_out.view(B, M, C, n_scales, L)
    return cwt_out

# ===================================================================
# 2. RoPE & 3. 基础组件 (保持不变)
# ===================================================================
# ... (RotaryEmbedding, apply_rotary_pos_emb, TensorizedLinear, 
#      DecomposedPatchEmbed, RoPEAttention, TensorizedBlock, Block 
#      代码与之前一致，此处省略) ...

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

class DecomposedPatchEmbed(nn.Module):
    def __init__(self, img_size=(64, 500), patch_size=(4, 50), in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
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
# 4. 主模型: Modality-Agnostic CWT-MAE
# ===================================================================

class CWT_MAE_RoPE(nn.Module):
    def __init__(
        self, 
        signal_len=500, 
        cwt_scales=64,
        patch_size_time=50,
        patch_size_freq=4,
        embed_dim=768, 
        depth=12, 
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mask_ratio=0.75,       
        mlp_rank_ratio=0.5,    
        norm_layer=nn.LayerNorm,
        time_loss_weight=1.0
    ):
        super().__init__()
        
        self.mask_ratio = mask_ratio
        self.cwt_scales = cwt_scales
        self.time_loss_weight = time_loss_weight
        self.patch_size_time = patch_size_time
        
        # 1. Patch Embed (共享权重，不区分通道)
        self.patch_embed = DecomposedPatchEmbed(
            img_size=(cwt_scales, signal_len),
            patch_size=(patch_size_freq, patch_size_time),
            in_chans=3,
            embed_dim=embed_dim,
            norm_layer=norm_layer
        )
        self.num_patches = self.patch_embed.num_patches
        self.grid_size = self.patch_embed.grid_size 

        # 2. Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Time Position Embedding (只编码时间，不编码通道)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # 【关键修改】移除了 channel_embed
        # 模型现在是 "Permutation Invariant" 的，它只看信号内容

        # RoPE Generators
        self.rope_encoder = RotaryEmbedding(dim=embed_dim // num_heads)
        self.rope_decoder = RotaryEmbedding(dim=decoder_embed_dim // decoder_num_heads)

        # 3. Encoder Blocks
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
        # 【关键修改】移除了 decoder_channel_embed
        
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, num_heads=decoder_num_heads, norm_layer=norm_layer) 
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        
        # 5. Heads
        self.patch_pixels = 3 * patch_size_freq * patch_size_time
        self.decoder_pred_spec = nn.Linear(decoder_embed_dim, self.patch_pixels, bias=True)

        self.time_reducer = nn.Sequential(
            nn.Conv2d(decoder_embed_dim, decoder_embed_dim, kernel_size=(self.grid_size[0], 1)),
            nn.GELU()
        )
        self.time_pred = nn.Linear(decoder_embed_dim, patch_size_time, bias=True)

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
        elif isinstance(m, nn.Conv2d):
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
        # x: (B, M, 3, Scales, Time)
        B, M, C, H, W = x.shape
        
        # 1. Patch Embedding (所有通道共享同一个 Embedder)
        x = x.view(B * M, C, H, W)
        x = self.patch_embed(x) # (B*M, N_patches, D)
        x = x.view(B, M, -1, x.shape[-1]) # (B, M, N_patches, D)
        N_patches = x.shape[2]

        # 2. Add Time Position Embeddings (只加时间，不加通道)
        # pos_embed: (1, N_patches+1, D)
        time_pos = self.pos_embed[:, 1:, :]
        x = x + time_pos.unsqueeze(1) # Broadcast to M

        # 3. Flatten (B, M*N_patches, D)
        # 这里我们将所有信号混合在一起，模型不知道哪个 Patch 来自哪个信号
        # 它只能通过 Patch 的内容（波形形状）来判断
        x = x.view(B, M * N_patches, -1)

        # 4. Masking
        if self.mask_ratio == 0.0:
            x_masked = x
            mask = torch.zeros(B, M * N_patches, device=x.device)
            ids_restore = torch.arange(M * N_patches, device=x.device).unsqueeze(0).expand(B, -1)
            ids_keep = ids_restore
        else:
            x_masked, mask, ids_restore, ids_keep = self.random_masking(x, self.mask_ratio)

        # 5. Append CLS Token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x_masked), dim=1)

        # 6. RoPE Position IDs
        # 关键：不同信号的同一时刻，拥有相同的 RoPE ID
        cls_pos = torch.zeros(B, 1, device=x.device, dtype=torch.long)
        patch_pos_indices = ids_keep % N_patches # 取模，获取时间索引
        patch_pos = patch_pos_indices + 1
        pos_ids = torch.cat((cls_pos, patch_pos), dim=1)

        # 7. Transformer
        rope_cos, rope_sin = self.rope_encoder(x, pos_ids)
        for blk in self.blocks:
            x = blk(x, rope_cos=rope_cos, rope_sin=rope_sin)
        x = self.norm(x)
        
        return x, mask, ids_restore, M

    def forward_decoder(self, x, ids_restore, M):
        x = self.decoder_embed(x)
        B, _, D_dec = x.shape
        N_patches = self.num_patches
        Total_Tokens = M * N_patches

        mask_tokens = self.mask_token.repeat(B, Total_Tokens + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D_dec))
        x = torch.cat([x[:, :1, :], x_], dim=1)

        # Add Decoder Time Pos (No Channel Pos)
        x_patches = x[:, 1:, :].view(B, M, N_patches, D_dec)
        x_patches = x_patches + self.decoder_pos_embed[:, 1:, :].unsqueeze(1)
        x_patches = x_patches.view(B, Total_Tokens, D_dec)
        x = torch.cat([x[:, :1, :], x_patches], dim=1)

        # RoPE
        patch_pos_indices = torch.arange(Total_Tokens, device=x.device) % N_patches
        patch_pos = patch_pos_indices + 1
        patch_pos = patch_pos.unsqueeze(0).expand(B, -1)
        cls_pos = torch.zeros(B, 1, device=x.device, dtype=torch.long)
        pos_ids = torch.cat((cls_pos, patch_pos), dim=1)
        
        rope_cos, rope_sin = self.rope_decoder(x, pos_ids)
        for blk in self.decoder_blocks:
            x = blk(x, rope_cos=rope_cos, rope_sin=rope_sin)
        x = self.decoder_norm(x)
        
        x = x[:, 1:, :].view(B, M, N_patches, D_dec)
        return x

    def forward_loss_spec(self, imgs, pred, mask):
        B, M, C, H, W = imgs.shape
        p_h, p_w = self.patch_embed.patch_size
        target = imgs.view(B, M, C, H // p_h, p_h, W // p_w, p_w)
        target = target.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
        target = target.view(B, M, -1, C * p_h * p_w)
        mask = mask.view(B, M, -1)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / (mask.sum() + 1e-6)
        return loss

    def forward_loss_time(self, x_raw, pred_time):
        x_raw = x_raw.float()
        mean = x_raw.mean(dim=-1, keepdim=True)
        std = x_raw.std(dim=-1, keepdim=True)
        std = torch.clamp(std, min=1e-5)
        target = (x_raw - mean) / std
        loss = F.mse_loss(pred_time.float(), target)
        return loss

    def forward(self, x):
        """
        x: (B, M, L) 或 (B, L)
        不需要 channel_ids。
        """
        if x.dim() == 2: x = x.unsqueeze(1)
        
        # 1. CWT & Norm
        imgs = cwt_wrap(x, num_scales=self.cwt_scales, lowest_scale=0.1, step=1.0)
        imgs_f32 = imgs.float() 
        mean = imgs_f32.mean(dim=(3, 4), keepdim=True)
        std = imgs_f32.std(dim=(3, 4), keepdim=True)
        std = torch.clamp(std, min=1e-5)
        imgs = (imgs_f32 - mean) / std
        
        # 数值鲁棒性增强
        imgs = torch.nan_to_num(imgs, nan=0.0, posinf=100.0, neginf=-100.0)
        imgs = torch.clamp(imgs, min=-100.0, max=100.0)

        # 确保输入数据类型与模型权重一致
        target_dtype = next(self.parameters()).dtype
        imgs = imgs.to(dtype=target_dtype)

        # 2. Encoder
        latent, mask, ids, M = self.forward_encoder(imgs)
        
        # 3. Decoder
        decoder_features = self.forward_decoder(latent, ids, M)
        
        # 4. Heads
        pred_spec = self.decoder_pred_spec(decoder_features)
        loss_spec = self.forward_loss_spec(imgs, pred_spec, mask)
        
        B, M, N, D = decoder_features.shape
        H_grid, W_grid = self.grid_size
        feat_2d = decoder_features.reshape(B * M, N, D).transpose(1, 2).reshape(B * M, D, H_grid, W_grid)
        feat_time_agg = self.time_reducer(feat_2d).squeeze(2).transpose(1, 2)
        pred_time = self.time_pred(feat_time_agg).flatten(1).view(B, M, -1)
        
        loss_time = self.forward_loss_time(x, pred_time)
        
        return loss_spec + self.time_loss_weight * loss_time, pred_spec, pred_time, imgs