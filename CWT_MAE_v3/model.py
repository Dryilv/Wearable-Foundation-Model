import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# ===================================================================
# 0. RevIN (Reversible Instance Normalization)
# ===================================================================
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _get_statistics(self, x):
        # x: [B, C, L] or [B, L] -> handle dimensions
        # Assuming input shape [B, C, L] or [B, 1, L]
        dim2reduce = (2,) # Reduce along time dimension L
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight.view(1, -1, 1) + self.affine_bias.view(1, -1, 1)
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias.view(1, -1, 1)) / (self.affine_weight.view(1, -1, 1) + 1e-10)
        x = x * self.stdev + self.mean
        return x

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
    # x: [B, 1, L]
    if x.dim() == 3 and x.shape[1] == 1:
        x = x.squeeze(1) # [B, L]
    
    # 简单的 CWT 包装，不再做多级分解，直接对原始信号做 CWT
    # 之前 v2 的逻辑是把 base, d1, d2 叠在一起，这里为了符合 Vertical Patching 和 效率，
    # 我们回归最纯粹的 CWT：只对原始信号变换。
    # 如果需要处理 d1, d2，可以在外部预处理或增加通道数。
    # 这里假设输入是单变量时间序列。
    
    scales = torch.arange(num_scales, device=x.device) * step + lowest_scale
    cwt_out = cwt_ricker(x, scales) # [B, Scales, L]
    
    # 增加 Channel 维度以适配 PatchEmbed (B, C, H, W) -> (B, 1, Scales, L)
    cwt_out = cwt_out.unsqueeze(1)
    return cwt_out

# ===================================================================
# 2. RoPE (Rotary Positional Embedding) 组件
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

# ===================================================================
# 3. 基础组件
# ===================================================================

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
    """
    Standard Conv2d Patch Embedding
    """
    def __init__(self, img_size=(64, 3000), patch_size=(64, 50), in_chans=1, embed_dim=384, norm_layer=None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # Vertical Patching: kernel_h = H (64), kernel_w = P_time (50)
        # Stride = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # x: (B, 1, 64, 3000)
        # proj -> (B, embed_dim, 1, 60) -> (B, embed_dim, 60)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2) # (B, N_patches, embed_dim)
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
# 4. 主模型: CWT-MAE v3 (Thorough Solution)
# ===================================================================

class CWT_MAE_RoPE(nn.Module):
    def __init__(
        self, 
        signal_len=3000, 
        cwt_scales=64,
        patch_size_time=50,
        # patch_size_freq is REMOVED/IGNORED, enforced to cwt_scales
        embed_dim=384,        # Downscaled
        depth=12, 
        num_heads=6,          # Downscaled
        decoder_embed_dim=256,
        decoder_depth=4,
        decoder_num_heads=8,  # Adjusted
        mask_ratio=0.8,       # Increased
        mlp_rank_ratio=0.5,    
        norm_layer=nn.LayerNorm,
        time_loss_weight=2.0, # Increased for dual objective
        **kwargs # consume extra args
    ):
        super().__init__()
        
        self.mask_ratio = mask_ratio
        self.cwt_scales = cwt_scales
        self.time_loss_weight = time_loss_weight
        self.patch_size_time = patch_size_time
        self.patch_size_freq = cwt_scales # Vertical Patching: covers full freq range
        
        # 0. RevIN
        self.revin = RevIN(num_features=1, affine=True) # Single channel input processing

        # 1. Patch Embed (Vertical Patching)
        self.patch_embed = DecomposedPatchEmbed(
            img_size=(cwt_scales, signal_len),
            patch_size=(self.patch_size_freq, patch_size_time),
            in_chans=1, # CWT output has 1 channel (if we ignore d1/d2)
            embed_dim=embed_dim,
            norm_layer=norm_layer
        )
        self.num_patches = self.patch_embed.num_patches
        self.grid_size = self.patch_embed.grid_size # (1, W)
        
        print(f"Model Config: Vertical Patching (1, {self.grid_size[1]}), Patches={self.num_patches}")

        # 2. RoPE Generator
        self.rope_encoder = RotaryEmbedding(dim=embed_dim // num_heads)
        self.rope_decoder = RotaryEmbedding(dim=decoder_embed_dim // decoder_num_heads)

        # 3. Encoder
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
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
        
        # 5. Heads (Dual Objective)
        # Head A: Reconstruct Spectrogram
        self.patch_pixels = 1 * self.patch_size_freq * patch_size_time # 1 channel * 64 * 50
        self.decoder_pred_spec = nn.Linear(decoder_embed_dim, self.patch_pixels, bias=True)
        
        # Head B: Reconstruct Raw Signal directly from latent patch
        # One patch corresponds to `patch_size_time` raw points
        self.decoder_pred_raw = nn.Linear(decoder_embed_dim, patch_size_time, bias=True)

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
        # 1. Patch Embedding & Position Embedding
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        
        # 2. Masking Logic
        if self.mask_ratio == 0.0:
            x_masked = x
            B, N, D = x.shape
            ids_keep = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
            ids_restore = ids_keep 
            mask = torch.zeros(B, N, device=x.device)
        else:
            x_masked, mask, ids_restore, ids_keep = self.random_masking(x, self.mask_ratio)
        
        # 3. Add CLS Token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x_masked), dim=1)
        
        # 4. RoPE IDs
        B = x.shape[0]
        cls_pos = torch.zeros(B, 1, device=x.device, dtype=torch.long)
        patch_pos = ids_keep + 1
        pos_ids = torch.cat((cls_pos, patch_pos), dim=1)
        
        # 5. RoPE
        rope_cos, rope_sin = self.rope_encoder(x, pos_ids)
        
        # 6. Transformer Blocks
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
        return x[:, 1:, :]

    def forward_loss_spec(self, imgs, pred, mask):
        # imgs: [B, 1, Scales, L]
        p_h, p_w = self.patch_embed.patch_size # (Scales, Patch_Time)
        B, C, H, W = imgs.shape
        
        # Vertical Patching: H == p_h, so H // p_h == 1
        target = imgs.view(B, C, H // p_h, p_h, W // p_w, p_w)
        target = target.permute(0, 2, 4, 1, 3, 5).contiguous()
        target = target.view(B, -1, C * p_h * p_w) # [B, N_patches, Pixels]
        
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / (mask.sum() + 1e-6)
        return loss

    def forward_loss_raw(self, x_raw, pred_raw, mask):
        """
        x_raw: [B, 1, L] (RevIN normalized)
        pred_raw: [B, N_patches, Patch_Time]
        """
        if x_raw.dim() == 2: x_raw = x_raw.unsqueeze(1)
        B, C, L = x_raw.shape
        patch_time = self.patch_size_time
        
        # Reshape raw signal to patches: [B, N_patches, Patch_Time]
        # Note: L must be divisible by patch_time
        target = x_raw.view(B, C, -1, patch_time).transpose(1, 2).squeeze(2) # [B, N, Patch_Time]
        
        loss = (pred_raw - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / (mask.sum() + 1e-6)
        return loss

    def forward(self, x):
        # x: [B, 1, L] or [B, L]
        if x.dim() == 2: x = x.unsqueeze(1)
        
        # 0. RevIN Normalization (Pre-processing)
        # Handle normalization statistics per-instance, reversible
        x = self.revin(x, 'norm') # [B, 1, L]
        
        # 1. CWT Transformation
        # No more manual normalization here, handled by RevIN
        imgs = cwt_wrap(x, num_scales=self.cwt_scales, lowest_scale=0.1, step=1.0)
        
        # 2. Encoder-Decoder
        latent, mask, ids = self.forward_encoder(imgs)
        decoder_features = self.forward_decoder(latent, ids)
        
        # 3. Dual Heads
        pred_spec = self.decoder_pred_spec(decoder_features)
        pred_raw = self.decoder_pred_raw(decoder_features)
        
        # 4. Dual Losses
        loss_spec = self.forward_loss_spec(imgs, pred_spec, mask)
        loss_raw = self.forward_loss_raw(x, pred_raw, mask) # Use RevIN-normalized x as target
        
        total_loss = loss_spec + self.time_loss_weight * loss_raw
        
        return total_loss, pred_spec, pred_raw, imgs
