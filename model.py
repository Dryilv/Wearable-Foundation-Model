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
    x_pad = F.pad(x, (1, 1), mode='replicate') 
    d1 = x_pad[:, 1:] - x_pad[:, :-1]
    d2 = d1[:, 1:] - d1[:, :-1]
    L = x.shape[1]
    base = x
    d1_cut = d1[:, :L]
    d2_cut = d2[:, :L]
    signals = torch.stack([base, d1_cut, d2_cut], dim=1)
    B, C, _ = signals.shape
    signals_flat = signals.view(B * C, L)
    scales = torch.arange(num_scales, device=x.device) * step + lowest_scale
    cwt_out = cwt_ricker(signals_flat, scales)
    _, n_scales, _ = cwt_out.shape
    cwt_out = cwt_out.view(B, C, n_scales, L)
    return cwt_out

# ===================================================================
# 2. RoPE (Rotary Positional Embedding) 组件 【新增】
# ===================================================================

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=6000):
        super().__init__()
        self.dim = dim
        # 预计算 theta
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        # 缓存 cos/sin
        self._update_cache(max_seq_len)

    def _update_cache(self, seq_len):
        self.max_seq_len = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        # 拼接成 (seq_len, dim)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype=torch.bfloat16), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype=torch.bfloat16), persistent=False)

    def forward(self, x, pos_ids):
        # x: [B, N, H, D/H]
        # pos_ids: [B, N]
        seq_len = torch.max(pos_ids) + 1
        if seq_len > self.max_seq_len:
            self._update_cache(int(seq_len * 1.5))
            
        # 根据 pos_ids 选取对应的 cos/sin
        # cos: [B, N, Dim]
        cos = F.embedding(pos_ids, self.cos_cached).to(x.dtype)
        sin = F.embedding(pos_ids, self.sin_cached).to(x.dtype)
        
        return cos.unsqueeze(2), sin.unsqueeze(2)

def apply_rotary_pos_emb(q, k, cos, sin):
    """
    应用 RoPE 旋转
    q, k: [B, N, H, D_head]
    cos, sin: [B, N, 1, D_head] (广播到 H)
    """
    # 将 q, k 切分为两半进行旋转
    # x = [x1, x2] -> [-x2, x1]
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# ===================================================================
# 3. 基础组件 (支持 RoPE 的 Attention 和 Block)
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
    def __init__(self, img_size=(64, 3000), patch_size=(4, 50), in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj_channel = nn.Conv2d(in_chans, embed_dim, kernel_size=1, stride=1)
        self.proj_freq = nn.Conv2d(embed_dim, embed_dim, kernel_size=(patch_size[0], 1), stride=(patch_size[0], 1), groups=embed_dim) 
        self.proj_time = nn.Conv2d(embed_dim, embed_dim, kernel_size=(1, patch_size[1]), stride=(1, patch_size[1]), groups=embed_dim) 
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj_channel(x)
        x = self.proj_freq(x)
        x = self.proj_time(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class RoPEAttention(nn.Module):
    """
    替代 nn.MultiheadAttention，支持 RoPE
    """
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
        # qkv: [B, N, 3*C] -> [B, N, 3, H, C/H] -> [3, B, N, H, C/H]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 应用 RoPE
        if rope_cos is not None and rope_sin is not None:
            q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin)

        # Flash Attention (PyTorch 2.0+)
        # q, k, v shape: [B, N, H, D] -> transpose to [B, H, N, D] for F.scaled_dot_product_attention
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
        # 使用自定义 RoPE Attention
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
    """Decoder Block (Standard Linear MLP)"""
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
# 4. 主模型: CWT-MAE with RoPE
# ===================================================================

class CWT_MAE_RoPE(nn.Module):
    def __init__(
        self, 
        signal_len=3000, 
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
        
        # 1. Patch Embed
        self.patch_embed = DecomposedPatchEmbed(
            img_size=(cwt_scales, signal_len),
            patch_size=(patch_size_freq, patch_size_time),
            in_chans=3,
            embed_dim=embed_dim,
            norm_layer=norm_layer
        )
        self.num_patches = self.patch_embed.num_patches
        self.grid_size = self.patch_embed.grid_size 

        # 2. RoPE Generator (Encoder & Decoder)
        # head_dim = embed_dim // num_heads
        self.rope_encoder = RotaryEmbedding(dim=embed_dim // num_heads)
        self.rope_decoder = RotaryEmbedding(dim=decoder_embed_dim // decoder_num_heads)

        # 3. Encoder
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 依然保留 Absolute Pos Embed，作为 Global Anchor，RoPE 作为 Relative 增强
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
        # x: [B, 3, H, W] -> [B, N, D]
        x = self.patch_embed(x)
        
        # 添加绝对位置编码 (Global Anchor)
        x = x + self.pos_embed[:, 1:, :]
        
        # Masking
        x_masked, mask, ids_restore, ids_keep = self.random_masking(x, self.mask_ratio)
        
        # 拼接 CLS Token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x_masked), dim=1)
        
        # --- 准备 RoPE 的 Position IDs ---
        # CLS token 位置设为 0
        # Patch tokens 位置设为 ids_keep + 1 (因为 0 被 CLS 占了)
        B = x.shape[0]
        cls_pos = torch.zeros(B, 1, device=x.device, dtype=torch.long)
        patch_pos = ids_keep + 1
        pos_ids = torch.cat((cls_pos, patch_pos), dim=1) # [B, N_keep + 1]
        
        # 生成 RoPE 旋转矩阵
        rope_cos, rope_sin = self.rope_encoder(x, pos_ids)
        
        for blk in self.blocks:
            # 传入 RoPE 参数
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
        
        # --- 准备 Decoder RoPE ---
        # Decoder 看到的是完整序列，位置是连续的 0, 1, 2, ... N
        B, N, _ = x.shape
        pos_ids = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        
        rope_cos, rope_sin = self.rope_decoder(x, pos_ids)
        
        for blk in self.decoder_blocks:
            x = blk(x, rope_cos=rope_cos, rope_sin=rope_sin)
            
        x = self.decoder_norm(x)
        return x[:, 1:, :]

    def forward_loss_spec(self, imgs, pred, mask):
        p_h, p_w = self.patch_embed.patch_size
        B, C, H, W = imgs.shape
        target = imgs.view(B, C, H // p_h, p_h, W // p_w, p_w)
        target = target.permute(0, 2, 4, 1, 3, 5).contiguous()
        target = target.view(B, -1, C * p_h * p_w)
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
        if x.dim() == 3: x = x.squeeze(1)
        
        imgs = cwt_wrap(x, num_scales=self.cwt_scales, lowest_scale=0.1, step=1.0)
        
        dtype_orig = imgs.dtype
        imgs_f32 = imgs.float() 
        mean = imgs_f32.mean(dim=(2, 3), keepdim=True)
        std = imgs_f32.std(dim=(2, 3), keepdim=True)
        std = torch.clamp(std, min=1e-5)
        imgs = (imgs_f32 - mean) / std
        imgs = imgs.to(dtype=dtype_orig)

        latent, mask, ids = self.forward_encoder(imgs)
        decoder_features = self.forward_decoder(latent, ids)
        
        pred_spec = self.decoder_pred_spec(decoder_features)
        loss_spec = self.forward_loss_spec(imgs, pred_spec, mask)
        
        B, N, D = decoder_features.shape
        H_grid, W_grid = self.grid_size
        feat_2d = decoder_features.transpose(1, 2).view(B, D, H_grid, W_grid)
        feat_time_agg = self.time_reducer(feat_2d)
        feat_time_agg = feat_time_agg.squeeze(2).transpose(1, 2)
        pred_time = self.time_pred(feat_time_agg).flatten(1)
        
        loss_time = self.forward_loss_time(x, pred_time)
        
        total_loss = loss_spec + self.time_loss_weight * loss_time
        
        return total_loss, pred_spec, pred_time, imgs

# ===================================================================
# 测试代码
# ===================================================================
if __name__ == "__main__":
    # 模拟输入: Batch=2, Length=3000
    x = torch.randn(2, 3000)
    
    model = CWT_MAE_RoPE(
        signal_len=3000,
        mask_ratio=0.75,     
        mlp_rank_ratio=0.5,  
        embed_dim=768
    )
    
    loss, pred_spec, pred_time, imgs = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Loss: {loss.item()}")
    print(f"Spec Pred shape: {pred_spec.shape}")
    print(f"Time Pred shape: {pred_time.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e6:.2f} M")