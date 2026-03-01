import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ===================================================================
# 1. CWT 模块 (保持不变，优秀的特征工程)
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
def cwt_wrap(x, num_scales=64, lowest_scale=0.1, step=1.0, use_diff=True):
    if x.dim() == 2:
        x = x.unsqueeze(1)
    B, M, L = x.shape
    x_flat = x.view(B * M, L)
    
    if use_diff:
        x_pad = F.pad(x_flat, (1, 1), mode='replicate') 
        d1 = x_pad[:, 1:] - x_pad[:, :-1]
        d2 = d1[:, 1:] - d1[:, :-1]
        
        base = x_flat
        d1_cut = d1[:, :L]
        d2_cut = d2[:, :L]
        
        signals = torch.stack([base, d1_cut, d2_cut], dim=1) 
    else:
        base = x_flat
        signals = base.unsqueeze(1) # (BM, 1, L)
        
    BM, C, _ = signals.shape
    signals_flat = signals.view(BM * C, L)
    
    scales = torch.arange(num_scales, device=x.device) * step + lowest_scale
    cwt_out = cwt_ricker(signals_flat, scales)
    _, n_scales, _ = cwt_out.shape
    
    cwt_out = cwt_out.view(B, M, C, n_scales, L)
    return cwt_out

# ===================================================================
# 2. 基础组件 & RoPE
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
        self.register_buffer("cos_cached", emb.cos().to(torch.float32), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(torch.float32), persistent=False)

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

class DecomposedPatchEmbed(nn.Module):
    """处理 2D CWT 图像的 Patch Embedding"""
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

class RawSignalPatchEmbed(nn.Module):
    """
    【新增】处理 1D 原始信号的 Patch Embedding
    将原始信号切分为与 CWT 时间步完全对齐的 Patch
    """
    def __init__(self, patch_size_time=50, embed_dim=768, norm_layer=None):
        super().__init__()
        # 使用 1D 卷积进行无重叠切分，步长与 CWT 的时间维度 patch_size 保持一致
        self.proj = nn.Conv1d(1, embed_dim, kernel_size=patch_size_time, stride=patch_size_time)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # Input: (B*M, 1, L)
        x = self.proj(x)              # -> (B*M, D, W_grid)
        x = x.transpose(1, 2)         # -> (B*M, W_grid, D)
        x = self.norm(x)
        return x

class RoPEAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope_cos=None, rope_sin=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        if rope_cos is not None and rope_sin is not None:
             q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin)
        
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# ===================================================================
# 3. 时空因子化 Block
# ===================================================================
class TrueFactorizedBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm_time = norm_layer(dim)
        self.time_attn = RoPEAttention(dim, num_heads=num_heads, proj_drop=drop)
        
        self.norm_channel = norm_layer(dim)
        self.temporal_smooth = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.channel_attn = RoPEAttention(dim, num_heads=num_heads, proj_drop=drop)
        
        self.norm_mlp = norm_layer(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop)
        )

    def forward(self, x, M, N, rope_cos=None, rope_sin=None):
        B, MN, D = x.shape
        
        # --- 1. Time Attention ---
        x_time = x.view(B * M, N, D)
        
        if rope_cos is not None and rope_sin is not None:
            cos_t = rope_cos if rope_cos.shape[0] == B * M else rope_cos.repeat_interleave(M, dim=0)
            sin_t = rope_sin if rope_sin.shape[0] == B * M else rope_sin.repeat_interleave(M, dim=0)
        else:
            cos_t, sin_t = None, None
            
        x_time = x_time + self.time_attn(self.norm_time(x_time), cos_t, sin_t)
        
        # --- 2. Cross-Channel Attention ---
        # 在单塔对比学习模式下，M=1，此部分会自动跳过，符合预期
        if M > 1:
            x_c = x_time.view(B, M, N, D)
            x_smooth = x_c.view(B * M, N, D).transpose(1, 2) 
            x_smooth = self.temporal_smooth(x_smooth).transpose(1, 2).view(B, M, N, D)
            x_channel = x_smooth.transpose(1, 2).reshape(B * N, M, D)
            attn_out = self.channel_attn(self.norm_channel(x_channel))
            x_c = x_c + attn_out.view(B, N, M, D).transpose(1, 2)
            x = x_c.reshape(B, MN, D)
        else:
            x = x_time.view(B, MN, D)
            
        # --- 3. MLP ---
        x = x + self.mlp(self.norm_mlp(x))
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = RoPEAttention(dim, num_heads=num_heads, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(drop)
        )
    def forward(self, x, rope_cos=None, rope_sin=None):
        x = x + self.attn(self.norm1(x), rope_cos, rope_sin)
        x = x + self.mlp(self.norm2(x))
        return x

# ===================================================================
# 4. 核心骨干网络: CWT-MAE (带残差融合)
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
        norm_layer=nn.LayerNorm,
        time_loss_weight=1.0,
        use_diff=False,
        diff_loss_weight=None,
        max_modalities=16 # 单塔模式下 M=1
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.cwt_scales = cwt_scales
        self.time_loss_weight = time_loss_weight
        self.patch_size_time = patch_size_time
        self.use_diff = use_diff
        
        if diff_loss_weight is None:
            self.diff_loss_weight =[1.0, 1.0, 1.0] if use_diff else [1.0]
        else:
            self.diff_loss_weight = diff_loss_weight
            
        self.register_buffer('channel_loss_weights', torch.tensor(self.diff_loss_weight).view(1, 1, 1, -1))
        
        in_chans = 3 if use_diff else 1
        
        # 2D CWT Patch Embedder
        self.patch_embed = DecomposedPatchEmbed(
            img_size=(cwt_scales, signal_len),
            patch_size=(patch_size_freq, patch_size_time),
            embed_dim=embed_dim, norm_layer=norm_layer,
            in_chans=in_chans
        )
        self.num_patches = self.patch_embed.num_patches
        self.grid_size = self.patch_embed.grid_size 

        # 【新增】1D Raw Signal Patch Embedder
        self.raw_patch_embed = RawSignalPatchEmbed(
            patch_size_time=patch_size_time,
            embed_dim=embed_dim,
            norm_layer=norm_layer
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        self.rope_encoder = RotaryEmbedding(dim=embed_dim // num_heads)
        self.rope_decoder = RotaryEmbedding(dim=decoder_embed_dim // decoder_num_heads)

        self.blocks = nn.ModuleList([
            TrueFactorizedBlock(embed_dim, num_heads, norm_layer=norm_layer) 
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_embed_dim))
        
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, num_heads=decoder_num_heads, norm_layer=norm_layer) 
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        
        self.patch_pixels = in_chans * patch_size_freq * patch_size_time
        self.decoder_pred_spec = nn.Linear(decoder_embed_dim, self.patch_pixels, bias=True)

        self.time_reducer = nn.Sequential(
            nn.Conv2d(decoder_embed_dim, decoder_embed_dim, kernel_size=(self.grid_size[0], 1)),
            nn.GELU(),
            norm_layer(decoder_embed_dim)
        )
        self.time_pred = nn.Linear(decoder_embed_dim, patch_size_time, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.trunc_normal_(self.pos_embed, std=.02)
        torch.nn.init.trunc_normal_(self.decoder_pos_embed, std=.02)
        torch.nn.init.trunc_normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def tubelet_masking(self, x, mask_ratio, M, N_patches):
        B, _, D = x.shape
        H_grid, W_grid = self.grid_size
        len_keep_w = int(W_grid * (1 - mask_ratio))
        len_keep = len_keep_w * H_grid
        
        noise_w = torch.rand(B, W_grid, device=x.device)
        ids_shuffle_w = torch.argsort(noise_w, dim=1)
        ids_restore_w = torch.argsort(ids_shuffle_w, dim=1)
        
        h_idx = torch.arange(H_grid, device=x.device).view(1, H_grid, 1)
        ids_restore_w_exp = ids_restore_w.unsqueeze(1)
        noise = (ids_restore_w_exp * H_grid + h_idx).reshape(B, N_patches)
        
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1) 
        ids_keep = ids_shuffle[:, :len_keep]            
        
        x_reshaped = x.view(B, M, N_patches, D)
        ids_keep_expanded = ids_keep.unsqueeze(1).unsqueeze(-1).expand(B, M, len_keep, D)
        x_masked = torch.gather(x_reshaped, dim=2, index=ids_keep_expanded)
        x_masked = x_masked.reshape(B, M * len_keep, D)
        
        mask = torch.ones([B, N_patches], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        mask = mask.unsqueeze(1).expand(B, M, N_patches).reshape(B, M * N_patches)
        
        global_ids_restore =[]
        for m in range(M):
            global_ids_restore.append(ids_restore + m * N_patches)
        global_ids_restore = torch.cat(global_ids_restore, dim=1) 
        
        return x_masked, mask, global_ids_restore, ids_keep

    def mixed_masking(self, x, mask_ratio, M, N_patches):
        return self.tubelet_masking(x, mask_ratio, M, N_patches) + (M,)

    # 【核心修改】forward_encoder 增加残差融合逻辑
    def forward_encoder(self, x_raw, imgs, mask_ratio=None):
        B, M, C, H, W = imgs.shape
        
        # 1. CWT 特征提取
        x_cwt = imgs.view(B * M, C, H, W)
        x_cwt = self.patch_embed(x_cwt) # (B*M, H*W, D)
        
        # 2. 原始信号处理
        if x_raw.dim() == 2:
            x_raw = x_raw.unsqueeze(1)  
        elif x_raw.dim() == 3 and x_raw.shape[1] != 1 and x_raw.shape[0] == B:
            x_raw = x_raw.view(B * M, 1, -1)
        elif x_raw.dim() == 3 and x_raw.shape[1] == M:
            x_raw = x_raw.view(B * M, 1, -1)
        else:
            x_raw = x_raw.reshape(B * M, 1, -1)
            
        # 实例归一化，防止原始信号数值过大
        mean_raw = x_raw.mean(dim=-1, keepdim=True)
        std_raw = torch.clamp(x_raw.std(dim=-1, keepdim=True), min=1e-5)
        x_raw_norm = (x_raw - mean_raw) / std_raw
        
        # 1D 特征提取
        x_raw_embed = self.raw_patch_embed(x_raw_norm.to(dtype=next(self.parameters()).dtype)) # (B*M, W_grid, D)
        
        # 3. 广播式残差融合 (Broadcasted Residual Connection)
        H_grid, W_grid = self.grid_size
        D = x_cwt.shape[-1]
        
        x_cwt_2d = x_cwt.view(B * M, H_grid, W_grid, D)
        x_raw_2d = x_raw_embed.unsqueeze(1) # (B*M, 1, W_grid, D)
        
        # 广播相加：1D 特征复制到所有频率带
        x_fused = x_cwt_2d + x_raw_2d 
        x = x_fused.view(B, M, -1, D) 
        N_patches = x.shape[2]

        # 4. 位置编码
        x = x + self.pos_embed.unsqueeze(1)
        x = x.view(B, M * N_patches, -1)

        # 5. Masking
        current_mask_ratio = mask_ratio if mask_ratio is not None else self.mask_ratio
        if current_mask_ratio == 0.0:
            x_masked = x
            mask = torch.zeros(B, M * N_patches, device=x.device)
            global_ids_restore = torch.arange(M * N_patches, device=x.device).unsqueeze(0).expand(B, -1)
            ids_keep = torch.arange(N_patches, device=x.device).unsqueeze(0).expand(B, -1)
            M_enc = M
        else:
            x_masked, mask, global_ids_restore, ids_keep, M_enc = self.mixed_masking(x, current_mask_ratio, M, N_patches)

        # 6. RoPE
        is_async = (ids_keep.dim() == 3)
        if is_async:
            pos_ids_flat = (ids_keep % W_grid).view(B * M_enc, -1) 
        else:
            pos_ids_flat = (ids_keep % W_grid) 
            
        rope_cos, rope_sin = self.rope_encoder(x_masked, pos_ids_flat)
        
        # 7. Transformer Blocks
        len_keep = ids_keep.shape[-1]
        for blk in self.blocks:
            x_masked = blk(x_masked, M_enc, len_keep, rope_cos=rope_cos, rope_sin=rope_sin)
        x_masked = self.norm(x_masked)
        
        return x_masked, mask, global_ids_restore, M

    def forward_decoder(self, x, ids_restore, M):
        x = self.decoder_embed(x)
        B, _, D_dec = x.shape
        N_patches = self.num_patches
        Total_Tokens = M * N_patches

        mask_tokens = self.mask_token.repeat(B, Total_Tokens - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D_dec))

        x_patches = x.view(B, M, N_patches, D_dec)
        x_patches = x_patches + self.decoder_pos_embed.unsqueeze(1)
        x = x_patches.view(B, Total_Tokens, D_dec)

        H_grid, W_grid = self.grid_size
        patch_pos = (torch.arange(N_patches, device=x.device) % W_grid).unsqueeze(0).expand(B, -1)
        patch_pos_expanded = patch_pos.repeat(1, M)
        rope_cos, rope_sin = self.rope_decoder(x, patch_pos_expanded)
        
        for blk in self.decoder_blocks:
            x = blk(x, rope_cos=rope_cos, rope_sin=rope_sin)
        x = self.decoder_norm(x)
        
        x = x.view(B, M, N_patches, D_dec)
        return x

    def forward_loss_spec(self, imgs, pred, mask):
        B, M, C, H, W = imgs.shape
        p_h, p_w = self.patch_embed.patch_size
        
        target = imgs.view(B, M, C, H // p_h, p_h, W // p_w, p_w)
        target = target.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
        target = target.view(B, M, -1, C * p_h * p_w)
        
        mask = mask.view(B, M, -1)
        loss = (pred - target) ** 2
        
        loss = loss.view(B, M, -1, C, p_h * p_w)
        loss = loss.mean(dim=-1) 
        loss = loss * self.channel_loss_weights
        loss = loss.sum(dim=-1) 
        
        mask_sum = mask.sum()
        if mask_sum > 0:
            loss = (loss * mask).sum() / mask_sum
        else:
            loss = loss.mean() * 0.0
        return loss

    def forward_loss_time(self, x_raw, pred_time, mask):
        x_raw = x_raw.float()
        mean = x_raw.mean(dim=-1, keepdim=True)
        std = torch.clamp(x_raw.std(dim=-1, keepdim=True), min=1e-5)
        target = torch.clamp((x_raw - mean) / std, min=-10.0, max=10.0)
        
        B, M, _ = pred_time.shape
        H_grid, W_grid = self.grid_size
        
        mask_2d = mask.view(B, M, H_grid, W_grid)
        mask_t = torch.round(mask_2d.mean(dim=2)) 
        
        mask_time = mask_t.unsqueeze(-1).expand(-1, -1, -1, self.patch_size_time)
        mask_time = mask_time.reshape(B, M, -1) 
        
        loss = (pred_time.float() - target) ** 2
        mask_time_sum = mask_time.sum()
        if mask_time_sum > 0:
            loss = (loss * mask_time).sum() / mask_time_sum
        else:
            loss = loss.mean() * 0.0
        return loss

    def prepare_tokens(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        imgs = cwt_wrap(x, num_scales=self.cwt_scales, lowest_scale=0.1, step=1.0, use_diff=self.use_diff)
        imgs_f32 = imgs.float() 
        mean = imgs_f32.mean(dim=(3, 4), keepdim=True)
        std = torch.clamp(imgs_f32.std(dim=(3, 4), keepdim=True), min=1e-5)
        imgs = (imgs_f32 - mean) / std
        imgs = torch.nan_to_num(imgs, nan=0.0, posinf=100.0, neginf=-100.0)
        imgs = torch.clamp(imgs, min=-100.0, max=100.0)
        return imgs.to(dtype=next(self.parameters()).dtype)

    def forward(self, x, mask_ratio=None):
        imgs = self.prepare_tokens(x)
        # 传入 x (原始信号) 和 imgs (CWT)
        latent, mask, ids, M = self.forward_encoder(x, imgs, mask_ratio=mask_ratio)
        decoder_features = self.forward_decoder(latent, ids, M)
        
        pred_spec = self.decoder_pred_spec(decoder_features)
        loss_spec = self.forward_loss_spec(imgs, pred_spec, mask)
        
        B, M_dec, N, D = decoder_features.shape
        H_grid, W_grid = self.grid_size
        feat_2d = decoder_features.reshape(B * M_dec, N, D).transpose(1, 2).reshape(B * M_dec, D, H_grid, W_grid)
        
        feat_time_agg = self.time_reducer[0](feat_2d) 
        feat_time_agg = self.time_reducer[1](feat_time_agg) 
        
        feat_time_agg = feat_time_agg.squeeze(2).transpose(1, 2)
        feat_time_agg = self.time_reducer[2](feat_time_agg) 
        
        pred_time = self.time_pred(feat_time_agg).flatten(1).view(B, M_dec, -1)
        
        loss_time = self.forward_loss_time(x, pred_time, mask)
        
        loss = loss_spec + self.time_loss_weight * loss_time
        loss_dict = {'loss_spec': loss_spec, 'loss_time': loss_time}

        return loss, loss_dict, pred_spec, pred_time, imgs, mask


# ===================================================================
# 5. 【新增】单塔对比学习包装器 (Single-Tower Contrastive Wrapper)
# ===================================================================
class SingleTowerContrastiveMAE(nn.Module):
    """
    方案一：单塔权重共享 + 批次内对比学习
    使用同一个 CWT_MAE_RoPE 骨干网络同时处理 PPG 和 ECG，并在隐空间对齐它们。
    """
    def __init__(self, base_model_config, proj_dim=256, temperature=0.07, alpha=1.0):
        super().__init__()
        # 强制 M=1，因为我们在 Batch 维度拼接信号，而不是通道维度
        base_model_config['max_modalities'] = 1
        self.encoder = CWT_MAE_RoPE(**base_model_config)
        
        embed_dim = base_model_config.get('embed_dim', 768)
        
        # 对比学习的非线性投影头 (Projection Head)
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, proj_dim)
        )
        self.temperature = temperature
        self.alpha = alpha # 对比损失的权重 (控制 MAE 重建和对比对齐的比例)

    def forward(self, x_ppg, x_ecg, mask_ratio=None):
        """
        训练时的前向传播
        x_ppg: (B, L) 或 (B, 1, L)
        x_ecg: (B, L) 或 (B, 1, L)
        """
        if x_ppg.dim() == 2: x_ppg = x_ppg.unsqueeze(1)
        if x_ecg.dim() == 2: x_ecg = x_ecg.unsqueeze(1)
        
        B = x_ppg.shape[0]
        
        # 1. 在 Batch 维度拼接，形成 (2B, 1, L) 的输入
        x_both = torch.cat([x_ppg, x_ecg], dim=0) 
        
        # 2. 计算 MAE 重建损失 (带有 Mask)
        # 这保证了模型依然能学到信号的底层细节
        loss_mae, loss_dict_mae, _, _, _, _ = self.encoder(x_both, mask_ratio=mask_ratio)
        
        # 3. 提取全局特征用于对比学习 (不带 Mask，保证特征完整性)
        # 准备 CWT tokens
        imgs_both = self.encoder.prepare_tokens(x_both)
        # 提取特征 (mask_ratio=0.0)
        latent, _, _, _ = self.encoder.forward_encoder(x_both, imgs_both, mask_ratio=0.0) 
        # latent shape: (2B, N_patches, D)
        
        # 全局平均池化 (Global Average Pooling) 得到全局表征
        global_feat = latent.mean(dim=1) # (2B, D)
        
        # 投影到对比空间并归一化
        z = self.proj_head(global_feat) # (2B, proj_dim)
        z = F.normalize(z, dim=-1)
        
        # 切分回 PPG 和 ECG 的特征
        z_ppg, z_ecg = z.chunk(2, dim=0) # 各自 shape: (B, proj_dim)
        
        # 4. 计算 InfoNCE 对比损失 (双向)
        # PPG 找 ECG
        logits_pe = torch.matmul(z_ppg, z_ecg.T) / self.temperature
        # ECG 找 PPG
        logits_ep = torch.matmul(z_ecg, z_ppg.T) / self.temperature
        
        labels = torch.arange(B, device=z.device)
        loss_nce_pe = F.cross_entropy(logits_pe, labels)
        loss_nce_ep = F.cross_entropy(logits_ep, labels)
        
        # 双向对比损失求平均
        loss_contrastive = (loss_nce_pe + loss_nce_ep) / 2
        
        # 5. 计算总损失
        total_loss = loss_mae + self.alpha * loss_contrastive
        
        loss_dict = {
            'loss_total': total_loss,
            'loss_mae': loss_mae,
            'loss_contrastive': loss_contrastive,
            **loss_dict_mae
        }
        
        return total_loss, loss_dict

    def extract_features(self, x):
        """
        推理/下游任务微调时使用：仅输入单模态信号 (如仅 PPG)
        返回对齐后的全局特征
        """
        if x.dim() == 2: x = x.unsqueeze(1)
        imgs = self.encoder.prepare_tokens(x)
        latent, _, _, _ = self.encoder.forward_encoder(x, imgs, mask_ratio=0.0)
        global_feat = latent.mean(dim=1)
        return global_feat