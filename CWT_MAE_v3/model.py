import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        output = x / keep_prob * random_tensor
        return output

# ===================================================================
# 0. CMAE 新增组件：生理信号增广 + 投影头
# ===================================================================
class PhysioAugment(nn.Module):
    def __init__(self, jitter_std=0.02, scale_range=(0.85, 1.15),
                 max_shift=50, wander_amp=0.05):
        super().__init__()
        self.jitter_std = jitter_std
        self.scale_range = scale_range
        self.max_shift = max_shift
        self.wander_amp = wander_amp

    @torch.no_grad()
    @torch.compiler.disable
    def forward(self, x):
        squeeze = (x.dim() == 2)
        if squeeze:
            x = x.unsqueeze(1)
        B, M, L = x.shape
        s = torch.empty(B, M, 1, device=x.device).uniform_(*self.scale_range)
        x = x * s
        x = x + torch.randn_like(x) * self.jitter_std
        t = torch.linspace(0, 1, L, device=x.device)
        f = torch.rand(B, M, 1, device=x.device) * 2 + 0.5
        ph = torch.rand(B, M, 1, device=x.device) * 6.283
        x = x + self.wander_amp * torch.sin(2 * math.pi * f * t + ph)
        shift = int(torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item())
        x = torch.roll(x, shifts=shift, dims=-1)
        return x.squeeze(1) if squeeze else x


class ProjectionHead(nn.Module):
    def __init__(self, dim, hidden=2048, out=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Linear(hidden, out)
        )

    def forward(self, x):
        return self.net(x)

# ===================================================================
# 1. CWT 模块 (保持不变，优秀的特征工程)
# ===================================================================
@torch.compiler.disable
def create_ricker_wavelets(points: int, scales: torch.Tensor):
    scales = scales.float()
    
    t = torch.arange(0, points, device=scales.device, dtype=torch.float32) - (points - 1.0) / 2.0
    t = t.reshape(1, 1, -1) 
    scales = scales.reshape(-1, 1, 1)
    
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
    
    scales = scales.to(x.device)
    largest_scale = scales[-1]
    
    if torch.onnx.is_in_onnx_export():
        wavelet_len_int = 631
    else:
        largest_scale_val = largest_scale.item()
        seq_len_val = sequence_length if isinstance(sequence_length, int) else sequence_length.item()
        wavelet_len_int = int(min(10.0 * largest_scale_val, float(seq_len_val)))
        if wavelet_len_int % 2 == 0:
            wavelet_len_int += 1
            
    wavelets = create_ricker_wavelets(wavelet_len_int, scales)
    wavelets = wavelets.to(dtype=x.dtype)
    
    pad_len = wavelet_len_int // 2
    x_padded = F.pad(x, (pad_len, pad_len), mode='reflect')
    
    cwt_output = F.conv1d(x_padded, wavelets)
    
    return cwt_output

@torch.compiler.disable
def cwt_wrap(x, num_scales=64, lowest_scale=0.1, step=1.0, use_diff=True):
    if x.dim() == 2:
        x = x.unsqueeze(1)
    B, M, L = x.shape
    x_flat = x.reshape(B * M, L)
    
    if use_diff:
        # 【修复】使用中心差分保持相位对齐
        # 一阶导数: f'(x) = (f(x+1) - f(x-1)) / 2
        x_pad = F.pad(x_flat, (1, 1), mode='replicate')
        d1 = (x_pad[:, 2:] - x_pad[:, :-2]) / 2.0

        # 二阶导数: f''(x) = f(x+1) - 2*f(x) + f(x-1)
        d2 = x_pad[:, 2:] - 2 * x_pad[:, 1:-1] + x_pad[:, :-2]

        base = x_flat
        d1_cut = d1[:, :L]
        d2_cut = d2[:, :L]

        signals = torch.stack([base, d1_cut, d2_cut], dim=1)
    else:
        base = x_flat
        signals = base.unsqueeze(1) 
        
    BM, C, _ = signals.shape
    signals_flat = signals.reshape(BM * C, L)
    
    scales = torch.arange(num_scales, device=x.device) * step + lowest_scale
    cwt_out = cwt_ricker(signals_flat, scales)
    _, n_scales, _ = cwt_out.shape
    
    cwt_out = cwt_out.reshape(B, M, C, n_scales, L)
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
        
        cos_tensor = emb.cos().to(torch.float32)
        sin_tensor = emb.sin().to(torch.float32)
        
        self.register_buffer("cos_cached", cos_tensor, persistent=False)
        self.register_buffer("sin_cached", sin_tensor, persistent=False)

    def forward(self, x, pos_ids):
        if self.cos_cached.device != x.device:
            self.cos_cached = self.cos_cached.to(x.device)
            self.sin_cached = self.sin_cached.to(x.device)
            
        pos_ids = pos_ids.to(x.device)

        if torch.onnx.is_in_onnx_export():
            cos = self.cos_cached[pos_ids].to(x.dtype)
            sin = self.sin_cached[pos_ids].to(x.dtype)
            return cos.unsqueeze(2), sin.unsqueeze(2)
            
        seq_len = torch.max(pos_ids) + 1
        
        with torch.no_grad():
            if seq_len > self.max_seq_len:
                new_len = (seq_len * 3 // 2).int().item()
                self._update_cache(new_len)

        # 【修复】避免在 forward 中覆盖自身 buffer，改为按需转换
        cos = self.cos_cached[pos_ids].to(device=x.device, dtype=x.dtype, non_blocking=True)
        sin = self.sin_cached[pos_ids].to(device=x.device, dtype=x.dtype, non_blocking=True)
        return cos.unsqueeze(2), sin.unsqueeze(2)

def apply_rotary_pos_emb(q, k, cos, sin):
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

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
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm(x)
        return x

class RawSignalPatchEmbed(nn.Module):
    def __init__(self, patch_size_time=50, embed_dim=768, norm_layer=None):
        super().__init__()
        self.proj = nn.Conv1d(1, embed_dim, kernel_size=patch_size_time, stride=patch_size_time)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)              
        x = x.transpose(1, 2)         
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
        
        q, k, v = q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous()
        
        dropout_val = float(self.attn_drop.p) if self.training else 0.0
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_val)
        
        x = x.transpose(1, 2).contiguous().reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# ===================================================================
# 3. 时空因子化 Block
# ===================================================================
class TrueFactorizedBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm_spatial_temporal = norm_layer(dim)
        self.spatial_temporal_attn = RoPEAttention(dim, num_heads=num_heads, proj_drop=drop)
        
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
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, M, N, rope_cos=None, rope_sin=None):
        B, MN, D = x.shape

        x_time = x.contiguous().reshape(B * M, N, D)

        if rope_cos is not None and rope_sin is not None:
            M_rope = (B * M) // rope_cos.shape[0]
            cos_t = rope_cos.repeat_interleave(M_rope, dim=0)
            sin_t = rope_sin.repeat_interleave(M_rope, dim=0)
        else:
            cos_t, sin_t = None, None

        attn_out = self.spatial_temporal_attn(self.norm_spatial_temporal(x_time), cos_t, sin_t)
        x_time = x_time + self.drop_path(attn_out)

        if M > 1:
            x_c = x_time.reshape(B, M, N, D)
            x_channel = x_c.transpose(1, 2).contiguous().reshape(B * N, M, D)
            attn_out = self.channel_attn(self.norm_channel(x_channel))
            x_c = x_c + self.drop_path(attn_out.reshape(B, N, M, D).transpose(1, 2))
            x = x_c.reshape(B, MN, D)
        else:
            x = x_time.reshape(B, MN, D)
            
        x = x + self.drop_path(self.mlp(self.norm_mlp(x)))
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = RoPEAttention(dim, num_heads=num_heads, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(drop)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, rope_cos=None, rope_sin=None):
        x = x + self.drop_path(self.attn(self.norm1(x), rope_cos, rope_sin))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# ===================================================================
# 5. 核心骨干网络: CWT-MAE (带 Attention Residuals)
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
        stats_loss_weight=1.0,
        contrast_loss_weight=1.0,
        use_diff=False,
        diff_loss_weight=None,
        max_modalities=16,
        drop_path_rate=0.0
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
            
        self.register_buffer('channel_loss_weights', torch.tensor(self.diff_loss_weight).reshape(1, 1, 1, -1))
        
        in_chans = 3 if use_diff else 1
        
        self.patch_embed = DecomposedPatchEmbed(
            img_size=(cwt_scales, signal_len),
            patch_size=(patch_size_freq, patch_size_time),
            embed_dim=embed_dim, norm_layer=norm_layer,
            in_chans=in_chans
        )
        self.num_patches = self.patch_embed.num_patches
        self.grid_size = self.patch_embed.grid_size 

        self.raw_patch_embed = RawSignalPatchEmbed(
            patch_size_time=patch_size_time,
            embed_dim=embed_dim,
            norm_layer=norm_layer
        )
        
        self.raw_signal_scale = nn.Parameter(torch.ones(1, 1, 1, embed_dim) * 0.1)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        self.rope_encoder = RotaryEmbedding(dim=embed_dim // num_heads)
        self.rope_decoder = RotaryEmbedding(dim=decoder_embed_dim // decoder_num_heads)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TrueFactorizedBlock(embed_dim, num_heads, drop_path=dpr[i], norm_layer=norm_layer) 
            for i in range(depth)
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

        # 【新增】统计量预测头 (预测 16 个统计特征)
        self.stats_pred_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 16)
        )
        self.stats_loss_weight = stats_loss_weight

        self.register_buffer('stats_running_mean', torch.zeros(16))
        self.register_buffer('stats_running_var', torch.ones(16))
        self.stats_momentum = 0.01

        self.contrast_loss_weight = contrast_loss_weight
        self.ema_decay = 0.999

        self.augment = PhysioAugment()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.student_projector = ProjectionHead(embed_dim, hidden=2048, out=256)
        self.student_predictor = ProjectionHead(256, hidden=1024, out=256)
        self.teacher_projector = ProjectionHead(embed_dim, hidden=2048, out=256)

        self.teacher_blocks = nn.ModuleList([
            TrueFactorizedBlock(embed_dim, num_heads, drop_path=0.0, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.teacher_norm = norm_layer(embed_dim)

        self.initialize_weights()
        self._init_teacher()

    def initialize_weights(self):
        torch.nn.init.trunc_normal_(self.pos_embed, std=.02)
        torch.nn.init.trunc_normal_(self.decoder_pos_embed, std=.02)
        torch.nn.init.trunc_normal_(self.mask_token, std=.02)
        torch.nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_teacher(self):
        for ps, pt in zip(self.blocks.parameters(), self.teacher_blocks.parameters()):
            pt.data.copy_(ps.data)
        for ps, pt in zip(self.norm.parameters(), self.teacher_norm.parameters()):
            pt.data.copy_(ps.data)
        for ps, pt in zip(self.student_projector.parameters(), self.teacher_projector.parameters()):
            pt.data.copy_(ps.data)
        for p in self.teacher_blocks.parameters():
            p.requires_grad_(False)
        for p in self.teacher_norm.parameters():
            p.requires_grad_(False)
        for p in self.teacher_projector.parameters():
            p.requires_grad_(False)

    def student_encoder_params(self):
        for p in self.blocks.parameters():
            yield p
        for p in self.norm.parameters():
            yield p
        for p in self.student_projector.parameters():
            yield p

    def teacher_encoder_params(self):
        for p in self.teacher_blocks.parameters():
            yield p
        for p in self.teacher_norm.parameters():
            yield p
        for p in self.teacher_projector.parameters():
            yield p

    def tubelet_masking(self, x, mask_ratio, M, N_patches, noise_w=None):
        B, _, D = x.shape
        H_grid, W_grid = self.grid_size
        len_keep_w = int(W_grid * (1 - mask_ratio))
        len_keep = len_keep_w * H_grid
        
        if noise_w is None:
            noise_w = torch.rand(B, W_grid, device=x.device)
        ids_shuffle_w = torch.argsort(noise_w, dim=1)
        ids_restore_w = torch.argsort(ids_shuffle_w, dim=1)
        
        h_idx = torch.arange(H_grid, device=x.device).reshape(1, H_grid, 1)
        ids_restore_w_exp = ids_restore_w.unsqueeze(1)
        noise = (ids_restore_w_exp * H_grid + h_idx).reshape(B, N_patches)
        
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1) 
        ids_keep = ids_shuffle[:, :len_keep]            
        
        x_reshaped = x.reshape(B, M, N_patches, D)
        ids_keep_expanded = ids_keep.unsqueeze(1).unsqueeze(-1).expand(B, M, len_keep, D)
        x_masked = torch.gather(x_reshaped, dim=2, index=ids_keep_expanded)
        x_masked = x_masked.reshape(B, M * len_keep, D)
        
        mask = torch.ones([B, N_patches], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        mask = mask.unsqueeze(1).expand(B, M, N_patches).reshape(B, M * N_patches)
        
        global_ids_restore = torch.cat([ids_restore + m * N_patches for m in range(M)], dim=1)

        return x_masked, mask, global_ids_restore, ids_keep, len_keep

    def mixed_masking(self, x, mask_ratio, M, N_patches, noise_w=None):
        return self.tubelet_masking(x, mask_ratio, M, N_patches, noise_w)

    def forward_encoder(self, x_raw, imgs, mask_ratio=None, noise_w=None, return_layer_indices=None):
        """
        参数:
            return_layer_indices: list[int], 需要返回的中间层索引，如 [3, 7, 11]
        """
        B, M, C, H, W = imgs.shape
        
        x_cwt = imgs.reshape(B * M, C, H, W)
        x_cwt = self.patch_embed(x_cwt) 
        
        if x_raw.dim() == 2:
            x_raw = x_raw.unsqueeze(1)
            
        x_raw = x_raw.reshape(B * M, 1, -1)
            
        mean_raw = x_raw.mean(dim=-1, keepdim=True)
        std_raw = torch.clamp(x_raw.std(dim=-1, keepdim=True), min=1e-5)
        x_raw_norm = (x_raw - mean_raw) / std_raw
        
        x_raw_embed = self.raw_patch_embed(x_raw_norm.to(dtype=next(self.parameters()).dtype)) 
        
        H_grid, W_grid = self.grid_size
        D = x_cwt.shape[-1]
        
        x_cwt_2d = x_cwt.reshape(B * M, H_grid, W_grid, D)
        x_raw_2d = x_raw_embed.unsqueeze(1) 
        
        if self.raw_signal_scale.device != x_raw_2d.device:
            raw_scale = self.raw_signal_scale.to(x_raw_2d.device)
        else:
            raw_scale = self.raw_signal_scale
            
        x_fused = x_cwt_2d + x_raw_2d * raw_scale
        
        x = x_fused.reshape(B, M, -1, D)
        N_patches = x.shape[2]

        if self.pos_embed.device != x.device:
            x = x + self.pos_embed.unsqueeze(1).to(x.device)
        else:
            x = x + self.pos_embed.unsqueeze(1)
            
        x = x.reshape(B, M * N_patches, -1)

        current_mask_ratio = mask_ratio if mask_ratio is not None else self.mask_ratio
        if current_mask_ratio == 0.0:
            x_masked = x
            mask = torch.zeros(B, M * N_patches, device=x.device)
            global_ids_restore = torch.arange(M * N_patches, device=x.device).unsqueeze(0).expand(B, -1)
            ids_keep = torch.arange(N_patches, device=x.device).unsqueeze(0).expand(B, -1)
            M_enc = M
            len_keep = N_patches  # 无 masking 时保留所有 patches
        else:
            x_masked, mask, global_ids_restore, ids_keep, len_keep = self.mixed_masking(x, current_mask_ratio, M, N_patches, noise_w)
            M_enc = M

        is_async = (ids_keep.dim() == 3)
        if is_async:
            pos_ids_flat = (ids_keep % W_grid).reshape(B * M_enc, -1)
        else:
            pos_ids_flat = (ids_keep % W_grid)

        # 【修复】使用 non_blocking 优化设备同步
        if pos_ids_flat.device != x_masked.device:
            pos_ids_flat = pos_ids_flat.to(x_masked.device, non_blocking=True)

        rope_cos, rope_sin = self.rope_encoder(x_masked, pos_ids_flat)

        intermediate_features = {}
        for i, blk in enumerate(self.blocks):
            x_masked = blk(x_masked, M_enc, len_keep, rope_cos=rope_cos, rope_sin=rope_sin)
            if return_layer_indices is not None and i in return_layer_indices:
                intermediate_features[i] = self.norm(x_masked)

        x_masked = self.norm(x_masked)
        
        if return_layer_indices is not None:
            return x_masked, mask, global_ids_restore, M, intermediate_features
        return x_masked, mask, global_ids_restore, M

    def forward_encoder_teacher(self, x_raw, imgs):
        B, M, C, H, W = imgs.shape

        x_cwt = imgs.reshape(B * M, C, H, W)
        x_cwt = self.patch_embed(x_cwt)

        if x_raw.dim() == 2:
            x_raw = x_raw.unsqueeze(1)
        x_raw = x_raw.reshape(B * M, 1, -1)
        mean_raw = x_raw.mean(dim=-1, keepdim=True)
        std_raw = torch.clamp(x_raw.std(dim=-1, keepdim=True), min=1e-5)
        x_raw_norm = (x_raw - mean_raw) / std_raw
        x_raw_embed = self.raw_patch_embed(x_raw_norm.to(dtype=next(self.parameters()).dtype))

        H_grid, W_grid = self.grid_size
        D = x_cwt.shape[-1]
        x_cwt_2d = x_cwt.reshape(B * M, H_grid, W_grid, D)
        x_raw_2d = x_raw_embed.unsqueeze(1)
        raw_scale = self.raw_signal_scale.to(x_raw_2d.device) if self.raw_signal_scale.device != x_raw_2d.device else self.raw_signal_scale
        x_fused = x_cwt_2d + x_raw_2d * raw_scale

        x = x_fused.reshape(B, M, -1, D)
        N_patches = x.shape[2]
        x = x + self.pos_embed.unsqueeze(1).to(x.device)
        x = x.reshape(B, M * N_patches, -1)

        pos_ids_flat = torch.arange(N_patches, device=x.device) % W_grid
        pos_ids_flat = pos_ids_flat.unsqueeze(0).expand(B, -1)
        rope_cos, rope_sin = self.rope_encoder(x, pos_ids_flat)

        for blk in self.teacher_blocks:
            x = blk(x, M, N_patches, rope_cos=rope_cos, rope_sin=rope_sin)
        x = self.teacher_norm(x)
        return x

    def forward_decoder(self, x, ids_restore, M):
        x = self.decoder_embed(x)
        B, _, D_dec = x.shape
        N_patches = self.num_patches
        Total_Tokens = M * N_patches

        mask_tokens = self.mask_token.repeat(B, Total_Tokens - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D_dec))

        x_patches = x.reshape(B, M, N_patches, D_dec)
        if self.decoder_pos_embed.device != x_patches.device:
            x_patches = x_patches + self.decoder_pos_embed.unsqueeze(1).to(x_patches.device)
        else:
            x_patches = x_patches + self.decoder_pos_embed.unsqueeze(1)
        x = x_patches.reshape(B, Total_Tokens, D_dec)

        H_grid, W_grid = self.grid_size
        patch_pos = (torch.arange(N_patches, device=x.device) % W_grid).unsqueeze(0).expand(B, -1)
        patch_pos_expanded = patch_pos.repeat(1, M)

        if patch_pos_expanded.device != x.device:
            patch_pos_expanded = patch_pos_expanded.to(x.device)

        rope_cos, rope_sin = self.rope_decoder(x, patch_pos_expanded)
        
        for blk in self.decoder_blocks:
            x = blk(x, rope_cos=rope_cos, rope_sin=rope_sin)
        x = self.decoder_norm(x)
        
        x = x.reshape(B, M, N_patches, D_dec)
        return x

    def forward_loss_spec(self, imgs, pred, mask, channel_mask=None):
        B, M, C, H, W = imgs.shape
        p_h, p_w = self.patch_embed.patch_size
        
        H_valid = (H // p_h) * p_h
        W_valid = (W // p_w) * p_w
        
        if H != H_valid or W != W_valid:
            imgs = imgs[..., :H_valid, :W_valid]
            H, W = H_valid, W_valid
            
        target = imgs.reshape(B, M, C, H // p_h, p_h, W // p_w, p_w)
        target = target.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
        target = target.reshape(B, M, -1, C * p_h * p_w)
        
        mask = mask.reshape(B, M, -1)
        loss = (pred - target) ** 2
        
        loss = loss.reshape(B, M, -1, C, p_h * p_w)
        loss = loss.mean(dim=-1) 
        loss = loss * self.channel_loss_weights
        loss = loss.sum(dim=-1) 
        
        if channel_mask is not None:
            ch_mask = channel_mask.unsqueeze(-1).float()
            loss = loss * ch_mask
            loss = (loss * mask).sum() / ((mask * ch_mask).sum() + 1e-8)
        else:
            loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        return loss

    def forward_loss_time(self, x_raw, pred_time, mask, channel_mask=None):
        x_raw = x_raw.float()
        mean = x_raw.mean(dim=-1, keepdim=True)
        std = torch.clamp(x_raw.std(dim=-1, keepdim=True), min=1e-5)
        target = torch.clamp((x_raw - mean) / std, min=-10.0, max=10.0)
        
        if target.shape[-1] > pred_time.shape[-1]:
            target = target[..., :pred_time.shape[-1]]
        
        B, M, _ = pred_time.shape
        H_grid, W_grid = self.grid_size
        
        mask_2d = mask.reshape(B, M, H_grid, W_grid)
        mask_t = torch.round(mask_2d.mean(dim=2)) 
        
        mask_time = mask_t.unsqueeze(-1).expand(-1, -1, -1, self.patch_size_time)
        mask_time = mask_time.reshape(B, M, -1) 
        
        loss = (pred_time.float() - target) ** 2
        
        if channel_mask is not None:
            ch_mask = channel_mask.unsqueeze(-1).float()
            loss = loss * ch_mask
            mask_time = mask_time * ch_mask
            loss = (loss * mask_time).sum() / (mask_time.sum() + 1e-8)
        else:
            loss = (loss * mask_time).sum() / (mask_time.sum() + 1e-8)
        return loss

    def forward_loss_stats(self, pred_stats, stats_target):
        """
        计算统计量预测的 MSE 损失，可考虑 Smooth L1 也可以。这里采用 MSE。
        由于特征尺度不一，建议外部已归一化，或这里进行 SmoothL1。
        """
        loss = F.smooth_l1_loss(pred_stats, stats_target)
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

    def forward(self, x, stats_target=None, mask_ratio=None, channel_mask=None):
        B = x.shape[0]
        current_mask_ratio = mask_ratio if mask_ratio is not None else self.mask_ratio

        # 提前计算 target CWT（只算一次，用于 loss）
        with torch.no_grad():
            imgs_target = self.prepare_tokens(x)

        if current_mask_ratio > 0.0:
            H_grid, W_grid = self.grid_size
            noise_w = torch.rand(B, W_grid, device=x.device)
            ids_shuffle_w = torch.argsort(noise_w, dim=1)
            len_keep_w = int(W_grid * (1 - current_mask_ratio))
            ids_keep_w = ids_shuffle_w[:, :len_keep_w]

            mask_w_bool = torch.zeros(B, W_grid, device=x.device)
            mask_w_bool.scatter_(1, ids_keep_w, 1.0)

            M_dim = x.shape[1] if x.dim() == 3 else 1
            L_dim = x.shape[-1]

            mask_raw = mask_w_bool.unsqueeze(1).unsqueeze(-1).repeat(1, M_dim, 1, self.patch_size_time)
            mask_raw = mask_raw.reshape(B, M_dim, W_grid * self.patch_size_time)
            if mask_raw.shape[-1] != L_dim:
                mask_raw = mask_raw[..., :L_dim]

            if x.dim() == 2:
                x_exp = x.unsqueeze(1)
                x_visible = x_exp * mask_raw
                local_mean = x_visible.sum(dim=-1, keepdim=True) / (mask_raw.sum(dim=-1, keepdim=True) + 1e-8)
                x_masked = x_visible + local_mean * (1 - mask_raw)
                x_masked = x_masked.squeeze(1)
            else:
                x_visible = x * mask_raw
                local_mean = x_visible.sum(dim=-1, keepdim=True) / (mask_raw.sum(dim=-1, keepdim=True) + 1e-8)
                x_masked = x_visible + local_mean * (1 - mask_raw)

            imgs_input = self.prepare_tokens(x_masked)
        else:
            x_masked = x
            noise_w = None
            # mask_ratio == 0 时，输入 == 原始，复用 target CWT
            imgs_input = imgs_target

        latent, mask, ids, M = self.forward_encoder(x_masked, imgs_input, mask_ratio=mask_ratio, noise_w=noise_w)

        # Decoder
        decoder_features = self.forward_decoder(latent, ids, M)
        pred_spec = self.decoder_pred_spec(decoder_features)

        loss_spec = self.forward_loss_spec(imgs_target, pred_spec, mask, channel_mask)

        B_dec, M_dec, N, D = decoder_features.shape
        H_grid, W_grid = self.grid_size
        feat_2d = decoder_features.reshape(B_dec * M_dec, N, D).transpose(1, 2).reshape(B_dec * M_dec, D, H_grid, W_grid)

        feat_time_agg = self.time_reducer[0](feat_2d)
        feat_time_agg = self.time_reducer[1](feat_time_agg)

        feat_time_agg = feat_time_agg.squeeze(2).transpose(1, 2)
        feat_time_agg = self.time_reducer[2](feat_time_agg)

        pred_time = self.time_pred(feat_time_agg).flatten(1).reshape(B_dec, M_dec, -1)

        loss_time = self.forward_loss_time(x, pred_time, mask, channel_mask)

        loss = loss_spec + self.time_loss_weight * loss_time
        loss_dict = {'loss_spec': loss_spec, 'loss_time': loss_time}

        latent_pooled = latent.mean(dim=1)
        pred_stats = self.stats_pred_head(latent_pooled)

        if stats_target is not None:
            stats_target = stats_target.float()
            if self.training:
                with torch.no_grad():
                    batch_mean = stats_target.mean(dim=0)
                    batch_var = stats_target.var(dim=0, unbiased=False)
                    self.stats_running_mean = (1 - self.stats_momentum) * self.stats_running_mean + self.stats_momentum * batch_mean
                    self.stats_running_var = (1 - self.stats_momentum) * self.stats_running_var + self.stats_momentum * batch_var
            stats_target_norm = (stats_target - self.stats_running_mean) / torch.sqrt(self.stats_running_var + 1e-5)
            stats_target_norm = torch.clamp(stats_target_norm, min=-10.0, max=10.0)

            loss_stats = self.forward_loss_stats(pred_stats, stats_target_norm)
            loss = loss + self.stats_loss_weight * loss_stats
            loss_dict['loss_stats'] = loss_stats

        if self.training and self.contrast_loss_weight > 0:
            z_student = self.student_predictor(self.student_projector(latent_pooled))
            with torch.no_grad():
                x_teacher = self.augment(x)
                if channel_mask is not None:
                    x_teacher = x_teacher * channel_mask.unsqueeze(-1).float()
                imgs_teacher = self.prepare_tokens(x_teacher)
                t_latent = self.forward_encoder_teacher(x_teacher, imgs_teacher)
                t_pooled = t_latent.mean(dim=1)
                z_teacher = self.teacher_projector(t_pooled)
            loss_contrast = 2 - 2 * F.cosine_similarity(
                F.normalize(z_student, dim=-1),
                F.normalize(z_teacher, dim=-1), dim=-1).mean()
            loss = loss + self.contrast_loss_weight * loss_contrast
            loss_dict['loss_contrast'] = loss_contrast

        return loss, loss_dict, pred_spec, pred_time, imgs_target, mask, latent, pred_stats