import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from tqdm import tqdm # 用于训练和推理的进度条

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
# 3. 基础组件 (修正 PatchEmbed，保持 TensorizedBlock)
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
    【修正版】内存安全的 Patch Embedding
    由于输入通道仅为 3，直接使用标准卷积是最优解。
    之前的分解版本会导致中间张量膨胀到 94 亿元素，引发 32-bit 索引错误。
    """
    def __init__(self, img_size=(64, 3000), patch_size=(4, 50), in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # 直接使用标准卷积，一步到位完成下采样和升维
        # 避免了中间产生 (B, 768, 64, 3000) 的巨大张量
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # x: (B, 3, 64, 3000)
        # proj -> (B, 768, 16, 60) -> 元素数量大幅减少，安全
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
# 4. 主模型: CWT-MAE with RoPE (MAE 部分)
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
        
        # 1. Patch Embed (修正为标准卷积)
        self.patch_embed = DecomposedPatchEmbed(
            img_size=(cwt_scales, signal_len),
            patch_size=(patch_size_freq, patch_size_time),
            in_chans=3,
            embed_dim=embed_dim,
            norm_layer=norm_layer
        )
        self.num_patches = self.patch_embed.num_patches
        self.grid_size = self.patch_embed.grid_size 

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
        # 1. Patch Embedding & Position Embedding
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        
        # 2. Masking Logic (优化：微调时 mask_ratio=0，跳过繁重的排序计算)
        if self.mask_ratio == 0.0:
            # 不进行 Mask，保留所有 Patch
            x_masked = x
            B, N, D = x.shape
            
            # 生成顺序索引 [0, 1, ..., N-1] 并扩展到 Batch 维度
            # ids_keep: [B, N]
            ids_keep = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
            
            # 在不打乱的情况下，恢复索引也是顺序的
            ids_restore = ids_keep 
            
            # 全 0 mask，表示没有被遮蔽
            mask = torch.zeros(B, N, device=x.device)
        else:
            # 预训练时进行随机 Mask
            x_masked, mask, ids_restore, ids_keep = self.random_masking(x, self.mask_ratio)
        
        # 3. 添加 CLS Token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x_masked), dim=1)
        
        # 4. 准备 RoPE 需要的位置索引 (Position IDs)
        B = x.shape[0]
        # CLS token 的位置设为 0
        cls_pos = torch.zeros(B, 1, device=x.device, dtype=torch.long)
        # Patch tokens 的位置设为 ids_keep + 1 (因为 0 被 CLS 占用了)
        # 如果是 mask_ratio=0，这里就是 1, 2, ..., N
        patch_pos = ids_keep + 1
        pos_ids = torch.cat((cls_pos, patch_pos), dim=1)
        
        # 5. 生成旋转位置编码 (RoPE)
        rope_cos, rope_sin = self.rope_encoder(x, pos_ids)
        
        # 6. Transformer Blocks
        for blk in self.blocks:
            x = blk(x, rope_cos=rope_cos, rope_sin=rope_sin)
            
        x = self.norm(x)
        
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        # mask_token 的数量应该与 x 中被 mask 的 patch 数量一致
        # x 的 shape 是 (B, N_keep+1, D_decoder)
        # ids_restore 的 shape 是 (B, N_patches+1)
        # N_patches = self.num_patches
        # N_keep = x.shape[1] - 1 (因为有 CLS token)
        # mask_token 需要填充到 N_patches + 1 的总长度
        
        # 确保 mask_token 的数量正确
        num_total_patches = self.num_patches + 1 # 包括 CLS token
        num_visible_tokens = x.shape[1] # 包括 CLS token
        num_mask_tokens = num_total_patches - num_visible_tokens
        
        mask_tokens = self.mask_token.repeat(x.shape[0], num_mask_tokens, 1)
        
        # 将 mask_token 插入到 x 的末尾 (在 CLS token 之后)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1) # 跳过 CLS token
        
        # 使用 ids_restore 将填充的 mask_token 放到正确的位置
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        
        # 重新加上 CLS token
        x = torch.cat([x[:, :1, :], x_], dim=1)
        
        x = x + self.decoder_pos_embed
        
        B, N, _ = x.shape
        pos_ids = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        rope_cos, rope_sin = self.rope_decoder(x, pos_ids)
        
        for blk in self.decoder_blocks:
            x = blk(x, rope_cos=rope_cos, rope_sin=rope_sin)
            
        x = self.decoder_norm(x)
        return x[:, 1:, :] # 返回去除 CLS token 后的 patch features

    def forward_loss_spec(self, imgs, pred, mask):
        # imgs: (B, 3, H, W)
        # pred: (B, num_patches, patch_pixels)
        # mask: (B, num_patches)
        
        p_h, p_w = self.patch_embed.patch_size # patch_size_freq, patch_size_time
        B, C, H, W = imgs.shape # C=3, H=cwt_scales, W=signal_len
        
        # 将预测的 patch_pixels 重建成原始图像的 patch
        # pred shape: (B, num_patches, C * p_h * p_w)
        # num_patches = H // p_h * W // p_w
        
        # 目标图像的 patch
        target = imgs.view(B, C, H // p_h, p_h, W // p_w, p_w)
        target = target.permute(0, 2, 4, 1, 3, 5).contiguous() # (B, H//p_h, W//p_w, C, p_h, p_w)
        target = target.view(B, -1, C * p_h * p_w) # (B, num_patches, patch_pixels)
        
        # 计算 loss
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1) # (B, num_patches)
        loss = (loss * mask).sum() / (mask.sum() + 1e-6) # Masked mean loss
        return loss

    def forward_loss_time(self, x_raw, pred_time):
        # x_raw: (B, signal_len) - 原始信号
        # pred_time: (B, signal_len) - 重建的信号
        x_raw = x_raw.float()
        pred_time = pred_time.float()
        
        # Z-score 归一化
        mean_raw = x_raw.mean(dim=-1, keepdim=True)
        std_raw = x_raw.std(dim=-1, keepdim=True)
        std_raw = torch.clamp(std_raw, min=1e-5)
        target = (x_raw - mean_raw) / std_raw
        
        mean_pred = pred_time.mean(dim=-1, keepdim=True)
        std_pred = pred_time.std(dim=-1, keepdim=True)
        std_pred = torch.clamp(std_pred, min=1e-5)
        pred = (pred_time - mean_pred) / std_pred
        
        loss = F.mse_loss(pred, target)
        return loss

    def forward(self, x):
        # x: (B, signal_len)
        if x.dim() == 3: x = x.squeeze(1) # 确保是 (B, signal_len)
        
        # 1. CWT 变换 & 归一化
        imgs = cwt_wrap(x, num_scales=self.cwt_scales, lowest_scale=0.1, step=1.0)
        
        dtype_orig = imgs.dtype
        imgs_f32 = imgs.float() 
        mean = imgs_f32.mean(dim=(2, 3), keepdim=True)
        std = imgs_f32.std(dim=(2, 3), keepdim=True)
        std = torch.clamp(std, min=1e-5)
        imgs = (imgs_f32 - mean) / std
        imgs = imgs.to(dtype=dtype_orig)

        # 2. Encoder
        latent, mask, ids_restore = self.forward_encoder(imgs)
        
        # 3. Decoder
        decoder_features = self.forward_decoder(latent, ids_restore)
        
        # 4. Reconstruction Heads
        # 频谱重建
        pred_spec = self.decoder_pred_spec(decoder_features)
        loss_spec = self.forward_loss_spec(imgs, pred_spec, mask)
        
        # 时间序列重建
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
# 5. Frozen MAE Wrapper (用于 Latent Diffusion)
# ===================================================================
class FrozenMAEWrapper(nn.Module):
    def __init__(self, pretrained_path, device='cuda', 
                 signal_len=3000, cwt_scales=64, patch_size_time=50, patch_size_freq=4,
                 embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16):
        super().__init__()
        self.device = device
        self.signal_len = signal_len
        self.cwt_scales = cwt_scales
        self.patch_size_time = patch_size_time
        self.patch_size_freq = patch_size_freq
        self.grid_size = (cwt_scales // patch_size_freq, signal_len // patch_size_time)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # 初始化 MAE 模型结构 (参数需要和你训练时一致)
        self.mae = CWT_MAE_RoPE(
            signal_len=signal_len,
            cwt_scales=cwt_scales,
            patch_size_time=patch_size_time,
            patch_size_freq=patch_size_freq,
            embed_dim=embed_dim, 
            depth=depth, 
            num_heads=num_heads,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mask_ratio=0.0  # 关键：推理和特征提取时不需要 Mask
        )
        
        # 加载权重
        print(f"Loading MAE weights from: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        # 处理可能存在的 'module.' 前缀 (如果是 DDP 训练的)
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        self.mae.load_state_dict(state_dict, strict=False) # strict=False 允许加载部分匹配的权重
        
        self.mae.to(device)
        self.mae.eval()
        
        # 冻结所有参数
        for param in self.mae.parameters():
            param.requires_grad = False
            
    def encode(self, x):
        """
        输入: 原始信号 (B, signal_len)
        输出: Latent Representation (B, N_patches+1, embed_dim)
        """
        with torch.no_grad():
            if x.dim() == 3: x = x.squeeze(1) # 确保是 (B, signal_len)
            
            # 1. CWT 变换 & 归一化 (复用原模型逻辑)
            imgs = cwt_wrap(x, num_scales=self.cwt_scales, lowest_scale=0.1, step=1.0)
            dtype_orig = imgs.dtype
            imgs_f32 = imgs.float()
            mean = imgs_f32.mean(dim=(2, 3), keepdim=True)
            std = imgs_f32.std(dim=(2, 3), keepdim=True)
            std = torch.clamp(std, min=1e-5)
            imgs = (imgs_f32 - mean) / std
            imgs = imgs.to(dtype=dtype_orig)

            # Encoder 前向传播 (mask_ratio=0)
            latent, _, ids_restore = self.mae.forward_encoder(imgs)
            
        return latent, ids_restore # latent shape: (B, N_patches+1, embed_dim)

    def decode(self, latent, ids_restore):
        """
        输入: Latent (B, N_patches+1, embed_dim)
        输出: 重建的时间序列信号 (B, signal_len)
        """
        with torch.no_grad():
            # Decoder 前向传播
            decoder_features = self.mae.forward_decoder(latent, ids_restore)
            
            # 时间域重建头 (Time Head)
            B, N, D = decoder_features.shape # N = num_patches
            H_grid, W_grid = self.grid_size
            
            # 变换维度以适应 Conv2d
            feat_2d = decoder_features.transpose(1, 2).view(B, D, H_grid, W_grid)
            feat_time_agg = self.mae.time_reducer(feat_2d)
            feat_time_agg = feat_time_agg.squeeze(2).transpose(1, 2)
            pred_time = self.mae.time_pred(feat_time_agg).flatten(1)
            
            # 注意：这里输出的是归一化后的信号，如果需要真实幅值，需要反归一化
            # 但通常生成模型生成归一化信号即可
            return pred_time

# ===================================================================
# 6. Latent Diffusion Model (1D UNet)
# ===================================================================

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        # 使用 log scale 来获得更平滑的频率变化
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, groups=8):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, out_channels),
            nn.SiLU() # 使用 SiLU 激活函数
        )
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.act1 = nn.SiLU()
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act2 = nn.SiLU()
        
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        # x: (B, C_in, L)
        # t: (B, time_emb_dim)
        
        h = self.conv1(x)
        # Add time embedding
        time_emb = self.time_mlp(t).unsqueeze(-1) # (B, C_out, 1)
        h = h + time_emb
        h = self.norm1(h)
        h = self.act1(h)
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act2(h)
        
        return h + self.shortcut(x)

class LatentDiffusion1D(nn.Module):
    def __init__(self, mae_embed_dim=768, mae_decoder_embed_dim=512, 
                 time_emb_dim=256, hidden_dim=128, num_groups=8):
        super().__init__()
        
        # 输入通道 = ECG Latent (D) + PPG Latent (D)
        # 注意：这里假设 MAE 的 embed_dim 和 decoder_embed_dim 是我们需要的通道数
        # 如果你的 MAE encoder 和 decoder embed_dim 不同，需要调整
        # 这里我们使用 decoder_embed_dim 作为 Diffusion Model 的通道数
        in_channels = mae_decoder_embed_dim * 2 # PPG latent + ECG latent
        out_channels = mae_decoder_embed_dim # 预测的噪声通道数
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim * 4),
        )

        # U-Net Structure
        # Downsampling Path
        self.down1 = ResidualBlock1D(in_channels, hidden_dim, time_emb_dim * 4, groups=num_groups)
        self.down2 = ResidualBlock1D(hidden_dim, hidden_dim * 2, time_emb_dim * 4, groups=num_groups)
        
        # Bottleneck
        self.mid = ResidualBlock1D(hidden_dim * 2, hidden_dim * 2, time_emb_dim * 4, groups=num_groups)
        
        # Upsampling Path
        # Skip connections concatenate features, so input channels increase
        self.up1 = ResidualBlock1D(hidden_dim * 2 + hidden_dim * 2, hidden_dim, time_emb_dim * 4, groups=num_groups) 
        self.up2 = ResidualBlock1D(hidden_dim + in_channels, out_channels, time_emb_dim * 4, groups=num_groups) 
        
        # Final layer to predict noise
        self.final_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def forward(self, x, cond, t):
        """
        x: Noisy ECG Latent (B, N_patches+1, D_decoder)
        cond: Clean PPG Latent (B, N_patches+1, D_decoder)
        t: Timestep tensor (B,)
        """
        # Permute to (B, D, N) for Conv1D
        x = x.transpose(1, 2)
        cond = cond.transpose(1, 2)
        
        # Concatenate conditioning (PPG latent) along channel dimension
        inp = torch.cat([x, cond], dim=1) # (B, D_decoder*2, N_patches+1)
        
        # Get time embeddings
        t_emb = self.time_mlp(t)
        
        # Downsampling
        d1 = self.down1(inp, t_emb) # (B, hidden_dim, N_patches+1)
        d2 = self.down2(d1, t_emb) # (B, hidden_dim*2, N_patches+1)
        
        # Bottleneck
        m = self.mid(d2, t_emb) # (B, hidden_dim*2, N_patches+1)
        
        # Upsampling with skip connections
        # Concatenate m with d2 (skip connection)
        u1 = self.up1(torch.cat([m, d2], dim=1), t_emb) # (B, hidden_dim, N_patches+1)
        # Concatenate u1 with inp (original input + condition)
        u2 = self.up2(torch.cat([u1, inp], dim=1), t_emb) # (B, out_channels, N_patches+1)
        
        # Final prediction
        out = self.final_conv(u2) # (B, out_channels, N_patches+1)
        
        return out.transpose(1, 2) # Transpose back to (B, N_patches+1, D_decoder)

# ===================================================================
# 7. Diffusion Training & Sampling Utilities
# ===================================================================

def extract(a, t, x_shape):
    """Extract coefficients and reshape to match input shape."""
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def get_diffusion_params(num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cuda'):
    """Precompute diffusion parameters."""
    betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod

# ===================================================================
# 8. Training & Inference Functions
# ===================================================================

def train_diffusion_model(
    mae_wrapper: FrozenMAEWrapper, 
    diffusion_model: LatentDiffusion1D, 
    train_loader: torch.utils.data.DataLoader, 
    optimizer: torch.optim.Optimizer,
    num_timesteps: int,
    betas: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    device: str,
    epochs: int = 10
):
    """Trains the Latent Diffusion Model."""
    diffusion_model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for ppg, ecg in progress_bar:
            ppg, ecg = ppg.to(device), ecg.to(device)
            
            # 1. Get Latent Representations from MAE Encoder
            z_ppg, _ = mae_wrapper.encode(ppg) # Condition (B, N+1, D_decoder)
            z_ecg, _ = mae_wrapper.encode(ecg) # Target (B, N+1, D_decoder)
            
            # 2. Sample random timesteps
            B = ppg.shape[0]
            t = torch.randint(0, num_timesteps, (B,), device=device).long()
            
            # 3. Add noise to ECG latent
            noise = torch.randn_like(z_ecg)
            alpha_bar_t = extract(alphas_cumprod, t, z_ecg.shape)
            z_ecg_noisy = torch.sqrt(alpha_bar_t) * z_ecg + torch.sqrt(1 - alpha_bar_t) * noise
            
            # 4. Predict the noise using the diffusion model
            noise_pred = diffusion_model(z_ecg_noisy, z_ppg, t)
            
            # 5. Calculate MSE loss
            loss = nn.functional.mse_loss(noise_pred, noise)
            
            # 6. Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            # Optional: Gradient clipping
            # torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.6f}")
        
        # Optional: Save model checkpoint periodically
        # torch.save(diffusion_model.state_dict(), f"diffusion_model_epoch_{epoch+1}.pth")

    print("Training finished.")

@torch.no_grad()
def generate_ecg_from_ppg(
    mae_wrapper: FrozenMAEWrapper, 
    diffusion_model: LatentDiffusion1D, 
    ppg_signal: torch.Tensor, 
    num_timesteps: int,
    betas: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    device: str
):
    """Generates ECG signal from PPG signal using the trained diffusion model."""
    diffusion_model.eval()
    ppg_signal = ppg_signal.to(device)
    
    # 1. Encode PPG signal to get conditional latent
    z_ppg, ids_restore = mae_wrapper.encode(ppg_signal)
    B, N, D = z_ppg.shape # N = num_patches + 1
    
    # 2. Initialize noisy ECG latent
    z_ecg = torch.randn((B, N, D), device=device)
    
    # 3. DDPM sampling loop (reverse process)
    for i in tqdm(reversed(range(num_timesteps)), desc='Sampling ECG', total=num_timesteps):
        t = torch.full((B,), i, device=device, dtype=torch.long)
        
        # Predict noise
        noise_pred = diffusion_model(z_ecg, z_ppg, t)
        
        # Calculate denoised latent using DDPM update rule
        beta_t = betas[i]
        alpha_t = 1.0 - beta_t
        alpha_bar_t = alphas_cumprod[i]
        
        # Standard DDPM update
        coeff = beta_t / torch.sqrt(1 - alpha_bar_t)
        mean = (1 / torch.sqrt(alpha_t)) * (z_ecg - coeff * noise_pred)
        
        if i > 0:
            # Add stochastic noise for steps > 0
            noise = torch.randn_like(z_ecg)
            sigma = torch.sqrt(beta_t)
            z_ecg = mean + sigma * noise
        else:
            # No noise added at the last step (t=0)
            z_ecg = mean

    # 4. Decode the final latent back to time-domain signal
    pred_ecg_signal = mae_wrapper.decode(z_ecg, ids_restore)
    
    return pred_ecg_signal