import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ===================================================================
# 1. 基础组件 & RoPE
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
        seq_len_val = seq_len.item()

        if seq_len_val > self.max_seq_len:
            self._update_cache(int(seq_len_val * 1.5))

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

class SignalPatchEmbed(nn.Module):
    """1D 信号 Patch Embedding (直接在原始信号上划分 token)"""
    def __init__(self, patch_size=50, in_chans=1, embed_dim=768, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # x: (B, M, L) -> (B*M, 1, L)
        B, M, L = x.shape
        x = x.reshape(B * M, 1, L)
        x = self.proj(x)  # (B*M, D, N)
        x = x.transpose(1, 2)  # (B*M, N, D)
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
# 2. Transformer Block
# ===================================================================
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
# 3. 核心骨干网络: Signal-MAE (基于原始信号 Token 划分)
# ===================================================================
class Signal_MAE_RoPE(nn.Module):
    def __init__(
        self,
        signal_len=3000,
        patch_size=50,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mask_ratio=0.75,
        norm_layer=nn.LayerNorm,
        in_chans=1,
        max_modalities=16
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.signal_len = signal_len
        self.in_chans = in_chans

        # 计算 patch 数量
        self.num_patches = signal_len // patch_size

        # 1D Patch Embedding
        self.patch_embed = SignalPatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer
        )

        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # 通道类型 Embedding (0=ECG, 1=PPG)
        self.channel_type_embed = nn.Embedding(2, embed_dim)

        # RoPE
        self.rope_encoder = RotaryEmbedding(dim=embed_dim // num_heads)
        self.rope_decoder = RotaryEmbedding(dim=decoder_embed_dim // decoder_num_heads)

        # Encoder Blocks
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, num_heads=decoder_num_heads, norm_layer=norm_layer)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)

        # 预测头: 重建原始信号 patch
        self.patch_pixels = in_chans * patch_size
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.patch_pixels, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.trunc_normal_(self.pos_embed, std=.02)
        torch.nn.init.trunc_normal_(self.decoder_pos_embed, std=.02)
        torch.nn.init.trunc_normal_(self.mask_token, std=.02)
        torch.nn.init.trunc_normal_(self.channel_type_embed.weight, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, mask_ratio, noise=None):
        """
        对 token 进行随机 masking
        x: (B, N, D)
        返回: x_masked, mask, ids_restore
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))

        if noise is None:
            noise = torch.rand(B, N, device=x.device)

        # 排序并获取索引
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # 保留的 token
        ids_keep = ids_shuffle[:, :len_keep]
        ids_keep_expanded = ids_keep.unsqueeze(-1).expand(-1, -1, D)
        x_masked = torch.gather(x, dim=1, index=ids_keep_expanded)

        # 生成 mask (0=keep, 1=mask)
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, channel_ids, mask_ratio=None, noise=None):
        """
        Encoder 前向传播
        x: (B, M, L) 原始信号
        channel_ids: (B,) tensor, 0=ECG, 1=PPG
        """
        B, M, L = x.shape

        # Patch Embedding
        x_patches = self.patch_embed(x)  # (B*M, N, D)
        N = x_patches.shape[1]
        D = x_patches.shape[2]

        # Reshape 回 (B, M, N, D)
        x_patches = x_patches.reshape(B, M, N, D)

        # 注入通道类型标识
        if channel_ids.dim() == 1:
            ch_embed = self.channel_type_embed(channel_ids)  # (B, D)
            ch_embed = ch_embed.unsqueeze(1).unsqueeze(1)    # (B, 1, 1, D)
            ch_embed = ch_embed.expand(-1, M, -1, -1)        # (B, M, 1, D)
        else:
            M_ids = channel_ids.shape[1]
            if M_ids != M:
                if M_ids < M:
                    channel_ids = torch.nn.functional.pad(channel_ids, (0, M - M_ids), value=0)
                else:
                    channel_ids = channel_ids[:, :M]
            ch_embed = self.channel_type_embed(channel_ids)  # (B, M, D)
            ch_embed = ch_embed.unsqueeze(2)                 # (B, M, 1, D)

        x_patches = x_patches + ch_embed

        # 添加位置编码
        if self.pos_embed.device != x_patches.device:
            x_patches = x_patches + self.pos_embed.to(x_patches.device)
        else:
            x_patches = x_patches + self.pos_embed

        # Reshape 为 (B, M*N, D)
        x = x_patches.reshape(B, M * N, D)

        # Masking
        current_mask_ratio = mask_ratio if mask_ratio is not None else self.mask_ratio
        if current_mask_ratio == 0.0:
            x_masked = x
            mask = torch.zeros(B, M * N, device=x.device)
            ids_restore = torch.arange(M * N, device=x.device).unsqueeze(0).expand(B, -1)
            ids_keep = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        else:
            x_masked, mask, ids_restore = self.random_masking(x, current_mask_ratio, noise)
            ids_keep = torch.argsort(ids_restore, dim=1)[:, :int(N * (1 - current_mask_ratio))]

        # RoPE 位置编码
        pos_ids = ids_keep % self.num_patches
        if pos_ids.device != x_masked.device:
            pos_ids = pos_ids.to(x_masked.device, non_blocking=True)

        rope_cos, rope_sin = self.rope_encoder(x_masked, pos_ids)

        # Transformer Blocks
        for blk in self.blocks:
            x_masked = blk(x_masked, rope_cos=rope_cos, rope_sin=rope_sin)

        x_masked = self.norm(x_masked)

        return x_masked, mask, ids_restore, M

    def forward_decoder(self, x, ids_restore, M):
        """
        Decoder 前向传播
        """
        x = self.decoder_embed(x)
        B, _, D_dec = x.shape
        N = self.num_patches
        Total_Tokens = M * N

        # 添加 mask tokens
        mask_tokens = self.mask_token.repeat(B, Total_Tokens - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D_dec))

        # 添加位置编码
        x_patches = x.reshape(B, M, N, D_dec)
        if self.decoder_pos_embed.device != x_patches.device:
            x_patches = x_patches + self.decoder_pos_embed.to(x_patches.device)
        else:
            x_patches = x_patches + self.decoder_pos_embed
        x = x_patches.reshape(B, Total_Tokens, D_dec)

        # RoPE
        patch_pos = torch.arange(N, device=x.device) % N
        patch_pos_expanded = patch_pos.repeat(1, M)
        if patch_pos_expanded.device != x.device:
            patch_pos_expanded = patch_pos_expanded.to(x.device)

        rope_cos, rope_sin = self.rope_decoder(x, patch_pos_expanded)

        # Transformer Blocks
        for blk in self.decoder_blocks:
            x = blk(x, rope_cos=rope_cos, rope_sin=rope_sin)
        x = self.decoder_norm(x)

        x = x.reshape(B, M, N, D_dec)
        return x

    def forward_loss(self, x, pred):
        """
        计算重建损失
        x: (B, M, L) 原始信号
        pred: (B, M, N, patch_size*in_chans) 预测的 patch
        """
        B, M, L = x.shape
        N = self.num_patches

        # 裁剪信号到 patch 对齐的长度
        L_valid = N * self.patch_size
        if L > L_valid:
            x = x[..., :L_valid]

        # 将信号重塑为 patch 形式
        x_patches = x.reshape(B, M, N, self.patch_size)

        # pred 是 (B, M, N, patch_size)
        pred = pred.reshape(B, M, N, self.patch_size)

        # MSE 损失
        loss = (pred - x_patches) ** 2
        loss = loss.mean()

        return loss

    def prepare_tokens(self, x):
        """
        预处理信号 (归一化等)
        x: (B, M, L)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # 归一化
        mean = x.mean(dim=-1, keepdim=True)
        std = torch.clamp(x.std(dim=-1, keepdim=True), min=1e-5)
        x_norm = (x - mean) / std

        x_norm = torch.nan_to_num(x_norm, nan=0.0, posinf=100.0, neginf=-100.0)
        x_norm = torch.clamp(x_norm, min=-100.0, max=100.0)

        return x_norm

    def forward(self, x, channel_ids, mask_ratio=None):
        """
        完整前向传播
        x: (B, M, L) 原始信号
        channel_ids: (B,) tensor, 0=ECG, 1=PPG
        """
        B = x.shape[0]
        current_mask_ratio = mask_ratio if mask_ratio is not None else self.mask_ratio

        # 1. 预处理信号
        x_norm = self.prepare_tokens(x)

        # 2. Encoder
        latent, mask, ids_restore, M = self.forward_encoder(
            x_norm, channel_ids, mask_ratio=current_mask_ratio
        )

        # 3. Decoder
        decoder_features = self.forward_decoder(latent, ids_restore, M)

        # 4. 预测头
        pred = self.decoder_pred(decoder_features)

        # 5. 计算重建损失 (使用未归一化的原始信号作为目标)
        loss = self.forward_loss(x, pred)

        loss_dict = {'loss': loss}

        return loss, loss_dict, pred, x, mask, latent, None
