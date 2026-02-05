import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# ===================================================================
# 1. CWT 模块 (用于 Loss 计算 - 可选)
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
    if x.dim() == 2:
        x = x.unsqueeze(1)
    B_in, M_in, L = x.shape
    x_flat = x.view(B_in * M_in, L)
    scales = torch.arange(num_scales, device=x.device) * step + lowest_scale
    cwt_out = cwt_ricker(x_flat, scales) 
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
    def __init__(self, signal_len=3000, patch_size=4, in_chans=1, embed_dim=768, norm_layer=None):
        super().__init__()
        self.signal_len = signal_len
        self.patch_size = patch_size
        self.num_patches = signal_len // patch_size
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
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

# ===================================================================
# 4. 主模型: Contrastive Learning (SimCLR Style)
# ===================================================================

class CWT_Contrastive_RoPE(nn.Module):
    def __init__(
        self, 
        signal_len=3000, 
        patch_size=4,          
        embed_dim=768, 
        depth=12, 
        num_heads=12,
        mlp_rank_ratio=0.5,    
        norm_layer=nn.LayerNorm,
        projection_dim=128,    # 对比学习投影头维度
        temperature=0.1,       # NT-Xent Loss 温度
    ):
        super().__init__()
        
        self.signal_len = signal_len
        self.patch_size = patch_size
        self.temperature = temperature
        
        # 1. 1D Point/Patch Embed
        self.patch_embed = PointPatchEmbed(
            signal_len=signal_len,
            patch_size=patch_size,
            in_chans=1, 
            embed_dim=embed_dim,
            norm_layer=norm_layer
        )
        self.num_patches = self.patch_embed.num_patches

        # 2. Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # RoPE
        self.rope_encoder = RotaryEmbedding(dim=embed_dim // num_heads)

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

        # 4. Projection Head (MLP) for Contrastive Learning
        # Standard SimCLR: Linear -> BN -> ReLU -> Linear
        # 这里用 LayerNorm 替代 BN 以适应 Transformer
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, projection_dim)
        )

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.pos_embed, std=.02)
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

    def forward_encoder(self, x):
        # x: (B*M, 1, L) -> (B*M, N, D)
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        
        # Append CLS Token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # RoPE IDs (No masking in Contrastive Learning usually)
        B, N_curr, _ = x.shape
        # ids_keep = [0, 1, ..., N-1]
        ids_keep = torch.arange(self.num_patches, device=x.device).unsqueeze(0).expand(B, -1)
        
        cls_pos = torch.zeros(B, 1, device=x.device, dtype=torch.long)
        patch_pos = ids_keep + 1
        pos_ids = torch.cat((cls_pos, patch_pos), dim=1)
        
        # Transformer
        rope_cos, rope_sin = self.rope_encoder(x, pos_ids)
        for blk in self.blocks:
            x = blk(x, rope_cos=rope_cos, rope_sin=rope_sin)
        x = self.norm(x)
        
        # Return CLS token representation
        return x[:, 0]

    def forward_loss_ntxent(self, z_i, z_j):
        """
        NT-Xent Loss (Normalized Temperature-scaled Cross Entropy Loss)
        z_i, z_j: (B, D) projections of two augmented views
        """
        batch_size = z_i.shape[0]
        
        # Normalize
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Similarity matrix
        # (2B, D)
        z = torch.cat([z_i, z_j], dim=0)
        sim_matrix = torch.matmul(z, z.T) / self.temperature
        
        # Create labels
        # Positive pairs are (i, i+B) and (i+B, i)
        labels = torch.cat([
            torch.arange(batch_size, device=z.device) + batch_size,
            torch.arange(batch_size, device=z.device)
        ], dim=0)
        
        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim_matrix.masked_fill_(mask, -9e15)
        
        loss = F.cross_entropy(sim_matrix, labels)
        return loss

    def forward(self, x1, x2=None):
        """
        Training: x1, x2 are two augmented views of the same batch.
        Inference: x1 is input, x2 is None.
        """
        # Inference / Fine-tuning Mode
        if x2 is None:
            if x1.dim() == 2: x1 = x1.unsqueeze(1)
            B, M, L = x1.shape
            x_flat = x1.view(B * M, 1, L)
            
            # Instance Norm
            mean = x_flat.mean(dim=2, keepdim=True)
            std = x_flat.std(dim=2, keepdim=True) + 1e-5
            x_norm = (x_flat - mean) / std
            
            feat = self.forward_encoder(x_norm)
            return feat # Return representation directly

        # Training Mode (Contrastive)
        # x1, x2: (B, M, L)
        B, M, L = x1.shape
        
        # Flatten & Norm
        x1_flat = x1.view(B * M, 1, L)
        x2_flat = x2.view(B * M, 1, L)
        
        def normalize(x):
            mean = x.mean(dim=2, keepdim=True)
            std = x.std(dim=2, keepdim=True) + 1e-5
            return (x - mean) / std
            
        x1_norm = normalize(x1_flat)
        x2_norm = normalize(x2_flat)
        
        # Encoder
        h1 = self.forward_encoder(x1_norm) # (BM, D)
        h2 = self.forward_encoder(x2_norm) # (BM, D)
        
        # Projection
        z1 = self.projection_head(h1) # (BM, P)
        z2 = self.projection_head(h2) # (BM, P)
        
        loss = self.forward_loss_ntxent(z1, z2)
        
        return loss, z1, z2
