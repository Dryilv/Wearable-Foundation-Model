import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import timm  # 【新增】依赖库

# -------------------------------------------------------------------
# 1. CWT 模块 (保持不变)
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# 2. 基础组件 (Decoder Block 仍需保留，但 Encoder Block 使用 timm)
# -------------------------------------------------------------------
class DecoderBlock(nn.Module):
    # 这是一个标准的 Transformer Block，用于 Decoder (我们不加载 decoder 的预训练权重)
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.norm2 = norm_layer(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

# -------------------------------------------------------------------
# 3. 辅助函数：位置编码插值
# -------------------------------------------------------------------
def interpolate_pos_embed(model, new_grid_size):
    """
    将预训练的 Position Embedding 插值到新的 Grid Size
    model: timm vision transformer model
    new_grid_size: tuple (H, W), e.g., (16, 60)
    """
    # 1. 获取预训练的 pos_embed (1, N+1, D)
    pos_embed_checkpoint = model.pos_embed
    embedding_size = pos_embed_checkpoint.shape[-1]
    
    # 【修复关键点】：直接从 pos_embed 张量形状获取 token 数量
    # 不要使用 model.patch_embed.num_patches，因为那个属性可能已经被修改了
    num_tokens = pos_embed_checkpoint.shape[1]
    
    # 假设有 1 个 CLS token (标准 ViT 都是 1 个，DeiT 可能是 2 个)
    # timm 模型通常有 num_prefix_tokens 属性，如果没有则默认为 1
    num_extra_tokens = getattr(model, 'num_prefix_tokens', 1)
    
    # 计算原始的 patch 数量
    num_patches_orig = num_tokens - num_extra_tokens
    
    # 获取原始 Grid Size (假设是方形，例如 sqrt(196) = 14)
    orig_size = int(math.sqrt(num_patches_orig))
    
    # 简单的校验
    if orig_size * orig_size != num_patches_orig:
        print(f"Warning: Original patch count {num_patches_orig} is not a perfect square. Interpolation might be wrong.")

    # 2. 分离 CLS token 和 Patch tokens
    cls_token = pos_embed_checkpoint[:, :num_extra_tokens]
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:] # (1, 196, 768)
    
    # 3. Reshape 为 2D 图像格式 (B, C, H, W) 以进行插值
    # 原始形状: (1, N, D) -> (1, H, W, D) -> (1, D, H, W)
    # 这里使用 orig_size (14) 而不是错误的 30
    pos_tokens = pos_tokens.reshape(1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
    
    # 4. 执行双三次插值 (Bicubic Interpolation)
    new_pos_tokens = F.interpolate(
        pos_tokens, 
        size=new_grid_size, # (16, 60)
        mode='bicubic', 
        align_corners=False
    )
    
    # 5. 还原形状 (1, D, H_new, W_new) -> (1, H_new, W_new, D) -> (1, H_new*W_new, D)
    new_pos_tokens = new_pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    
    # 6. 拼接回 CLS token
    new_pos_embed = torch.cat((cls_token, new_pos_tokens), dim=1)
    
    # 7. 赋值给模型
    model.pos_embed = nn.Parameter(new_pos_embed)
    print(f"Positional Embeddings interpolated from {orig_size}x{orig_size} to {new_grid_size}.")

# -------------------------------------------------------------------
# 4. 主模型: Pretrained CWT-MAE
# -------------------------------------------------------------------
class CWT_MAE_Pretrained(nn.Module):
    def __init__(
        self, 
        signal_len=3000, 
        cwt_scales=64,
        patch_size_time=50,
        patch_size_freq=4,
        model_name='vit_base_patch16_224', 
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mask_ratio=0.6,
        norm_layer=nn.LayerNorm,
        time_loss_weight=1.0
    ):
        super().__init__()
        
        self.mask_ratio = mask_ratio
        self.cwt_scales = cwt_scales
        self.time_loss_weight = time_loss_weight
        self.patch_size_time = patch_size_time
        self.patch_size_freq = patch_size_freq
        
        # 计算 Grid Size: (16, 60)
        self.grid_size = (cwt_scales // patch_size_freq, signal_len // patch_size_time)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        # --- 1. Encoder (借助 timm 加载预训练) ---
        print(f"Loading pretrained weights from {model_name}...")
        self.encoder = timm.create_model(model_name, pretrained=True)
        embed_dim = self.encoder.embed_dim
        
        # 【关键修改 A】: 替换 Patch Embedding 层
        self.encoder.patch_embed.proj = nn.Conv2d(
            in_channels=3, 
            out_channels=embed_dim, 
            kernel_size=(patch_size_freq, patch_size_time), 
            stride=(patch_size_freq, patch_size_time)
        )
        
        # 【修复点】：更新 img_size 以通过 timm 的断言检查
        self.encoder.patch_embed.img_size = (cwt_scales, signal_len)
        
        self.encoder.patch_embed.patch_size = (patch_size_freq, patch_size_time)
        self.encoder.patch_embed.grid_size = self.grid_size
        self.encoder.patch_embed.num_patches = self.num_patches
        
        # 【关键修改 B】: 插值 Positional Embedding
        # 注意：请确保使用我上一条回答中修复过的 interpolate_pos_embed 函数
        interpolate_pos_embed(self.encoder, self.grid_size)
        
        # 移除分类头
        self.encoder.head = nn.Identity()
        self.encoder.fc_norm = nn.Identity()
        
        # --- 2. Decoder ---
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim))
        
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(decoder_embed_dim, num_heads=decoder_num_heads, norm_layer=norm_layer) 
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        
        # --- 3. Prediction Heads ---
        self.patch_pixels = 3 * patch_size_freq * patch_size_time
        self.decoder_pred_spec = nn.Linear(decoder_embed_dim, self.patch_pixels, bias=True)

        self.time_reducer = nn.Sequential(
            nn.Conv2d(decoder_embed_dim, decoder_embed_dim, kernel_size=(self.grid_size[0], 1)),
            nn.GELU()
        )
        self.time_pred = nn.Linear(decoder_embed_dim, patch_size_time, bias=True)

        self.initialize_decoder_weights()

    def initialize_decoder_weights(self):
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed, std=.02)
        # 初始化 Patch Embed (因为是新层)
        torch.nn.init.xavier_uniform_(self.encoder.patch_embed.proj.weight)
        self.apply(self._init_decoder_submodules)

    def _init_decoder_submodules(self, m):
        # 只初始化 Decoder 相关的 Linear，避免重置 Encoder
        # 这里简单处理，因为 Encoder 的 Linear 已经有权重了，xavier_uniform 再次初始化也没大问题(如果是微调的话最好跳过)
        # 但为了安全，我们通常依赖 PyTorch 默认初始化或者只针对 decoder_blocks 遍历
        pass 

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
        return x_masked, mask, ids_restore

    def forward_encoder(self, x):
        # x: (B, 3, 64, 3000)
        # 1. Patch Embed (现在 img_size 匹配了，不会报错)
        x = self.encoder.patch_embed(x) 
        
        # 2. Add Pos Embed
        pos_embed = self.encoder.pos_embed[:, 1:, :]
        x = x + pos_embed
        
        # 3. Masking
        x_masked, mask, ids_restore = self.random_masking(x, self.mask_ratio)
        
        # 4. Append CLS Token
        cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x_masked), dim=1)
        
        # 5. Transformer Blocks
        x = self.encoder.blocks(x)
        x = self.encoder.norm(x)
        
        return x, mask, ids_restore

    # ... (forward_decoder, forward_loss_spec, forward_loss_time, forward 保持不变) ...
    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], self.num_patches + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        return x[:, 1:, :]

    def forward_loss_spec(self, imgs, pred, mask):
        p_h, p_w = self.patch_size_freq, self.patch_size_time
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

# 测试代码
if __name__ == "__main__":
    # 模拟输入 (Batch=2, Length=3000)
    x = torch.randn(2, 3000).bfloat16() # 支持 BF16
    
    # 实例化模型
    model = CWT_MAE_Pretrained(
        signal_len=3000, 
        cwt_scales=64,
        patch_size_time=50, 
        patch_size_freq=4
    ).bfloat16()
    
    # 前向传播
    loss, _, _, _ = model(x)
    print(f"Loss: {loss.item()}")