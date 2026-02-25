# patchtst_model.py
import torch
import torch.nn as nn
import math

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        Reversible Instance Normalization
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
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + 1e-10)
            x = x * self.stdev + self.mean
            return x

class PositionalEncoding(nn.Module):
    """
    Learnable Positional Encoding (as per PatchTST paper)
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Paper says: "W_pos is the learnable positional embedding"
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        # x shape: [Batch*Channels, Num_Patches, d_model]
        return x + self.pe[:, :x.size(1), :]

class PatchTST_Pretrain(nn.Module):
    def __init__(
            self,
            seq_len=3000,
            patch_len=50,
            stride=50,
            in_channels=3,
            d_model=768,
            n_heads=12,
            e_layers=12,
            d_ff=3072,
            dropout=0.1,
            mask_ratio=0.75,
            use_revin=True
    ):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.in_channels = in_channels
        self.d_model = d_model
        self.mask_ratio = mask_ratio
        self.use_revin = use_revin

        # RevIN Module
        if self.use_revin:
            self.revin = RevIN(in_channels, affine=True)

        # Patch Num
        self.patch_num = int((seq_len - patch_len) / stride + 1)

        # 1. Patch Embedding
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_embedding = PositionalEncoding(d_model, max_len=self.patch_num)
        self.dropout = nn.Dropout(dropout)

        # 2. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True # Usually better for deep transformers
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)

        # 3. Reconstruction Head
        self.head = nn.Linear(d_model, patch_len)

    def random_masking(self, x, mask_ratio):
        """生成随机掩码"""
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))

        noise = torch.rand(B, N, device=x.device)  # 随机噪声
        ids_shuffle = torch.argsort(noise, dim=1)  # 升序排列
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # 恢复索引

        # 保持未被 Mask 的部分
        ids_keep = ids_shuffle[:, :len_keep]
        x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # 生成 Mask 矩阵 (1 表示被遮挡，0 表示保留)
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # PatchTST: Zero-fill masked patches
        mask_tokens = torch.zeros(B, N - len_keep, D, device=x.device)
        x_masked = torch.cat([x_kept, mask_tokens], dim=1)
        x_masked = torch.gather(x_masked, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))

        return x_masked, mask

    def forward(self, x):
        # Input x: [Batch, Seq_Len, Channels]
        # or [Batch, Channels, Seq_Len] if dataset outputs that?
        # Standard convention: [Batch, Seq_Len, Channels] if channels are last.
        # But wait, my dataset collate_fn outputs (B, C, L).
        # In train.py: batch = batch.permute(0, 2, 1) -> (B, L, C)
        # So here x is (B, L, C).
        
        B, L, M = x.shape

        # 1. RevIN Normalization (Per-Channel)
        if self.use_revin:
            # RevIN expects (B, L, M) or (B, M, L)?
            # My RevIN implementation uses dim2reduce=tuple(range(1, x.ndim-1))
            # If x is (B, L, M), dim2reduce is (1,).
            # So mean is (B, 1, M).
            # (B, L, M) - (B, 1, M) -> Works.
            x = self.revin(x, 'norm')

        # ==============================================================
        # Channel Independence
        # [Batch, Seq_Len, Channels] -> [Batch * Channels, Seq_Len, 1]
        # ==============================================================
        x = x.permute(0, 2, 1).contiguous().view(B * M, L, 1)

        # 2. Patching
        # unfold: (B*M, L, 1) -> (B*M, N, P)
        x = x.unfold(dimension=1, size=self.patch_len, step=self.stride).squeeze(2)

        # Save Target
        target_patches = x.clone()

        # 3. Embedding
        x = self.value_embedding(x)  # -> [B*M, N, d_model]
        x = self.position_embedding(x)
        x = self.dropout(x)

        # 4. Masking
        x_masked, mask = self.random_masking(x, self.mask_ratio)

        # 5. Transformer Encoder
        enc_out = self.encoder(x_masked)

        # 6. Reconstruction
        pred_patches = self.head(enc_out)

        # 7. Loss
        # Loss is calculated on normalized patches (if RevIN used)
        loss = (pred_patches - target_patches) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / (mask.sum() + 1e-6)

        return loss, pred_patches, target_patches, mask
