# patchtst_model.py
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """传统的绝对位置编码"""

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [Batch*Channels, Num_Patches, d_model]
        return x + self.pe[:x.size(1), :].unsqueeze(0)


class PatchTST_Pretrain(nn.Module):
    def __init__(
            self,
            seq_len=3000,  # 你的序列长度
            patch_len=50,  # 对应你模型的 patch_size_time
            stride=50,  # 步长等于 patch_len，无重叠 (与你的 MAE 一致)
            in_channels=3,  # 输入通道数 (比如 ECG, PPG)
            d_model=768,  # 对齐你的 embed_dim
            n_heads=12,  # 对齐你的 num_heads
            e_layers=12,  # 对齐你的 depth
            d_ff=3072,  # Transformer 前馈网络维度 (通常是 4*d_model)
            dropout=0.1,
            mask_ratio=0.75  # 掩码比例，与你保持一致
    ):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.in_channels = in_channels
        self.d_model = d_model
        self.mask_ratio = mask_ratio

        # 计算 Patch 数量
        self.patch_num = int((seq_len - patch_len) / stride + 1)

        # 1. Patch Embedding (仅时域 Linear 投影，这就是和你的核心区别之一)
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_embedding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

        # 2. Transformer Encoder (纯净版 Vanilla Transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True  # 注意这里用 batch_first
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)

        # 3. 预训练重建头 (Reconstruction Head)
        # 把 d_model 映射回 patch_len 的原始波形数值
        self.head = nn.Linear(d_model, patch_len)

    def random_masking(self, x, mask_ratio):
        """生成随机掩码"""
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))

        noise = torch.rand(B, N, device=x.device)  # 随机噪声
        ids_shuffle = torch.argsort(noise, dim=1)  # 升序排列，获取打乱的索引
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # 恢复索引

        # 保持未被 Mask 的部分
        ids_keep = ids_shuffle[:, :len_keep]
        x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # 生成 Mask 矩阵 (1 表示被遮挡，0 表示保留)
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # PatchTST 预训练中通常用全 0 向量填充被 mask 的位置 (或者用 learnable token)
        # 这里用全 0 填充以保持轻量化
        mask_tokens = torch.zeros(B, N - len_keep, D, device=x.device)
        x_masked = torch.cat([x_kept, mask_tokens], dim=1)
        x_masked = torch.gather(x_masked, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))

        return x_masked, mask

    def forward(self, x):
        # 输入 x: [Batch, Seq_Len, Channels]
        # (注意，和你的模型直接吃 3D 甚至 2D 不一样，这里吃纯 1D 序列)
        B, L, M = x.shape

        # ==============================================================
        # 【核心精髓】：Channel Independence (通道独立)
        # 把 Batch 和 Channels 融合成同一个维度，让它们共享权重但互不干扰
        # [Batch, Seq_Len, Channels] -> [Batch * Channels, Seq_Len, 1]
        # ==============================================================
        x = x.permute(0, 2, 1).contiguous().view(B * M, L, 1)

        # 1. Patching 过程 (切块)
        # 使用 unfold 提取块: [B*M, Patch_Num, Patch_Len]
        x = x.unfold(dimension=1, size=self.patch_len, step=self.stride).squeeze(2)

        # 保存原始的 Patch 用于计算 Loss
        target_patches = x.clone()

        # 2. Embedding
        x = self.value_embedding(x)  # -> [B*M, Patch_Num, d_model]
        x = self.position_embedding(x)
        x = self.dropout(x)

        # 3. Masking
        x_masked, mask = self.random_masking(x, self.mask_ratio)

        # 4. Transformer 编码
        # 注意: 这里的 Encoder 接收的是 [B*M, Patch_Num, d_model]
        # 意味着 ECG 和 PPG 是被当作两个完全不认识的序列来处理的！
        enc_out = self.encoder(x_masked)

        # 5. 重建输出
        # -> [B*M, Patch_Num, Patch_Len]
        pred_patches = self.head(enc_out)

        # 6. 计算 MSE Loss (仅计算被 Mask 掉的部分)
        loss = (pred_patches - target_patches) ** 2
        loss = loss.mean(dim=-1)  # 对每个 Patch 内的数值求均值
        loss = (loss * mask).sum() / (mask.sum() + 1e-6)  # 仅保留被 mask 掉的 patch 的 loss

        # 返回 Loss 和掩码信息 (后续可以把 pred_patches 拼回去做可视化)
        return loss, pred_patches, target_patches, mask