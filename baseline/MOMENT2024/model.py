import torch
from torch import nn
from transformers import T5Config, T5EncoderModel


class MOMENTPretrain(nn.Module):
    def __init__(self, config_size='base', seq_len=512, patch_len=8):
        super().__init__()

        # ====================================================
        # 1. 配置模型规模 (Small / Base / Large)
        # ====================================================
        if config_size == 'small':
            d_model = 512
            n_layers = 6
            n_heads = 8
            d_ff = 2048
        elif config_size == 'base':
            d_model = 768
            n_layers = 12
            n_heads = 12
            d_ff = 3072
        elif config_size == 'large':
            d_model = 1024
            n_layers = 24
            n_heads = 16
            d_ff = 4096
        else:
            raise ValueError("config_size must be 'small', 'base', or 'large'")

        # ====================================================
        # 2. 定义核心组件
        # ====================================================
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.n_patches = seq_len // patch_len  # 512/8 = 64

        # A. Patch Embedding 层
        self.patch_embedding = nn.Linear(patch_len, d_model)

        # B. Mask Token
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))

        # C. 骨干网络 (T5 Encoder)
        config = T5Config(
            d_model=d_model,
            num_layers=n_layers,
            num_heads=n_heads,
            d_kv=d_model // n_heads,
            d_ff=d_ff,
            dropout_rate=0.1,
            vocab_size=1,
            use_cache=False
        )
        self.encoder = T5EncoderModel(config)

        # D. 重建头
        self.head = nn.Linear(d_model, patch_len)

        # E. 损失函数 (定义在这里更规范)
        self.loss_fct = nn.MSELoss()

    def forward(self, x, mask=None):
        """
        输入:
        - x: [Batch, N_Patches, Patch_Len]
        - mask: [Batch, N_Patches] (True 表示被掩盖)
        """
        batch_size, n_patches, _ = x.shape

        # 1. 投影: [B, N, P] -> [B, N, D]
        x_embed = self.patch_embedding(x)

        # 2. 应用掩码
        mask_tokens = self.mask_token.expand(batch_size, n_patches, -1)

        # 如果提供了 mask，则应用替换逻辑
        if mask is not None:
            # mask 扩展为 [B, N, D]
            mask_expanded = mask.unsqueeze(-1).expand_as(x_embed)
            # 核心替换: Mask 为 True 的地方用 mask_token
            input_embeds = torch.where(mask_expanded, mask_tokens, x_embed)
        else:
            input_embeds = x_embed

        # 3. T5 Encoder
        outputs = self.encoder(inputs_embeds=input_embeds)
        hidden_states = outputs.last_hidden_state  # [B, N, D]

        # 4. 重建输出
        pred_patches = self.head(hidden_states)  # [B, N, P]

        # 5. 计算 Loss (只计算 Mask 部分)
        loss = torch.tensor(0.0, device=x.device)  # 初始化 loss
        if mask is not None:
            # 利用 PyTorch 的布尔索引，直接取出被 Mask 的部分
            # target 形状变更为: [Total_Masked_Count, Patch_Len]
            target = x[mask]
            pred = pred_patches[mask]

            # 计算 MSE
            loss = self.loss_fct(pred, target)

        # 返回元组: (Loss, 预测结果)
        # 这符合 HuggingFace 和常用训练习惯
        return loss, pred_patches


# # ====================================================
# # 测试代码 (修正了元组解包问题)
# # ====================================================
# if __name__ == "__main__":
#     # 1. 实例化
#     model = MOMENTPretrain(config_size='base')
#     print(f"✅ 模型初始化成功 | 参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
#
#     # 2. 模拟数据
#     batch_size = 32
#     dummy_input = torch.randn(batch_size, 64, 8)
#
#     # 3. 模拟掩码 (30% True)
#     dummy_mask = torch.rand(batch_size, 64) < 0.3
#
#     # 4. 前向传播
#     # 【关键修改】这里必须用两个变量接收返回值！
#     loss, output = model(dummy_input, dummy_mask)
#
#     print(f"\n输入形状: {dummy_input.shape}")
#     print(f"输出形状: {output.shape}")
#     print(f"Loss 值: {loss.item()}")
#
#     # 5. 检查
#     assert output.shape == dummy_input.shape, "❌ 输入输出形状不匹配！"
#     assert not torch.isnan(loss), "❌ Loss 为 NaN，检查数据或归一化！"
#
#     print("\n🎉 架构验证通过！")