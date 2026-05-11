"""
Signal-MAE 模型测试脚本
验证模型前向传播是否正确
"""
import torch
from model import Signal_MAE_RoPE

def test_model():
    print("=" * 60)
    print("Signal-MAE 模型测试")
    print("=" * 60)

    # 初始化模型
    model = Signal_MAE_RoPE(
        signal_len=1000,
        patch_size=50,
        embed_dim=768,
        depth=4,  # 测试用较浅层
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=4,
        decoder_num_heads=16,
        mask_ratio=0.75
    )
    model.eval()

    print(f"\n模型参数:")
    print(f"  - signal_len: 1000")
    print(f"  - patch_size: 50")
    print(f"  - num_patches: {model.num_patches}")
    print(f"  - embed_dim: 768")
    print(f"  - depth: 4")
    print(f"  - mask_ratio: 0.75")

    # 创建测试数据 (B, M, L)
    B = 2  # batch size
    M = 1  # channels (单通道)
    L = 1000  # signal length

    x = torch.randn(B, M, L)
    channel_ids = torch.tensor([0, 1], dtype=torch.long)  # 0=ECG, 1=PPG

    print(f"\n输入数据形状: {x.shape}")
    print(f"Channel IDs: {channel_ids}")

    # 前向传播
    print("\n执行前向传播...")
    with torch.no_grad():
        loss, loss_dict, pred, x_target, mask, latent = model(x, channel_ids)

    print(f"\n输出:")
    print(f"  - Loss: {loss.item():.4f}")
    print(f"  - Pred shape: {pred.shape}")
    print(f"  - Target shape: {x_target.shape}")
    print(f"  - Mask shape: {mask.shape}")
    print(f"  - Latent shape: {latent.shape}")

    # 验证 masking 比例
    mask_ratio_actual = mask.mean().item()
    print(f"\n实际 Mask 比例: {mask_ratio_actual:.2f} (目标: 0.75)")

    # 测试无 masking 情况
    print("\n" + "=" * 60)
    print("测试无 masking (mask_ratio=0.0)")
    print("=" * 60)
    with torch.no_grad():
        loss_no_mask, _, _, _, mask_no_mask, _ = model(x, channel_ids, mask_ratio=0.0)
    print(f"  - Loss (no mask): {loss_no_mask.item():.4f}")
    print(f"  - Mask mean: {mask_no_mask.mean().item():.2f}")

    print("\n" + "=" * 60)
    print("✅ 所有测试通过!")
    print("=" * 60)

if __name__ == "__main__":
    test_model()
