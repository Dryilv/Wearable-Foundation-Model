import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# 导入纯净版 PatchTST
from patchtst_model import PatchTST_Pretrain

# ==========================================
# 1. 配置参数 (Configurations)
# ==========================================
DATA_PATH = "real_ecg_ppg_data.npy"  # 替换为你的.npy真实数据文件名
SAVE_DIR = "checkpoints_patchtst"  # 模型权重保存路径

SEQ_LEN = 512
PATCH_LEN = 8  # 切块大小
STRIDE = 8  # 步长 (与你的 CWT-MAE 保持一致)
IN_CHANNELS = 2  # ECG 和 PPG 两个通道

# 训练超参数
BATCH_SIZE = 128  # 物理 Batch Size
ACCUMULATION_STEPS = 16  # 梯度累积，逻辑 Batch Size = 128 * 16 = 2048
LR = 1e-4
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 2. 极速内存映射 Dataset
# ==========================================
class PatchTSTDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        # 使用 mmap_mode='r'，20GB 数据瞬间加载，不占内存
        self.data = np.load(data_path, mmap_mode='r')
        self.total_samples = self.data.shape[0]
        print(f"✅ 成功加载数据集，总样本数: {self.total_samples}")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # 读取单条数据，形状为 (2, 512) -> (Channels, Seq_Len)
        sample = self.data[idx].copy()  # copy 防止修改只读的 mmap

        # 1. 独立通道归一化 (Instance Normalization / RevIN 风格)
        # PatchTST 对数据的尺度非常敏感，必须在切块前进行 Z-score 归一化
        mean = sample.mean(axis=1, keepdims=True)
        std = sample.std(axis=1, keepdims=True)
        std = np.clip(std, a_min=1e-5, a_max=None)
        sample = (sample - mean) / std

        # 2. 转换形状以适配 PatchTST 模型
        # PatchTST 要求的输入是 [Seq_Len, Channels]，所以需要转置
        sample = np.transpose(sample, (1, 0))  # 变成 (512, 2)

        return torch.tensor(sample, dtype=torch.float32)


# ==========================================
# 3. 核心训练循环 (Training Loop)
# ==========================================
def train():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 初始化 Dataset 和 DataLoader
    dataset = PatchTSTDataset(DATA_PATH)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,  # 开启多进程加速读取
        pin_memory=True
    )

    # 初始化模型 (参数量与你的 CWT-MAE 对齐)
    model = PatchTST_Pretrain(
        seq_len=SEQ_LEN,
        patch_len=PATCH_LEN,
        stride=STRIDE,
        in_channels=IN_CHANNELS,
        d_model=768,  # 隐藏层维度
        n_heads=12,  # 注意力头数
        e_layers=12,  # Transformer 层数
        mask_ratio=0.75  # 掩住 75% 的数据让它猜
    ).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = GradScaler()  # 混合精度加速器

    print(f"🚀 开始 PatchTST 预训练 (设备: {DEVICE})...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        # 使用 tqdm 打印进度条
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for step, x in pbar:
            x = x.to(DEVICE)  # x shape: [Batch, 512, 2]

            # 开启自动混合精度 (AMP)，显存减半，速度翻倍
            with autocast():
                # 前向传播，PatchTST_Pretrain 返回的是 (loss, 预测patch, 真实patch, mask)
                loss, _, _, _ = model(x)

                # 梯度累积：对 loss 进行缩放
                loss = loss / ACCUMULATION_STEPS

            # 反向传播
            scaler.scale(loss).backward()

            # 当累积到指定的步数时，更新一次权重
            if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == len(dataloader):
                scaler.unscale_(optimizer)
                # 梯度裁剪，防止 Transformer 训练初期梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # 记录 Loss (由于之前除了 ACCUMULATION_STEPS，这里乘回来以便显示真实的 Loss)
            current_loss = loss.item() * ACCUMULATION_STEPS
            total_loss += current_loss

            # 实时更新进度条上的 Loss 显示
            pbar.set_postfix({"Loss": f"{current_loss:.4f}"})

        # 打印 Epoch 统计信息
        avg_loss = total_loss / len(dataloader)
        print(f"✅ Epoch [{epoch + 1}/{EPOCHS}] Average Loss: {avg_loss:.4f}")

        # 保存 Checkpoint
        save_path = os.path.join(SAVE_DIR, f"patchtst_pretrain_epoch_{epoch + 1}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, save_path)
        print(f"💾 模型已保存至 {save_path}\n")


if __name__ == "__main__":
    # 提醒：运行前确保你的 real_ecg_ppg_data.npy 路径正确
    train()