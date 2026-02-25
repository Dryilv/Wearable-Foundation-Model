# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# 导入你自己写的模块
from model import MOMENTPretrain  # 导入模型
from dataset import MOMENTPhysioDataset, DataSplitter  # 导入新数据管道

from transformers import get_cosine_schedule_with_warmup  # 导入 Scheduler

# =================配置参数=================
DATA_PATH = "/home/bml/storage/mnt/v-044d0fb740b04ad3/org/WFM/vit16trans/CWT_MAE_v3/train_index_cleaned.json"  # 现在是 JSON 索引文件
MODEL_SIZE = 'base'  # 设定模型大小为 base
BATCH_SIZE = 768  # 你的 4090 单次能吃的量 (根据显存调整)
STRIDE = 256 # 滑动窗口步长 (50% 重叠)
Accumulation_Steps = 1  # 64 * 32 = 2048 (模拟论文的大 Batch)
LEARNING_RATE = 1e-4
WARMUP_STEPS = 1000 # 预热步数
EPOCHS = 10
SAVE_DIR = "checkpoints"


# =========================================

def train():
    # 1. 准备分布式环境
    # Windows 下 nccl 不可用，建议使用 gloo
    backend = "gloo" if os.name == "nt" else "nccl"
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    # 只有主进程打印信息
    is_master = (dist.get_rank() == 0)
    
    if is_master:
        os.makedirs(SAVE_DIR, exist_ok=True)
        print(f"正在分布式训练，World Size: {dist.get_world_size()}")
        print(f"正在使用设备: {device}")

    # 2. 加载数据
    if is_master:
        print(f"正在从索引加载数据: {DATA_PATH}...")
    
    # 使用 DataSplitter 进行切分
    splitter = DataSplitter(DATA_PATH, split_ratio=0.1, seed=42)
    train_indices, val_indices = splitter.get_split()

    train_dataset = MOMENTPhysioDataset(DATA_PATH, indices=train_indices, mode='train', stride=STRIDE)
    # val_dataset = MOMENTPhysioDataset(DATA_PATH, indices=val_indices, mode='val', stride=STRIDE) # 如果需要验证集
    
    # 分布式采样器
    sampler = DistributedSampler(train_dataset, shuffle=True)
    
    # 注意：分布式下 batch_size 是每个 GPU 的大小
    dataloader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=sampler,
        num_workers=4, 
        pin_memory=True
    )
    
    if is_master:
        print(f"数据加载完毕，训练集样本数: {len(train_dataset)}")

    # 3. 初始化模型
    if is_master:
        print(f"正在初始化模型 (Size: {MODEL_SIZE})...")
    model = MOMENTPretrain(config_size=MODEL_SIZE).to(device)
    
    # 包装为分布式模型
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # 4. 优化器、Scheduler 与混合精度
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    
    # 计算总步数用于 Scheduler
    total_steps = len(dataloader) * EPOCHS // Accumulation_Steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=WARMUP_STEPS, 
        num_training_steps=total_steps
    )
    
    scaler = GradScaler()  # 4090 必开神器

    # 5. 开始训练循环
    model.train()
    for epoch in range(EPOCHS):
        # 告诉 sampler 当前 epoch，保证 shuffle 随机性
        sampler.set_epoch(epoch)
        
        total_loss = 0
        if is_master:
            progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}/{EPOCHS}")
        else:
            progress_bar = enumerate(dataloader)

        optimizer.zero_grad()

        for step, (patches, mask) in progress_bar:
            patches = patches.to(device)  # [B, 64, 8]
            mask = mask.to(device)  # [B, 64]

            # 混合精度前向传播
            with autocast():
                # Model 返回 (loss, pred_patches)
                loss, pred_patches = model(patches, mask)

                # 梯度累积标准化
                loss_scaled = loss / Accumulation_Steps

            # 反向传播
            scaler.scale(loss_scaled).backward()

            # 梯度累积：每 Accumulation_Steps 步才更新一次权重
            if (step + 1) % Accumulation_Steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()  # 更新学习率

            # 记录日志
            current_loss = loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            total_loss += current_loss
            if is_master:
                progress_bar.set_postfix({
                    'loss': f"{current_loss:.4f}",
                    'lr': f"{current_lr:.2e}"
                })

        # 每个 Epoch 结束保存模型
        avg_loss = total_loss / len(dataloader)
        
        # 汇总所有卡的 Loss (可选，这里只打印主进程的)
        if is_master:
            print(f"Epoch {epoch + 1} 结束，平均 Loss: {avg_loss:.6f}")

            save_path = os.path.join(SAVE_DIR, f"moment_base_epoch_{epoch + 1}.pth")
            # DDP 包装后保存 model.module
            torch.save(model.module.state_dict(), save_path)
            print(f"模型已保存至: {save_path}")

    # 6. 销毁进程组
    dist.destroy_process_group()


if __name__ == "__main__":
    train()