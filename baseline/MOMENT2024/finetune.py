import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import os
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report

from finetune_model import MOMENTClassification
from finetune_dataset import DownstreamDataset

# ================= 配置参数 =================
DATA_DIR = "/home/bml/storage/mnt/v-044d0fb740b04ad3/org/WFM/wearable_FM/downstream_data/ppgguanxinbing/sample_for_downstream"  # pkl 文件所在目录
SPLIT_FILE = "/home/bml/storage/mnt/v-044d0fb740b04ad3/org/WFM/wearable_FM/downstream_data/ppgguanxinbing/train_test_split.json"
PRETRAINED_PATH = "./checkpoints/moment_base_epoch_10.pth"
SAVE_DIR = "finetune_checkpoints"

MODEL_SIZE = 'base'
SEQ_LEN = 512
PATCH_LEN = 8
NUM_CLASSES = 2 # 根据实际分类任务修改

BATCH_SIZE = 128
LEARNING_RATE = 5e-5
EPOCHS = 20

def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # torchrun / distributed launch
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        # 非分布式模式
        return False, 0, 1, 0

    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    return True, rank, world_size, local_rank

def cleanup_distributed():
    dist.destroy_process_group()

def evaluate(model, dataloader, device, is_distributed=False, rank=0, epoch=0, epochs=0):
    model.eval()
    total_loss = 0
    all_probs = []
    all_labels = []
    
    criterion = nn.CrossEntropyLoss()
    
    # 只在主进程显示测试进度条
    if rank == 0:
        progress_bar = tqdm(dataloader, desc=f"Test  {epoch+1}/{epochs}")
    else:
        progress_bar = dataloader

    with torch.no_grad():
        for patches, labels in progress_bar:
            patches, labels = patches.to(device), labels.to(device)
            logits = model(patches)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            # 获取概率 (对于二分类，取 index 1 的概率)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            if rank == 0:
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    if is_distributed:
        # 收集所有进程的结果
        world_size = dist.get_world_size()
        
        # 收集 Loss
        avg_loss = torch.tensor(total_loss / len(dataloader)).to(device)
        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss.item() / world_size

        gathered_probs = [None] * world_size
        gathered_labels = [None] * world_size
        dist.all_gather_object(gathered_probs, all_probs)
        dist.all_gather_object(gathered_labels, all_labels)
        
        if dist.get_rank() == 0:
            all_probs = np.concatenate(gathered_probs, axis=0)
            all_labels = np.concatenate(gathered_labels, axis=0)
            return avg_loss, all_probs, all_labels
        else:
            return None, None, None
    
    return total_loss / len(dataloader), all_probs, all_labels

def find_best_threshold(labels, probs):
    """
    寻找最佳 Macro F1 对应的阈值
    probs: [N, 2] 的概率数组
    """
    best_threshold = 0.5
    best_f1 = 0
    
    # 在 0.1 到 0.9 之间搜索阈值
    thresholds = np.arange(0.1, 0.9, 0.01)
    for t in thresholds:
        preds = (probs[:, 1] >= t).astype(int)
        f1 = f1_score(labels, preds, average='macro')
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            
    return best_threshold, best_f1

def train():
    # 0. 分布式初始化
    is_distributed, rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    # 1. 创建保存目录 (仅在主进程)
    if rank == 0:
        os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 2. 初始化数据集
    train_dataset = DownstreamDataset(DATA_DIR, SPLIT_FILE, mode='train', seq_len=SEQ_LEN, patch_len=PATCH_LEN)
    test_dataset = DownstreamDataset(DATA_DIR, SPLIT_FILE, mode='test', seq_len=SEQ_LEN, patch_len=PATCH_LEN)
    
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        test_sampler = None

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=(train_sampler is None), 
        num_workers=4,
        sampler=train_sampler,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        sampler=test_sampler,
        pin_memory=True
    )
    
    # 3. 初始化模型并加载预训练权重
    model = MOMENTClassification(
        config_size=MODEL_SIZE, 
        seq_len=SEQ_LEN, 
        patch_len=PATCH_LEN, 
        num_classes=NUM_CLASSES
    ).to(device)
    
    # 加载预训练权重 (应该在 DDP 包装之前)
    if os.path.exists(PRETRAINED_PATH):
        model.load_pretrained_weights(PRETRAINED_PATH)
        if rank == 0:
            print(f"Loaded pretrained weights from {PRETRAINED_PATH}")
    else:
        if rank == 0:
            print(f"Warning: Pretrained weight not found at {PRETRAINED_PATH}. Training from scratch.")
        
    # 包装 DDP
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    # 4. 优化器和损失函数
    criterion = nn.CrossEntropyLoss()
    # 注意：在分布式训练中，学习率有时需要根据 world_size 调整，这里保持不变或根据需要缩放
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # 5. 训练循环
    best_metric = 0.0
    for epoch in range(EPOCHS):
        if is_distributed:
            train_sampler.set_epoch(epoch)
            
        model.train()
        train_loss = 0
        
        # 只在主进程显示进度条
        if rank == 0:
            progress_bar = tqdm(train_loader, desc=f"Train {epoch+1}/{EPOCHS}")
        else:
            progress_bar = train_loader
            
        for patches, labels in progress_bar:
            patches, labels = patches.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(patches)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            if rank == 0:
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        scheduler.step()
        
        # 评估
        val_loss, val_probs, val_labels = evaluate(model, test_loader, device, is_distributed, rank, epoch, EPOCHS)
        
        # 仅在主进程进行指标计算和保存
        if rank == 0:
            num_val_classes = val_probs.shape[1]
            if num_val_classes == 2:
                # 1. 二分类：搜索最佳阈值
                best_threshold, best_macro_f1 = find_best_threshold(val_labels, val_probs)
                final_preds = (val_probs[:, 1] >= best_threshold).astype(int)
                auc = roc_auc_score(val_labels, val_probs[:, 1])
                print(f"[Threshold Search] Best Threshold: {best_threshold:.2f} | Best Macro F1: {best_macro_f1:.4f}")
            else:
                # 2. 多分类：直接使用 argmax
                final_preds = np.argmax(val_probs, axis=1)
                # 多分类 AUC 需要指定 multi_class
                auc = roc_auc_score(val_labels, val_probs, multi_class='ovr')
                best_threshold = "N/A (Multi-class)"
            
            acc = accuracy_score(val_labels, final_preds)
            report = classification_report(val_labels, final_preds, digits=4)
            
            print("\n" + "="*50)
            print(f"Train Loss: {train_loss/len(train_loader):.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print("-" * 50)
            print(f"Applied Threshold: {best_threshold}")
            print(f"最终测试集准确率 (Accuracy): {acc:.4f}")
            print(f"AUC Score: {auc:.4f}")
            print("-" * 50)
            print(f"最终测试集分类报告 (Classification Report):\n{report}")
            print("-" * 50)
            
            # 保存最佳模型
            if auc > best_metric:
                best_metric = auc
                save_path = os.path.join(SAVE_DIR, "best_model.pth")
                # 保存 DDP 包装内的模型 state_dict
                model_to_save = model.module if is_distributed else model
                torch.save(model_to_save.state_dict(), save_path)
                print(f">>> Best model saved! (Metric: {auc:.4f})")

    if is_distributed:
        cleanup_distributed()

if __name__ == "__main__":
    train()

