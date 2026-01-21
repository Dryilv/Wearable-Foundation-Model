import os
import argparse
import yaml
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# 导入你的模型和数据集
from model_ppg2ecg import PPG2ECG_Translator
from dataset_paired import PairedPhysioDataset

# 启用 TensorFloat-32
torch.set_float32_matmul_precision('high')

def find_lr(args, config):
    # 1. 设置设备 (单卡运行即可)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 加载数据集
    print("Loading dataset...")
    dataset = PairedPhysioDataset(
        index_file=config['data']['index_path'],
        signal_len=config['data']['signal_len'],
        mode='train',
        row_ppg=4, # Input
        row_ecg=0  # Target
    )
    
    # 不需要 DistributedSampler，普通 DataLoader 即可
    dataloader = DataLoader(
        dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    # 3. 加载模型
    print("Initializing model...")
    model = PPG2ECG_Translator(
        pretrained_path=args.pretrained,
        signal_len=config['data']['signal_len'],
        cwt_scales=config['model'].get('cwt_scales', 64),
        embed_dim=config['model']['embed_dim'],
        depth=config['model']['depth'],
        num_heads=config['model']['num_heads'],
        decoder_embed_dim=config['model']['decoder_embed_dim'],
        decoder_depth=config['model']['decoder_depth'],
        decoder_num_heads=config['model']['decoder_num_heads'],
        # 这里的权重不重要，因为我们只看相对变化，但保持一致最好
        time_loss_weight=config['model'].get('time_loss_weight', 5.0) 
    )
    model.to(device)
    
    # 注意：LR Finder 不需要 torch.compile，编译反而会拖慢测试启动速度
    
    # 4. 定义优化器 (初始 LR 设置得很小)
    start_lr = 1e-7
    end_lr = 10.0
    optimizer = optim.AdamW(model.parameters(), lr=start_lr, weight_decay=0.05)
    
    # 5. LR Finder 核心逻辑
    num_batches = len(dataloader)
    # 我们只跑 100-200 个 step 就足够画图了，不需要跑完整个 Epoch
    num_test_steps = min(num_batches, 200) 
    
    # 计算每个 Step 的倍增因子
    gamma = (end_lr / start_lr) ** (1 / num_test_steps)
    
    lrs = []
    losses = []
    best_loss = float('inf')
    
    model.train()
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    print(f"Starting LR Finder (Steps: {num_test_steps}, Range: {start_lr} -> {end_lr})...")
    
    progress_bar = tqdm(range(num_test_steps))
    iter_loader = iter(dataloader)
    
    current_lr = start_lr
    
    for step in progress_bar:
        try:
            ppg, ecg = next(iter_loader)
        except StopIteration:
            break
            
        ppg = ppg.to(device)
        ecg = ecg.to(device)
        
        # Forward
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            loss, _, _ = model(ppg, ecg_target=ecg)
        
        # 记录
        loss_val = loss.item()
        
        # 如果 Loss 爆炸 (变成 NaN 或比最小值大 4 倍)，提前停止
        if step > 10 and (np.isnan(loss_val) or loss_val > 4 * best_loss):
            print(f"Loss exploded at step {step}, stopping early.")
            break
            
        if loss_val < best_loss:
            best_loss = loss_val
            
        lrs.append(current_lr)
        losses.append(loss_val)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        optimizer.step()
        
        # 更新 LR
        current_lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
            
        progress_bar.set_description(f"LR: {current_lr:.5f} | Loss: {loss_val:.4f}")

    # 6. 绘图
    print("Plotting results...")
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate (Log Scale)')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    # 简单的平滑处理，让曲线更好看
    def smooth(scalars, weight=0.8):
        last = scalars[0]
        smoothed = []
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    if len(losses) > 20:
        smoothed_losses = smooth(losses)
        plt.plot(lrs, smoothed_losses, 'r--', label='Smoothed')
        plt.legend()

    save_path = 'lr_finder_result.png'
    plt.savefig(save_path)
    print(f"LR curve saved to {save_path}")
    print("Done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', type=str)
    parser.add_argument('--pretrained', type=str, required=True)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    find_lr(args, config)