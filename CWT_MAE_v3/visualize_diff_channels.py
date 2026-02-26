import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import yaml
import os
import sys
import random

# 尝试导入项目模块
try:
    from model import cwt_ricker
    from dataset import PhysioSignalDataset
except ImportError:
    print("请确保将此脚本放在 CWT_MAE_v3 目录下运行，或者正确设置 PYTHONPATH")
    sys.exit(1)

def visualize_all_channels():
    # 1. 加载配置
    config_path = 'config.yaml'
    if not os.path.exists(config_path):
        print(f"找不到配置文件: {config_path}")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 2. 初始化数据集
    print(f"正在加载数据集索引: {config['data']['index_path']} ...")
    dataset = PhysioSignalDataset(
        index_file=config['data']['index_path'],
        signal_len=config['data']['signal_len'],
        mode='train',
        data_ratio=1.0 # 加载所有数据以随机选择
    )
    
    # 3. 随机选择一个样本
    idx = random.randint(0, len(dataset) - 1)
    # 或者指定一个索引
    # idx = 0 
    
    print(f"正在分析样本 Index: {idx} ...")
    signal_tensor, modality_ids, label = dataset[idx]
    # signal_tensor shape: (5, 3000)
    
    num_channels = signal_tensor.shape[0]
    signal_len = signal_tensor.shape[1]
    
    # CWT 参数
    num_scales = config['model'].get('cwt_scales', 64)
    scales = torch.arange(num_scales) * 1.0 + 0.1
    
    # 定义通道名称 (假设顺序)
    # Original: ["ECG", "ACC_X", "ACC_Y", "ACC_Z", "PPG"]
    # Kept only: 0:ECG and 4:PPG
    channel_names = ["ECG", "PPG"]
    
    if num_channels != 2:
        channel_names = [f"Channel {i}" for i in range(num_channels)]

    # 4. 循环为每个通道绘图
    save_dir = "vis_results"
    os.makedirs(save_dir, exist_ok=True)

    for ch in range(num_channels):
        print(f"正在处理通道 {ch}: {channel_names[ch]} ...")
        
        # 准备数据
        x = signal_tensor[ch:ch+1, :] # (1, L)
        
        # 计算差分
        x_pad = F.pad(x, (1, 1), mode='replicate')
        d1 = (x_pad[:, 1:] - x_pad[:, :-1])[:, :signal_len]
        d2 = (d1[:, 1:] - d1[:, :-1])[:, :signal_len] # 注意：再次对 d1 差分
        # 修正：代码中 d2 计算通常基于 d1_full，这里简化处理，确保形状对齐即可
        
        # 计算 CWT
        cwt_base = cwt_ricker(x, scales)
        cwt_d1 = cwt_ricker(d1, scales)
        cwt_d2 = cwt_ricker(d2, scales)
        
        # 绘图: 3行2列 (左边波形，右边频谱)
        fig, axes = plt.subplots(3, 2, figsize=(15, 10))
        fig.suptitle(f"Sample {idx} - Channel {ch} ({channel_names[ch]}) Analysis", fontsize=16)
        
        # --- Row 1: Original ---
        axes[0, 0].plot(x[0].numpy(), 'k', lw=1)
        axes[0, 0].set_title("Original Signal")
        axes[0, 0].grid(True, alpha=0.3)
        
        im0 = axes[0, 1].imshow(cwt_base[0].abs().numpy(), aspect='auto', cmap='turbo', origin='lower')
        axes[0, 1].set_title("Original Spectrogram")
        plt.colorbar(im0, ax=axes[0, 1])
        
        # --- Row 2: 1st Diff ---
        axes[1, 0].plot(d1[0].numpy(), 'b', lw=1)
        axes[1, 0].set_title("1st Order Diff (Velocity)")
        axes[1, 0].grid(True, alpha=0.3)
        
        im1 = axes[1, 1].imshow(cwt_d1[0].abs().numpy(), aspect='auto', cmap='turbo', origin='lower')
        axes[1, 1].set_title("1st Order Spectrogram")
        plt.colorbar(im1, ax=axes[1, 1])
        
        # --- Row 3: 2nd Diff ---
        axes[2, 0].plot(d2[0].numpy(), 'r', lw=1)
        axes[2, 0].set_title("2nd Order Diff (Acceleration)")
        axes[2, 0].grid(True, alpha=0.3)
        
        im2 = axes[2, 1].imshow(cwt_d2[0].abs().numpy(), aspect='auto', cmap='turbo', origin='lower')
        axes[2, 1].set_title("2nd Order Spectrogram")
        plt.colorbar(im2, ax=axes[2, 1])
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"sample_{idx}_channel_{ch}_{channel_names[ch]}.png")
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        
    print(f"\n所有通道的可视化图片已保存至目录: {os.path.abspath(save_dir)}")

if __name__ == "__main__":
    visualize_all_channels()