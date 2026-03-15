import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import glob

# 导入模型中的 CWT 函数，确保逻辑一致
from model import cwt_wrap

def load_sample(file_path, signal_len=3000):
    with open(file_path, 'rb') as f:
        content = pickle.load(f)
    
    if isinstance(content, dict) and 'data' in content:
        raw_data = content['data']
    else:
        raw_data = content
        
    if isinstance(raw_data, list):
        raw_data = np.array(raw_data)
    if raw_data.ndim == 1:
        raw_data = raw_data[np.newaxis, :]
        
    # 裁剪/填充到指定长度
    M, L = raw_data.shape
    if L > signal_len:
        start = (L - signal_len) // 2
        raw_data = raw_data[:, start:start+signal_len]
    elif L < signal_len:
        pad_len = signal_len - L
        raw_data = np.pad(raw_data, ((0, 0), (0, pad_len)), mode='edge')
        
    return torch.from_numpy(raw_data.astype(np.float32))

def visualize_cwt(args):
    # 1. 查找文件
    if os.path.isdir(args.input_path):
        files = glob.glob(os.path.join(args.input_path, "*.pkl"))
        if not files:
            print(f"No .pkl files found in {args.input_path}")
            return
        target_file = files[0] # 默认取第一个
    else:
        target_file = args.input_path
        
    print(f"Visualizing: {target_file}")
    
    # 2. 加载数据
    signal_tensor = load_sample(target_file, args.signal_len) # (M, L)
    print(f"Original Signal Shape: {signal_tensor.shape}")
    
    # 3. 归一化 (模拟 Dataset 中的 Robust Norm)
    # 这一步非常关键，如果训练时做了归一化，可视化也必须做
    median = torch.median(signal_tensor, dim=1, keepdim=True).values
    q25 = torch.quantile(signal_tensor, 0.25, dim=1, keepdim=True)
    q75 = torch.quantile(signal_tensor, 0.75, dim=1, keepdim=True)
    iqr = q75 - q25
    iqr = torch.where(iqr < 1e-6, torch.tensor(1.0), iqr)
    signal_norm = (signal_tensor - median) / iqr
    signal_norm = torch.clamp(signal_norm, -20.0, 20.0)
    
    # 4. 执行 CWT
    # cwt_wrap expects (B, M, L)
    input_tensor = signal_norm.unsqueeze(0) # (1, M, L)
    
    with torch.no_grad():
        # 调用模型中的 CWT 函数
        cwt_out = cwt_wrap(
            input_tensor, 
            num_scales=args.scales, 
            lowest_scale=0.1, 
            step=1.0, 
            use_diff=args.use_diff
        )
        # cwt_out shape: (B, M, C, Scales, L)
        # C=3 if use_diff else 1
    
    print(f"CWT Output Shape: {cwt_out.shape}")
    
    # 5. 绘图
    B, M, C, S, L = cwt_out.shape
    
    # 我们只画第一个样本
    # 如果 M > 1，画前两个通道 (通常是 ECG 和 PPG)
    num_channels_to_plot = min(M, 2)
    
    fig, axes = plt.subplots(num_channels_to_plot * 2, 1, figsize=(15, 6 * num_channels_to_plot), sharex=True)
    if num_channels_to_plot == 1: axes = np.array(axes).reshape(-1) # 统一为数组
    
    for m in range(num_channels_to_plot):
        # --- Plot 1: Raw Signal ---
        ax_raw = axes[m * 2]
        ax_raw.plot(signal_norm[m].numpy(), label=f'Ch {m} (Norm)', color='black', linewidth=0.8)
        ax_raw.set_title(f"Channel {m} - Normalized Signal")
        ax_raw.legend(loc='upper right')
        ax_raw.grid(True, alpha=0.3)
        ax_raw.set_ylim(-10, 10) # 限制 y 轴，看清细节
        
        # --- Plot 2: CWT Spectrogram ---
        ax_cwt = axes[m * 2 + 1]
        
        # 取 CWT 的第一个分量 (Base) 进行可视化，忽略差分分量
        # spec shape: (S, L)
        spec = cwt_out[0, m, 0, :, :].numpy()
        
        # 对数变换增强对比度 (可选，但模型看到的是原始值)
        # spec_vis = np.log1p(np.abs(spec))
        spec_vis = spec # 直接看原始值
        
        im = ax_cwt.imshow(spec_vis, aspect='auto', cmap='jet', origin='lower', 
                           extent=[0, L, 0, S])
        ax_cwt.set_title(f"Channel {m} - CWT (Scales={args.scales})")
        ax_cwt.set_ylabel("Scale")
        fig.colorbar(im, ax=ax_cwt, orientation='vertical', fraction=0.02, pad=0.04)

    plt.tight_layout()
    output_png = "cwt_visualization.png"
    plt.savefig(output_png, dpi=150)
    print(f"\nVisualization saved to: {os.path.abspath(output_png)}")
    print("Please check this image. If CWT is all blue/red or black, the scales are wrong.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help="Path to a .pkl file or directory")
    parser.add_argument('--signal_len', type=int, default=3000)
    parser.add_argument('--scales', type=int, default=64, help="Number of CWT scales")
    parser.add_argument('--use_diff', action='store_true', help="Enable diff channel")
    args = parser.parse_args()
    
    visualize_cwt(args)
