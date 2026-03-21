import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch

def load_inference_signal(file_path, signal_len=3000, stride=1500):
    """
    加载推理数据并进行滑动窗口切分和归一化
    与 inference.py 中的 AdaptivePatientDataset 逻辑保持一致
    """
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
        
        raw_data = raw_data.astype(np.float32)

    M, n_samples = raw_data.shape
    print(f"Raw data shape: {raw_data.shape}")

    if n_samples < signal_len:
        print(f"Warning: Signal length ({n_samples}) is shorter than required ({signal_len}).")
        return [raw_data]

    segments = []
    normalized_segments = []
    
    # 滑动窗口切分
    for start in range(0, n_samples - signal_len + 1, stride):
        segment = raw_data[:, start : start + signal_len]
        segments.append(segment)
        
        # Robust Normalization (与推理一致)
        median = np.median(segment, axis=1, keepdims=True)
        q25 = np.percentile(segment, 25, axis=1, keepdims=True)
        q75 = np.percentile(segment, 75, axis=1, keepdims=True)
        iqr_val = q75 - q25
        iqr_val = np.where(iqr_val < 1e-6, 1.0, iqr_val)
        
        segment_norm = (segment - median) / iqr_val
        segment_norm = np.clip(segment_norm, -20.0, 20.0)
        normalized_segments.append(segment_norm)

    return segments, normalized_segments

def visualize_inference_signal(args):
    print(f"Loading data from: {args.input_path}")
    raw_segments, norm_segments = load_inference_signal(args.input_path, args.signal_len, args.stride)
    
    num_segments = len(raw_segments)
    print(f"Generated {num_segments} segments using stride {args.stride}")
    
    if num_segments == 0:
        print("No segments generated.")
        return

    # 限制可视化的片段数量，防止图太大
    vis_count = min(num_segments, args.max_segments)
    M = raw_segments[0].shape[0] # 通道数

    # 为每个片段画图
    for i in range(vis_count):
        raw_seg = raw_segments[i]
        norm_seg = norm_segments[i]
        
        fig, axes = plt.subplots(M, 2, figsize=(15, 3 * M))
        if M == 1:
            axes = axes[np.newaxis, :] # 统一维度以便索引
            
        fig.suptitle(f"Segment {i+1}/{num_segments} (Start={i*args.stride}, End={i*args.stride+args.signal_len})", fontsize=14)
        
        for m in range(M):
            # 左侧：原始波形
            ax_raw = axes[m, 0]
            ax_raw.plot(raw_seg[m], color='blue', linewidth=0.8)
            ax_raw.set_title(f"Channel {m} - Raw Signal")
            ax_raw.grid(True, alpha=0.3)
            
            # 右侧：归一化后的波形 (送入模型的实际数据)
            ax_norm = axes[m, 1]
            ax_norm.plot(norm_seg[m], color='red', linewidth=0.8)
            
            # 计算实际范围
            norm_min, norm_max = norm_seg[m].min(), norm_seg[m].max()
            ax_norm.set_title(f"Channel {m} - Normalized (Model Input) | Range: [{norm_min:.2f}, {norm_max:.2f}]")
            ax_norm.grid(True, alpha=0.3)
            
            # 移除强制的 -20 和 20 的参考线，让 matplotlib 自动缩放 y 轴，以看清波形细节
            # 如果依然想看到 0 刻度，可以加一条 0 的虚线
            ax_norm.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        plt.tight_layout()
        output_file = os.path.join(args.output_dir, f"inference_vis_seg_{i:03d}.png")
        plt.savefig(output_file, dpi=150)
        plt.close(fig)
        print(f"Saved visualization to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize raw and normalized signals for inference data.")
    parser.add_argument('--input_path', type=str, required=True, help="Path to the patient .pkl file")
    parser.add_argument('--output_dir', type=str, default="./vis_inference", help="Directory to save the plots")
    parser.add_argument('--signal_len', type=int, default=3000, help="Signal length per window")
    parser.add_argument('--stride', type=int, default=1500, help="Sliding window stride")
    parser.add_argument('--max_segments', type=int, default=5, help="Maximum number of segments to plot")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    visualize_inference_signal(args)