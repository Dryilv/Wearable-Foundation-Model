import json
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import argparse
import sys

def load_index(json_path):
    if not os.path.exists(json_path):
        print(f"错误: 找不到索引文件 {json_path}")
        sys.exit(1)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 提取所有唯一的文件路径 (去重)
    # 假设 json 结构是 [{'path': '...', ...}, ...]
    paths = list(set([item['path'] for item in data]))
    print(f"成功加载索引，共包含 {len(paths)} 个唯一文件。")
    return paths

def load_signal(file_path):
    try:
        with open(file_path, 'rb') as f:
            content = pickle.load(f)
            
        # 假设数据在 'data' 键下，且形状为 [Channels, Length]
        if isinstance(content, dict) and 'data' in content:
            data = content['data']
        else:
            data = content # 如果 pickle 直接存的是数组
            
        return data
    except Exception as e:
        print(f"读取文件出错 {file_path}: {e}")
        return None

def visualize_sample(file_path, signal_data, sample_id=None):
    """
    可视化前5个通道的数据
    """
    # 确保数据至少有5行
    num_channels = signal_data.shape[0]
    if num_channels < 5:
        print(f"警告: 数据只有 {num_channels} 行，无法显示前5行。")
        return

    # 创建画布
    fig, axes = plt.subplots(5, 1, figsize=(15, 12), sharex=True)
    
    # 设置标题
    title_str = f"File: {os.path.basename(file_path)}"
    if sample_id is not None:
        title_str = f"Sample Index: {sample_id} | " + title_str
    fig.suptitle(title_str, fontsize=16)

    # 定义通道名称 (根据你的描述)
    channel_names = {
        0: "Row 1: PPG (Input)",
        1: "Row 2: Channel 2",
        2: "Row 3: Channel 3",
        3: "Row 4: Channel 4",
        4: "Row 5: ECG (Target)"
    }
    
    colors = ['green', 'blue', 'blue', 'blue', 'red'] # PPG绿，ECG红，其他蓝

    # 循环绘制前5行
    for i in range(5):
        ax = axes[i]
        signal = signal_data[i]
        
        # 简单的统计信息
        mean_val = np.mean(signal)
        std_val = np.std(signal)
        
        ax.plot(signal, color=colors[i], linewidth=1)
        ax.set_ylabel(f"Amp", fontsize=10)
        ax.set_title(f"{channel_names[i]} | Mean: {mean_val:.2f}, Std: {std_val:.2f}", loc='left', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 移除多余的边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.xlabel("Time Steps", fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92) # 给总标题留空间
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize first 5 rows of physiological signals from PKL files.")
    parser.add_argument('--index', type=str, required=True, help='Path to the JSON index file')
    parser.add_argument('--idx', type=int, default=None, help='Specific index in the file list to visualize (optional)')
    args = parser.parse_args()

    # 1. 加载索引
    file_paths = load_index(args.index)

    # 2. 交互式查看循环
    while True:
        # 选择文件
        if args.idx is not None:
            current_idx = args.idx
            if current_idx < 0 or current_idx >= len(file_paths):
                print(f"索引越界 (0 - {len(file_paths)-1})")
                break
        else:
            current_idx = random.randint(0, len(file_paths) - 1)
        
        file_path = file_paths[current_idx]
        print(f"\n正在查看 [{current_idx}/{len(file_paths)}]: {file_path}")

        # 加载数据
        data = load_signal(file_path)
        
        if data is not None:
            # 可视化
            visualize_sample(file_path, data, sample_id=current_idx)
        
        # 如果指定了特定索引，只看一次就退出
        if args.idx is not None:
            break
            
        # 询问是否继续
        user_input = input("按 [Enter] 查看下一个随机样本，输入 'q' 退出: ")
        if user_input.lower() == 'q':
            break

if __name__ == "__main__":
    main()