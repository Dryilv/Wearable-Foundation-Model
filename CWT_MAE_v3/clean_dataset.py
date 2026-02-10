import os
import json
import pickle
import numpy as np
from tqdm import tqdm
import argparse

def clean_dataset(input_index, output_index, expected_channels=5):
    if not os.path.exists(input_index):
        print(f"Error: Input index file {input_index} not found.")
        return

    print(f"Loading index from {input_index}...")
    with open(input_index, 'r') as f:
        data_list = json.load(f)

    cleaned_list = []
    stats = {
        'total': len(data_list),
        'valid': 0,
        'invalid_channels': 0,
        'has_nan_inf': 0,
        'invalid_std': 0,
        'file_not_found': 0,
        'read_error': 0
    }

    print("Starting dataset cleaning...")
    for item in tqdm(data_list):
        file_path = item['path']
        
        if not os.path.exists(file_path):
            stats['file_not_found'] += 1
            continue

        try:
            with open(file_path, 'rb') as f:
                content = pickle.load(f)
            
            # 1. 检查数据结构
            if 'data' not in content:
                stats['read_error'] += 1
                continue
                
            raw_signal = content['data']
            
            # 2. 检查通道数 (M, L)
            if raw_signal.ndim == 1:
                raw_signal = raw_signal[np.newaxis, :]
            
            if raw_signal.shape[0] != expected_channels:
                stats['invalid_channels'] += 1
                continue
            
            # 3. 检查 NaN 和 Inf
            if np.isnan(raw_signal).any() or np.isinf(raw_signal).any():
                stats['has_nan_inf'] += 1
                continue
            
            # 4. 检查数值有效性 (防止死线或极端异常值)
            # 参考 PhysioSignalDataset 的过滤标准
            std_vals = np.std(raw_signal, axis=1)
            if np.any(std_vals < 1e-4) or np.any(std_vals > 5000.0):
                stats['invalid_std'] += 1
                continue
            
            if np.max(np.abs(raw_signal)) > 1e5:
                stats['has_nan_inf'] += 1 # 归类为异常数值
                continue

            # 通过所有检查，保留该样本
            # 可以在此处更新长度信息，确保索引准确
            item['len'] = raw_signal.shape[1]
            cleaned_list.append(item)
            stats['valid'] += 1

        except Exception as e:
            # print(f"\nError reading {file_path}: {e}")
            stats['read_error'] += 1
            continue

    print("\n" + "="*30)
    print("Cleaning Statistics:")
    print(f"Total samples:     {stats['total']}")
    print(f"Valid samples:     {stats['valid']} ({(stats['valid']/stats['total'])*100:.2f}%)")
    print(f"Invalid channels:  {stats['invalid_channels']}")
    print(f"NaN or Inf:        {stats['has_nan_inf']}")
    print(f"Invalid Std (flat): {stats['invalid_std']}")
    print(f"File not found:    {stats['file_not_found']}")
    print(f"Read error:        {stats['read_error']}")
    print("="*30)

    print(f"Saving cleaned index to {output_index}...")
    with open(output_index, 'w') as f:
        json.dump(cleaned_list, f, indent=2)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean PhysioSignal dataset and generate a new index.")
    parser.add_argument("--input", default="train_index_new.json", help="Path to the original index file")
    parser.add_argument("--output", default="train_index_cleaned.json", help="Path to save the cleaned index file")
    parser.add_argument("--channels", type=int, default=5, help="Expected number of channels")
    
    args = parser.parse_args()
    clean_dataset(args.input, args.output, args.channels)
