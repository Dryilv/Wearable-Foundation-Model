import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
import random
import pickle

class PhysioSignalDataset(Dataset):
    def __init__(self, index_file, signal_len=3000, mode='train', 
                 min_std_threshold=1e-4,   # 降低阈值，避免过滤掉微弱但有效的生理信号
                 max_std_threshold=5000.0 
                 ):
        self.signal_len = signal_len
        self.mode = mode
        self.min_std_threshold = min_std_threshold
        self.max_std_threshold = max_std_threshold
        
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"Index file not found: {index_file}")
            
        print(f"Loading index from: {index_file} ...")
        with open(index_file, 'r') as f:
            self.index_data = json.load(f)
        print(f"Loaded {len(self.index_data)} samples.")

    def __len__(self):
        return len(self.index_data)

    def __getitem__(self, idx):
        # 尝试加载数据的重试机制 (最多 3 次)
        for _ in range(3):
            try:
                item_info = self.index_data[idx]
                file_path = item_info['path']
                row_idx = item_info['row']
                
                with open(file_path, 'rb') as f:
                    content = pickle.load(f)
                    # 假设数据存储格式为 {'data': np.array(N, Length)}
                    raw_signal = content['data'][row_idx]
                    
                    if raw_signal.dtype != np.float32:
                        raw_signal = raw_signal.astype(np.float32)
                
                # 1. 检查 NaN / Inf
                if np.isnan(raw_signal).any() or np.isinf(raw_signal).any():
                    # 随机换一个样本
                    idx = random.randint(0, len(self.index_data) - 1)
                    continue

                # 2. 裁剪或填充到固定长度
                processed_signal = self._process_signal(raw_signal)

                # 3. 计算标准差并过滤异常值
                std_val = np.std(processed_signal)
                
                if std_val < self.min_std_threshold:
                    idx = random.randint(0, len(self.index_data) - 1)
                    continue

                if std_val > self.max_std_threshold:
                    idx = random.randint(0, len(self.index_data) - 1)
                    continue
                
                # 4. 基础 Z-Score 归一化 (关键修改)
                # 这有助于 CWT 的数值稳定性，并让模型专注于波形形态而非绝对幅度
                mean_val = np.mean(processed_signal)
                processed_signal = (processed_signal - mean_val) / (std_val + 1e-6)

                # 返回 [1, L] 格式，适配 Conv1d 输入
                signal_tensor = torch.from_numpy(processed_signal).unsqueeze(0)
                return signal_tensor

            except Exception as e:
                # print(f"[Warning] Error loading {file_path} row {row_idx}: {e}")
                idx = random.randint(0, len(self.index_data) - 1)
                continue
        
        # 兜底信号：生成微弱的高斯噪声，防止 DataLoader 崩溃
        fallback_signal = np.random.randn(self.signal_len).astype(np.float32) * 1e-3
        return torch.from_numpy(fallback_signal).unsqueeze(0)

    def _process_signal(self, signal):
        current_len = len(signal)
        target_len = self.signal_len

        if current_len == target_len:
            return signal

        if current_len > target_len:
            if self.mode == 'train':
                # 随机裁剪
                start = np.random.randint(0, current_len - target_len)
            else:
                # 中心裁剪
                start = (current_len - target_len) // 2
            return signal[start : start + target_len]
        else:
            # 零填充
            pad_len = target_len - current_len
            return np.pad(signal, (0, pad_len), 'constant', constant_values=0)