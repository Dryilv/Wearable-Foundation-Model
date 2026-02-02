import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
import random
import pickle

class PhysioSignalDataset(Dataset):
    def __init__(self, index_file, signal_len=3000, mode='train', 
                 min_std_threshold=1e-4,
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
        # 重试机制
        for _ in range(3):
            try:
                item_info = self.index_data[idx]
                file_path = item_info['path']
                row_idx = item_info['row'] # 这里假设 row_idx 指向一个多通道样本
                
                with open(file_path, 'rb') as f:
                    content = pickle.load(f)
                    # 【关键假设】：
                    # content['data'] 的形状应该是 (Num_Samples, Channels, Length)
                    # 或者 content['data'][row_idx] 直接返回 (Channels, Length)
                    raw_signal = content['data'][row_idx]
                    
                    # 如果数据是 (Length,) 或 (1, Length)，统一转为 (M, Length)
                    if raw_signal.ndim == 1:
                        raw_signal = raw_signal[np.newaxis, :]
                    
                    if raw_signal.dtype != np.float32:
                        raw_signal = raw_signal.astype(np.float32)
                
                # 1. 检查 NaN / Inf (只要任意通道有坏值，就换一个样本)
                if np.isnan(raw_signal).any() or np.isinf(raw_signal).any():
                    idx = random.randint(0, len(self.index_data) - 1)
                    continue

                # 2. 同步裁剪或填充 (所有通道处理方式必须一致)
                # processed_signal shape: (M, signal_len)
                processed_signal = self._process_signal(raw_signal)

                # 3. 逐通道计算标准差并过滤
                # axis=1 表示沿时间轴计算
                std_vals = np.std(processed_signal, axis=1, keepdims=True) # (M, 1)
                
                # 如果所有通道都太平（可能是脱落）或太剧烈（可能是伪影），则跳过
                # 这里策略比较宽松：只要有一个通道是好的，就保留样本
                # 或者你可以改为：必须所有通道都满足条件
                if np.all(std_vals < self.min_std_threshold) or np.any(std_vals > self.max_std_threshold):
                    idx = random.randint(0, len(self.index_data) - 1)
                    continue
                
                # 4. 逐通道 Z-Score 归一化
                # 这一步至关重要，消除 ECG/PPG/ACC 之间的量纲差异
                mean_vals = np.mean(processed_signal, axis=1, keepdims=True)
                processed_signal = (processed_signal - mean_vals) / (std_vals + 1e-6)

                # 转为 Tensor
                signal_tensor = torch.from_numpy(processed_signal) # (M, L)

                # 5. 【核心修改】通道乱序 (Channel Shuffling)
                # 仅在训练模式下进行，迫使模型学习 "Bag of Signals"
                if self.mode == 'train':
                    M = signal_tensor.shape[0]
                    # 生成随机排列索引
                    perm_indices = torch.randperm(M)
                    signal_tensor = signal_tensor[perm_indices]

                return signal_tensor

            except Exception as e:
                # print(f"[Warning] Error loading {file_path} row {row_idx}: {e}")
                idx = random.randint(0, len(self.index_data) - 1)
                continue
        
        # 兜底信号：生成 (1, L) 的噪声，避免崩溃
        # 注意：如果你的模型期望 M=5，这里返回 1 可能会报错，
        # 但由于模型是变长输入的，返回 (1, L) 也是合法的。
        fallback_signal = np.random.randn(1, self.signal_len).astype(np.float32) * 1e-3
        return torch.from_numpy(fallback_signal)

    def _process_signal(self, signal):
        """
        输入 signal 形状: (M, Current_Len)
        输出 signal 形状: (M, Target_Len)
        保证所有通道使用相同的裁剪区间。
        """
        M, current_len = signal.shape
        target_len = self.signal_len

        if current_len == target_len:
            return signal

        if current_len > target_len:
            if self.mode == 'train':
                # 随机裁剪：计算一次 start，应用到所有通道
                start = np.random.randint(0, current_len - target_len)
            else:
                # 中心裁剪
                start = (current_len - target_len) // 2
            
            # 切片操作：[:, start:end]
            return signal[:, start : start + target_len]
        else:
            # 零填充
            pad_len = target_len - current_len
            # np.pad 格式: ((top, bottom), (left, right))
            # 我们只在时间轴 (axis 1) 的右侧填充
            return np.pad(signal, ((0, 0), (0, pad_len)), 'constant', constant_values=0)