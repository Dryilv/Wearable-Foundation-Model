import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
import random
import pickle
import scipy.signal

class PhysioSignalDataset(Dataset):
    def __init__(self, index_file, signal_len=3000, mode='train', 
                 min_std_threshold=1e-4,
                 max_std_threshold=5000.0,
                 max_abs_value=1e5, 
                 stride=None,       
                 original_len=3000,
                 contrastive_mode=False # v4 新增：开启对比学习模式
                 ):
        self.signal_len = signal_len
        self.mode = mode
        self.min_std_threshold = min_std_threshold
        self.max_std_threshold = max_std_threshold
        self.max_abs_value = max_abs_value
        self.stride = stride
        self.original_len = original_len
        self.contrastive_mode = contrastive_mode
        
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"Index file not found: {index_file}")
            
        print(f"Loading index from: {index_file} ...")
        with open(index_file, 'r') as f:
            self.index_data = json.load(f)
        
        # 预生成样本索引
        self.samples = []
        if self.stride is not None:
            for i in range(len(self.index_data)):
                for start in range(0, self.original_len - self.signal_len + 1, self.stride):
                    self.samples.append({'idx': i, 'start': start})
            print(f"Sliding window enabled (stride={stride}). Expanded {len(self.index_data)} samples to {len(self.samples)} windows.")
        else:
            for i in range(len(self.index_data)):
                self.samples.append({'idx': i, 'start': None})
            print(f"Loaded {len(self.index_data)} samples.")

    def __len__(self):
        return len(self.samples)
    
    # -----------------------------------------------------------
    # v4 数据增强函数
    # -----------------------------------------------------------
    def _augment_signal(self, signal):
        """
        对信号进行强增强，生成两个视图
        signal: (M, L) numpy array, normalized
        """
        M, L = signal.shape
        aug_signal = signal.copy()
        
        # 1. Random Resized Crop (Resize-Back)
        # 模拟不同心率或采样率波动
        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.2)
            new_len = int(L * scale)
            # 使用 scipy.signal.resample 
            # 注意：resample 默认沿着最后一维 (time)
            aug_signal = scipy.signal.resample(aug_signal, new_len, axis=1)
            
            # Crop or Pad back to L
            if new_len > L:
                start = random.randint(0, new_len - L)
                aug_signal = aug_signal[:, start:start+L]
            else:
                pad_len = L - new_len
                aug_signal = np.pad(aug_signal, ((0,0), (0, pad_len)), 'constant')
                
        # 2. Random Gaussian Noise
        if random.random() < 0.5:
            noise = np.random.normal(0, 0.1, aug_signal.shape).astype(np.float32)
            aug_signal += noise
            
        # 3. Random Channel Masking (Drop Channel)
        # 如果通道数 > 1，随机将某些通道置零
        if M > 1 and random.random() < 0.3:
            drop_idx = random.randint(0, M-1)
            aug_signal[drop_idx, :] = 0.0
            
        # 4. Random Amplitude Scale (Global)
        if random.random() < 0.5:
            scale_amp = random.uniform(0.5, 2.0)
            aug_signal *= scale_amp
            
        # 5. Random Flip (Time Reverse)
        if random.random() < 0.2:
            aug_signal = np.flip(aug_signal, axis=1).copy()
            
        return aug_signal

    def __getitem__(self, idx):
        # 重试机制
        for _ in range(3):
            try:
                sample_info = self.samples[idx]
                original_idx = sample_info['idx']
                fixed_start = sample_info['start']

                item_info = self.index_data[original_idx]
                file_path = item_info['path']
                row_idx = item_info['row'] 
                label = item_info.get('label', 0) 
                
                with open(file_path, 'rb') as f:
                    content = pickle.load(f)
                    raw_signal = content['data'][row_idx]
                    
                    if raw_signal.ndim == 1:
                        raw_signal = raw_signal[np.newaxis, :]
                    
                    if raw_signal.dtype != np.float32:
                        raw_signal = raw_signal.astype(np.float32)
                
                # 1. 基础检查
                if np.isnan(raw_signal).any() or np.isinf(raw_signal).any():
                    idx = random.randint(0, len(self.samples) - 1)
                    continue
                
                if np.max(np.abs(raw_signal)) > self.max_abs_value:
                    idx = random.randint(0, len(self.samples) - 1)
                    continue

                # 2. 同步裁剪或填充 (使用固定起始位置或随机起始位置)
                processed_signal = self._process_signal(raw_signal, fixed_start)

                # 3. 逐通道质量检查
                std_vals = np.std(processed_signal, axis=1, keepdims=True) # (M, 1)
                
                if np.isnan(std_vals).any() or \
                   np.any(std_vals > self.max_std_threshold) or \
                   np.any(std_vals < self.min_std_threshold):
                    idx = random.randint(0, len(self.samples) - 1)
                    continue
                
                # 4. 逐通道 Z-Score 归一化
                mean_vals = np.mean(processed_signal, axis=1, keepdims=True)
                processed_signal = (processed_signal - mean_vals) / (std_vals + 1e-5)
                processed_signal = np.clip(processed_signal, -10, 10) 
                
                # --- v4 Contrastive Logic ---
                if self.contrastive_mode:
                    # 生成两个增强视图
                    view1 = self._augment_signal(processed_signal)
                    view2 = self._augment_signal(processed_signal)
                    
                    t_view1 = torch.from_numpy(view1.astype(np.float32))
                    t_view2 = torch.from_numpy(view2.astype(np.float32))
                    
                    # 返回 tuple: (view1, view2), label
                    # Dataset 返回 tuple，Collate Fn 会处理
                    return (t_view1, t_view2), torch.tensor(label, dtype=torch.long)
                
                # --- Normal Mode ---
                signal_tensor = torch.from_numpy(processed_signal) 

                # 5. 通道乱序 (仅训练模式)
                if self.mode == 'train':
                    M = signal_tensor.shape[0]
                    perm_indices = torch.randperm(M)
                    signal_tensor = signal_tensor[perm_indices]

                return signal_tensor, torch.tensor(label, dtype=torch.long)

            except Exception as e:
                idx = random.randint(0, len(self.samples) - 1)
                continue
        
        # 兜底
        fallback = torch.zeros((1, self.signal_len), dtype=torch.float32)
        if self.contrastive_mode:
            return (fallback, fallback), torch.tensor(0, dtype=torch.long)
        return fallback, torch.tensor(0, dtype=torch.long)

    def _process_signal(self, signal, fixed_start=None):
        M, current_len = signal.shape
        target_len = self.signal_len

        if current_len == target_len:
            return signal

        if current_len > target_len:
            if fixed_start is not None:
                start = fixed_start
            elif self.mode == 'train':
                start = np.random.randint(0, current_len - target_len)
            else:
                start = (current_len - target_len) // 2
            return signal[:, start : start + target_len]
        else:
            pad_len = target_len - current_len
            return np.pad(signal, ((0, 0), (0, pad_len)), 'constant', constant_values=0)
