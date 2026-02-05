import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
import random
import pickle

class PhysioSignalDataset(Dataset):
    def __init__(self, index_file, signal_len=500, mode='train', 
                 min_std_threshold=1e-4,
                 max_std_threshold=5000.0,
                 max_abs_value=1e5, # 新增：绝对值上限过滤
                 stride=None,       # 新增：滑动窗口步长
                 original_len=3000, # 新增：原始信号长度（用于滑动窗口计算）
                 contrastive=False  # 新增：是否启用对比学习模式
                 ):
        self.signal_len = signal_len
        self.mode = mode
        self.contrastive = contrastive
        self.min_std_threshold = min_std_threshold
        self.max_std_threshold = max_std_threshold
        self.max_abs_value = max_abs_value
        self.stride = stride
        self.original_len = original_len
        
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

    def _augment_signal(self, signal):
        """
        Apply random augmentations to the signal:
        1. Amplitude scaling (0.8 - 1.2)
        2. Gaussian noise
        """
        if self.mode != 'train':
            return signal
            
        # Amplitude scaling
        scale = np.random.uniform(0.8, 1.2)
        signal = signal * scale
        
        # Gaussian noise (scaled by signal std)
        std = np.std(signal, axis=1, keepdims=True)
        noise = np.random.normal(0, 0.05, signal.shape).astype(np.float32) * (std + 1e-6)
        signal = signal + noise
        
        return signal

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

                # 辅助函数：处理单个视图
                def process_view(signal_data, start_pos):
                    # 2. 同步裁剪或填充
                    proc_sig = self._process_signal(signal_data, start_pos)
                    
                    # 增强 (仅在对比学习模式下，且是训练阶段)
                    if self.contrastive and self.mode == 'train':
                        proc_sig = self._augment_signal(proc_sig)

                    # 3. 逐通道质量检查
                    std_v = np.std(proc_sig, axis=1, keepdims=True)
                    if np.isnan(std_v).any() or \
                       np.any(std_v > self.max_std_threshold) or \
                       np.any(std_v < self.min_std_threshold):
                        return None, None
                    
                    # 4. 逐通道 Z-Score 归一化
                    mean_v = np.mean(proc_sig, axis=1, keepdims=True)
                    proc_sig = (proc_sig - mean_v) / (std_v + 1e-5)
                    proc_sig = np.clip(proc_sig, -10, 10)
                    
                    return torch.from_numpy(proc_sig), std_v

                # 如果是对比学习模式，生成两个视图
                if self.contrastive:
                    # View 1
                    sig_tensor1, std1 = process_view(raw_signal, fixed_start)
                    if sig_tensor1 is None: raise ValueError("Bad signal view 1")
                    
                    # View 2 (如果是训练模式，尝试不同的随机裁剪)
                    # 注意：如果 fixed_start 不为 None，则 View 2 位置也固定，主要靠 augment_signal 增强
                    sig_tensor2, std2 = process_view(raw_signal, fixed_start)
                    if sig_tensor2 is None: raise ValueError("Bad signal view 2")
                    
                    # 5. 通道乱序 (仅训练模式) - 两个视图保持相同的乱序还是不同？
                    # 通常对比学习希望视图间有差异，但通道对应关系是否重要？
                    # 如果通道代表不同的导联，打乱通道顺序可能会破坏语义对应。
                    # 但在这里的模型是 "Permutation Invariant" (Modality Agnostic)，所以打乱是可以的。
                    # 为了增加难度，可以让两个视图有不同的通道顺序，或者相同的。
                    # 暂时保持两个视图使用 相同的 乱序（如果应用的话），或者 不同的？
                    # 这里的 MAE 实现中，Forward Encoder 混合了所有通道。
                    # 为了简单起见，如果乱序，对两个视图分别乱序。
                    if self.mode == 'train':
                        M = sig_tensor1.shape[0]
                        perm_indices1 = torch.randperm(M)
                        sig_tensor1 = sig_tensor1[perm_indices1]
                        
                        perm_indices2 = torch.randperm(M)
                        sig_tensor2 = sig_tensor2[perm_indices2]
                    
                    return (sig_tensor1, sig_tensor2), torch.tensor(label, dtype=torch.long)

                else:
                    # 原始模式
                    signal_tensor, std_val = process_view(raw_signal, fixed_start)
                    if signal_tensor is None:
                        idx = random.randint(0, len(self.samples) - 1)
                        continue
                        
                    if self.mode == 'train':
                        M = signal_tensor.shape[0]
                        perm_indices = torch.randperm(M)
                        signal_tensor = signal_tensor[perm_indices]

                    return signal_tensor, torch.tensor(label, dtype=torch.long)
                
            except Exception as e:
                idx = random.randint(0, len(self.samples) - 1)
                continue
        
        # 兜底
        fallback_signal = torch.zeros((1, self.signal_len), dtype=torch.float32)
        if self.contrastive:
             return (fallback_signal, fallback_signal), torch.tensor(0, dtype=torch.long)
        return fallback_signal, torch.tensor(0, dtype=torch.long)

    def _process_signal(self, signal, fixed_start=None):
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
            if fixed_start is not None:
                start = fixed_start
            elif self.mode == 'train':
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