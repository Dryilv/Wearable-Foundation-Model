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
                 max_std_threshold=5000.0,
                 max_abs_value=1e5 # 新增：绝对值上限过滤
                 ):
        self.signal_len = signal_len
        self.mode = mode
        self.min_std_threshold = min_std_threshold
        self.max_std_threshold = max_std_threshold
        self.max_abs_value = max_abs_value
        
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
                row_idx = item_info['row'] 
                label = item_info.get('label', 0) # 加载标签，默认为 0
                
                with open(file_path, 'rb') as f:
                    content = pickle.load(f)
                    raw_signal = content['data'][row_idx]
                    
                    if raw_signal.ndim == 1:
                        raw_signal = raw_signal[np.newaxis, :]
                    
                    if raw_signal.dtype != np.float32:
                        raw_signal = raw_signal.astype(np.float32)
                
                # 1. 基础检查：NaN/Inf 和 极端异常值 (Sensor Saturation)
                if np.isnan(raw_signal).any() or np.isinf(raw_signal).any():
                    idx = random.randint(0, len(self.index_data) - 1)
                    continue
                
                if np.max(np.abs(raw_signal)) > self.max_abs_value:
                    idx = random.randint(0, len(self.index_data) - 1)
                    continue

                # 2. 同步裁剪或填充
                processed_signal = self._process_signal(raw_signal)

                # 3. 逐通道质量检查
                std_vals = np.std(processed_signal, axis=1, keepdims=True) # (M, 1)
                
                # 深度过滤逻辑：
                # 1. 检查 std 是否包含 NaN (np.std 在输入包含 NaN 时会返回 NaN)
                # 2. 检查是否有任意通道标准差过大 (伪影)
                # 3. 检查是否有任意通道标准差过小 (死线/脱落) -> 这里改为 np.any，只要有一个通道不行就换
                if np.isnan(std_vals).any() or \
                   np.any(std_vals > self.max_std_threshold) or \
                   np.any(std_vals < self.min_std_threshold):
                    idx = random.randint(0, len(self.index_data) - 1)
                    continue
                
                # 检查标签是否合法
                if not isinstance(label, (int, float)) or np.isnan(label) or np.isinf(label):
                    label = 0

                # 4. 逐通道 Z-Score 归一化 (增加稳定性控制)
                mean_vals = np.mean(processed_signal, axis=1, keepdims=True)
                # 使用稍大的 epsilon 并在归一化后裁剪
                processed_signal = (processed_signal - mean_vals) / (std_vals + 1e-5)
                processed_signal = np.clip(processed_signal, -10, 10) # 限制在 [-10, 10] 标准差范围内

                # 转为 Tensor
                signal_tensor = torch.from_numpy(processed_signal) 

                # 5. 通道乱序 (仅训练模式)
                if self.mode == 'train':
                    M = signal_tensor.shape[0]
                    perm_indices = torch.randperm(M)
                    signal_tensor = signal_tensor[perm_indices]

                return signal_tensor, torch.tensor(label, dtype=torch.long)

            except Exception as e:
                idx = random.randint(0, len(self.index_data) - 1)
                continue
        
        # 兜底：返回全零信号和 0 标签
        fallback_signal = torch.zeros((1, self.signal_len), dtype=torch.float32)
        return fallback_signal, torch.tensor(0, dtype=torch.long)

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