import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
import random
import pickle
import hashlib
from sklearn.model_selection import StratifiedShuffleSplit
from functools import lru_cache

# 独立的缓存函数，避免实例绑定的序列化问题
@lru_cache(maxsize=8)
def load_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

class DataSplitter:
    def __init__(self, index_file, split_ratio=0.1, seed=42):
        self.index_file = index_file
        self.split_ratio = split_ratio
        self.seed = seed
        
        with open(index_file, 'r') as f:
            self.full_data = json.load(f)
            
        self.total_samples = len(self.full_data)
        self.split_meta_file = index_file.replace('.json', f'_split_seed{seed}.json')

    def get_split(self):
        # 1. Checksum verification
        data_str = json.dumps(self.full_data, sort_keys=True)
        current_hash = hashlib.md5(data_str.encode('utf-8')).hexdigest()
        
        # 2. Check if split meta exists and matches
        if os.path.exists(self.split_meta_file):
            print(f"Loading existing split from {self.split_meta_file}")
            with open(self.split_meta_file, 'r') as f:
                meta = json.load(f)
            
            if meta['hash'] == current_hash and meta['split_ratio'] == self.split_ratio:
                print("Split checksum verified.")
                return meta['train_indices'], meta['val_indices']
            else:
                print("Split metadata mismatch or outdated. Re-splitting...")
        
        # 3. Create new split
        print(f"Creating new split (val_ratio={self.split_ratio})...")
        indices = np.arange(self.total_samples)
        labels = [item.get('label', 0) for item in self.full_data]
        
        # Stratified Split
        try:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=self.split_ratio, random_state=self.seed)
            train_idx, val_idx = next(sss.split(indices, labels))
        except ValueError:
            # Fallback to random split if only 1 class or labels issue
            print("Stratified split failed (maybe single class), falling back to random split.")
            np.random.seed(self.seed)
            np.random.shuffle(indices)
            split_point = int(self.total_samples * (1 - self.split_ratio))
            train_idx = indices[:split_point]
            val_idx = indices[split_point:]
            
        train_indices = train_idx.tolist()
        val_indices = val_idx.tolist()
        
        # 4. Save metadata
        meta = {
            'hash': current_hash,
            'split_ratio': self.split_ratio,
            'seed': self.seed,
            'train_indices': train_indices,
            'val_indices': val_indices,
            'timestamp': str(np.datetime64('now'))
        }
        
        with open(self.split_meta_file, 'w') as f:
            json.dump(meta, f, indent=2)
            
        return train_indices, val_indices

class PhysioSignalDataset(Dataset):
    def __init__(self, index_file=None, data_source=None, indices=None, signal_len=500, mode='train', 
                 min_std_threshold=1e-4,
                 max_std_threshold=5000.0,
                 max_abs_value=1e5,
                 ):
        self.signal_len = signal_len
        self.mode = mode
        self.min_std_threshold = min_std_threshold
        self.max_std_threshold = max_std_threshold
        self.max_abs_value = max_abs_value
        
        # 支持直接传入 data_source (list) 和 indices (list) 以避免重复加载
        if data_source is not None:
            self.index_data = data_source
        elif index_file is not None:
            if not os.path.exists(index_file):
                raise FileNotFoundError(f"Index file not found: {index_file}")
            print(f"Loading index from: {index_file} ...")
            with open(index_file, 'r') as f:
                self.index_data = json.load(f)
        else:
            raise ValueError("Must provide either index_file or data_source")
        
        # 如果指定了 indices，则只使用该子集
        if indices is not None:
            self.active_indices = indices
        else:
            self.active_indices = list(range(len(self.index_data)))
            
        # 预生成样本索引 (Mapping local_idx -> global_idx)
        self.samples = []
        for i in self.active_indices:
            self.samples.append({'idx': i, 'start': None}) # idx points to global index in self.index_data
            
        print(f"[{mode.upper()}] Dataset initialized with {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)


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
                
                # 使用缓存加载，避免频繁 IO 和反序列化
                content = load_pickle_file(file_path)
                raw_signal = content['data'][row_idx]
                
                if raw_signal.ndim == 1:
                    raw_signal = raw_signal[np.newaxis, :]
                    
                    if raw_signal.dtype != np.float32:
                        raw_signal = raw_signal.astype(np.float32)
                
                # 1. 基础检查
                # 注意：raw_signal 可能是只读的（来自缓存），如果后续有原地修改操作需要 copy
                # 目前的代码逻辑主要是读取和计算，或者 create new tensor，是安全的。
                # 但为了保险起见，如果需要修改 raw_signal，建议 raw_signal = raw_signal.copy()
                
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
                
                # 深度过滤逻辑：
                # 1. 检查 std 是否包含 NaN (np.std 在输入包含 NaN 时会返回 NaN)
                # 2. 检查是否有任意通道标准差过大 (伪影)
                # 3. 检查是否有任意通道标准差过小 (死线/脱落) -> 这里改为 np.any，只要有一个通道不行就换
                if np.isnan(std_vals).any() or \
                   np.any(std_vals > self.max_std_threshold) or \
                   np.any(std_vals < self.min_std_threshold):
                    idx = random.randint(0, len(self.samples) - 1)
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
                idx = random.randint(0, len(self.samples) - 1)
                continue
        
        # 兜底：返回全零信号和 0 标签
        fallback_signal = torch.zeros((1, self.signal_len), dtype=torch.float32)
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