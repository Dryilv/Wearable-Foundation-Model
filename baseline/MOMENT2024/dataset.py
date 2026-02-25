import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
import random
import pickle
import hashlib
from functools import lru_cache

# 独立的缓存函数，避免实例绑定的序列化问题
@lru_cache(maxsize=128)
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
        
        # 使用普通随机切分，不再依赖标签
        np.random.seed(self.seed)
        np.random.shuffle(indices)
        
        split_point = int(self.total_samples * (1 - self.split_ratio))
        train_indices = indices[:split_point].tolist()
        val_indices = indices[split_point:].tolist()
        
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

class MOMENTPhysioDataset(Dataset):
    def __init__(self, index_file=None, indices=None, seq_len=512, patch_len=8, mask_ratio=0.3, mode='train', stride=None):
        """
        基于索引的生理信号预训练数据集
        参数:
        - index_file: JSON 索引文件路径
        - indices: 选用的索引列表 (由 DataSplitter 提供)
        - seq_len: 序列长度 (默认 512)
        - patch_len: 每个 Patch 的长度 (默认 8)
        - mask_ratio: 掩码比例 (默认 0.3)
        - stride: 滑动窗口步长 (默认等于 seq_len，即无重叠)
        """
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.n_patches = seq_len // patch_len
        self.mask_ratio = mask_ratio
        self.mode = mode
        self.stride = stride if stride is not None else seq_len

        if index_file is not None:
            with open(index_file, 'r') as f:
                self.index_data = json.load(f)
        else:
            raise ValueError("Must provide index_file")

        if indices is not None:
            self.active_indices = indices
        else:
            self.active_indices = list(range(len(self.index_data)))

        # 预计算所有窗口
        self.samples = []
        print(f"[{mode.upper()}] Pre-calculating windows with stride {self.stride}...")
        for record_idx in self.active_indices:
            item_info = self.index_data[record_idx]
            # 假设 index_data 中包含序列长度 'length'，如果没有则需要加载文件获取
            # 为了效率，如果 index_data 没记长度，这里先假设一个基础长度或动态获取
            # 这里我们动态获取一次长度（会有一定初始化开销，但保证了窗口化的准确性）
            try:
                # 尝试从索引获取长度，避免频繁 IO
                total_l = item_info.get('length')
                if total_l is None:
                    # 如果索引没存长度，读取文件头（如果是 pickle 可能得全读，建议索引存长度）
                    content = load_pickle_file(item_info['path'])
                    total_l = content['data'].shape[1]
                
                # 为 ECG (0) 和 PPG (1) 分别计算窗口
                for channel_type in [0, 1]:
                    if total_l <= self.seq_len:
                        self.samples.append((record_idx, channel_type, 0))
                    else:
                        for start in range(0, total_l - self.seq_len + 1, self.stride):
                            self.samples.append((record_idx, channel_type, start))
                        # 确保覆盖末尾 (可选)
                        if (total_l - self.seq_len) % self.stride != 0:
                            self.samples.append((record_idx, channel_type, total_l - self.seq_len))
            except Exception as e:
                continue

        self.total_len = len(self.samples)
        print(f"[{mode.upper()}] MOMENTPhysioDataset initialized: {len(self.active_indices)} records -> {self.total_len} windows")

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        # 1. 获取预计算的窗口信息
        actual_record_idx, channel_type, start_offset = self.samples[idx]
        
        item_info = self.index_data[actual_record_idx]
        file_path = item_info['path']

        # 2. 加载数据
        try:
            content = load_pickle_file(file_path)
            raw_signal = content['data'] # 预期是 [5, L]
            
            # 选择通道: 1st (0) 是 ECG, 5th (4) 是 PPG
            channel_idx = 0 if channel_type == 0 else 4
            series_numpy = raw_signal[channel_idx, start_offset : start_offset + self.seq_len]
            
            # 3. 填充 (如果长度不足 seq_len，虽然窗口计算已尽量避免，但为了健壮性保留)
            if len(series_numpy) < self.seq_len:
                pad_len = self.seq_len - len(series_numpy)
                series_numpy = np.pad(series_numpy, (0, pad_len), 'constant')
                
            series = torch.from_numpy(series_numpy).float()

            # 4. 异常处理
            if torch.isnan(series).any() or torch.isinf(series).any():
                series = torch.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)

            # 5. RevIN (可逆实例归一化)
            mean = series.mean()
            std = series.std() + 1e-5
            series = (series - mean) / std

            # 6. Patching
            patches = series.view(self.n_patches, self.patch_len)

            # 7. Masking
            n_masked = int(self.n_patches * self.mask_ratio)
            mask = torch.zeros(self.n_patches, dtype=torch.bool)
            
            # 训练模式随机掩码，验证模式固定或随机（通常预训练也随机）
            perm = torch.randperm(self.n_patches)
            masked_indices = perm[:n_masked]
            mask[masked_indices] = True

            return patches, mask

        except Exception as e:
            # 发生错误时随机找一个样本重试
            # print(f"Error loading {file_path}: {e}")
            new_idx = random.randint(0, self.total_len - 1)
            return self.__getitem__(new_idx)

    def _process_signal(self, signal):
        """处理单通道信号的裁剪和填充"""
        current_len = len(signal)
        if current_len == self.seq_len:
            return signal
        
        if current_len > self.seq_len:
            if self.mode == 'train':
                start = np.random.randint(0, current_len - self.seq_len)
            else:
                start = (current_len - self.seq_len) // 2
            return signal[start : start + self.seq_len]
        else:
            pad_len = self.seq_len - current_len
            return np.pad(signal, (0, pad_len), 'constant', constant_values=0)
