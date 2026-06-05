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
from utils_features import extract_features

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
                 expected_channels=5,  # 1) 将默认通道数参数设为 5
                 data_ratio=1.0,  # 新增: 数据使用比例 (0.0 - 1.0)
                 use_sliding_window=False, # 新增: 是否启用滑动窗口
                 window_stride=500        # 新增: 滑动窗口步长
                 ):
        self.signal_len = signal_len
        self.mode = mode
        self.min_std_threshold = min_std_threshold
        self.max_std_threshold = max_std_threshold
        self.max_abs_value = max_abs_value
        self.expected_channels = expected_channels # 保存期望通道数
        self.use_sliding_window = use_sliding_window
        self.window_stride = window_stride
        
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
            
        # 根据 data_ratio 进行采样
        if 0.0 < data_ratio < 1.0:
            total_samples = len(self.active_indices)
            keep_num = int(total_samples * data_ratio)
            # 使用固定 seed 确保可复现性，或者根据 mode 决定
            # 这里简单做顺序截断，或者随机采样
            if mode == 'train':
                 random.seed(42)
                 self.active_indices = random.sample(self.active_indices, keep_num)
            else:
                 self.active_indices = self.active_indices[:keep_num]
            print(f"[{mode.upper()}] Data Ratio: {data_ratio:.2f} | Using {len(self.active_indices)}/{total_samples} samples.")
            
        # 预生成样本索引 (Mapping local_idx -> global_idx)
        self.samples = []
        for i in self.active_indices:
            item_info = self.index_data[i]
            
            # 如果开启滑动窗口且 index 中包含长度信息
            if self.use_sliding_window and 'len' in item_info:
                total_len = item_info['len']
                if total_len > self.signal_len:
                    # 计算窗口起始点
                    starts = range(0, total_len - self.signal_len + 1, self.window_stride)
                    for s in starts:
                        self.samples.append({'idx': i, 'start': s})
                else:
                    self.samples.append({'idx': i, 'start': 0})
            else:
                # 默认行为：每个条目作为一个样本，start 为 None (触发随机或中心裁剪)
                self.samples.append({'idx': i, 'start': None})
            
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
                # row_idx = item_info.get('row', 0) # Deprecated
                label = item_info.get('label', 0) 
                
                # 使用缓存加载，避免频繁 IO 和反序列化
                content = load_pickle_file(file_path)
                
                # [Modified] 直接读取 data，不再使用 row 索引
                raw_signal = content['data']

                if raw_signal.ndim == 1:
                    raw_signal = raw_signal[np.newaxis, :]
                    
                if raw_signal.dtype != np.float32:
                    raw_signal = raw_signal.astype(np.float32)

                num_channels = raw_signal.shape[0]
                if num_channels < 1:
                    print(f"Warning: Sample {original_idx} has {num_channels} channels, skipping...")
                    idx = random.randint(0, len(self.samples) - 1)
                    continue

                if np.isnan(raw_signal).any() or np.isinf(raw_signal).any():
                    idx = random.randint(0, len(self.samples) - 1)
                    continue
                
                if np.max(np.abs(raw_signal)) > self.max_abs_value:
                    idx = random.randint(0, len(self.samples) - 1)
                    continue

                # 2. 同步裁剪或填充 (使用固定起始位置或随机起始位置)
                processed_signal = self._process_signal(raw_signal, fixed_start)

                # 计算统计特征 (跨通道聚合)
                fs = self.index_data[original_idx].get('fs', 100)
                all_feats = []
                for ch in range(num_channels):
                    ch_signal = processed_signal[ch:ch+1, :]
                    ch_stats = extract_features(ch_signal, ch, fs=fs, item_info=item_info)
                    all_feats.append(ch_stats[:-1])
                stats = np.mean(np.stack(all_feats), axis=0)
                age = 0.0
                if item_info is not None and 'age' in item_info:
                    age = float(item_info['age'])
                stats = np.append(stats, [age])

                # 3. 逐通道质量检查
                std_vals = np.std(processed_signal, axis=1, keepdims=True) # (M, 1)
                
                # 深度过滤逻辑：
                # 1. 检查 std 是否包含 NaN (np.std 在输入包含 NaN 时会返回 NaN)
                # 2. 检查是否有任意通道标准差过大 (伪影)
                # 3. 检查是否有任意通道标准差过小 (死线/脱落) -> 这里改为 np.any，只要有一个通道不行就换
                if np.isnan(std_vals).any() or \
                   np.any(std_vals > self.max_std_threshold) or \
                   np.any(std_vals < self.min_std_threshold):
                    # print(f"Skipping sample {original_idx}: std_vals={std_vals.flatten()}")
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

                # Double check for flat signal after normalization (should not happen if std check passed)
                if np.abs(processed_signal).max() < 1e-6:
                     idx = random.randint(0, len(self.samples) - 1)
                     continue

                # 转为 Tensor
                signal_tensor = torch.from_numpy(processed_signal)  # (M, L)
                stats_tensor = torch.from_numpy(stats)

                return signal_tensor, torch.tensor(label, dtype=torch.long), stats_tensor

            except Exception as e:
                print(f"Error loading sample {idx}: {e}")
                idx = random.randint(0, len(self.samples) - 1)
                continue
        
        # 兜底：如果重试 3 次依然失败，尝试从数据集开头找一个可能有效的样本
        # 避免返回全零信号导致模型梯度冲击
        print(f"Warning: Fallback triggered for idx {idx} after 3 retries. Attempting to return a safe sample.")
        try:
            sample_info = self.samples[0]
            content = load_pickle_file(self.index_data[sample_info['idx']]['path'])
            safe_signal = content['data']
            if safe_signal.ndim == 1:
                safe_signal = safe_signal[np.newaxis, :]
            if safe_signal.dtype != np.float32:
                safe_signal = safe_signal.astype(np.float32)
            safe_signal = safe_signal[:, :self.signal_len]
            
            fs = self.index_data[sample_info['idx']].get('fs', 100)
            num_ch = safe_signal.shape[0]
            all_feats = []
            for ch in range(num_ch):
                ch_signal = safe_signal[ch:ch+1, :]
                ch_stats = extract_features(ch_signal, ch, fs=fs, item_info=self.index_data[sample_info['idx']])
                all_feats.append(ch_stats[:-1])
            stats = np.mean(np.stack(all_feats), axis=0)
            age = 0.0
            if 'age' in self.index_data[sample_info['idx']]:
                age = float(self.index_data[sample_info['idx']]['age'])
            stats = np.append(stats, [age])
            
            safe_signal = (safe_signal - np.mean(safe_signal, axis=1, keepdims=True)) / (np.std(safe_signal, axis=1, keepdims=True) + 1e-5)
            return torch.from_numpy(safe_signal).float(), torch.tensor(0, dtype=torch.long), torch.from_numpy(stats)
        except:
            fallback_signal = torch.ones((1, self.signal_len), dtype=torch.float32) * 0.01
            fallback_stats = torch.zeros(16, dtype=torch.float32)
            return fallback_signal, torch.tensor(0, dtype=torch.long), fallback_stats

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

def fixed_channel_collate_fn(batch):
    """
    单通道数据的 Collate Function。
    Output: (B, 1, L), (B,) channel_ids, (B,) labels, (B, 16) stats
    """
    signals = [item[0] for item in batch]
    channel_ids = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    stats = [item[3] for item in batch]

    padded_signals = torch.stack(signals)  # (B, 1, L)
    channel_ids = torch.stack(channel_ids)  # (B,)
    labels = torch.stack(labels)  # (B,)
    stats_tensor = torch.stack(stats) # (B, 16)

    return padded_signals, channel_ids, labels, stats_tensor

def multi_channel_collate_fn(batch):
    """
    多通道预训练的 Collate Function，处理不同样本通道数不一致的情况。
    Batch: list of tuples (signal_tensor, label, stats)
    signal_tensor shape: (M_i, L)
    Output: padded_signals (B, Max_M, L), None, labels (B,), stats (B, 16), channel_mask (B, Max_M)
    """
    signals = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    stats = [item[2] for item in batch]

    max_m = max([s.shape[0] for s in signals])
    signal_len = signals[0].shape[1]
    batch_size = len(batch)

    padded_signals = torch.zeros((batch_size, max_m, signal_len), dtype=signals[0].dtype)
    channel_mask = torch.zeros((batch_size, max_m), dtype=torch.bool)

    for i, s in enumerate(signals):
        m = s.shape[0]
        padded_signals[i, :m, :] = s
        channel_mask[i, :m] = True

    labels = torch.stack(labels)
    stats_tensor = torch.stack(stats)

    return padded_signals, None, labels, stats_tensor, channel_mask