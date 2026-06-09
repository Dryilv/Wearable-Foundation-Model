import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
import random
import pickle


class PhysioSignalDataset(Dataset):
    def __init__(self, index_file=None, data_source=None, indices=None, signal_len=500, mode='train',
                 min_std_threshold=1e-4,
                 max_std_threshold=5000.0,
                 max_abs_value=1e5,
                 expected_channels=5,
                 data_ratio=1.0,
                 use_sliding_window=False,
                 window_stride=500
                 ):
        self.signal_len = signal_len
        self.mode = mode
        self.min_std_threshold = min_std_threshold
        self.max_std_threshold = max_std_threshold
        self.max_abs_value = max_abs_value
        self.expected_channels = expected_channels
        self.use_sliding_window = use_sliding_window
        self.window_stride = window_stride

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

        if indices is not None:
            self.active_indices = indices
        else:
            self.active_indices = list(range(len(self.index_data)))

        if 0.0 < data_ratio < 1.0:
            total_samples = len(self.active_indices)
            keep_num = int(total_samples * data_ratio)
            if mode == 'train':
                 random.seed(42)
                 self.active_indices = random.sample(self.active_indices, keep_num)
            else:
                 self.active_indices = self.active_indices[:keep_num]
            print(f"[{mode.upper()}] Data Ratio: {data_ratio:.2f} | Using {len(self.active_indices)}/{total_samples} samples.")

        # 预加载所有数据到内存
        unique_paths = list(set(self.index_data[i]['path'] for i in self.active_indices))
        self._cache = {}
        from tqdm import tqdm as _tqdm
        for path in _tqdm(unique_paths, desc=f"[{mode.upper()}] Loading data into memory"):
            with open(path, 'rb') as f:
                self._cache[path] = pickle.load(f)
        print(f"[{mode.upper()}] Loaded {len(self._cache)} unique files into memory.")

        # 预生成样本索引
        self.samples = []
        for i in self.active_indices:
            item_info = self.index_data[i]

            if self.use_sliding_window and 'len' in item_info:
                total_len = item_info['len']
                if total_len > self.signal_len:
                    starts = range(0, total_len - self.signal_len + 1, self.window_stride)
                    for s in starts:
                        self.samples.append({'idx': i, 'start': s})
                else:
                    self.samples.append({'idx': i, 'start': 0})
            else:
                self.samples.append({'idx': i, 'start': None})

        print(f"[{mode.upper()}] Dataset initialized with {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        for _ in range(3):
            try:
                sample_info = self.samples[idx]
                original_idx = sample_info['idx']
                fixed_start = sample_info['start']

                item_info = self.index_data[original_idx]
                file_path = item_info['path']
                label = item_info.get('label', 0)

                content = self._cache[file_path]
                raw_signal = content['data']

                if raw_signal.ndim == 1:
                    raw_signal = raw_signal[np.newaxis, :]

                if raw_signal.dtype != np.float32:
                    raw_signal = raw_signal.astype(np.float32)

                num_channels = raw_signal.shape[0]
                if num_channels < 1:
                    idx = random.randint(0, len(self.samples) - 1)
                    continue

                if np.isnan(raw_signal).any() or np.isinf(raw_signal).any():
                    idx = random.randint(0, len(self.samples) - 1)
                    continue

                if np.max(np.abs(raw_signal)) > self.max_abs_value:
                    idx = random.randint(0, len(self.samples) - 1)
                    continue

                processed_signal = self._process_signal(raw_signal, fixed_start)

                if not isinstance(label, (int, float)) or np.isnan(label) or np.isinf(label):
                    label = 0

                signal_tensor = torch.from_numpy(processed_signal)  # (M, L)
                age = float(item_info.get('age', 0.0))

                return signal_tensor, torch.tensor(label, dtype=torch.long), torch.tensor(age, dtype=torch.float32)

            except Exception as e:
                print(f"Error loading sample {idx}: {e}")
                idx = random.randint(0, len(self.samples) - 1)
                continue

        # 兜底
        fallback_signal = torch.ones((1, self.signal_len), dtype=torch.float32) * 0.01
        return fallback_signal, torch.tensor(0, dtype=torch.long), torch.tensor(0.0, dtype=torch.float32)

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


def multi_channel_collate_fn(batch):
    """
    多通道预训练 Collate Function。
    Input: list of (signal_tensor, label, age)
    Output: (padded_signals, labels, ages, channel_mask)
    """
    signals = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    ages = [item[2] for item in batch]

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
    ages = torch.stack(ages)

    return padded_signals, labels, ages, channel_mask
