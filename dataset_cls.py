import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import json
import os

class DownstreamClassificationDataset(Dataset):
    def __init__(self, data_root, split_file, mode='train', signal_len=3000, task_index=0):
        self.data_root = data_root
        self.signal_len = signal_len
        self.mode = mode
        self.task_index = task_index

        with open(split_file, 'r') as f:
            splits = json.load(f)
        
        if mode not in splits:
            raise ValueError(f"Mode {mode} not found in split file.")
        
        self.file_list = splits[mode]
        print(f"[{mode}] Loaded {len(self.file_list)} samples.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        file_path = os.path.join(self.data_root, filename)

        try:
            with open(file_path, 'rb') as f:
                content = pickle.load(f)
                
            raw_data = content['data']
            if raw_data.ndim == 2:
                raw_signal = raw_data[0, :] 
            else:
                raw_signal = raw_data
            
            raw_signal = raw_signal.astype(np.float32)

            # 获取标签
            label_dict = content['label'][self.task_index]
            if 'class' in label_dict:
                label = int(label_dict['class'])
            else:
                raise ValueError(f"Sample {filename} label[{self.task_index}] does not have 'class' key.")

            # 预处理
            processed_signal = self._process_signal(raw_signal)

            # 转 Tensor [1, L]
            signal_tensor = torch.from_numpy(processed_signal).unsqueeze(0)
            
            return signal_tensor, torch.tensor(label, dtype=torch.long)

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return torch.zeros(1, self.signal_len), torch.tensor(0, dtype=torch.long)

    def _process_signal(self, signal):
        """
        处理逻辑：裁剪/填充 -> Instance Norm
        注意：必须进行 Instance Norm，否则 CWT 的数值范围会异常
        """
        current_len = len(signal)
        target_len = self.signal_len

        # 1. 裁剪
        if current_len > target_len:
            if self.mode == 'train':
                start = np.random.randint(0, current_len - target_len)
            else:
                start = (current_len - target_len) // 2
            signal = signal[start : start + target_len]
        
        # 2. Instance Normalization (关键修改)
        # 必须将信号归一化到均值0方差1，与预训练保持一致
        mean = np.mean(signal)
        std = np.std(signal)
        if std < 1e-6: std = 1e-6
        signal = (signal - mean) / std

        # 3. 填充 (用0填充，因为均值已为0)
        if len(signal) < target_len:
            pad_len = target_len - len(signal)
            signal = np.pad(signal, (0, pad_len), 'constant', constant_values=0)
            
        return signal