import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import json
import os

class DownstreamClassificationDataset(Dataset):
    def __init__(self, data_root, split_file, mode='train', signal_len=1000, task_index=0, num_classes=2):
        """
        Args:
            num_classes: 
                如果为 2: 将执行二分类映射 (0->0, 1~5->1)
                如果 > 2: 保持原始标签 (0~5)
        """
        self.data_root = data_root
        self.signal_len = signal_len
        self.mode = mode
        self.task_index = task_index
        self.num_classes = num_classes

        with open(split_file, 'r') as f:
            splits = json.load(f)
        
        if mode not in splits:
            raise ValueError(f"Mode {mode} not found in split file.")
        
        self.file_list = splits[mode]
        print(f"[{mode}] Loaded {len(self.file_list)} samples. (Num Classes: {num_classes})")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        file_path = os.path.join(self.data_root, filename)

        # 【修改】移除 try-except，让错误暴露出来，防止静默失败导致 AUC=0
        with open(file_path, 'rb') as f:
            content = pickle.load(f)
            
        raw_data = content['data']
        # 处理可能的多维度情况
        if raw_data.ndim == 2:
            raw_signal = raw_data[0, :] 
        else:
            raw_signal = raw_data
        
        raw_signal = raw_signal.astype(np.float32)

        # 获取原始标签
        label_dict = content['label'][self.task_index]
        if 'class' in label_dict:
            original_label = int(label_dict['class'])
        else:
            raise ValueError(f"Sample {filename} label[{self.task_index}] does not have 'class' key.")

        # 【关键修改】标签映射逻辑
        if self.num_classes == 2:
            # 二分类：0是正常，1-5是异常 -> 映射为 0 和 1
            label = 1 if original_label > 0 else 0
        else:
            # 多分类：保持 0-5
            label = original_label

        # 预处理 (Instance Norm)
        processed_signal = self._process_signal(raw_signal)

        # 转 Tensor [1, L]
        signal_tensor = torch.from_numpy(processed_signal).unsqueeze(0)
        
        return signal_tensor, torch.tensor(label, dtype=torch.long)

    def _process_signal(self, signal):
        """
        处理逻辑：裁剪/填充 -> Instance Norm
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
        
        # 2. Instance Normalization (关键：防止数值范围异常)
        mean = np.mean(signal)
        std = np.std(signal)
        if std < 1e-6: std = 1e-6
        signal = (signal - mean) / std

        # 3. 填充
        if len(signal) < target_len:
            pad_len = target_len - len(signal)
            signal = np.pad(signal, (0, pad_len), 'constant', constant_values=0)
            
        return signal