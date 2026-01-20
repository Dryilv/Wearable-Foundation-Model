import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import json
import os
import random
from scipy import signal as scipy_signal

# ===================================================================
# 1. 信号增强器 (保持之前的冠心病优化策略)
# ===================================================================
class SignalAugmentor:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, signal):
        # 1. 随机缩放
        if random.random() < self.p:
            scale_factor = np.random.uniform(0.8, 1.2)
            signal = signal * scale_factor

        # 2. 基线漂移
        if random.random() < self.p:
            L = len(signal)
            t = np.linspace(0, L / 100.0, L)
            freq = np.random.uniform(0.1, 0.5)
            amp = np.std(signal) * np.random.uniform(0.1, 0.5)
            drift = np.sin(2 * np.pi * freq * t) * amp
            signal = signal + drift

        # 3. 高斯噪声
        if random.random() < self.p:
            noise_amp = np.std(signal) * np.random.uniform(0.01, 0.05)
            noise = np.random.normal(0, noise_amp, len(signal))
            signal = signal + noise

        # 4. 随机掩码
        if random.random() < self.p:
            mask_len = int(len(signal) * np.random.uniform(0.05, 0.15))
            start = np.random.randint(0, len(signal) - mask_len)
            signal[start : start + mask_len] = 0.0

        # 5. 局部垂直平移 (ST段增强)
        if random.random() < 0.3:
            seg_len = int(len(signal) * 0.1)
            if len(signal) > seg_len:
                start = np.random.randint(0, len(signal) - seg_len)
                offset = np.random.uniform(-0.3, 0.3) * (np.std(signal) + 1e-6)
                signal[start : start + seg_len] += offset

        return signal

# ===================================================================
# 2. Dataset 定义
# ===================================================================
class DownstreamClassificationDataset(Dataset):
    def __init__(self, data_root, split_file, mode='train', signal_len=3000, task_index=0, num_classes=2):
        """
        新增 num_classes 参数，用于接收 finetune.py 传入的类别数，并进行标签校验。
        """
        self.data_root = data_root
        self.signal_len = signal_len
        self.mode = mode
        self.task_index = task_index
        self.num_classes = num_classes  # 【新增】保存类别数

        # 训练模式开启增强
        self.augmentor = SignalAugmentor(p=0.7) if mode == 'train' else None

        with open(split_file, 'r') as f:
            splits = json.load(f)
        
        if mode not in splits:
            raise ValueError(f"Mode {mode} not found in split file.")
        
        self.file_list = splits[mode]
        print(f"[{mode}] Loaded {len(self.file_list)} samples. (Num Classes check: {num_classes})")

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
                raise ValueError(f"Sample {filename} label[{self.task_index}] missing 'class'.")

            # 【新增】标签合法性检查
            if label >= self.num_classes or label < 0:
                # 如果标签越界，打印警告并返回全0数据（或者你可以选择抛出异常）
                # print(f"Warning: Label {label} out of bounds (0-{self.num_classes-1}) in {filename}")
                return torch.zeros(1, self.signal_len), torch.tensor(0, dtype=torch.long)

            # --- 处理流程 ---
            # 1. 裁剪/填充
            processed_signal = self._crop_or_pad(raw_signal)
            
            # 2. 数据增强
            if self.augmentor is not None:
                processed_signal = self.augmentor(processed_signal)

            # 3. 归一化 (Robust Scaling)
            processed_signal = self._robust_norm(processed_signal)

            # 转 Tensor
            signal_tensor = torch.from_numpy(processed_signal).unsqueeze(0)
            
            return signal_tensor, torch.tensor(label, dtype=torch.long)

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return torch.zeros(1, self.signal_len), torch.tensor(0, dtype=torch.long)

    def _crop_or_pad(self, signal):
        current_len = len(signal)
        target_len = self.signal_len

        if current_len > target_len:
            if self.mode == 'train':
                start = np.random.randint(0, current_len - target_len)
            else:
                start = (current_len - target_len) // 2
            signal = signal[start : start + target_len]
        
        if len(signal) < target_len:
            pad_len = target_len - len(signal)
            signal = np.pad(signal, (0, pad_len), 'constant', constant_values=0)
            
        return signal

    def _robust_norm(self, signal):
        q25 = np.percentile(signal, 25)
        q75 = np.percentile(signal, 75)
        iqr = q75 - q25
        median = np.median(signal)
        
        if iqr < 1e-6:
            return signal - np.mean(signal)
            
        return (signal - median) / iqr