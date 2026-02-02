
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import json
import os
import random
from scipy import signal as scipy_signal

# ===================================================================
# 1. 信号增强器 (针对 ECG 调整)
# ===================================================================
class ECGAugmentor:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, signal):
        # 注意：ECG 通常不建议垂直翻转，因为 T 波倒置等有病理意义
        
        # 随机缩放
        if random.random() < self.p:
            scale = np.random.uniform(0.7, 1.3)
            signal = signal * scale
            
        # 基线漂移 (模拟呼吸影响)
        if random.random() < self.p:
            L = len(signal)
            t = np.linspace(0, L/100.0, L)
            freq = np.random.uniform(0.1, 0.5) # 呼吸频率
            amp = np.std(signal) * np.random.uniform(0.1, 0.5)
            drift = np.sin(2 * np.pi * freq * t) * amp
            signal = signal + drift
            
        # 随机掩码 (模拟接触不良)
        if random.random() < self.p:
            mask_len = int(len(signal) * 0.1)
            start = np.random.randint(0, len(signal) - mask_len)
            signal[start : start + mask_len] = 0 # 或者用 mean
            
        # 添加高斯噪声
        if random.random() < self.p:
            noise = np.random.normal(0, 0.05 * np.std(signal), len(signal))
            signal = signal + noise
            
        return signal

# ===================================================================
# 2. Dataset 定义 (单通道 ECG 版本)
# ===================================================================
class ECGDataset(Dataset):
    def __init__(self, data_root, split_file, mode='train', signal_len=3000, task_index=0, num_classes=2):
        self.data_root = data_root
        self.signal_len = signal_len
        self.mode = mode
        self.task_index = task_index
        self.num_classes = num_classes

        # 训练模式下启用增强
        self.augmentor = ECGAugmentor(p=0.5) if mode == 'train' else None

        with open(split_file, 'r') as f:
            splits = json.load(f)
        self.file_list = splits[mode]
        print(f"[{mode}] Loaded {len(self.file_list)} ECG samples.")

    def __len__(self):
        return len(self.file_list)

    def _ecg_filter(self, signal):
        """
        【ECG 清洗】
        0.5Hz - 40Hz 带通滤波
        """
        fs = 100.0 
        b, a = scipy_signal.butter(4, [0.5 / (0.5*fs), 40.0 / (0.5*fs)], btype='band')
        clean_signal = scipy_signal.filtfilt(b, a, signal)
        return clean_signal.copy()

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        file_path = os.path.join(self.data_root, filename)

        try:
            with open(file_path, 'rb') as f:
                content = pickle.load(f)
            
            # 【改动】只加载 ECG 通道 (Index 1)
            raw_data = content['data']
            if raw_data.ndim == 2:
                raw_ecg = raw_data[1, :].astype(np.float32)
            else:
                # 兼容可能已经是单通道的数据
                raw_ecg = raw_data.astype(np.float32)

            # 加载标签
            label_dict = content['label'][self.task_index]
            label = int(label_dict['class'])
            if label >= self.num_classes or label < 0:
                return None

            # 1. 滤波
            clean_ecg = self._ecg_filter(raw_ecg)
            clean_ecg = np.nan_to_num(clean_ecg, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 2. 裁剪或填充
            processed_ecg = self._crop_or_pad(clean_ecg)
            
            # 3. 数据增强
            if self.augmentor is not None:
                processed_ecg = self.augmentor(processed_ecg)

            # 4. 归一化 (Z-Score)
            norm_ecg = self._z_score_norm(processed_ecg)
            
            # 返回 Tensor, Shape: [signal_len]
            # DataLoader 会将其堆叠为 [Batch, signal_len]
            return torch.from_numpy(norm_ecg).float(), torch.tensor(label, dtype=torch.long)

        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            return None

    def _crop_or_pad(self, signal):
        current_len = len(signal)
        target_len = self.signal_len
        
        if current_len > target_len:
            if self.mode == 'train':
                start = np.random.randint(0, current_len - target_len)
            else:
                start = (current_len - target_len) // 2
            signal = signal[start : start + target_len]
            
        elif current_len < target_len:
            pad_len = target_len - current_len
            signal = np.pad(signal, (0, pad_len), 'constant', constant_values=0)
            
        return signal

    def _z_score_norm(self, signal):
        mean = np.mean(signal)
        std = np.std(signal)
        if std < 1e-6:
            return signal - mean
        return (signal - mean) / std

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return torch.tensor([]), torch.tensor([])
    signals, labels = zip(*batch)
    return torch.stack(signals), torch.stack(labels)