import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import json
import os
import random
from scipy import signal as scipy_signal

# ===================================================================
# 1. 通用信号增强器 (适配多通道)
# ===================================================================
class MultiChannelAugmentor:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, signal):
        # signal shape: (M, L)
        M, L = signal.shape
        
        # 1. 随机翻转 (针对每个通道独立判定)
        # 某些信号(如ACC)翻转意味着方向改变，生理信号翻转意味着极性改变
        if random.random() < 0.5:
            # 生成一个 (M, 1) 的随机符号掩码
            flip_mask = np.random.choice([-1.0, 1.0], size=(M, 1))
            signal = signal * flip_mask

        # 2. 随机缩放 (模拟信号强度波动)
        if random.random() < self.p:
            scale = np.random.uniform(0.8, 1.2, size=(M, 1))
            signal = signal * scale

        # 3. 随机高斯噪声
        if random.random() < self.p:
            noise = np.random.normal(0, 0.05, size=(M, L))
            signal = signal + noise

        # 4. 随机通道丢弃 (模拟传感器接触不良)
        # 只有当 M > 1 时才执行，避免数据全空
        if M > 1 and random.random() < 0.2:
            drop_idx = np.random.randint(0, M)
            signal[drop_idx, :] = 0.0

        return signal

# ===================================================================
# 2. Dataset 定义
# ===================================================================
class DownstreamClassificationDataset(Dataset):
    def __init__(self, data_root, split_file, mode='train', signal_len=3000, task_index=0, num_classes=2):
        self.data_root = data_root
        self.signal_len = signal_len
        self.mode = mode
        self.task_index = task_index
        self.num_classes = num_classes

        # 训练集开启增强
        self.augmentor = MultiChannelAugmentor(p=0.5) if mode == 'train' else None

        with open(split_file, 'r') as f:
            splits = json.load(f)
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
            
            # --- 1. 加载数据 ---
            # 假设 content['data'] 可能是 (M, L) 或 (L,)
            raw_data = content['data']
            if raw_data.ndim == 1:
                raw_data = raw_data[np.newaxis, :] # (1, L)
            
            raw_signal = raw_data.astype(np.float32) # (M, L_raw)

            # 加载标签
            # 假设 label 结构是 list of dicts
            if isinstance(content['label'], list):
                label_dict = content['label'][self.task_index]
                label = int(label_dict['class'])
            else:
                # 兼容直接是 label 的情况
                label = int(content['label'])

            if label >= self.num_classes or label < 0:
                # 异常标签处理
                return torch.zeros(1, self.signal_len), torch.tensor(0, dtype=torch.long)

            # --- 2. 同步裁剪/填充 ---
            processed_signal = self._sync_crop_or_pad(raw_signal)
            
            # --- 3. 归一化 (Per-Channel Robust Scaling) ---
            # 避免不同模态(ECG/ACC)量纲不同
            processed_signal = self._robust_norm(processed_signal)

            # --- 4. 数据增强 ---
            if self.augmentor is not None:
                processed_signal = self.augmentor(processed_signal)

            # 转为 Tensor: (M, L)
            signal_tensor = torch.from_numpy(processed_signal)

            # --- 5. 通道乱序 (Channel Shuffling) ---
            # 关键：保持与预训练一致的 "Bag of Signals" 策略
            if self.mode == 'train':
                M = signal_tensor.shape[0]
                perm_indices = torch.randperm(M)
                signal_tensor = signal_tensor[perm_indices]

            return signal_tensor, torch.tensor(label, dtype=torch.long)

        except Exception as e:
            print(f"Error loading {filename}: {e}")
            # 返回全0兜底
            return torch.zeros(1, self.signal_len), torch.tensor(0, dtype=torch.long)

    def _sync_crop_or_pad(self, signal):
        """
        保证所有通道裁剪相同的时间窗口
        signal: (M, Current_Len)
        """
        M, current_len = signal.shape
        target_len = self.signal_len

        if current_len == target_len:
            return signal

        if current_len > target_len:
            if self.mode == 'train':
                start = np.random.randint(0, current_len - target_len)
            else:
                start = (current_len - target_len) // 2
            return signal[:, start : start + target_len]
        else:
            pad_len = target_len - current_len
            # 只在时间轴右侧填充
            return np.pad(signal, ((0, 0), (0, pad_len)), 'constant', constant_values=0)

    def _robust_norm(self, signal):
        """
        对每个通道独立进行 Robust Normalization
        signal: (M, L)
        """
        # axis=1 表示沿时间轴计算统计量
        median = np.median(signal, axis=1, keepdims=True)
        q25 = np.percentile(signal, 25, axis=1, keepdims=True)
        q75 = np.percentile(signal, 75, axis=1, keepdims=True)
        iqr = q75 - q25
        
        # 避免除以零
        iqr = np.where(iqr < 1e-6, 1.0, iqr)
        
        # 如果 IQR 极小，退化为减均值
        normalized = (signal - median) / iqr
        return normalized