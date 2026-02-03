import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import json
import os
import random
from scipy import signal as scipy_signal

# ===================================================================
# 1. 通用信号增强器 (适配多通道 & 增强多样性)
# ===================================================================
class MultiChannelAugmentor:
    def __init__(self, p=0.5, noise_std=0.02, scale_range=(0.8, 1.2)):
        self.p = p
        self.noise_std = noise_std
        self.scale_range = scale_range

    def __call__(self, signal):
        # signal shape: (M, L)
        M, L = signal.shape
        
        # 1. 随机翻转 (针对每个通道独立判定)
        # 【修改】禁用随机翻转，防止破坏生物信号极性（如 ECG T波倒置）
        # if random.random() < self.p:
        #     flip_mask = np.random.choice([-1.0, 1.0], size=(M, 1))
        #     signal = signal * flip_mask

        # 2. 随机缩放 (模拟信号强度波动)
        if random.random() < self.p:
            scale = np.random.uniform(self.scale_range[0], self.scale_range[1], size=(M, 1))
            signal = signal * scale

        # 3. 随机高斯噪声
        if random.random() < self.p:
            noise = np.random.normal(0, self.noise_std, size=(M, L))
            signal = signal + noise

        # 4. 随机时间偏移 (Phase Shift) - 模拟采样起始点微差
        if random.random() < self.p:
            shift = np.random.randint(-20, 20)
            signal = np.roll(signal, shift, axis=1)

        # 5. 随机通道丢弃 (模拟传感器接触不良)
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

        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")

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
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Data file not found: {file_path}")

            with open(file_path, 'rb') as f:
                content = pickle.load(f)
            
            # --- 1. 加载数据 ---
            # 兼容不同结构的 content
            if isinstance(content, dict) and 'data' in content:
                raw_data = content['data']
            else:
                raw_data = content # 假设直接是数据
                
            if isinstance(raw_data, list):
                raw_data = np.array(raw_data)

            if raw_data.ndim == 1:
                raw_data = raw_data[np.newaxis, :] # (1, L)
            
            raw_signal = raw_data.astype(np.float32) # (M, L_raw)

            # --- 2. 加载标签 ---
            label = 0
            if isinstance(content, dict) and 'label' in content:
                target_label = content['label']
                if isinstance(target_label, list):
                    if self.task_index < len(target_label):
                        label_item = target_label[self.task_index]
                        label = int(label_item['class']) if isinstance(label_item, dict) else int(label_item)
                else:
                    label = int(target_label)
            
            # 异常标签限制
            label = max(0, min(label, self.num_classes - 1))

            # --- 3. 同步裁剪/填充 ---
            processed_signal = self._sync_crop_or_pad(raw_signal)
            
            # --- 4. 归一化 (Per-Channel Robust Scaling) ---
            # 核心：消除不同传感器量纲差异
            processed_signal = self._robust_norm(processed_signal)

            # --- 5. 数据增强 ---
            if self.augmentor is not None:
                processed_signal = self.augmentor(processed_signal)

            # 转为 Tensor: (M, L)
            signal_tensor = torch.from_numpy(processed_signal)

            # --- 6. 通道乱序 (Channel Shuffling) ---
            # 关键：微调时也保持 Bag-of-Signals 逻辑，增强对通道顺序的不敏感性
            # 【修改】禁用通道乱序，因为下游任务通常依赖固定的通道顺序（空间信息）
            # if self.mode == 'train':
            #     M = signal_tensor.shape[0]
            #     if M > 1:
            #         perm_indices = torch.randperm(M)
            #         signal_tensor = signal_tensor[perm_indices]

            return signal_tensor, torch.tensor(label, dtype=torch.long)

        except Exception as e:
            # print(f"Error loading {filename}: {e}") # 生产环境建议 log 而非 print
            # 返回全0兜底，防止训练中断
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
            # 使用边缘填充而非 0 填充，减少 CWT 伪影
            return np.pad(signal, ((0, 0), (0, pad_len)), mode='edge')

    def _robust_norm(self, signal):
        """
        对每个通道独立进行 Robust Normalization (Median & IQR)
        """
        # axis=1 表示沿时间轴
        median = np.median(signal, axis=1, keepdims=True)
        q25 = np.percentile(signal, 25, axis=1, keepdims=True)
        q75 = np.percentile(signal, 75, axis=1, keepdims=True)
        iqr = q75 - q25
        
        # 避免除以零，对于全 0 通道赋予默认尺度
        iqr = np.where(iqr < 1e-6, 1.0, iqr)
        
        normalized = (signal - median) / iqr
        
        # 数值截断，防止离群值
        normalized = np.clip(normalized, -20.0, 20.0)
        return normalized
