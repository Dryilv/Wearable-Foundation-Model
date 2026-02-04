import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import json
import os
import random
from scipy import signal as scipy_signal

# ===================================================================
# 1. PPG 专用信号增强器
# ===================================================================
class PPGAugmentor:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, signal):
        # 1. 垂直翻转 (PPG 传感器佩戴差异)
        # 这是一个非常重要的增强，防止模型死记硬背波峰方向
        if random.random() < 0.5: 
            signal = -signal

        # 2. 随机缩放 (模拟灌注指数 PI 变化)
        if random.random() < self.p:
            scale = np.random.uniform(0.7, 1.3)
            signal = signal * scale

        # 3. 强基线漂移 (PPG 最怕呼吸干扰)
        # 我们故意加入漂移，训练模型去适应它
        if random.random() < self.p:
            L = len(signal)
            t = np.linspace(0, L/100.0, L)
            freq = np.random.uniform(0.1, 0.4) # 呼吸频率
            amp = np.std(signal) * np.random.uniform(0.2, 0.8)
            drift = np.sin(2 * np.pi * freq * t) * amp
            signal = signal + drift

        # 4. 随机掩码 (模拟运动伪影导致的信号丢失)
        if random.random() < self.p:
            mask_len = int(len(signal) * 0.1)
            start = np.random.randint(0, len(signal) - mask_len)
            # PPG 掉线通常是变成一条直线（基线），而不是 0
            signal[start : start + mask_len] = np.mean(signal)

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

        # PPG 增强器
        self.augmentor = PPGAugmentor(p=0.6) if mode == 'train' else None

        with open(split_file, 'r') as f:
            splits = json.load(f)
        self.file_list = splits[mode]
        print(f"[{mode}] Loaded {len(self.file_list)} PPG samples.")

    def __len__(self):
        return len(self.file_list)

    def _ppg_filter(self, signal):
        """
        【PPG 强力清洗】
        PPG 的有效频率比 ECG 低得多。
        0.5Hz - 8Hz 是 PPG 形态学特征（如重搏波）的核心区域。
        高于 8Hz 的通常是肌电噪声或环境光干扰。
        """
        fs = 100.0
        # 4阶巴特沃斯带通
        b, a = scipy_signal.butter(4, [0.5 / (0.5*fs), 8.0 / (0.5*fs)], btype='band')
        # 零相移滤波
        clean_signal = scipy_signal.filtfilt(b, a, signal)
        return clean_signal.copy()

    def _calculate_apg(self, signal):
        """
        计算加速度脉搏波 (APG) - 二阶导数
        这对于冠心病（血管硬化）识别至关重要。
        """
        # 一阶导数 (速度)
        vpg = np.gradient(signal)
        # 二阶导数 (加速度)
        apg = np.gradient(vpg)
        return apg

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        file_path = os.path.join(self.data_root, filename)

        try:
            with open(file_path, 'rb') as f:
                content = pickle.load(f)
            
            raw_data = content['data']
            if raw_data.ndim == 2: raw_signal = raw_data[0, :] 
            else: raw_signal = raw_data
            raw_signal = raw_signal.astype(np.float32)

            label_dict = content['label'][self.task_index]
            label = int(label_dict['class'])
            if label >= self.num_classes or label < 0:
                return torch.zeros(1, self.signal_len), torch.tensor(0, dtype=torch.long)

            # --- PPG 处理流程 ---
            
            # 1. 强力滤波 (0.5-8Hz)
            # 这一步能去掉绝大多数高频噪声，让波形变圆润
            clean_signal = self._ppg_filter(raw_signal)
            
            # 2. 裁剪/填充
            processed_signal = self._crop_or_pad(clean_signal)
            
            # 3. 数据增强
            if self.augmentor is not None:
                processed_signal = self.augmentor(processed_signal)

            # 4. 归一化 (Robust Scaling)
            processed_signal = self._robust_norm(processed_signal)
            
            # 【可选策略】
            # 如果模型一直学不会，可以尝试直接返回 APG (二阶导数)
            # processed_signal = self._calculate_apg(processed_signal)
            # processed_signal = self._robust_norm(processed_signal) # 再次归一化

            signal_tensor = torch.from_numpy(processed_signal).unsqueeze(0)
            return signal_tensor, torch.tensor(label, dtype=torch.long)

        except Exception as e:
            print(f"Error: {e}")
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
        q25, q75 = np.percentile(signal, [25, 75])
        iqr = q75 - q25
        median = np.median(signal)
        if iqr < 1e-6: return signal - np.mean(signal)
        return (signal - median) / iqr