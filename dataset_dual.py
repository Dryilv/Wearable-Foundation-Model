import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import json
import os
import random
from scipy import signal as scipy_signal

# ===================================================================
# 1. 信号增强器 (保持不变，但只对 PPG 应用)
# ===================================================================
class PPGAugmentor:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, signal):
        # 垂直翻转
        if random.random() < 0.5: 
            signal = -signal
        # 随机缩放
        if random.random() < self.p:
            scale = np.random.uniform(0.7, 1.3)
            signal = signal * scale
        # 强基线漂移
        if random.random() < self.p:
            L = len(signal)
            t = np.linspace(0, L/100.0, L)
            freq = np.random.uniform(0.1, 0.4)
            amp = np.std(signal) * np.random.uniform(0.2, 0.8)
            drift = np.sin(2 * np.pi * freq * t) * amp
            signal = signal + drift
        # 随机掩码
        if random.random() < self.p:
            mask_len = int(len(signal) * 0.1)
            start = np.random.randint(0, len(signal) - mask_len)
            signal[start : start + mask_len] = np.mean(signal)
        return signal

# ===================================================================
# 2. Dataset 定义 (双通道版本)
# ===================================================================
class DualChannelDataset(Dataset):
    def __init__(self, data_root, split_file, mode='train', signal_len=3000, task_index=0, num_classes=2):
        self.data_root = data_root
        self.signal_len = signal_len
        self.mode = mode
        self.task_index = task_index
        self.num_classes = num_classes

        # PPG 增强器 (只对 PPG 通道使用)
        self.ppg_augmentor = PPGAugmentor(p=0.6) if mode == 'train' else None

        with open(split_file, 'r') as f:
            splits = json.load(f)
        self.file_list = splits[mode]
        print(f"[{mode}] Loaded {len(self.file_list)} dual-channel (PPG+ECG) samples.")

    def __len__(self):
        return len(self.file_list)

    # --- PPG 专用处理函数 ---
    def _ppg_filter(self, signal):
        """PPG 强力清洗 (0.5Hz - 8Hz)"""
        fs = 100.0 # 假设采样率为 100Hz
        b, a = scipy_signal.butter(4, [0.5 / (0.5*fs), 8.0 / (0.5*fs)], btype='band')
        clean_signal = scipy_signal.filtfilt(b, a, signal)
        return clean_signal.copy()

    # --- ECG 专用处理函数 ---
    def _ecg_filter(self, signal):
        """
        【ECG 清洗】
        ECG 的频率范围更广，特别是 QRS 波群包含大量高频成分。
        滤波范围通常设为 0.5Hz - 40Hz，以去除基线漂移和高频噪声。
        """
        fs = 100.0 # 假设采样率为 100Hz
        # 4阶巴特沃斯带通
        b, a = scipy_signal.butter(4, [0.5 / (0.5*fs), 40.0 / (0.5*fs)], btype='band')
        # 零相移滤波
        clean_signal = scipy_signal.filtfilt(b, a, signal)
        return clean_signal.copy()

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        file_path = os.path.join(self.data_root, filename)

        try:
            with open(file_path, 'rb') as f:
                content = pickle.load(f)
            
            # 【改动 1】加载两个通道的数据
            # 假设 pkl 文件中 'data' 的 shape 是 [2, sequence_length]
            # 通道 0: PPG, 通道 1: ECG
            raw_data = content['data']
            if raw_data.ndim != 2 or raw_data.shape[0] < 2:
                raise ValueError(f"Data in {filename} is not 2-channel. Shape is {raw_data.shape}")
            
            raw_ppg = raw_data[0, :].astype(np.float32)
            raw_ecg = raw_data[1, :].astype(np.float32)

            # 加载标签 (保持不变)
            label_dict = content['label'][self.task_index]
            label = int(label_dict['class'])
            if label >= self.num_classes or label < 0:
                # 返回一个占位符，之后在 collate_fn 中可以过滤掉
                return None

            # --- 【改动 2】分离处理流程 ---
            
            # 1. 独立滤波
            clean_ppg = self._ppg_filter(raw_ppg)
            clean_ecg = self._ecg_filter(raw_ecg)
            
            clean_ppg = np.nan_to_num(clean_ppg, nan=0.0, posinf=0.0, neginf=0.0)
            clean_ecg = np.nan_to_num(clean_ecg, nan=0.0, posinf=0.0, neginf=0.0)
            # 2. 同步裁剪/填充 (关键步骤，保证对齐)
            processed_ppg, processed_ecg = self._sync_crop_or_pad(clean_ppg, clean_ecg)
            
            # 3. 只对 PPG 进行数据增强
            if self.ppg_augmentor is not None:
                processed_ppg = self.ppg_augmentor(processed_ppg)

            # 4. 独立归一化
            # 使用 Z-Score 标准化，因为它对 ECG 的 QRS 波峰更鲁棒
            norm_ppg = self._z_score_norm(processed_ppg)
            norm_ecg = self._z_score_norm(processed_ecg)
            
            # 【改动 3】堆叠成一个张量
            # 将处理好的 numpy 数组堆叠起来
            final_signal = np.stack([norm_ppg, norm_ecg], axis=0)
            
            signal_tensor = torch.from_numpy(final_signal) # Shape: [2, signal_len]
            return signal_tensor, torch.tensor(label, dtype=torch.long)

        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            # 返回 None，方便后续过滤
            return None

    def _sync_crop_or_pad(self, signal1, signal2):
        """
        【新函数】对两个信号进行同步的裁剪或填充
        """
        # 假设两个信号输入时长度一致
        current_len = len(signal1)
        target_len = self.signal_len
        
        if current_len > target_len:
            if self.mode == 'train':
                # 随机选择一个起始点
                start = np.random.randint(0, current_len - target_len)
            else:
                # 测试时从中间裁剪
                start = (current_len - target_len) // 2
            
            # 对两个信号使用完全相同的起始点和结束点
            signal1 = signal1[start : start + target_len]
            signal2 = signal2[start : start + target_len]
            
        if current_len < target_len:
            pad_len = target_len - current_len
            # 对两个信号进行相同的填充
            signal1 = np.pad(signal1, (0, pad_len), 'constant', constant_values=0)
            signal2 = np.pad(signal2, (0, pad_len), 'constant', constant_values=0)
            
        return signal1, signal2

    def _z_score_norm(self, signal):
        """
        【新函数】使用 Z-Score 标准化
        (x - mean) / std
        """
        mean = np.mean(signal)
        std = np.std(signal)
        if std < 1e-6:
            return signal - mean # 避免除以零
        return (signal - mean) / std

# ===================================================================
# 3. 自定义 Collate Function (过滤坏样本)
# ===================================================================
def collate_fn(batch):
    """
    由于 __getitem__ 在出错时会返回 None，
    这个函数会过滤掉这些 None 值，确保 DataLoader 不会报错。
    """
    # 过滤掉批次中的 None
    batch = [b for b in batch if b is not None]
    if not batch:
        # 如果整个批次都是坏数据，返回空的 tensor
        return torch.tensor([]), torch.tensor([])
    
    # 正常地将数据和标签解包并堆叠
    signals, labels = zip(*batch)
    return torch.stack(signals), torch.stack(labels)