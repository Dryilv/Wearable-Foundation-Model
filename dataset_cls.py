import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import json
import os
import random
from scipy import signal as scipy_signal

# ===================================================================
# 1. 信号增强器 (针对生理信号定制)
# ===================================================================
class SignalAugmentor:
    """
    包含针对 ECG/PPG 的物理增强策略。
    重点增加了针对冠心病 (ST段改变) 的局部平移增强。
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, signal):
        # 1. 随机缩放 (模拟增益变化)
        if random.random() < self.p:
            scale_factor = np.random.uniform(0.8, 1.2)
            signal = signal * scale_factor

        # 2. 基线漂移 (模拟呼吸/运动)
        if random.random() < self.p:
            L = len(signal)
            t = np.linspace(0, L / 100.0, L)
            freq = np.random.uniform(0.1, 0.5) # 呼吸频率
            amp = np.std(signal) * np.random.uniform(0.1, 0.5)
            drift = np.sin(2 * np.pi * freq * t) * amp
            signal = signal + drift

        # 3. 高斯噪声 (模拟传感器噪声)
        if random.random() < self.p:
            noise_amp = np.std(signal) * np.random.uniform(0.01, 0.05)
            noise = np.random.normal(0, noise_amp, len(signal))
            signal = signal + noise

        # 4. 随机掩码 (模拟接触不良)
        if random.random() < self.p:
            mask_len = int(len(signal) * np.random.uniform(0.05, 0.15))
            start = np.random.randint(0, len(signal) - mask_len)
            signal[start : start + mask_len] = 0.0

        # 5. 【关键】局部垂直平移 (模拟 ST 段抬高/压低)
        # 这对于冠心病识别非常有帮助，强迫模型关注局部电位变化
        if random.random() < 0.3: # 概率设低一点，避免破坏太多正常波形
            seg_len = int(len(signal) * 0.1) # 假设一段波形的长度
            if len(signal) > seg_len:
                start = np.random.randint(0, len(signal) - seg_len)
                # 偏移量: ±0.1 ~ ±0.3 倍标准差
                offset = np.random.uniform(-0.3, 0.3) * (np.std(signal) + 1e-6)
                signal[start : start + seg_len] += offset

        return signal

# ===================================================================
# 2. Dataset 定义
# ===================================================================
class DownstreamClassificationDataset(Dataset):
    def __init__(self, data_root, split_file, mode='train', signal_len=3000, task_index=0):
        self.data_root = data_root
        self.signal_len = signal_len
        self.mode = mode
        self.task_index = task_index

        # 针对 20人小样本，训练时开启强增强 (p=0.7)
        self.augmentor = SignalAugmentor(p=0.7) if mode == 'train' else None

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

            # --- 处理流程 ---
            
            # 1. 裁剪/填充 (确保长度对齐)
            processed_signal = self._crop_or_pad(raw_signal)
            
            # 2. 数据增强 (在归一化之前进行，模拟物理干扰)
            if self.augmentor is not None:
                processed_signal = self.augmentor(processed_signal)

            # 3. 归一化 (使用 Robust Scaling 保护 ST 段)
            processed_signal = self._robust_norm(processed_signal)

            # 转 Tensor [1, L]
            signal_tensor = torch.from_numpy(processed_signal).unsqueeze(0)
            
            return signal_tensor, torch.tensor(label, dtype=torch.long)

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return torch.zeros(1, self.signal_len), torch.tensor(0, dtype=torch.long)

    def _crop_or_pad(self, signal):
        current_len = len(signal)
        target_len = self.signal_len

        # 裁剪
        if current_len > target_len:
            if self.mode == 'train':
                # 训练时随机裁剪，增加样本多样性
                start = np.random.randint(0, current_len - target_len)
            else:
                # 测试时中心裁剪，保证确定性
                start = (current_len - target_len) // 2
            signal = signal[start : start + target_len]
        
        # 填充
        if len(signal) < target_len:
            pad_len = target_len - len(signal)
            signal = np.pad(signal, (0, pad_len), 'constant', constant_values=0)
            
        return signal

    def _robust_norm(self, signal):
        """
        【关键改进】Robust Scaling
        使用中位数和四分位距 (IQR) 进行归一化。
        相比于 (x-mean)/std，这种方法对大幅度的 R 波不敏感，
        能更好地保留 ST 段和 T 波的相对幅度特征，对冠心病识别更友好。
        """
        q25 = np.percentile(signal, 25)
        q75 = np.percentile(signal, 75)
        iqr = q75 - q25
        median = np.median(signal)
        
        # 防止除以零 (如果是死线)
        if iqr < 1e-6:
            # 如果信号几乎是直线，退化为减去均值
            return signal - np.mean(signal)
            
        return (signal - median) / iqr

    # 备用：标准 Instance Norm (如果 Robust Norm 效果不好可切回)
    def _instance_norm(self, signal):
        mean = np.mean(signal)
        std = np.std(signal)
        if std < 1e-6: std = 1e-6
        return (signal - mean) / std