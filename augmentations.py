import torch
import numpy as np

class SignalAugmentation:
    def __init__(self, signal_len=3000, mode='train'):
        self.signal_len = signal_len
        self.mode = mode

    def __call__(self, raw_signal):
        # [关键修复] 强制转为 float32
        raw_signal = raw_signal.astype(np.float32)
        
        # 生成两个高度重叠但略有不同的 View
        view1_data = self._augment(raw_signal)
        view2_data = self._augment(raw_signal)
        
        view1 = torch.from_numpy(view1_data).unsqueeze(0)
        view2 = torch.from_numpy(view2_data).unsqueeze(0)
        
        return [view1, view2]

    def _augment(self, signal):
        current_len = len(signal)
        target_len = self.signal_len
        
        # 1. 裁剪: 中心抖动 (Center Jitter)
        # 确保两个 View 至少有 90% 的重叠，这是 Loss 能下降的关键
        if current_len > target_len:
            center = current_len // 2
            # 允许左右偏移 50 个点 (0.5秒)
            shift = np.random.randint(-50, 50) 
            start = center - (target_len // 2) + shift
            
            # 边界保护
            start = max(0, min(start, current_len - target_len))
            crop = signal[start : start + target_len]
        else:
            pad_len = target_len - current_len
            crop = np.pad(signal, (0, pad_len), 'constant')

        # 2. 缩放: 非常温和 (0.95 - 1.05)
        scale = np.random.uniform(0.95, 1.05)
        crop = crop * scale

        # 3. 噪声: 极小的高斯噪声 (防止模型死记硬背数值)
        # 之前可能噪声太大导致波形畸变
        noise = np.random.normal(0, 0.001, size=crop.shape).astype(np.float32)
        crop = crop + noise
        
        # 4. Instance Norm (至关重要)
        # 消除幅度差异，让模型关注波形形状
        mean = np.mean(crop)
        std = np.std(crop)
        if std < 1e-6: std = 1e-6
        crop = (crop - mean) / std
        
        # 再次确保返回 float32
        return crop.astype(np.float32)