import torch
import numpy as np

class SignalAugmentation:
    def __init__(self, signal_len=3000, mode='train'):
        self.signal_len = signal_len
        self.mode = mode

    def __call__(self, raw_signal):
        """
        输入: raw_signal (numpy array), 长度可能不等于 signal_len
        输出: [view1, view2] (List of Tensor)
        """
        # 确保输入是 float32
        raw_signal = raw_signal.astype(np.float32)
        
        # 生成两个不同的 View
        view1 = self._augment(raw_signal)
        view2 = self._augment(raw_signal)
        
        return [torch.from_numpy(view1).unsqueeze(0), torch.from_numpy(view2).unsqueeze(0)]

    def _augment(self, signal, seed_crop=None):
        # 1. 裁剪逻辑修改：确保两个 View 的物理位置接近
        current_len = len(signal)
        
        if current_len > self.signal_len:
            # 【关键修改】限制随机裁剪的范围，或者让两个 View 共享同一个中心
            # 策略：先定一个中心，然后在中心附近微调
            center = current_len // 2
            max_shift = self.signal_len // 10 # 允许 10% 的错位
            
            # 随机偏移
            shift = np.random.randint(-max_shift, max_shift)
            start = center - (self.signal_len // 2) + shift
            
            # 边界保护
            start = max(0, min(start, current_len - self.signal_len))
            
            crop = signal[start : start + self.signal_len]
        else:
            pad_len = self.signal_len - current_len
            crop = np.pad(signal, (0, pad_len), 'constant')

        # 2. 缩放 (保持)
        scale = np.random.uniform(0.9, 1.1) # 稍微温和一点
        crop = crop * scale

        # 3. 噪声 (减小)
        # 初始阶段建议先关掉噪声，或者加很小的噪声
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 0.001, size=crop.shape).astype(np.float32)
            crop = crop + noise

        # 4. Norm
        mean = np.mean(crop)
        std = np.std(crop)
        if std < 1e-6: std = 1e-6
        crop = (crop - mean) / std

        return crop