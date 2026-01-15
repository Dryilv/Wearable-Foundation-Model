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

    def _augment(self, signal):
        # 1. 随机裁剪 (Random Crop)
        current_len = len(signal)
        if current_len > self.signal_len:
            # 随机选起点
            start = np.random.randint(0, current_len - self.signal_len)
            crop = signal[start : start + self.signal_len]
        else:
            # 填充
            pad_len = self.signal_len - current_len
            crop = np.pad(signal, (0, pad_len), 'constant')

        # 2. 随机幅度缩放 (Random Amplitude Scaling)
        scale = np.random.uniform(0.8, 1.2)
        crop = crop * scale

        # 3. 随机高斯噪声 (Gaussian Noise)
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 0.01, size=crop.shape).astype(np.float32)
            crop = crop + noise

        # 4. Instance Norm (必须做，保证分布一致)
        mean = np.mean(crop)
        std = np.std(crop)
        if std < 1e-6: std = 1e-6
        crop = (crop - mean) / std

        return crop