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
        # 1. 强制两个 View 高度重叠
        current_len = len(signal)
        target_len = self.signal_len
        
        if current_len > target_len:
            # 找到中心点
            center = current_len // 2
            # 允许左右偏移不超过 50 个点 (0.5秒)
            # 这样两个 View 至少有 95% 是重叠的
            shift = np.random.randint(-50, 50) 
            start = center - (target_len // 2) + shift
            
            # 边界检查
            start = max(0, min(start, current_len - target_len))
            crop = signal[start : start + target_len]
        else:
            pad_len = target_len - current_len
            crop = np.pad(signal, (0, pad_len), 'constant')

        # 2. 加上微弱的噪声 (增加难度，防止 Loss 变成 0)
        noise = np.random.normal(0, 0.01, size=crop.shape)
        crop = crop + noise
        
        # 3. 归一化 (必须)
        crop = (crop - crop.mean()) / (crop.std() + 1e-6)
        
        return crop