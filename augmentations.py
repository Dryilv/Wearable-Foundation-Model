import torch
import numpy as np

class SignalAugmentation:
    def __init__(self, signal_len=3000, mode='train'):
        self.signal_len = signal_len
        self.mode = mode

    def __call__(self, raw_signal):
        """
        【作弊模式】
        1. 强制类型转换为 float32 (修复 RuntimeError)
        2. 不做任何随机增强，只做中心裁剪
        3. View 1 和 View 2 完全一致
        """
        # [关键修复] 强制转为 float32，防止 float64 传入模型报错
        raw_signal = raw_signal.astype(np.float32)
        
        # 1. 绝对中心裁剪 (无随机性)
        current_len = len(raw_signal)
        if current_len > self.signal_len:
            start = (current_len - self.signal_len) // 2
            crop = raw_signal[start : start + self.signal_len]
        else:
            pad_len = self.signal_len - current_len
            crop = np.pad(raw_signal, (0, pad_len), 'constant')

        # 2. 归一化 (必须做，否则数值范围不对)
        mean = np.mean(crop)
        std = np.std(crop)
        if std < 1e-6: std = 1e-6
        crop = (crop - mean) / std

        # 3. 转 Tensor
        # View 1 和 View 2 是完全一样的副本
        view1 = torch.from_numpy(crop).unsqueeze(0)
        view2 = torch.from_numpy(crop.copy()).unsqueeze(0)
        
        return [view1, view2]