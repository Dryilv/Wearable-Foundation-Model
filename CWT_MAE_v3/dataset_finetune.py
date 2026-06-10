import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import json
import os
import random
from utils import is_main_process

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
        
        # 1. 随机缩放 (模拟信号强度波动)
        if random.random() < self.p:
            scale = np.random.uniform(self.scale_range[0], self.scale_range[1], size=(M, 1))
            signal = signal * scale

        # 2. 随机高斯噪声
        if random.random() < self.p:
            noise = np.random.normal(0, self.noise_std, size=(M, L))
            signal = signal + noise

        # 3. 随机时间偏移 (Phase Shift) - 模拟采样起始点微差
        if random.random() < self.p:
            shift = np.random.randint(-20, 20)
            signal = np.roll(signal, shift, axis=1)

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
    def __init__(self, data_root, split_file, mode='train', signal_len=3000, task_index=0, num_classes=2, on_error='raise', max_error_logs=20, refined_labels_path=None):
        self.data_root = data_root
        self.signal_len = signal_len
        self.mode = mode
        self.task_index = task_index
        self.num_classes = num_classes
        self.on_error = on_error
        self.max_error_logs = max_error_logs
        self.error_count = 0
        
        # 加载软标签 (如果提供)
        self.refined_labels = None
        if refined_labels_path and os.path.exists(refined_labels_path) and mode == 'train':
            if is_main_process():
                print(f"[{mode}] Loading refined soft labels from {refined_labels_path}")
            with open(refined_labels_path, 'r') as f:
                # JSON keys are strings, need to convert to int if using as list index
                # But JSON usually loads keys as strings.
                loaded_labels = json.load(f)
                # Convert keys to int for easier access
                self.refined_labels = {int(k): v for k, v in loaded_labels.items()}

        # 训练集开启增强
        self.augmentor = MultiChannelAugmentor(p=0.5) if mode == 'train' else None

        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, 'r') as f:
            splits = json.load(f)
        
        if mode not in splits:
            raise ValueError(f"Mode {mode} not found in split file.")
            
        self.file_list = splits[mode]
        if is_main_process():
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
                # 处理 numpy 版本不兼容问题
                try:
                    content = pickle.load(f)
                except ModuleNotFoundError as e:
                    if 'numpy' in str(e):
                        # 使用自定义 Unpickler 修复 numpy 模块路径
                        class NumpyUnpickler(pickle.Unpickler):
                            def find_class(self, module, name):
                                # 将旧的 numpy 内部模块映射到当前可用模块
                                if module.startswith('numpy._core') or module.startswith('numpy.core'):
                                    module = 'numpy'
                                return super().find_class(module, name)
                        f.seek(0)
                        content = NumpyUnpickler(f).load()
                    else:
                        raise
            
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
            LABEL_NAMES = [
            '高血压', '高血糖', '高血脂', 
            '冠心病', '心律失常（房颤、频发早搏等）', '糖尿病', 
            '颈动脉斑块'
        ]
            
            if isinstance(content, dict) and 'label' in content:
                target_label = content['label']
                if isinstance(target_label, dict):
                    # Multi-label dict
                    label_vector = [float(target_label.get(name, 0)) for name in LABEL_NAMES]
                else:
                    # fallback
                    label_vector = [0.0] * self.num_classes
            else:
                label_vector = [0.0] * self.num_classes

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

            # --- 6. 通道自适应处理 (Modality-Agnostic) ---
            # 保留所有可用通道，模型通过 channel_mask 和跨通道注意力处理变长通道输入
            
            M = signal_tensor.shape[0]

            # 所有通道共享 modality_id=0 (通用生理信号)
            modality_ids = torch.zeros(M, dtype=torch.long)
            
            # --- 7. 返回标签 (硬标签 or 软标签) ---
            if self.refined_labels is not None and idx in self.refined_labels:
                soft_label = torch.tensor(self.refined_labels[idx], dtype=torch.float32)
                if soft_label.numel() == self.num_classes and torch.isfinite(soft_label).all():
                    soft_sum = soft_label.sum()
                    if soft_sum > 0:
                        soft_label = soft_label / soft_sum
                    return signal_tensor, modality_ids, soft_label
            else:
                # 返回多标签 FloatTensor
                return signal_tensor, modality_ids, torch.tensor(label_vector, dtype=torch.float32)

        except Exception as e:
            self.error_count += 1
            if self.error_count <= self.max_error_logs:
                print(f"[{self.mode}] Error loading {filename}: {e}")
            if self.on_error == 'raise':
                raise
            modality_ids = torch.zeros(1, dtype=torch.long)
            return torch.zeros(1, self.signal_len), modality_ids, torch.zeros(self.num_classes, dtype=torch.float32)

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
                # 随机裁剪
                start = np.random.randint(0, current_len - target_len + 1)
                
                # 增强策略：对于训练集，有一定概率使用能量最大的片段
                # 这有助于在长程信号中捕捉到那些真正有意义的病理波动，而不是裁剪到平庸的直线
                if random.random() < 0.3: # 30% 概率寻找高能量区域
                    # 快速滑动窗口寻找能量最高（方差最大）的区域
                    num_windows = min(10, (current_len - target_len) // (target_len // 2) + 1)
                    if num_windows > 1:
                        best_start = start
                        max_var = -1
                        for i in range(num_windows):
                            s = np.random.randint(0, current_len - target_len + 1)
                            window = signal[:, s : s + target_len]
                            var = np.mean(np.var(window, axis=1))
                            if var > max_var:
                                max_var = var
                                best_start = s
                        start = best_start
            else:
                # 验证/测试：中心裁剪
                start = (current_len - target_len) // 2
            return signal[:, start : start + target_len]
        else:
            pad_len = target_len - current_len
            # 使用边缘填充而非 0 填充，减少 CWT 伪影
            return np.pad(signal, ((0, 0), (0, pad_len)), mode='edge')

    def _robust_norm(self, signal):
        """
        与 dataset.py (预训练) 保持完全一致的归一化方式：Z-Score
        消除预训练与微调之间的 Distribution Gap
        """
        # 计算全局或逐通道的 mean 和 std，与 dataset.py 保持一致
        # 在 dataset.py 中，是逐通道 Z-score，然后裁剪到 [-10, 10]
        std_vals = np.std(signal, axis=1, keepdims=True)
        mean_vals = np.mean(signal, axis=1, keepdims=True)
        
        normalized = (signal - mean_vals) / (std_vals + 1e-5)
        normalized = np.clip(normalized, -10.0, 10.0)
        
        return normalized
