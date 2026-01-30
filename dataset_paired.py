import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
import random
import pickle

class PairedPhysioDataset(Dataset):
    def __init__(self, index_file, signal_len=3000, mode='train', 
                 row_ppg=4, row_ecg=0,  # 指定 PPG 和 ECG 的行号
                 min_std_threshold=1e-4,
                 max_std_threshold=5000.0,
                 use_ratio=0.1,  # 新参数：使用数据的比例 (0.0-1.0)
                 max_samples=None,  # 新参数：最大样本数（优先级高于use_ratio）
                 random_seed=42  # 新参数：随机种子，确保可重复性
                 ):
        self.signal_len = signal_len
        self.mode = mode
        self.row_ppg = row_ppg
        self.row_ecg = row_ecg
        self.min_std_threshold = min_std_threshold
        self.max_std_threshold = max_std_threshold
        
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"Index file not found: {index_file}")
            
        print(f"Loading index from: {index_file} ...")
        with open(index_file, 'r') as f:
            self.index_data = json.load(f)
        
        # 保存原始数据量
        self.original_size = len(self.index_data)
        print(f"Loaded {self.original_size} samples.")
        
        # 根据参数限制数据量
        self.index_data = self._limit_data_size(
            self.index_data, 
            use_ratio, 
            max_samples, 
            random_seed
        )
        
        # 记录实际使用的数据量
        self.actual_size = len(self.index_data)
        print(f"Using {self.actual_size} samples ({(self.actual_size/self.original_size)*100:.1f}% of total).")
        
        # 设置随机种子
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # 用于记录已使用样本的集合（可选，用于调试）
        self.used_samples = set()

    def _limit_data_size(self, index_data, use_ratio, max_samples, random_seed):
        """
        根据参数限制数据量
        """
        # 先复制一份，避免修改原始数据
        limited_data = index_data.copy()
        
        # 随机打乱，确保随机性但可重复
        random.seed(random_seed)
        random.shuffle(limited_data)
        
        # 计算实际使用的样本数
        if max_samples is not None:
            # 使用 max_samples 参数
            actual_samples = min(len(limited_data), max_samples)
        elif use_ratio < 1.0:
            # 使用 use_ratio 参数
            actual_samples = int(len(limited_data) * use_ratio)
        else:
            # 使用全部数据
            actual_samples = len(limited_data)
        
        # 截取前 actual_samples 个样本
        limited_data = limited_data[:actual_samples]
        
        # 再次随机打乱，避免潜在的顺序偏差
        random.shuffle(limited_data)
        
        return limited_data

    def __len__(self):
        return len(self.index_data)

    def __getitem__(self, idx):
        # 记录已使用的样本（用于调试）
        if idx not in self.used_samples:
            self.used_samples.add(idx)
        
        # 重试机制
        for _ in range(3):
            try:
                item_info = self.index_data[idx]
                file_path = item_info['path']
                # 注意：我们忽略 item_info['row']，因为我们要强制读取 row_ppg 和 row_ecg
                
                with open(file_path, 'rb') as f:
                    content = pickle.load(f)
                    # content['data'] shape: (Channels, Length)
                    raw_ppg = content['data'][self.row_ppg]
                    raw_ecg = content['data'][self.row_ecg]
                    
                    if raw_ppg.dtype != np.float32: raw_ppg = raw_ppg.astype(np.float32)
                    if raw_ecg.dtype != np.float32: raw_ecg = raw_ecg.astype(np.float32)
                
                # 1. 检查 NaN / Inf (任意一个坏了都要换)
                if (np.isnan(raw_ppg).any() or np.isinf(raw_ppg).any() or 
                    np.isnan(raw_ecg).any() or np.isinf(raw_ecg).any()):
                    # 在当前数据集中随机选择另一个样本
                    idx = random.randint(0, len(self.index_data) - 1)
                    continue

                # 2. 同步裁剪/填充 (关键步骤)
                ppg_proc, ecg_proc = self._process_paired_signal(raw_ppg, raw_ecg)

                # 3. 检查标准差 (分别检查)
                std_ppg = np.std(ppg_proc)
                std_ecg = np.std(ecg_proc)
                
                if (std_ppg < self.min_std_threshold or std_ppg > self.max_std_threshold or
                    std_ecg < self.min_std_threshold or std_ecg > self.max_std_threshold):
                    # 在当前数据集中随机选择另一个样本
                    idx = random.randint(0, len(self.index_data) - 1)
                    continue
                
                # 4. 独立归一化 (Z-Score)
                # PPG 和 ECG 幅值差异巨大，必须分别归一化
                ppg_proc = (ppg_proc - np.mean(ppg_proc)) / (std_ppg + 1e-6)
                ecg_proc = (ecg_proc - np.mean(ecg_proc)) / (std_ecg + 1e-6)

                # 返回 [1, L] 格式
                ppg_tensor = torch.from_numpy(ppg_proc).unsqueeze(0)
                ecg_tensor = torch.from_numpy(ecg_proc).unsqueeze(0)
                
                return ppg_tensor, ecg_tensor

            except Exception as e:
                # 打印错误信息用于调试
                print(f"Error loading {file_path}: {e}")
                # 在当前数据集中随机选择另一个样本
                idx = random.randint(0, len(self.index_data) - 1)
                continue
        
        # 兜底数据
        fallback = torch.randn(1, self.signal_len).float() * 1e-3
        return fallback, fallback

    def _process_paired_signal(self, ppg, ecg):
        """
        确保 PPG 和 ECG 进行完全相同的裁剪，保持时间对齐
        """
        current_len = len(ppg) # 假设 ppg 和 ecg 长度一致
        target_len = self.signal_len

        if current_len == target_len:
            return ppg, ecg

        if current_len > target_len:
            if self.mode == 'train':
                # 随机裁剪：生成一次随机数，应用到两个信号
                start = np.random.randint(0, current_len - target_len)
            else:
                # 中心裁剪
                start = (current_len - target_len) // 2
            
            return ppg[start : start + target_len], ecg[start : start + target_len]
        else:
            # 零填充
            pad_len = target_len - current_len
            ppg_pad = np.pad(ppg, (0, pad_len), 'constant', constant_values=0)
            ecg_pad = np.pad(ecg, (0, pad_len), 'constant', constant_values=0)
            return ppg_pad, ecg_pad

    def get_data_stats(self):
        """
        返回数据集的统计信息
        """
        return {
            "original_size": self.original_size,
            "actual_size": self.actual_size,
            "use_percentage": (self.actual_size / self.original_size) * 100,
            "used_samples_count": len(self.used_samples)
        }