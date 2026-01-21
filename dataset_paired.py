import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
import random
import pickle

class PairedPhysioDataset(Dataset):
    def __init__(self, index_file, signal_len=3000, mode='train', 
                 row_ppg=0, # 第1行 (索引0)
                 row_ecg=4, # 第5行 (索引4)
                 min_std_threshold=1e-4,
                 max_std_threshold=5000.0):
        
        self.signal_len = signal_len
        self.mode = mode
        self.row_ppg = row_ppg
        self.row_ecg = row_ecg
        self.min_std_threshold = min_std_threshold
        self.max_std_threshold = max_std_threshold
        
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"Index file not found: {index_file}")
            
        print(f"Loading paired index from: {index_file} ...")
        with open(index_file, 'r') as f:
            # 假设 index_data 是列表，每个元素包含 'path'
            # 我们不再关心 index 中的 'row' 字段，因为我们固定读取 0 和 4
            raw_index = json.load(f)
        
        # 去重：因为原 index 可能包含同一个文件的多个 row，我们只需要文件路径
        self.file_paths = list(set([item['path'] for item in raw_index]))
        print(f"Loaded {len(self.file_paths)} unique files for paired training.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # 重试机制
        for _ in range(3):
            try:
                file_path = self.file_paths[idx]
                
                with open(file_path, 'rb') as f:
                    content = pickle.load(f)
                    # content['data'] shape: [Channels, Length]
                    data = content['data']
                    
                    # 获取原始信号
                    raw_ppg = data[self.row_ppg].astype(np.float32)
                    raw_ecg = data[self.row_ecg].astype(np.float32)
                
                # 1. 检查 NaN / Inf
                if (np.isnan(raw_ppg).any() or np.isinf(raw_ppg).any() or 
                    np.isnan(raw_ecg).any() or np.isinf(raw_ecg).any()):
                    idx = random.randint(0, len(self.file_paths) - 1)
                    continue

                # 2. 同步处理 (裁剪/填充)
                # 必须传入两个信号，保证裁剪位置一致
                ppg_proc, ecg_proc = self._process_paired_signal(raw_ppg, raw_ecg)

                # 3. 质量检查 (基于 PPG)
                # 通常我们只检查输入的质量，如果输入是噪声，翻译就没有意义
                std_ppg = np.std(ppg_proc)
                if std_ppg < self.min_std_threshold or std_ppg > self.max_std_threshold:
                    idx = random.randint(0, len(self.file_paths) - 1)
                    continue
                
                # 4. 独立归一化 (Z-Score)
                # PPG 和 ECG 幅值差异巨大，必须分别归一化
                ppg_norm = (ppg_proc - np.mean(ppg_proc)) / (np.std(ppg_proc) + 1e-6)
                ecg_norm = (ecg_proc - np.mean(ecg_proc)) / (np.std(ecg_proc) + 1e-6)

                # 返回 [1, L]
                ppg_tensor = torch.from_numpy(ppg_norm).unsqueeze(0)
                ecg_tensor = torch.from_numpy(ecg_norm).unsqueeze(0)
                
                return ppg_tensor, ecg_tensor

            except Exception as e:
                # print(f"Error loading {file_path}: {e}")
                idx = random.randint(0, len(self.file_paths) - 1)
                continue
        
        # 兜底
        fallback = torch.randn(1, self.signal_len).float() * 1e-3
        return fallback, fallback

    def _process_paired_signal(self, ppg, ecg):
        """确保 PPG 和 ECG 在时间轴上进行完全相同的裁剪"""
        current_len = len(ppg) # 假设 ppg 和 ecg 长度一致
        target_len = self.signal_len

        if current_len == target_len:
            return ppg, ecg

        if current_len > target_len:
            if self.mode == 'train':
                # 随机裁剪，但起点相同
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