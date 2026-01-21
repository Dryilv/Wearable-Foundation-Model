import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
import random
import pickle

class PairedPhysioDataset(Dataset):
    def __init__(self, index_file, signal_len=3000, mode='train', 
                 row_ppg=4, # 【修正】第5行 (索引4) 是 PPG -> 作为输入
                 row_ecg=0, # 【修正】第1行 (索引0) 是 ECG -> 作为标签
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
            raw_index = json.load(f)
        
        self.file_paths = list(set([item['path'] for item in raw_index]))
        print(f"Loaded {len(self.file_paths)} unique files. PPG index: {row_ppg}, ECG index: {row_ecg}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        for _ in range(3):
            try:
                file_path = self.file_paths[idx]
                
                with open(file_path, 'rb') as f:
                    content = pickle.load(f)
                    data = content['data']
                    
                    # 【关键】根据修正后的索引读取
                    raw_ppg = data[self.row_ppg].astype(np.float32)
                    raw_ecg = data[self.row_ecg].astype(np.float32)
                
                # 1. 检查 NaN / Inf
                if (np.isnan(raw_ppg).any() or np.isinf(raw_ppg).any() or 
                    np.isnan(raw_ecg).any() or np.isinf(raw_ecg).any()):
                    idx = random.randint(0, len(self.file_paths) - 1)
                    continue

                # 2. 同步裁剪
                ppg_proc, ecg_proc = self._process_paired_signal(raw_ppg, raw_ecg)

                # 3. 质量检查 (基于输入 PPG)
                std_ppg = np.std(ppg_proc)
                if std_ppg < self.min_std_threshold or std_ppg > self.max_std_threshold:
                    idx = random.randint(0, len(self.file_paths) - 1)
                    continue
                
                # 4. 归一化
                ppg_norm = (ppg_proc - np.mean(ppg_proc)) / (np.std(ppg_proc) + 1e-6)
                ecg_norm = (ecg_proc - np.mean(ecg_proc)) / (np.std(ecg_proc) + 1e-6)

                ppg_tensor = torch.from_numpy(ppg_norm).unsqueeze(0)
                ecg_tensor = torch.from_numpy(ecg_norm).unsqueeze(0)
                
                return ppg_tensor, ecg_tensor

            except Exception as e:
                idx = random.randint(0, len(self.file_paths) - 1)
                continue
        
        fallback = torch.randn(1, self.signal_len).float() * 1e-3
        return fallback, fallback

    def _process_paired_signal(self, ppg, ecg):
        current_len = len(ppg)
        target_len = self.signal_len

        if current_len == target_len:
            return ppg, ecg

        if current_len > target_len:
            if self.mode == 'train':
                start = np.random.randint(0, current_len - target_len)
            else:
                start = (current_len - target_len) // 2
            return ppg[start : start + target_len], ecg[start : start + target_len]
        else:
            pad_len = target_len - current_len
            ppg_pad = np.pad(ppg, (0, pad_len), 'constant', constant_values=0)
            ecg_pad = np.pad(ecg, (0, pad_len), 'constant', constant_values=0)
            return ppg_pad, ecg_pad