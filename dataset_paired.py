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
                 max_std_threshold=5000.0 
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
        
        # 过滤逻辑：这里假设 index_data 里的每个文件都包含所需的 PPG 和 ECG 行
        # 如果你的 index_data 是按单行索引的，这里可能需要去重，只保留文件路径
        # 为了简单起见，我们假设 index_data 里的每一项代表一个可用的记录文件
        print(f"Loaded {len(self.index_data)} samples.")

    def __len__(self):
        return len(self.index_data)

    def __getitem__(self, idx):
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
                    idx = random.randint(0, len(self.index_data) - 1)
                    continue

                # 2. 同步裁剪/填充 (关键步骤)
                ppg_proc, ecg_proc = self._process_paired_signal(raw_ppg, raw_ecg)

                # 3. 检查标准差 (分别检查)
                std_ppg = np.std(ppg_proc)
                std_ecg = np.std(ecg_proc)
                
                if (std_ppg < self.min_std_threshold or std_ppg > self.max_std_threshold or
                    std_ecg < self.min_std_threshold or std_ecg > self.max_std_threshold):
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
                # print(f"Error loading {file_path}: {e}")
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