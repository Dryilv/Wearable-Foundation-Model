import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
import random
import pickle

class PairedPhysioDataset(Dataset):
    def __init__(self, index_file, 
                 signal_len=150,      # 训练用的窗口大小
                 mode='train', 
                 row_ppg=4, 
                 row_ecg=0, 
                 iqr_scale=1.5,       # IQR 严格程度
                 stat_sample_count=1000): 
        
        self.signal_len = signal_len
        self.mode = mode
        self.row_ppg = row_ppg
        self.row_ecg = row_ecg
        self.iqr_scale = iqr_scale
        
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"Index file not found: {index_file}")
            
        print(f"Loading paired index from: {index_file} ...")
        with open(index_file, 'r') as f:
            raw_index = json.load(f)
        
        self.file_paths = list(set([item['path'] for item in raw_index]))
        print(f"Loaded {len(self.file_paths)} unique files.")

        # ==========================================
        # 初始化：基于【整段原始信号】计算阈值
        # ==========================================
        self.ppg_bounds, self.ecg_bounds = self._calculate_global_thresholds(
            sample_count=stat_sample_count
        )
        
        print(f"Global Thresholds (IQR={iqr_scale}) - Applied to FULL signal:")
        print(f"  PPG Global STD Bounds: [{self.ppg_bounds[0]:.4f}, {self.ppg_bounds[1]:.4f}]")
        print(f"  ECG Global STD Bounds: [{self.ecg_bounds[0]:.4f}, {self.ecg_bounds[1]:.4f}]")

    def _calculate_global_thresholds(self, sample_count):
        """
        随机采样部分文件，计算【整段】信号的标准差分布。
        这样可以捕捉到那些因为局部噪声导致全局STD飙升的坏样本。
        """
        print("Calculating global thresholds based on raw file statistics...")
        ppg_stds = []
        ecg_stds = []
        
        sample_paths = self.file_paths
        if len(self.file_paths) > sample_count:
            sample_paths = random.sample(self.file_paths, sample_count)
            
        for fp in sample_paths:
            try:
                with open(fp, 'rb') as f:
                    content = pickle.load(f)
                    data = content['data']
                    # 读取整段数据
                    raw_ppg = data[self.row_ppg].astype(np.float32)
                    raw_ecg = data[self.row_ecg].astype(np.float32)
                    
                    # 基础检查
                    if self._check_basic_validity(raw_ppg) and self._check_basic_validity(raw_ecg):
                        ppg_stds.append(np.std(raw_ppg))
                        ecg_stds.append(np.std(raw_ecg))
            except:
                continue
        
        def get_bounds(std_list):
            if not std_list: return 1e-4, 5000.0
            arr = np.array(std_list)
            q1 = np.percentile(arr, 25)
            q3 = np.percentile(arr, 75)
            iqr = q3 - q1
            # 下界防止死线，上界防止像你图中那样的剧烈噪声
            lower = max(1e-4, q1 - self.iqr_scale * iqr)
            upper = q3 + self.iqr_scale * iqr
            return lower, upper

        return get_bounds(ppg_stds), get_bounds(ecg_stds)

    def _check_basic_validity(self, signal):
        if len(signal) == 0: return False
        if not np.isfinite(signal).all(): return False
        if np.std(signal) < 1e-6: return False 
        return True

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # 重试机制
        for _ in range(3):
            try:
                file_path = self.file_paths[idx]
                
                with open(file_path, 'rb') as f:
                    content = pickle.load(f)
                    data = content['data']
                    raw_ppg = data[self.row_ppg].astype(np.float32)
                    raw_ecg = data[self.row_ecg].astype(np.float32)
                
                # 1. 基础 NaN / Inf 检查
                if not (self._check_basic_validity(raw_ppg) and self._check_basic_validity(raw_ecg)):
                    idx = random.randint(0, len(self.file_paths) - 1)
                    continue

                # ============================================================
                # 2. 【核心修改】全局质量检查 (Global Quality Check)
                # 在裁剪之前，先检查整段信号的 STD。
                # 如果图中那种局部大噪声存在，整段信号的 STD 会显著高于正常值，
                # 从而在这里被拦截，直接丢弃整个文件。
                # ============================================================
                global_std_ppg = np.std(raw_ppg)
                global_std_ecg = np.std(raw_ecg)

                ppg_bad = global_std_ppg < self.ppg_bounds[0] or global_std_ppg > self.ppg_bounds[1]
                ecg_bad = global_std_ecg < self.ecg_bounds[0] or global_std_ecg > self.ecg_bounds[1]

                if ppg_bad or ecg_bad:
                    # 只要整段信号里有“脏东西”拉坏了统计指标，就直接换文件
                    idx = random.randint(0, len(self.file_paths) - 1)
                    continue

                # 3. 同步裁剪 (此时我们确信 raw_ppg/ecg 整体是干净的)
                ppg_proc, ecg_proc = self._process_paired_signal(raw_ppg, raw_ecg)

                # 4. 局部再次检查 (可选，防止裁剪到了极其平坦的死线区域)
                # 虽然全局通过了，但为了保险，确保裁剪出的小片段不是死线
                if np.std(ppg_proc) < 1e-6 or np.std(ecg_proc) < 1e-6:
                    idx = random.randint(0, len(self.file_paths) - 1)
                    continue
                
                # 5. 归一化 (使用局部统计量进行归一化，效果通常更好)
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