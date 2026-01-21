import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
import random
import pickle

class PairedPhysioDataset(Dataset):
    def __init__(self, index_file, 
                 signal_len=150,      # 【修改】窗口大小改为 150
                 mode='train', 
                 row_ppg=4, 
                 row_ecg=0, 
                 iqr_scale=1.5,       # 【新增】IQR 缩放系数
                 stat_sample_count=1000): # 【新增】用于计算统计量的采样文件数，防止初始化太慢
        
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
        # 【新增】初始化阶段：计算全局 IQR 阈值
        # ==========================================
        self.ppg_bounds, self.ecg_bounds = self._calculate_adaptive_thresholds(
            sample_count=stat_sample_count
        )
        
        print(f"Adaptive Thresholds (IQR={iqr_scale}):")
        print(f"  PPG STD Bounds: [{self.ppg_bounds[0]:.4f}, {self.ppg_bounds[1]:.4f}]")
        print(f"  ECG STD Bounds: [{self.ecg_bounds[0]:.4f}, {self.ecg_bounds[1]:.4f}]")

    def _calculate_adaptive_thresholds(self, sample_count):
        """
        随机采样部分文件，计算 PPG 和 ECG 的标准差分布，
        生成基于 IQR 的动态阈值。
        """
        print("Calculating adaptive thresholds based on data distribution...")
        ppg_stds = []
        ecg_stds = []
        
        # 随机选择一部分文件进行统计，避免遍历整个数据集太慢
        sample_paths = self.file_paths
        if len(self.file_paths) > sample_count:
            sample_paths = random.sample(self.file_paths, sample_count)
            
        for fp in sample_paths:
            try:
                with open(fp, 'rb') as f:
                    content = pickle.load(f)
                    data = content['data']
                    raw_ppg = data[self.row_ppg].astype(np.float32)
                    raw_ecg = data[self.row_ecg].astype(np.float32)
                    
                    # 简单切片用于统计（取中间一段，避免全零填充影响统计）
                    if len(raw_ppg) >= self.signal_len:
                        mid = len(raw_ppg) // 2
                        start = max(0, mid - self.signal_len // 2)
                        end = start + self.signal_len
                        
                        seg_ppg = raw_ppg[start:end]
                        seg_ecg = raw_ecg[start:end]
                        
                        # 基础有效性检查
                        if self._check_basic_validity(seg_ppg) and self._check_basic_validity(seg_ecg):
                            ppg_stds.append(np.std(seg_ppg))
                            ecg_stds.append(np.std(seg_ecg))
            except:
                continue
        
        # 计算 IQR 边界的辅助函数
        def get_bounds(std_list):
            if not std_list:
                return 1e-4, 5000.0 # Fallback
            arr = np.array(std_list)
            q1 = np.percentile(arr, 25)
            q3 = np.percentile(arr, 75)
            iqr = q3 - q1
            lower = max(1e-4, q1 - self.iqr_scale * iqr) # 保证下界不为负或0
            upper = q3 + self.iqr_scale * iqr
            return lower, upper

        return get_bounds(ppg_stds), get_bounds(ecg_stds)

    def _check_basic_validity(self, signal):
        """基础信号检查：非空、有限数值、非平坦"""
        if len(signal) == 0: return False
        if not np.isfinite(signal).all(): return False
        if np.std(signal) < 1e-6: return False 
        return True

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # 【重试机制】尝试 3 次，如果随机到的样本质量差，就换一个
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

                # 2. 同步裁剪 (Random Crop or Center Crop)
                ppg_proc, ecg_proc = self._process_paired_signal(raw_ppg, raw_ecg)

                # 3. 【关键修改】基于 IQR 的双重质量检查
                # 只要有一个信号的 STD 不在动态范围内，就视为废弃样本
                std_ppg = np.std(ppg_proc)
                std_ecg = np.std(ecg_proc)
                
                ppg_bad = std_ppg < self.ppg_bounds[0] or std_ppg > self.ppg_bounds[1]
                ecg_bad = std_ecg < self.ecg_bounds[0] or std_ecg > self.ecg_bounds[1]
                
                if ppg_bad or ecg_bad:
                    # 质量差，重新随机索引重试
                    idx = random.randint(0, len(self.file_paths) - 1)
                    continue
                
                # 4. 归一化 (Z-Score)
                ppg_norm = (ppg_proc - np.mean(ppg_proc)) / (std_ppg + 1e-6)
                ecg_norm = (ecg_proc - np.mean(ecg_proc)) / (std_ecg + 1e-6)

                ppg_tensor = torch.from_numpy(ppg_norm).unsqueeze(0)
                ecg_tensor = torch.from_numpy(ecg_norm).unsqueeze(0)
                
                return ppg_tensor, ecg_tensor

            except Exception as e:
                # 读取出错，重试
                idx = random.randint(0, len(self.file_paths) - 1)
                continue
        
        # 如果重试 3 次都失败，返回全 0 或噪声兜底，防止 DataLoader 崩溃
        fallback = torch.randn(1, self.signal_len).float() * 1e-3
        return fallback, fallback

    def _process_paired_signal(self, ppg, ecg):
        current_len = len(ppg)
        target_len = self.signal_len

        if current_len == target_len:
            return ppg, ecg

        if current_len > target_len:
            if self.mode == 'train':
                # 训练模式：随机裁剪
                start = np.random.randint(0, current_len - target_len)
            else:
                # 验证模式：中心裁剪
                start = (current_len - target_len) // 2
            return ppg[start : start + target_len], ecg[start : start + target_len]
        else:
            # 长度不足：补零
            pad_len = target_len - current_len
            ppg_pad = np.pad(ppg, (0, pad_len), 'constant', constant_values=0)
            ecg_pad = np.pad(ecg, (0, pad_len), 'constant', constant_values=0)
            return ppg_pad, ecg_pad