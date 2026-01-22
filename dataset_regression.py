import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class BPDataset(Dataset):
    def __init__(self, 
                 signal_dir, 
                 excel_path, 
                 seq_len=1400,  # 修改默认长度：14s * 100Hz = 1400
                 normalize_labels=True):
        self.signal_dir = signal_dir
        self.seq_len = seq_len
        self.normalize_labels = normalize_labels

        print(f"Loading metadata from {excel_path}...")
        self.df = pd.read_excel(excel_path)
        
        # --- 1. 表格数据预处理 ---
        # 性别编码
        self.df['Sex_Encoded'] = self.df['Sex(M/F)'].map(lambda x: 1 if str(x).upper().strip() == 'F' else 0)
        
        self.feature_cols = [
            'Sex_Encoded', 'Age(year)', 'Height(cm)', 
            'Weight(kg)', 'Heart Rate(b/m)', 'BMI(kg/m^2)'
        ]
        self.num_tabular_features = len(self.feature_cols)

        # 计算表格特征统计量 (用于归一化)
        tabular_data = self.df[self.feature_cols].values.astype(np.float32)
        # 忽略第一列性别(0/1)，对后面5列进行统计
        self.tab_mean = np.mean(tabular_data[:, 1:], axis=0)
        self.tab_std = np.std(tabular_data[:, 1:], axis=0) + 1e-6

        # 建立索引
        self.subject_info = {}
        for _, row in self.df.iterrows():
            sid = int(row['subject_ID'])
            feats = row[self.feature_cols].values.astype(np.float32)
            # 归一化
            feats[1:] = (feats[1:] - self.tab_mean) / self.tab_std
            
            self.subject_info[sid] = {
                'sbp': float(row['Systolic Blood Pressure(mmHg)']),
                'dbp': float(row['Diastolic Blood Pressure(mmHg)']),
                'tabular_features': feats
            }

        # --- 2. 匹配文件 ---
        self.samples = []
        if os.path.exists(signal_dir):
            files = [f for f in os.listdir(signal_dir) if f.endswith('.txt')]
            for fname in files:
                try:
                    sid = int(fname.split('_')[0])
                    if sid in self.subject_info:
                        self.samples.append({
                            'path': os.path.join(signal_dir, fname),
                            'sid': sid
                        })
                except: pass
        
        print(f"Matched {len(self.samples)} samples. Target Seq Len: {self.seq_len} (100Hz)")

        # --- 3. 血压标签统计量 ---
        if self.normalize_labels and len(self.samples) > 0:
            sbp_vals = [self.subject_info[s['sid']]['sbp'] for s in self.samples]
            dbp_vals = [self.subject_info[s['sid']]['dbp'] for s in self.samples]
            self.bp_mean = torch.tensor([np.mean(sbp_vals), np.mean(dbp_vals)], dtype=torch.float32)
            self.bp_std = torch.tensor([np.std(sbp_vals), np.std(dbp_vals)], dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def _load_signal(self, path):
        """
        读取数据 -> 降采样 (1000Hz to 100Hz) -> 截断/填充
        """
        try:
            # 1. 读取原始数据 (Tab分割)
            df = pd.read_csv(path, sep='\t', header=None, engine='python')
            raw_signal = df.values.flatten()
            raw_signal = raw_signal[~np.isnan(raw_signal)] # 去除 NaN
            
            # 2. 执行降采样: 1000Hz -> 100Hz (Factor = 10)
            # 使用平均池化 (Average Pooling) 防止混叠
            factor = 10
            if len(raw_signal) >= factor:
                # 裁掉末尾不足 10 的部分，确保能整除
                limit = (len(raw_signal) // factor) * factor
                raw_signal = raw_signal[:limit]
                # Reshape 为 [N/10, 10] 然后求平均
                signal = raw_signal.reshape(-1, factor).mean(axis=1)
            else:
                # 如果数据极短（几乎不可能），直接用原数据
                signal = raw_signal

        except Exception as e:
            print(f"Error reading {path}: {e}")
            return np.zeros(self.seq_len)

        # 3. 长度处理：截断或填充到 seq_len (1400)
        if len(signal) > self.seq_len:
            signal = signal[:self.seq_len]
        elif len(signal) < self.seq_len:
            pad_len = self.seq_len - len(signal)
            signal = np.pad(signal, (0, pad_len), 'constant')
        
        return signal

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        sid = sample_info['sid']
        info = self.subject_info[sid]

        # 1. 信号
        raw_signal = self._load_signal(sample_info['path'])
        # Z-Score 归一化 (针对单条信号)
        norm_signal = (raw_signal - np.mean(raw_signal)) / (np.std(raw_signal) + 1e-6)
        
        # 2. 表格特征
        tabular_feats = torch.tensor(info['tabular_features'], dtype=torch.float32)

        # 3. 标签
        target = torch.tensor([info['sbp'], info['dbp']], dtype=torch.float32)
        if self.normalize_labels:
            target = (target - self.bp_mean) / self.bp_std

        return {
            'signal': torch.from_numpy(norm_signal).float(),
            'tabular': tabular_feats,
            'target': target,
            'raw_bp': torch.tensor([info['sbp'], info['dbp']])
        }