import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
import pickle

class DownstreamDataset(Dataset):
    def __init__(self, data_dir, split_file, mode='train', seq_len=512, patch_len=8):
        """
        下游任务数据集
        参数:
        - data_dir: 包含所有 pkl 文件的目录
        - split_file: train_test_split.json 的路径
        - mode: 'train' 或 'test'
        - seq_len: 期望的序列长度
        - patch_len: Patch 长度
        """
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.n_patches = seq_len // patch_len
        
        with open(split_file, 'r') as f:
            self.split_info = json.load(f)
            
        self.file_list = self.split_info[mode]
        print(f"[{mode.upper()}] Loaded {len(self.file_list)} samples.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.data_dir, file_name)
        
        # 1. 加载 pkl 文件
        with open(file_path, 'rb') as f:
            content = pickle.load(f)
            
        # 2. 提取数据 [num_channels, sequence_length]
        # 根据用户描述，data 是 float16 numpy array
        raw_data = content['data'].astype(np.float32)
        
        # 3. 处理长度 (裁剪或填充)
        # 假设 raw_data 形状为 [C, L]
        processed_data = []
        for i in range(raw_data.shape[0]):
            channel_data = raw_data[i]
            if len(channel_data) >= self.seq_len:
                # 裁剪
                start = 0 # 或者随机裁剪
                channel_data = channel_data[start : start + self.seq_len]
            else:
                # 填充
                pad_len = self.seq_len - len(channel_data)
                channel_data = np.pad(channel_data, (0, pad_len), 'constant')
            
            # 归一化 (Z-score)
            mean = channel_data.mean()
            std = channel_data.std() + 1e-5
            channel_data = (channel_data - mean) / std
            
            processed_data.append(channel_data)
            
        # 堆叠通道 [C, seq_len]
        data_tensor = torch.from_numpy(np.stack(processed_data)).float()
        
        # 4. Patching: [C, seq_len] -> [C, n_patches, patch_len]
        patches = data_tensor.view(data_tensor.shape[0], self.n_patches, self.patch_len)
        
        # 5. 提取标签
        # 格式: "label": [{"class": label}, {"reg": label}, ...]
        # 我们假设取第一个 class 标签进行分类
        label = -1
        for item in content['label']:
            if 'class' in item:
                label = item['class']
                break
        
        return patches, torch.tensor(label).long()
