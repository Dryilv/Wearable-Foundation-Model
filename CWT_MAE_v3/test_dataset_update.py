import torch
import numpy as np
import os
import json
import pickle
import unittest
from torch.utils.data import DataLoader
from dataset import PhysioSignalDataset, fixed_channel_collate_fn, load_pickle_file
from model import CWT_MAE_RoPE

class TestDatasetUpdate(unittest.TestCase):
    def setUp(self):
        # Clear cache
        load_pickle_file.cache_clear()
        
        # Create dummy data
        self.temp_dir = 'temp_test_data'
        os.makedirs(self.temp_dir, exist_ok=True)
        
        self.pkl_path = os.path.abspath(os.path.join(self.temp_dir, 'sample.pkl'))
        # Create 5-channel data (1 sample in file)
        # Shape: (1, 5, 1000)
        data = np.random.randn(1, 5, 1000).astype(np.float32)
        with open(self.pkl_path, 'wb') as f:
            pickle.dump({'data': data}, f)
            
        self.index_file = os.path.abspath(os.path.join(self.temp_dir, 'index.json'))
        # Point to row 0
        index_data = [{'path': self.pkl_path, 'row': 0, 'label': 1}]
        with open(self.index_file, 'w') as f:
            json.dump(index_data, f)
            
    def tearDown(self):
        # Cleanup
        if os.path.exists(self.pkl_path):
            os.remove(self.pkl_path)
        if os.path.exists(self.index_file):
            os.remove(self.index_file)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    def test_dataset_loading(self):
        # 3) 验证加载后的样本 shape 为 (5, 1000)
        ds = PhysioSignalDataset(index_file=self.index_file, signal_len=1000, expected_channels=5)
        item, label = ds[0]
        self.assertEqual(item.shape, (5, 1000))
        print(f"Sample shape: {item.shape}")

    def test_collate_fn(self):
        print("\n--- Testing Collate Function ---")
        ds = PhysioSignalDataset(index_file=self.index_file, signal_len=1000, expected_channels=5)
        # Mock more samples by duplicating index
        ds.samples = ds.samples * 4 
        
        loader = DataLoader(ds, batch_size=2, collate_fn=fixed_channel_collate_fn)
        batch, labels = next(iter(loader))
        print(f"Batch shape: {batch.shape}, Labels shape: {labels.shape}")
        self.assertEqual(batch.shape, (2, 5, 1000))
        self.assertEqual(labels.shape, (2,))

    def test_sliding_window(self):
        print("\n--- Testing Sliding Window ---")
        # 模拟包含长度信息的 index
        index_data = [{
            'path': self.pkl_path,
            'row': 0,
            'label': 1,
            'len': 2000 # 2000 点总长
        }]
        
        # 设定 signal_len=1000, stride=500
        # 预期窗口起始点: 0, 500, 1000 (因为 1000 + 1000 = 2000, 1500 + 1000 > 2000 不行)
        # range(0, 2000 - 1000 + 1, 500) -> [0, 500, 1000]
        ds = PhysioSignalDataset(
            data_source=index_data, 
            signal_len=1000, 
            use_sliding_window=True, 
            window_stride=500
        )
        
        print(f"Number of samples generated: {len(ds)}")
        self.assertEqual(len(ds), 3)
        
        # 验证每个窗口的内容是否正确
        for i in range(len(ds)):
            item, label = ds[i]
            self.assertEqual(item.shape, (5, 1000))
            print(f"Window {i} loaded successfully.")

    def test_model_forward(self):
        # 4) 运行完整前向步骤
        print("\n--- Testing Model Forward Pass ---")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        from model import CWT_MAE_RoPE
        model = CWT_MAE_RoPE(signal_len=1000, use_factorized_attn=True).to(device)
        model.eval()
        
        # 构造 5 通道输入 (B, 5, 1000)
        batch = torch.randn(2, 5, 1000).to(device)
        with torch.no_grad():
            loss, pred_spec, pred_time, imgs, mask = model(batch)
            
        print(f"Loss: {loss.item()}")
        print(f"Reconstructed image shape: {imgs.shape}")
        print(f"Mask shape: {mask.shape}")
        # imgs shape: (B, 5, C, H, W) -> (2, 5, 3, 64, 1000)
        self.assertEqual(imgs.shape, (2, 5, 3, 64, 1000))

if __name__ == '__main__':
    unittest.main()
