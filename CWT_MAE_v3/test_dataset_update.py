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
        # Shape: (1, 5, 3000)
        data = np.random.randn(1, 5, 3000).astype(np.float32)
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
        print("\n--- Testing Dataset Loading (5 Channels) ---")
        ds = PhysioSignalDataset(index_file=self.index_file, signal_len=3000, expected_channels=5)
        item, label = ds[0]
        print(f"Item shape: {item.shape}, Label: {label}")
        self.assertEqual(item.shape, (5, 3000))
        self.assertEqual(label, 1)

    def test_collate_fn(self):
        print("\n--- Testing Collate Function ---")
        ds = PhysioSignalDataset(index_file=self.index_file, signal_len=3000, expected_channels=5)
        # Mock more samples by duplicating index
        ds.samples = ds.samples * 4 
        
        loader = DataLoader(ds, batch_size=2, collate_fn=fixed_channel_collate_fn)
        batch, labels = next(iter(loader))
        print(f"Batch shape: {batch.shape}, Labels shape: {labels.shape}")
        self.assertEqual(batch.shape, (2, 5, 3000))
        self.assertEqual(labels.shape, (2,))

    def test_model_forward(self):
        print("\n--- Testing Model Forward Pass ---")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
            
        model = CWT_MAE_RoPE(
            signal_len=3000,
            cwt_scales=64,
            patch_size_time=25,
            patch_size_freq=4,
            embed_dim=384,
            depth=2, 
            num_heads=6,
            decoder_embed_dim=192,
            decoder_depth=2,
            decoder_num_heads=6,
            mask_ratio=0.75,
            mlp_rank_ratio=0.5,
            use_factorized_attn=True
        )
        model.to(device)
        
        # Create dummy batch
        batch = torch.randn(2, 5, 3000).to(device)
        with torch.no_grad():
            loss, pred_spec, pred_time, imgs = model(batch)
            
        print(f"Forward pass successful. Loss: {loss.item()}")
        self.assertTrue(loss.item() > 0)
        # imgs shape: (B, M, 3, H, W) -> (2, 5, 3, 64, 3000)
        self.assertEqual(imgs.shape, (2, 5, 3, 64, 3000))

if __name__ == '__main__':
    unittest.main()
