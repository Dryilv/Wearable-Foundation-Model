import unittest
import os
import json
import numpy as np
import pickle
import tempfile
import shutil
import sys
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import PhysioSignalDataset

class TestDataset(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        
        # Create mock data files
        self.data_files = []
        for i in range(3):
            data_path = os.path.join(self.test_dir, f"data_{i}.pkl")
            # Create a mock signal of length 3000 (original_len was 3000)
            mock_data = {
                'data': np.random.randn(1, 3000).astype(np.float32),
                'label': 0
            }
            with open(data_path, 'wb') as f:
                pickle.dump(mock_data, f)
            self.data_files.append(data_path)
            
        # Create index file
        self.index_file = os.path.join(self.test_dir, "index.json")
        index_data = [
            {'path': self.data_files[0], 'row': 0, 'label': 0},
            {'path': self.data_files[1], 'row': 0, 'label': 0},
            {'path': self.data_files[2], 'row': 0, 'label': 0}
        ]
        with open(self.index_file, 'w') as f:
            json.dump(index_data, f)

    def tearDown(self):
        # Remove temporary directory
        shutil.rmtree(self.test_dir)

    def test_dataset_no_sliding_window(self):
        # Initialize dataset without stride (as it is removed)
        dataset = PhysioSignalDataset(
            index_file=self.index_file,
            signal_len=1000,
            mode='train'
        )
        
        # Length should be equal to number of files (3), not expanded
        self.assertEqual(len(dataset), 3)
        
        # Test __getitem__
        signal, label = dataset[0]
        self.assertEqual(signal.shape, (1, 1000))
        
    def test_dataset_init_signature(self):
        # Verify that passing stride raises TypeError (confirming removal)
        with self.assertRaises(TypeError):
            PhysioSignalDataset(
                index_file=self.index_file,
                signal_len=1000,
                stride=500 # This should fail
            )

if __name__ == '__main__':
    unittest.main()
