import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import time
from collections import defaultdict

class ExperimentTracker:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.metrics_file = os.path.join(save_dir, "metrics.csv")
        self.metrics = defaultdict(list)
        
        # Init CSV header
        if not os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'w') as f:
                f.write("epoch,train_loss,val_loss,grad_norm,gpu_mem_mb,throughput\n")
                
    def log(self, epoch, metrics_dict):
        # Update internal dict
        for k, v in metrics_dict.items():
            self.metrics[k].append(v)
            
        # Write to CSV
        with open(self.metrics_file, 'a') as f:
            f.write(f"{epoch},{metrics_dict.get('train_loss', 0):.4f},{metrics_dict.get('val_loss', 0):.4f},"
                    f"{metrics_dict.get('grad_norm', 0):.4f},{metrics_dict.get('gpu_mem_mb', 0):.1f},"
                    f"{metrics_dict.get('throughput', 0):.2f}\n")

    def check_early_stopping(self, patience=3):
        return False
