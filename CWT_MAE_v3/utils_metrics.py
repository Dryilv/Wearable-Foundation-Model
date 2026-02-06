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

class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.fc(x)

def compute_reconstruction_loss(pred_spec, target_spec, pred_time, target_time, time_loss_weight=1.0):
    """
    计算验证集上的重建损失 (MSE)
    """
    # 这里简单起见，假设 Loss 已经在 Model forward 里算好了
    # 如果需要在外面单独算，逻辑会很复杂（涉及到 CWT 等）
    # 因此我们主要依赖 Model forward 返回的 Loss
    pass

def train_linear_probe(model, train_loader, val_loader, device, num_classes=2, epochs=5, lr=1e-3, limit_batches=None):
    """
    在冻结的 Encoder 上训练线性分类头
    """
    model.eval()
    
    # 1. 提取特征
    # 为了效率，我们可以先提取所有特征存到内存（如果显存/内存够用）
    # 或者直接在线训练
    
    # 这里为了简单和通用，采用在线提取+训练的方式，虽然会重复前向传播 Encoder
    # 考虑到 Linear Probe 只跑几个 Epoch，且是验证环节，可以接受
    
    # 自动推断 embed_dim
    with torch.no_grad():
        dummy_batch, _ = next(iter(train_loader))
        dummy_batch = dummy_batch.to(device)
        # 兼容 CWT_MAE_RoPE 的 forward_encoder 输出
        # forward_encoder 返回: x, mask, ids, M
        # x: (B, M*N + 1, D)
        imgs = model.module.prepare_tokens(dummy_batch) if hasattr(model, 'module') else model.prepare_tokens(dummy_batch)
        latent, _, _, _ = model.module.forward_encoder(imgs) if hasattr(model, 'module') else model.forward_encoder(imgs)
        embed_dim = latent.shape[-1]
        
    probe = LinearProbe(embed_dim, num_classes).to(device)
    optimizer = optim.AdamW(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Train Loop
    for ep in range(epochs):
        probe.train()
        for i, (batch, labels) in enumerate(train_loader):
            if limit_batches is not None and i >= limit_batches:
                break
                
            batch = batch.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                imgs = model.module.prepare_tokens(batch) if hasattr(model, 'module') else model.prepare_tokens(batch)
                latent, _, _, _ = model.module.forward_encoder(imgs) if hasattr(model, 'module') else model.forward_encoder(imgs)
                # Global Average Pooling over tokens (excluding CLS if used, or just mean)
                # latent: (B, SeqLen, D)
                features = latent[:, 1:, :].mean(dim=1) 
            
            optimizer.zero_grad()
            logits = probe(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
    # Eval Loop
    probe.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch, labels in val_loader:
            batch = batch.to(device)
            labels = labels.to(device)
            
            imgs = model.module.prepare_tokens(batch) if hasattr(model, 'module') else model.prepare_tokens(batch)
            latent, _, _, _ = model.module.forward_encoder(imgs) if hasattr(model, 'module') else model.forward_encoder(imgs)
            features = latent[:, 1:, :].mean(dim=1)
            
            logits = probe(features)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    return acc

def evaluate_features_quality(model, val_loader, device, save_dir, epoch, num_samples=2000):
    """
    计算特征聚类指标并绘制 t-SNE
    """
    model.eval()
    all_features = []
    all_labels = []
    
    count = 0
    with torch.no_grad():
        for batch, labels in val_loader:
            batch = batch.to(device)
            imgs = model.module.prepare_tokens(batch) if hasattr(model, 'module') else model.prepare_tokens(batch)
            latent, _, _, _ = model.module.forward_encoder(imgs) if hasattr(model, 'module') else model.forward_encoder(imgs)
            # GAP
            features = latent[:, 1:, :].mean(dim=1).cpu().numpy()
            all_features.append(features)
            all_labels.append(labels.numpy())
            
            count += batch.shape[0]
            if count >= num_samples:
                break
                
    X = np.concatenate(all_features)[:num_samples]
    y = np.concatenate(all_labels)[:num_samples]
    
    # 1. Clustering Metrics
    # 只有当类别数 > 1 时才计算
    if len(np.unique(y)) > 1:
        sil_score = silhouette_score(X, y)
        db_score = davies_bouldin_score(X, y)
    else:
        sil_score = -1.0
        db_score = -1.0
        
    # 2. t-SNE Visualization
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_emb = tsne.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_emb[:, 0], X_emb[:, 1], c=y, cmap='tab10', alpha=0.6, s=10)
    plt.colorbar(scatter)
    plt.title(f"t-SNE Epoch {epoch} (Sil={sil_score:.3f})")
    plt.savefig(os.path.join(save_dir, f"tsne_epoch_{epoch}.png"))
    plt.close()
    
    return sil_score, db_score

class ExperimentTracker:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.metrics_file = os.path.join(save_dir, "metrics.csv")
        self.metrics = defaultdict(list)
        
        # Init CSV header
        if not os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'w') as f:
                f.write("epoch,train_loss,val_loss,grad_norm,gpu_mem_mb,throughput,linear_acc,sil_score,db_score\n")
                
    def log(self, epoch, metrics_dict):
        # Update internal dict
        for k, v in metrics_dict.items():
            self.metrics[k].append(v)
            
        # Write to CSV
        with open(self.metrics_file, 'a') as f:
            f.write(f"{epoch},{metrics_dict.get('train_loss', 0):.4f},{metrics_dict.get('val_loss', 0):.4f},"
                    f"{metrics_dict.get('grad_norm', 0):.4f},{metrics_dict.get('gpu_mem_mb', 0):.1f},"
                    f"{metrics_dict.get('throughput', 0):.2f},{metrics_dict.get('linear_acc', 0):.4f},"
                    f"{metrics_dict.get('sil_score', 0):.4f},{metrics_dict.get('db_score', 0):.4f}\n")

    def check_early_stopping(self, patience=3):
        """
        检查特征质量指标是否连续 patience 轮无提升
        这里主要看 Silhouette Score (越大越好)
        """
        scores = self.metrics.get('sil_score', [])
        # 过滤掉 -1 (未计算的轮次)
        valid_scores = [s for s in scores if s > -1]
        
        if len(valid_scores) < patience + 1:
            return False
            
        recent = valid_scores[-patience:]
        best_prev = max(valid_scores[:-patience])
        
        # 如果最近 patience 轮的最大值都没有超过之前的最佳值，且有下降趋势
        if max(recent) <= best_prev:
            return True
            
        return False
