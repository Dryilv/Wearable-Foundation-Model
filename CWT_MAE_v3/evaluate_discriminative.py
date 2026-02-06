import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt

from dataset_finetune import DownstreamClassificationDataset
from model_finetune import TF_MAE_Classifier
from finetune import variable_channel_collate_fn_cls

def calculate_iv(feature, target, n_bins=10):
    """
    计算单个特征的 IV 值 (Information Value)
    feature: numpy array, continuous
    target: numpy array, binary (0/1)
    """
    try:
        df = pd.DataFrame({'val': feature, 'y': target})
        # 使用 qcut 进行等频分箱
        df['bin'] = pd.qcut(df['val'], n_bins, duplicates='drop')
        
        stats = df.groupby('bin', observed=True)['y'].agg(['count', 'sum'])
        stats['good'] = stats['sum'] # 假设 1 是 Good/Positive
        stats['bad'] = stats['count'] - stats['sum'] # 0 是 Bad/Negative
        
        total_good = stats['good'].sum()
        total_bad = stats['bad'].sum()
        
        if total_good == 0 or total_bad == 0:
            return 0.0

        stats['dist_good'] = stats['good'] / total_good
        stats['dist_bad'] = stats['bad'] / total_bad
        
        # WOE & IV
        # Add small epsilon to avoid div by zero or log(0)
        epsilon = 1e-6
        stats['woe'] = np.log((stats['dist_good'] + epsilon) / (stats['dist_bad'] + epsilon))
        stats['iv'] = (stats['dist_good'] - stats['dist_bad']) * stats['woe']
        
        return stats['iv'].sum()
    except Exception as e:
        # print(f"IV calc error: {e}")
        return 0.0

def calculate_ks(y_true, y_prob):
    """
    计算 KS 值 (Kolmogorov-Smirnov)
    """
    try:
        data = pd.DataFrame({'label': y_true, 'prob': y_prob})
        data0 = data[data['label'] == 0]['prob']
        data1 = data[data['label'] == 1]['prob']
        ks_stat, p_value = ks_2samp(data0, data1)
        return ks_stat
    except:
        return 0.0

def extract_features(model, loader, device):
    model.eval()
    all_features = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Extracting Features"):
            x = x.to(device)
            # 获取特征 (Embedding)
            features = model(x, return_features=True)
            
            # 获取预测概率 (用于计算 AUC/KS)
            # 如果是 ArcFace，logits 已经是 cosine * s
            # 这里的 probs 可能不直接代表概率，但用于排序计算 AUC/KS 是可以的
            # 或者我们可以再过一个 Softmax
            logits = model(x, return_features=False)
            probs = torch.softmax(logits, dim=1)
            
            all_features.append(features.cpu().numpy())
            all_labels.append(y.numpy())
            all_probs.append(probs.cpu().numpy())
            
    return np.concatenate(all_features), np.concatenate(all_labels), np.concatenate(all_probs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--split_file', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--signal_len', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_classes', type=int, default=2)
    
    # Model Args needed for reconstruction
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--use_cot', action='store_true')
    parser.add_argument('--use_arcface', action='store_true')
    parser.add_argument('--arcface_s', type=float, default=30.0)
    parser.add_argument('--arcface_m', type=float, default=0.50)
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    print(f"Loading model from {args.model_path}...")
    model = TF_MAE_Classifier(
        pretrained_path=None, # Don't load pretrain weights, load full checkpoint later
        num_classes=args.num_classes,
        signal_len=args.signal_len,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        use_cot=args.use_cot,
        use_arcface=args.use_arcface,
        arcface_s=args.arcface_s,
        arcface_m=args.arcface_m
    )
    
    # Load state dict
    checkpoint = torch.load(args.model_path, map_location='cpu')
    # Handle DDP keys
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    
    # 2. Load Data
    ds = DownstreamClassificationDataset(
        args.data_root, args.split_file, mode='test', 
        signal_len=args.signal_len, num_classes=args.num_classes
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, 
        num_workers=4, collate_fn=variable_channel_collate_fn_cls
    )
    
    # 3. Extract Features
    features, labels, probs = extract_features(model, loader, device)
    print(f"Features shape: {features.shape}")
    
    # 4. Calculate Metrics
    # Global Metrics
    if args.num_classes == 2:
        auc = roc_auc_score(labels, probs[:, 1])
        ks = calculate_ks(labels, probs[:, 1])
        print(f"\n[Global Metrics] AUC: {auc:.4f} | KS: {ks:.4f}")
    
    # 5. Feature Analysis (IV)
    print("\n[Feature Analysis] Calculating IV for each embedding dimension...")
    iv_scores = []
    for i in range(features.shape[1]):
        iv = calculate_iv(features[:, i], labels)
        iv_scores.append((i, iv))
    
    # Sort by IV
    iv_scores.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 20 Discriminative Features (Dimensions):")
    print(f"{'Dim':<10} | {'IV':<10}")
    print("-" * 25)
    for dim, iv in iv_scores[:20]:
        print(f"{dim:<10} | {iv:.4f}")
        
    # Count High IV Features
    high_iv_count = sum(1 for _, iv in iv_scores if iv >= 0.3)
    print(f"\nNumber of features with IV >= 0.3: {high_iv_count}")
    
    # 6. AutoFeature / Feature Combination Search (Simple Demo)
    # Try combining Top 2 features
    if len(iv_scores) >= 2:
        idx1, iv1 = iv_scores[0]
        idx2, iv2 = iv_scores[1]
        
        f1 = features[:, idx1]
        f2 = features[:, idx2]
        
        # Interactions
        comb_add = f1 + f2
        comb_sub = f1 - f2
        comb_mul = f1 * f2
        
        iv_add = calculate_iv(comb_add, labels)
        iv_sub = calculate_iv(comb_sub, labels)
        iv_mul = calculate_iv(comb_mul, labels)
        
        print(f"\n[Feature Combination Search]")
        print(f"Dim {idx1} (IV={iv1:.4f}) & Dim {idx2} (IV={iv2:.4f})")
        print(f"Add IV: {iv_add:.4f}")
        print(f"Sub IV: {iv_sub:.4f}")
        print(f"Mul IV: {iv_mul:.4f}")

if __name__ == "__main__":
    main()
