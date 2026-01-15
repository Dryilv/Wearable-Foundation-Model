import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import sys

# 引入你的模型
from model_supcon import SupCon_CWT_MAE
from losses import SupConLoss

# ==========================================
# 配置
# ==========================================
DEVICE = torch.device("cuda:0")
BATCH_SIZE = 16  # 小 Batch
LR = 1e-3
TEMP = 0.1

print(f"🚀 Starting Nuclear Debug on {DEVICE}...")

# 1. 初始化模型 (强制 float32)
print("\n[1] Initializing Model...")
try:
    model = SupCon_CWT_MAE(
        signal_len=3000,
        embed_dim=768,
        depth=4,        # 减少层数，方便调试
        num_heads=4,
        cwt_scales=64,
        patch_size_time=50,
        patch_size_freq=4,
        mlp_rank_ratio=0.5
    ).to(DEVICE).float() # 强制 float32
    print("✅ Model initialized.")
except Exception as e:
    print(f"❌ Model init failed: {e}")
    sys.exit(1)

# 2. 构造合成数据 (作弊模式：View1 == View2)
print("\n[2] Generating Synthetic Data (Identity Views)...")
# 随机生成信号 [B, 3000]
raw_data = torch.randn(BATCH_SIZE, 3000).to(DEVICE)
# 模拟 View 1 和 View 2 完全一样
images1 = raw_data.unsqueeze(1) # [B, 1, 3000]
images2 = raw_data.unsqueeze(1) # [B, 1, 3000]
# 标签：每个样本自成一类，或者随机分类
labels = torch.arange(BATCH_SIZE).to(DEVICE) # [0, 1, 2, ... 15]

# 3. 检查 CWT 输出 (关键疑点)
print("\n[3] Checking CWT Output...")
try:
    from model import cwt_wrap
    with torch.no_grad():
        cwt_out = cwt_wrap(raw_data, num_scales=64)
        print(f"   CWT Shape: {cwt_out.shape}")
        print(f"   CWT Mean: {cwt_out.mean().item():.4f}")
        print(f"   CWT Std:  {cwt_out.std().item():.4f}")
        print(f"   CWT Max:  {cwt_out.max().item():.4f}")
        
        if torch.isnan(cwt_out).any():
            print("❌ CRITICAL: CWT output contains NaN!")
            sys.exit(1)
        if cwt_out.std() < 1e-6:
            print("❌ CRITICAL: CWT output is constant (Zero/Dead)!")
            sys.exit(1)
        print("✅ CWT looks healthy.")
except Exception as e:
    print(f"❌ CWT check failed: {e}")
    sys.exit(1)

# 4. 运行训练循环 (Overfit Test)
print("\n[4] Starting Overfit Loop (Loss MUST drop)...")
optimizer = optim.AdamW(model.parameters(), lr=LR)
criterion = SupConLoss(temperature=TEMP).to(DEVICE)

model.train()

for step in range(20):
    optimizer.zero_grad()
    
    # 拼接
    images = torch.cat([images1, images2], dim=0) # [32, 1, 3000]
    
    # Forward
    features = model(images) # [32, 128]
    
    # 检查特征是否坍塌
    if step == 0:
        feat_std = features.std(dim=0).mean().item()
        print(f"   Step 0 Feature Std: {feat_std:.6f}")
        if feat_std < 1e-6:
            print("⚠️ WARNING: Initial features are collapsed (all same).")

    # Split
    f1, f2 = torch.split(features, [BATCH_SIZE, BATCH_SIZE], dim=0)
    features_global = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
    
    # Loss
    loss = criterion(features_global, labels)
    
    # Backward
    loss.backward()
    
    # 检查梯度
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm().item()
            
    print(f"Step {step+1:02d} | Loss: {loss.item():.6f} | Grad Norm: {grad_norm:.6f}")
    
    if torch.isnan(loss):
        print("❌ Loss is NaN!")
        break
        
    optimizer.step()

print("\n[5] Diagnosis Finished.")