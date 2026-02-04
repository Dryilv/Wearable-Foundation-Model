import os
import argparse
import yaml
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.amp import autocast
import sys
import os
# Default to v2 model for visualization
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../CWT_MAE_v2')))

# 导入你的模型定义和数据集
from model import CWT_MAE_RoPE , cwt_wrap
from dataset import PhysioSignalDataset

def load_model(config_path, checkpoint_path, device):
    """加载配置和模型权重"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Initializing model...")
    model = CWT_MAE_RoPE(
        signal_len=config['data']['signal_len'],
        cwt_scales=config['model'].get('cwt_scales', 64),
        patch_size_time=config['model'].get('patch_size_time', 50),
        patch_size_freq=config['model'].get('patch_size_freq', 4),
        embed_dim=config['model']['embed_dim'],
        depth=config['model']['depth'],
        num_heads=config['model']['num_heads'],
        decoder_embed_dim=config['model']['decoder_embed_dim'],
        decoder_depth=config['model']['decoder_depth'],
        decoder_num_heads=config['model']['decoder_num_heads'],
        mask_ratio=config['model']['mask_ratio'],
        time_loss_weight=config['model'].get('time_loss_weight', 1.0)
    )
    
    if os.path.isfile(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 处理权重键名 (去除 DDP 和 Compile 前缀)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k
            if name.startswith('module.'): name = name[7:]
            if name.startswith('_orig_mod.'): name = name[10:]
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict)
        print("Weights loaded successfully.")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model.to(device)
    model.eval()
    return model, config

def visualize_sample(model, dataset, index, device, save_path="recon_result.png"):
    """可视化单个样本的重建效果"""
    print(f"Visualizing sample index: {index}")
    
    # 1. 获取数据
    # dataset[index] 返回 [1, L]
    sample = dataset[index]
    sample = sample.to(device) # [1, L]
    
    # 2. 模型前向推理
    with torch.no_grad():
        # 自动选择精度
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        with autocast(device_type='cuda', dtype=amp_dtype):
            # A. 手动执行 CWT 和 归一化 (模拟 forward 内部逻辑)
            if sample.dim() == 3: sample = sample.squeeze(1)
            imgs = cwt_wrap(sample, num_scales=model.cwt_scales, lowest_scale=0.1, step=1.0)
            
            # Instance Norm
            imgs_f32 = imgs.float()
            mean = imgs_f32.mean(dim=(2, 3), keepdim=True)
            std = imgs_f32.std(dim=(2, 3), keepdim=True)
            std = torch.clamp(std, min=1e-5)
            imgs_norm = (imgs_f32 - mean) / std
            
            # B. Encoder (获取 Mask)
            latent, mask, ids_restore = model.forward_encoder(imgs_norm)
            
            # C. Decoder (获取重建特征)
            pred_features = model.forward_decoder(latent, ids_restore)
            
            # D. Heads (获取最终预测)
            pred_spec = model.decoder_pred_spec(pred_features)
            
            # 时域重建
            B, N, D = pred_features.shape
            H_grid, W_grid = model.grid_size
            feat_2d = pred_features.transpose(1, 2).view(B, D, H_grid, W_grid)
            feat_time_agg = model.time_reducer(feat_2d)
            feat_time_agg = feat_time_agg.squeeze(2).transpose(1, 2)
            pred_time = model.time_pred(feat_time_agg).flatten(1)

    # 3. 数据后处理 (转 Numpy 用于绘图)
    
    # --- 时域 ---
    orig_signal = sample[0].float().cpu().numpy()
    recon_signal = pred_time[0].float().cpu().numpy()
    
    # 简单反归一化 (对齐均值方差，方便视觉对比)
    orig_mean = orig_signal.mean()
    orig_std = orig_signal.std()
    recon_signal = recon_signal * (orig_std + 1e-6) + orig_mean

    # --- 频域 ---
    # 获取 Patch 参数
    p_h, p_w = model.patch_embed.patch_size
    B, C, H, W = imgs_norm.shape
    
    # 原始图谱 (取 Channel 0)
    orig_spec = imgs_norm[0, 0, :, :].float().cpu().numpy()
    
    # 重建图谱 (Unpatchify)
    # pred_spec: [1, Num_Patches, C*ph*pw]
    pred_patches = pred_spec[0].view(H // p_h, W // p_w, C, p_h, p_w)
    pred_patches = pred_patches.permute(2, 0, 3, 1, 4) # [C, Grid_H, p_h, Grid_W, p_w]
    recon_img = pred_patches.reshape(C, H, W)
    recon_spec = recon_img[0].float().cpu().numpy()
    
    # Mask 可视化
    # mask: [1, Num_Patches]
    mask_patch = mask[0].view(H // p_h, W // p_w)
    # 放大回像素级
    mask_pixel = mask_patch.repeat_interleave(p_h, dim=0).repeat_interleave(p_w, dim=1)
    mask_pixel = mask_pixel.float().cpu().numpy()
    
    # 生成 Masked Input (被遮挡的部分置为 NaN 或 0，这里用 0 方便显示)
    masked_input_spec = orig_spec.copy()
    masked_input_spec[mask_pixel == 1] = np.min(orig_spec) # 用最小值填充，显示为深色

    # 4. 绘图
    plt.figure(figsize=(16, 12))
    
    # Row 1: Time Domain
    plt.subplot(4, 1, 1)
    plt.plot(orig_signal, label='Original', color='black', alpha=0.6, linewidth=1)
    plt.plot(recon_signal, label='Reconstructed', color='red', alpha=0.6, linewidth=1)
    plt.title(f"Time Domain Reconstruction (Index {index})")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Row 2: Original CWT
    plt.subplot(4, 1, 2)
    plt.imshow(orig_spec, aspect='auto', origin='lower', cmap='jet')
    plt.title("Original CWT Spectrogram")
    plt.colorbar()

    # Row 3: Masked Input
    plt.subplot(4, 1, 3)
    plt.imshow(masked_input_spec, aspect='auto', origin='lower', cmap='jet')
    plt.title("Masked Input (What Model Sees)")
    plt.colorbar()

    # Row 4: Reconstructed CWT
    plt.subplot(4, 1, 4)
    plt.imshow(recon_spec, aspect='auto', origin='lower', cmap='jet')
    plt.title("Reconstructed CWT (Model Prediction)")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Result saved to {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize CWT-MAE Reconstruction")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--index', type=int, default=None, help='Sample index to visualize (random if None)')
    parser.add_argument('--output', type=str, default='recon_vis.png', help='Output image path')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 加载模型
    model, config = load_model(args.config, args.checkpoint, device)

    # 2. 加载数据集
    print("Loading dataset...")
    dataset = PhysioSignalDataset(
        index_file=config['data']['index_path'],
        signal_len=config['data']['signal_len'],
        mode='train' # 使用训练集查看拟合情况，也可以改 'test'
    )
    
    # 3. 选择样本
    if args.index is None:
        import random
        idx = random.randint(0, len(dataset) - 1)
    else:
        idx = args.index
        
    # 4. 可视化
    visualize_sample(model, dataset, idx, device, save_path=args.output)

if __name__ == "__main__":
    main()