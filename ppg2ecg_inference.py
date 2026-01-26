import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import random

# 引入你的模型定义
from model_ppg2ecg import PPG2ECG_Translator
from model import cwt_wrap # 用于可视化频谱

# -------------------------------------------------------------------
# 1. 配置参数 (需要与训练时的 config.yaml 保持一致)
# -------------------------------------------------------------------
MODEL_CONFIG = {
    'signal_len': 3000,
    'cwt_scales': 64,
    'embed_dim': 768,
    'depth': 12,
    'num_heads': 12,
    'decoder_embed_dim': 512,
    'decoder_depth': 8,
    'decoder_num_heads': 16,
    'patch_size_time': 50,
    'patch_size_freq': 4,
    
}

# 数据行号映射 (根据你之前的修正)
ROW_PPG = 4
ROW_ECG = 0

# -------------------------------------------------------------------
# 2. 模型加载函数
# -------------------------------------------------------------------
def load_model(checkpoint_path, device):
    print(f"Creating model with config: {MODEL_CONFIG}")
    model = PPG2ECG_Translator(**MODEL_CONFIG)
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 兼容性处理：支持加载仅保存了 state_dict 的文件，也支持保存了完整 info 的文件
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    
    # 清洗键名 (去除 DDP 和 Compile 前缀)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '').replace('_orig_mod.', '')
        new_state_dict[name] = v
        
    msg = model.mae.load_state_dict(new_state_dict, strict=False)
    print(f"Weights loaded. Missing keys: {msg.missing_keys}")
    
    model.to(device)
    model.eval()
    return model

# -------------------------------------------------------------------
# 3. 数据预处理函数 (必须与训练保持一致)
# -------------------------------------------------------------------
def preprocess_signal(signal):
    """
    输入: numpy array (L,)
    输出: tensor (1, 1, L)
    """
    signal = signal.astype(np.float32)
    
    # 1. 长度处理 (裁剪或填充)
    target_len = MODEL_CONFIG['signal_len']
    current_len = len(signal)
    
    if current_len > target_len:
        # 推理时通常取中间一段，或者从头开始
        start = (current_len - target_len) // 2
        signal = signal[start : start + target_len]
    elif current_len < target_len:
        pad_len = target_len - current_len
        signal = np.pad(signal, (0, pad_len), 'constant')
        
    # 2. Z-Score 归一化
    mean = np.mean(signal)
    std = np.std(signal)
    norm_signal = (signal - mean) / (std + 1e-6)
    
    # 3. 异常值截断 (与训练时的优化建议保持一致)
    norm_signal = np.clip(norm_signal, -5.0, 5.0)
    
    # 转为 Tensor [Batch=1, Channel=1, Length]
    return torch.from_numpy(norm_signal).unsqueeze(0).unsqueeze(0)

# -------------------------------------------------------------------
# 4. 推理与可视化
# -------------------------------------------------------------------
def run_inference(model, ppg_raw, ecg_raw, device, save_path="result.png"):
    # 预处理
    ppg_tensor = preprocess_signal(ppg_raw).to(device)
    
    # 如果有真实 ECG，也做同样的预处理以便对比
    ecg_gt_norm = None
    if ecg_raw is not None:
        ecg_gt_tensor = preprocess_signal(ecg_raw) # 仅用于归一化
        ecg_gt_norm = ecg_gt_tensor.squeeze().numpy()
    
    # 推理
    print("Running inference...")
    with torch.no_grad():
        # 使用 float32 或 bfloat16
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            # model 返回 (pred_time, pred_spec)
            pred_ecg_time, _ = model(ppg_tensor, ecg_target=None)
            
    # 转回 Numpy
    pred_ecg = pred_ecg_time.squeeze().float().cpu().numpy()
    ppg_input = ppg_tensor.squeeze().float().cpu().numpy()
    
    # --- 绘图 ---
    plt.figure(figsize=(18, 12))
    
    # 子图 1: 输入 PPG
    plt.subplot(3, 1, 1)
    plt.plot(ppg_input, color='green', label='Input PPG (Normalized)')
    plt.title("Input: PPG Signal")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图 2: ECG 对比
    plt.subplot(3, 1, 2)
    if ecg_gt_norm is not None:
        plt.plot(ecg_gt_norm, color='black', alpha=0.5, label='Ground Truth ECG')
    plt.plot(pred_ecg, color='red', linewidth=1.5, label='Generated ECG')
    plt.title("Output: Generated ECG vs Ground Truth")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图 3: 生成 ECG 的频谱 (检查是否包含高频噪声)
    plt.subplot(3, 1, 3)
    # 临时计算 CWT
    with torch.no_grad():
        cwt_out = cwt_wrap(torch.from_numpy(pred_ecg).unsqueeze(0).unsqueeze(0).to(device), 
                           num_scales=64, lowest_scale=0.1, step=1.0)
        spec_img = cwt_out[0, 0].float().cpu().numpy() # 取 Channel 0
        
    plt.imshow(spec_img, aspect='auto', origin='lower', cmap='jet')
    plt.title("Spectrogram of Generated ECG")
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Result saved to {save_path}")

# -------------------------------------------------------------------
# 主程序
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="PPG to ECG Inference")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained checkpoint (.pth)')
    parser.add_argument('--input_file', type=str, default=None, help='Path to a pickle file containing data')
    parser.add_argument('--index_file', type=str, default=None, help='Path to train_index.json (to pick random sample)')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--output', type=str, default='inference_result.png', help='Output image path')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # 1. 加载模型
    model = load_model(args.checkpoint, device)
    
    # 2. 获取数据
    ppg_data = None
    ecg_data = None
    
    if args.input_file:
        # 模式 A: 指定单个文件
        print(f"Loading file: {args.input_file}")
        with open(args.input_file, 'rb') as f:
            content = pickle.load(f)
            data = content['data']
            ppg_data = data[ROW_PPG]
            ecg_data = data[ROW_ECG]
            
    elif args.index_file:
        # 模式 B: 从 Index 中随机选一个
        print(f"Picking random sample from: {args.index_file}")
        with open(args.index_file, 'r') as f:
            index_list = json.load(f)
        
        # 随机尝试直到成功读取
        while ppg_data is None:
            item = random.choice(index_list)
            path = item['path']
            try:
                with open(path, 'rb') as f:
                    content = pickle.load(f)
                    data = content['data']
                    ppg_data = data[ROW_PPG]
                    ecg_data = data[ROW_ECG]
                    print(f"Selected sample: {path}")
            except Exception as e:
                print(f"Error reading {path}, retrying...")
                continue
    else:
        print("Error: Please provide either --input_file or --index_file")
        sys.exit(1)
        
    # 3. 运行推理
    run_inference(model, ppg_data, ecg_data, device, save_path=args.output)

if __name__ == "__main__":
    main()