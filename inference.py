import os
import glob
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast 

from model_finetune import TF_MAE_Classifier

# ==========================================
# 1. 基础信号检查 (仅保留最基础的非空检查)
# ==========================================
def check_basic_validity(signal):
    if len(signal) == 0: return False
    if not np.isfinite(signal).all(): return False
    # 极小值检查，防止除以0，但主要过滤交给自适应逻辑
    if np.std(signal) < 1e-6: return False 
    return True

# ==========================================
# 2. 自适应推理数据集 (核心修改)
# ==========================================
class AdaptivePatientDataset(Dataset):
    def __init__(self, file_paths, signal_len=3000, stride=1500, iqr_scale=1.5):
        """
        iqr_scale: 离群值判定的严格程度。
                   1.5 是标准值 (Tukey's fences)。
                   如果想保留更多数据，可以设为 2.0 或 3.0。
        """
        self.windows = []
        self.sqa_stats = {
            "total_raw": 0, 
            "valid_final": 0, 
            "threshold_low": 0.0, 
            "threshold_high": 0.0,
            "dropped_low": 0,
            "dropped_high": 0
        }
        
        # --- 第一阶段：收集所有原始窗口及其统计特征 ---
        raw_segments = []
        std_values = []
        
        for fp in file_paths:
            try:
                with open(fp, 'rb') as f:
                    content = pickle.load(f)
                    raw_data = content['data'] if isinstance(content, dict) else content
                    if raw_data.ndim == 2: raw_data = raw_data[0]
                    raw_data = raw_data.astype(np.float32)
                
                n_samples = len(raw_data)
                if n_samples < signal_len: continue
                
                for start in range(0, n_samples - signal_len + 1, stride):
                    segment = raw_data[start : start + signal_len]
                    
                    # 基础检查 (NaN, 空值, 纯死线)
                    if check_basic_validity(segment):
                        std_val = np.std(segment)
                        raw_segments.append(segment)
                        std_values.append(std_val)
                        self.sqa_stats["total_raw"] += 1
                        
            except Exception as e:
                print(f"Error reading {fp}: {e}")

        # 如果没有数据，直接返回
        if not std_values:
            return

        # --- 第二阶段：计算自适应阈值 (IQR Method) ---
        std_array = np.array(std_values)
        
        # 计算分位数
        q1 = np.percentile(std_array, 25)
        q3 = np.percentile(std_array, 75)
        iqr = q3 - q1
        
        # 定义上下限
        # 下限至少要是 0.001 (防止保留完全平直的线)
        lower_bound = max(0.001, q1 - iqr_scale * iqr)
        # 上限
        upper_bound = q3 + iqr_scale * iqr
        
        self.sqa_stats["threshold_low"] = lower_bound
        self.sqa_stats["threshold_high"] = upper_bound
        
        # --- 第三阶段：过滤并标准化 ---
        for segment, std_val in zip(raw_segments, std_values):
            
            if std_val < lower_bound:
                self.sqa_stats["dropped_low"] += 1
                continue
            
            if std_val > upper_bound:
                self.sqa_stats["dropped_high"] += 1
                continue
            
            # 通过筛选，进行归一化
            mean = np.mean(segment)
            # std_val 已经在上面算过了，直接用
            segment_norm = (segment - mean) / (std_val + 1e-6)
            
            self.windows.append(segment_norm)
            self.sqa_stats["valid_final"] += 1

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        sig = self.windows[idx]
        return torch.from_numpy(sig).unsqueeze(0)

# ==========================================
# 3. 主推理逻辑
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_csv', type=str, default="inference_report_adaptive.csv")
    
    # 模型参数
    parser.add_argument('--signal_len', type=int, default=1000)
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--cwt_scales', type=int, default=64)
    parser.add_argument('--patch_size_time', type=int, default=50)
    parser.add_argument('--patch_size_freq', type=int, default=4)

    # 推理参数
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--stride', type=int, default=500)
    
    # 自适应过滤参数
    parser.add_argument('--iqr_scale', type=float, default=0.8, 
                        help="IQR倍数。1.5是标准去噪，3.0是宽松去噪。值越小过滤越严格。")

    # 双阈值诊断参数
    parser.add_argument('--segment_threshold', type=float, default=0.5)
    parser.add_argument('--ratio_threshold', type=float, default=0.1)

    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载模型
    print(f"Loading model from {args.checkpoint}...")
    model = TF_MAE_Classifier(
        pretrained_path=None,
        num_classes=args.num_classes,
        signal_len=args.signal_len,
        cwt_scales=args.cwt_scales,
        patch_size_time=args.patch_size_time,
        patch_size_freq=args.patch_size_freq,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        decoder_embed_dim=128, decoder_depth=2, decoder_num_heads=4
    )
    
    state_dict = torch.load(args.checkpoint, map_location='cpu')
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    
    # 2. 扫描患者
    patient_folders = [f for f in glob.glob(os.path.join(args.data_root, "*")) if os.path.isdir(f)]
    print(f"Found {len(patient_folders)} patients.")
    print(f"Strategy: Adaptive IQR Filtering (Scale={args.iqr_scale})")
    
    results = []
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # 3. 逐个患者处理
    for patient_path in tqdm(patient_folders, desc="Processing Patients"):
        patient_id = os.path.basename(patient_path)
        pkl_files = glob.glob(os.path.join(patient_path, "*.pkl"))
        
        if not pkl_files: continue
            
        # 使用新的自适应 Dataset
        dataset = AdaptivePatientDataset(
            pkl_files, 
            signal_len=args.signal_len, 
            stride=args.stride, 
            iqr_scale=args.iqr_scale
        )
        
        stats = dataset.sqa_stats
        
        if len(dataset) == 0:
            results.append({
                "PatientID": patient_id, 
                "Diagnosis": "Insufficient Data",
                "Total_Raw": stats["total_raw"],
                "Valid_Final": 0,
                "Std_Threshold_High": round(stats["threshold_high"], 4)
            })
            continue
            
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        all_pos_probs = []
        
        with torch.no_grad():
            for x in loader:
                x = x.to(device)
                with autocast(device_type='cuda', dtype=amp_dtype):
                    logits = model(x)
                    probs = F.softmax(logits, dim=1)
                all_pos_probs.extend(probs[:, 1].float().cpu().numpy())

        all_pos_probs = np.array(all_pos_probs)

        # 4. 诊断逻辑
        total_valid = len(dataset)
        pos_segment_mask = all_pos_probs > args.segment_threshold
        num_pos_segments = np.sum(pos_segment_mask)
        pos_ratio = num_pos_segments / total_valid
        diagnosis = 1 if pos_ratio > args.ratio_threshold else 0
        
        results.append({
            "PatientID": patient_id,
            "Total_Raw": stats["total_raw"],
            "Valid_Final": total_valid,
            "Std_Threshold_Low": round(stats["threshold_low"], 4),
            "Std_Threshold_High": round(stats["threshold_high"], 4), # 记录自动计算出的阈值
            "Dropped_High_Std": stats["dropped_high"],
            "Pos_Ratio": round(pos_ratio, 4),
            "Diagnosis": diagnosis
        })
        
    # 5. 保存结果
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"\nInference done. Report saved to {args.output_csv}")

if __name__ == "__main__":
    main()