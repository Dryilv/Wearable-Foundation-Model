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

# 导入 v1 版本的模型定义
from model_finetune import TF_MAE_Classifier

# ==========================================
# 1. 基础信号检查 (保持不变)
# ==========================================
def check_basic_validity(signal):
    if len(signal) == 0: return False
    if not np.isfinite(signal).all(): return False
    if np.std(signal) < 1e-6: return False 
    return True

# ==========================================
# 2. 自适应推理数据集 (保持不变)
# ==========================================
class AdaptivePatientDataset(Dataset):
    def __init__(self, file_paths, signal_len=3000, stride=1500, iqr_scale=1.5):
        self.windows = []
        self.sqa_stats = {
            "total_raw": 0, 
            "valid_final": 0, 
            "threshold_low": 0.0, 
            "threshold_high": 0.0,
            "dropped_low": 0,
            "dropped_high": 0
        }
        
        raw_segments = []
        std_values = []
        
        for fp in file_paths:
            try:
                with open(fp, 'rb') as f:
                    content = pickle.load(f)
                    # 兼容不同结构的 content
                    if isinstance(content, dict) and 'data' in content:
                        raw_data = content['data']
                    else:
                        raw_data = content # 假设直接是数据
                        
                    if isinstance(raw_data, list):
                        raw_data = np.array(raw_data)

                    if raw_data.ndim == 1:
                        raw_data = raw_data[np.newaxis, :] # (1, L)
                    
                    raw_data = raw_data.astype(np.float32) # (M, L_raw)
                
                # 假设我们只处理第一个通道，或者需要修改这里以支持多通道推理逻辑
                # 这里为了兼容 v1 的多通道特性，我们可能需要调整逻辑
                # 但原 inference.py 似乎是针对单通道或已经展平的数据
                # 如果是多通道模型，输入应该是 (M, L)
                
                # 这里假设 raw_data 是 (M, L_raw)
                M, n_samples = raw_data.shape
                if n_samples < signal_len: continue
                
                for start in range(0, n_samples - signal_len + 1, stride):
                    segment = raw_data[:, start : start + signal_len] # (M, L)
                    
                    if check_basic_validity(segment):
                        # 对每个通道计算 std，这里简单取平均或最大作为过滤依据
                        std_val = np.mean(np.std(segment, axis=1))
                        
                        raw_segments.append(segment)
                        std_values.append(std_val)
                        self.sqa_stats["total_raw"] += 1
                        
            except Exception as e:
                print(f"Error reading {fp}: {e}")

        if not std_values:
            return

        std_array = np.array(std_values)
        q1 = np.percentile(std_array, 25)
        q3 = np.percentile(std_array, 75)
        iqr = q3 - q1
        
        lower_bound = max(0.001, q1 - iqr_scale * iqr)
        upper_bound = q3 + iqr_scale * iqr
        
        self.sqa_stats["threshold_low"] = lower_bound
        self.sqa_stats["threshold_high"] = upper_bound
        
        for segment, std_val in zip(raw_segments, std_values):
            if std_val < lower_bound:
                self.sqa_stats["dropped_low"] += 1
                continue
            if std_val > upper_bound:
                self.sqa_stats["dropped_high"] += 1
                continue
            
            # Robust Normalization per channel
            # segment: (M, L)
            median = np.median(segment, axis=1, keepdims=True)
            q25 = np.percentile(segment, 25, axis=1, keepdims=True)
            q75 = np.percentile(segment, 75, axis=1, keepdims=True)
            iqr_val = q75 - q25
            iqr_val = np.where(iqr_val < 1e-6, 1.0, iqr_val)
            
            segment_norm = (segment - median) / iqr_val
            segment_norm = np.clip(segment_norm, -20.0, 20.0)
            
            self.windows.append(segment_norm)
            self.sqa_stats["valid_final"] += 1

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        sig = self.windows[idx]
        return torch.from_numpy(sig) # (M, L)

# ==========================================
# 3. 主推理逻辑 (多分类修改版)
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_csv', type=str, default="inference_report_v1.csv")
    
    # 模型参数
    parser.add_argument('--signal_len', type=int, default=3000) # v1 默认 3000
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--num_classes', type=int, default=2, help="分类数量")
    parser.add_argument('--cwt_scales', type=int, default=64)
    parser.add_argument('--patch_size_time', type=int, default=50)
    parser.add_argument('--patch_size_freq', type=int, default=4)
    parser.add_argument('--mlp_rank_ratio', type=float, default=0.5)

    # 推理参数
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--stride', type=int, default=1500)
    
    # 自适应过滤参数
    parser.add_argument('--iqr_scale', type=float, default=1.5)

    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载模型
    print(f"Loading model from {args.checkpoint} (Num Classes: {args.num_classes})...")
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
        mlp_rank_ratio=args.mlp_rank_ratio,
        use_cot=True # v1 默认开启 CoT
    )
    
    state_dict = torch.load(args.checkpoint, map_location='cpu')
    # 处理 DDP 保存的 'module.' 前缀
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    
    # 2. 扫描患者
    patient_folders = [f for f in glob.glob(os.path.join(args.data_root, "*")) if os.path.isdir(f)]
    print(f"Found {len(patient_folders)} patients.")
    
    results = []
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # 3. 逐个患者处理
    for patient_path in tqdm(patient_folders, desc="Processing Patients"):
        patient_id = os.path.basename(patient_path)
        pkl_files = glob.glob(os.path.join(patient_path, "*.pkl"))
        
        if not pkl_files: continue
            
        dataset = AdaptivePatientDataset(
            pkl_files, 
            signal_len=args.signal_len, 
            stride=args.stride, 
            iqr_scale=args.iqr_scale
        )
        
        stats = dataset.sqa_stats
        
        if len(dataset) == 0:
            # 数据不足时的记录
            res = {
                "PatientID": patient_id, 
                "Predicted_Class": -1, # -1 表示无效
                "Confidence": 0.0,
                "Total_Raw": stats["total_raw"],
                "Valid_Final": 0
            }
            # 填充空的概率列
            for c in range(args.num_classes):
                res[f"Prob_Class_{c}"] = 0.0
            results.append(res)
            continue
            
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        all_probs_list = []
        
        with torch.no_grad():
            for x in loader:
                x = x.to(device)
                with autocast(device_type='cuda', dtype=amp_dtype):
                    logits = model(x)
                    # 强制 float32 保证精度
                    probs = F.softmax(logits.float(), dim=1)
                all_probs_list.append(probs.cpu().numpy())

        # 拼接所有 batch: Shape [Total_Segments, Num_Classes]
        all_probs = np.vstack(all_probs_list)

        # =========================================================
        # 4. 多分类诊断逻辑 (Soft Voting)
        # =========================================================
        
        # 方法 A: Soft Voting (推荐)
        # 计算该患者所有片段的平均概率分布
        mean_probs = np.mean(all_probs, axis=0) # Shape [Num_Classes]
        
        # 最终预测类别为平均概率最大的那个
        final_pred_class = np.argmax(mean_probs)
        confidence = mean_probs[final_pred_class]
        
        # 构建结果字典
        res_dict = {
            "PatientID": patient_id,
            "Total_Raw": stats["total_raw"],
            "Valid_Final": len(dataset),
            "Std_Threshold_High": round(stats["threshold_high"], 4),
            "Predicted_Class": final_pred_class,
            "Confidence": round(confidence, 4)
        }
        
        # 记录每个类别的平均概率 (反映了该类别的风险程度)
        for c in range(args.num_classes):
            res_dict[f"Prob_Class_{c}"] = round(mean_probs[c], 4)
            
        results.append(res_dict)
        
    # 5. 保存结果
    df = pd.DataFrame(results)
    
    # 调整列顺序，把预测结果放前面
    cols = ["PatientID", "Predicted_Class", "Confidence"] + \
           [f"Prob_Class_{c}" for c in range(args.num_classes)] + \
           ["Total_Raw", "Valid_Final", "Std_Threshold_High"]
    
    # 确保只取存在的列
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    
    df.to_csv(args.output_csv, index=False)
    print(f"\nInference done. Report saved to {args.output_csv}")

if __name__ == "__main__":
    main()
