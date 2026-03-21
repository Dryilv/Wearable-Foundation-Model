import os
import glob
import pickle
import argparse
import contextlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast 

# 导入 v1 版本的模型定义
from model_finetune import TF_MAE_Classifier
from finetune import variable_channel_collate_fn_cls, move_batch_to_device

# ==========================================
# 1. 基础信号检查 (保持不变)
# ==========================================
def check_basic_validity(signal):
    if len(signal) == 0: return False
    if not np.isfinite(signal).all(): return False
    if np.std(signal) < 1e-6: return False 
    return True

def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0

def get_world_size():
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1

def is_main_process():
    return get_rank() == 0

def setup_distributed():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size <= 1:
        return False, rank, world_size, local_rank

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    else:
        backend = "gloo"

    dist.init_process_group(
        backend=backend,
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )
    dist.barrier()
    return True, rank, world_size, local_rank

def cleanup_distributed():
    if is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()

def all_gather_pyobj(data, device):
    if not is_dist_avail_and_initialized():
        return [data]

    world_size = get_world_size()
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    local_size = torch.tensor([tensor.numel()], device=device, dtype=torch.long)

    size_list = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    max_size = int(torch.stack(size_list).max().item())

    if tensor.numel() < max_size:
        pad = torch.zeros(max_size - tensor.numel(), dtype=torch.uint8, device=device)
        tensor = torch.cat([tensor, pad], dim=0)

    tensor_list = [torch.empty(max_size, dtype=torch.uint8, device=device) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for t, sz in zip(tensor_list, size_list):
        n = int(sz.item())
        bytes_ = t[:n].cpu().numpy().tobytes()
        data_list.append(pickle.loads(bytes_))
    return data_list

def build_balanced_shards(patient_folders, world_size):
    patient_items = []
    for patient_path in patient_folders:
        pkl_files = sorted(glob.glob(os.path.join(patient_path, "*.pkl")))
        weight = max(1, len(pkl_files))
        patient_items.append((patient_path, pkl_files, weight))

    patient_items.sort(key=lambda x: x[2], reverse=True)
    shards = [[] for _ in range(world_size)]
    loads = [0 for _ in range(world_size)]
    for item in patient_items:
        target = min(range(world_size), key=lambda i: loads[i])
        shards[target].append(item)
        loads[target] += item[2]
    return shards, loads

# ==========================================
# 2. 自适应推理数据集 (Lazy Loading 优化版)
# ==========================================
class AdaptivePatientDataset(Dataset):
    def __init__(self, file_paths, signal_len=3000, stride=1500, iqr_scale=1.5):
        self.signal_len = signal_len
        self.stride = stride
        self.iqr_scale = iqr_scale
        self.windows_meta = [] # 存储元数据：(file_path, start_idx, std_val)
        
        self.sqa_stats = {
            "total_raw": 0, 
            "valid_final": 0, 
            "threshold_low": 0.0, 
            "threshold_high": 0.0,
            "dropped_low": 0,
            "dropped_high": 0
        }
        
        std_values = []
        
        # 第一阶段：轻量级扫描，仅记录有效切片的位置和标准差
        for fp in file_paths:
            try:
                with open(fp, 'rb') as f:
                    content = pickle.load(f)
                    if isinstance(content, dict) and 'data' in content:
                        raw_data = content['data']
                    else:
                        raw_data = content
                        
                    if isinstance(raw_data, list):
                        raw_data = np.array(raw_data)

                    if raw_data.ndim == 1:
                        raw_data = raw_data[np.newaxis, :]
                    
                    raw_data = raw_data.astype(np.float32)
                    raw_data = np.nan_to_num(raw_data, nan=0.0, posinf=0.0, neginf=0.0)
                
                M, n_samples = raw_data.shape
                
                if n_samples < signal_len:
                    pad_len = signal_len - n_samples
                    raw_data = np.pad(raw_data, ((0, 0), (0, pad_len)), mode='edge')
                    n_samples = signal_len
                
                for start in range(0, n_samples - signal_len + 1, stride):
                    segment = raw_data[:, start : start + signal_len] # (M, L)
                    
                    if check_basic_validity(segment):
                        std_val = np.mean(np.std(segment, axis=1))
                        self.windows_meta.append({
                            'file_path': fp,
                            'start': start,
                            'std': std_val
                        })
                        std_values.append(std_val)
                        self.sqa_stats["total_raw"] += 1
                        
            except Exception as e:
                print(f"Error scanning {fp}: {e}")

        if not std_values:
            return

        std_array = np.array(std_values)
        
        q1 = np.percentile(std_array, 5)
        q3 = np.percentile(std_array, 95)
        iqr = q3 - q1
        
        self.lower_bound = max(0.0001, q1 - iqr_scale * iqr)
        self.upper_bound = q3 + iqr_scale * iqr
        
        self.sqa_stats["threshold_low"] = self.lower_bound
        self.sqa_stats["threshold_high"] = self.upper_bound
        
        # 过滤元数据
        valid_meta = []
        for meta in self.windows_meta:
            std_val = meta['std']
            if std_val < self.lower_bound or std_val > self.upper_bound:
                if std_val < self.lower_bound:
                    self.sqa_stats["dropped_low"] += 1
                else:
                    self.sqa_stats["dropped_high"] += 1
                # 根据之前的逻辑，这里不丢弃，保留让模型判断，或者你可以选择 continue
                # valid_meta.append(meta) 
            # 即使超界我们也保留它，记录在 valid_meta 中
            valid_meta.append(meta)
            self.sqa_stats["valid_final"] += 1
            
        self.windows_meta = valid_meta

    def __len__(self):
        return len(self.windows_meta)

    def __getitem__(self, idx):
        meta = self.windows_meta[idx]
        
        # 实时读取和切片
        with open(meta['file_path'], 'rb') as f:
            content = pickle.load(f)
            if isinstance(content, dict) and 'data' in content:
                raw_data = content['data']
            else:
                raw_data = content
                
            if isinstance(raw_data, list):
                raw_data = np.array(raw_data)

            if raw_data.ndim == 1:
                raw_data = raw_data[np.newaxis, :]
            
            raw_data = raw_data.astype(np.float32)
            raw_data = np.nan_to_num(raw_data, nan=0.0, posinf=0.0, neginf=0.0)
            
        M, n_samples = raw_data.shape
        if n_samples < self.signal_len:
            pad_len = self.signal_len - n_samples
            raw_data = np.pad(raw_data, ((0, 0), (0, pad_len)), mode='edge')
            
        start = meta['start']
        segment = raw_data[:, start : start + self.signal_len]
        
        # 实时归一化 (使用与 dataset.py 一致的 Z-Score，抛弃 IQR)
        std_vals = np.std(segment, axis=1, keepdims=True)
        mean_vals = np.mean(segment, axis=1, keepdims=True)
        
        segment_norm = (segment - mean_vals) / (std_vals + 1e-5)
        segment_norm = np.clip(segment_norm, -10.0, 10.0)

        sig_tensor = torch.from_numpy(segment_norm) # (M, L)
        modality_ids = torch.zeros(M, dtype=torch.long)
        
        return sig_tensor, modality_ids, torch.tensor(-1, dtype=torch.long)

# ==========================================
# 3. 主推理逻辑 (多分类修改版)
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_csv', type=str, default="inference_report_v1.csv")
    
    # 模型参数
    parser.add_argument('--signal_len', type=int, default=1000) # v1 默认 3000
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--num_classes', type=int, default=2, help="分类数量")
    parser.add_argument('--cwt_scales', type=int, default=64)
    parser.add_argument('--patch_size_time', type=int, default=25)
    parser.add_argument('--patch_size_freq', type=int, default=8)

    # 推理参数
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--stride', type=int, default=1000)
    parser.add_argument('--confidence_threshold', type=float, default=0.75, help="单片段置信度阈值，低于此值归为 Class 0")
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--prefetch_factor', type=int, default=2)
    
    # 高级聚合与抗噪参数
    parser.add_argument('--top_p_ratio', type=float, default=0.15, help="Top-P% 均值聚合比例")
    parser.add_argument('--fallback_top_k', type=int, default=5, help="自适应阈值降级时保留的 Top-K 片段数")
    parser.add_argument('--trim_ratio', type=float, default=0.05, help="截尾均值剔除极值的比例 (单侧)")
    
    # 自适应过滤参数
    parser.add_argument('--iqr_scale', type=float, default=1.5)

    args = parser.parse_args()

    distributed, rank, world_size, local_rank = setup_distributed()
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank if distributed else 0)
    else:
        device = torch.device("cpu")
    
    # 1. 加载模型
    if is_main_process():
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
        use_cot=True # v1 默认开启 CoT
    )
    
    state_dict = torch.load(args.checkpoint, map_location='cpu')
    # 处理 DDP 保存的 'module.' 前缀
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    
    # 2. 扫描患者
    patient_folders = sorted([f for f in glob.glob(os.path.join(args.data_root, "*")) if os.path.isdir(f)])
    if is_main_process():
        print(f"Found {len(patient_folders)} patients.")

    shards, shard_loads = build_balanced_shards(patient_folders, world_size if distributed else 1)
    rank_items = shards[rank] if distributed else shards[0]
    local_load = sum(item[2] for item in rank_items)

    if distributed:
        load_tensor = torch.tensor([local_load], device=device, dtype=torch.long)
        all_loads = [torch.zeros_like(load_tensor) for _ in range(world_size)]
        dist.all_gather(all_loads, load_tensor)
        if is_main_process():
            loads_text = ", ".join([f"rank{i}={int(t.item())}" for i, t in enumerate(all_loads)])
            print(f"Balanced workload by pkl count: {loads_text}")

    results_local = []
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    patient_pbar = None
    batch_pbar = None
    if is_main_process():
        patient_pbar = tqdm(total=len(rank_items), desc=f"Rank{rank} Patients", dynamic_ncols=True, smoothing=0.05)
        batch_pbar = tqdm(total=0, desc=f"Rank{rank} Batches", dynamic_ncols=True, smoothing=0.05, leave=False)

    for patient_path, pkl_files, _ in rank_items:
        patient_id = os.path.basename(patient_path)
        
        if not pkl_files:
            if patient_pbar is not None:
                patient_pbar.update(1)
            continue
            
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
                "Hard_Pred": -1,
                "Hard_Conf": 0.0,
                "Filtered_Rate": 0.0,
                "Confidence_Threshold": args.confidence_threshold,
                "Strict_Pred": -1,
                "Strict_Status": "No_Data",
                "Total_Raw": stats["total_raw"],
                "Valid_Final": 0,
                "Std_Threshold_High": round(stats["threshold_high"], 4)
            }
            # 填充空的概率列
            for c in range(args.num_classes):
                res[f"Prob_Mean_{c}"] = 0.0
                res[f"Prob_Max_{c}"] = 0.0
                res[f"Count_Hard_{c}"] = 0
            results_local.append(res)
            if patient_pbar is not None:
                patient_pbar.update(1)
            continue
            
        loader_kwargs = {
            "dataset": dataset,
            "batch_size": args.batch_size,
            "shuffle": False,
            "num_workers": args.num_workers,
            "pin_memory": (device.type == "cuda"),
            "collate_fn": variable_channel_collate_fn_cls # --- 关键修复：加入变长通道的 collate_fn ---
        }
        if args.num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = max(1, args.prefetch_factor)

        loader = DataLoader(
            **loader_kwargs
        )

        if batch_pbar is not None:
            batch_pbar.total += len(loader)
            batch_pbar.refresh()
        
        all_probs_list = []
        
        with torch.no_grad():
            for batch in loader:
                # --- 关键修复：使用与 finetune 一致的 batch 提取逻辑 ---
                x, modality_ids, y, channel_mask = move_batch_to_device(batch, device)

                amp_ctx = autocast(device_type="cuda", dtype=amp_dtype) if use_amp else contextlib.nullcontext()
                with amp_ctx:
                    # 推理时也要传入 channel_mask
                    logits = model(x, channel_mask=channel_mask)
                probs = F.softmax(logits.float(), dim=1)
                all_probs_list.append(probs.cpu().numpy())
                if batch_pbar is not None:
                    batch_pbar.update(1)

        # 拼接所有 batch: Shape [Total_Segments, Num_Classes]
        all_probs = np.vstack(all_probs_list)

        # =========================================================
        # 4. 多维度诊断逻辑 (Advanced Enhanced)
        # =========================================================
        
        # 1. 获取每个切片的最大概率和预测类别
        per_segment_max_probs = np.max(all_probs, axis=1)
        per_segment_preds = np.argmax(all_probs, axis=1)
        
        # 2. 应用置信度阈值过滤
        valid_mask = per_segment_max_probs >= args.confidence_threshold
        
        # 统计被过滤的切片比例
        filtered_count = np.sum(~valid_mask)
        total_segments = len(per_segment_preds)
        filtered_rate = filtered_count / max(1, total_segments)
        
        # --- 3. 动态/自适应置信度阈值降级 (Adaptive Thresholding) ---
        fallback_used = False
        if np.sum(valid_mask) == 0 and total_segments > 0:
            fallback_used = True
            # 取全局置信度最高的 Top-K
            k = min(args.fallback_top_k, total_segments)
            top_k_indices = np.argsort(per_segment_max_probs)[-k:]
            valid_mask = np.zeros(total_segments, dtype=bool)
            valid_mask[top_k_indices] = True
            
        # 提取有效片段
        valid_probs = all_probs[valid_mask]
        valid_preds = per_segment_preds[valid_mask]
        valid_confidences = per_segment_max_probs[valid_mask]
        
        if len(valid_probs) == 0:
            # 极端情况：没有任何数据 (可能本身 total_segments=0)
            soft_pred_class = -1
            soft_conf = 0.0
            hard_pred_class = -1
            hard_conf = 0.0
            mean_probs = np.zeros(args.num_classes)
            max_probs = np.zeros(args.num_classes)
            counts = np.zeros(args.num_classes, dtype=int)
            strict_pred_class = -1
            strict_status = "No_Data"
        else:
            num_valid = len(valid_probs)
            
            # --- 4. 引入截尾均值 (Trimmed Mean) 抵抗极端假阳性 ---
            # 如果有效片段数量足够多 (例如 >10)，则在每个类别的维度上剔除极值
            trim_probs = valid_probs.copy()
            if num_valid > 10 and args.trim_ratio > 0:
                trim_count = int(num_valid * args.trim_ratio)
                if trim_count > 0:
                    for c in range(args.num_classes):
                        c_probs = trim_probs[:, c]
                        sorted_indices = np.argsort(c_probs)
                        # 将极高和极低值替换为有效边界内的中值，或者直接把它们的权重设为0
                        # 简单起见，我们直接计算均值时排除它们
                        pass # 这里我们留到聚合时处理，为了和 Top-P 不冲突，或者直接修改 trim_probs
            
            # 改进的截尾机制：不直接修改 trim_probs，而是在计算 mean 时进行过滤
            
            mean_probs = np.zeros(args.num_classes)
            for c in range(args.num_classes):
                c_probs = valid_probs[:, c]
                c_confs = valid_confidences
                
                # --- 4. 引入截尾均值 (Trimmed Mean) 抵抗极端假阳性 ---
                # 先进行截尾过滤
                if num_valid > 10 and args.trim_ratio > 0:
                    trim_count = int(num_valid * args.trim_ratio)
                    if trim_count > 0:
                        # 找到排名的索引
                        sorted_indices = np.argsort(c_probs)
                        # 剔除最低和最高 trim_count 个样本
                        valid_indices_after_trim = sorted_indices[trim_count:-trim_count]
                        c_probs = c_probs[valid_indices_after_trim]
                        c_confs = c_confs[valid_indices_after_trim]
                
                # 更新当前有效数量
                current_valid_count = len(c_probs)
                
                # --- 1. Top-P% 均值聚合 (Top-P Mean Pooling) ---
                # 对于每个类别，只取该类别概率排名前 P% 的片段
                p_count = max(1, int(current_valid_count * args.top_p_ratio))
                top_p_indices = np.argsort(c_probs)[-p_count:]
                selected_probs = c_probs[top_p_indices]
                
                if fallback_used:
                     # 如果 fallback，则 confs 的顺序可能与 probs 不完全一致，但由于 fallback 通常片段很少，这里安全起见直接使用 c_confs
                     selected_confs = c_confs[top_p_indices]
                else:
                     selected_confs = c_confs[top_p_indices]
                    
                # --- 2. 置信度加权软投票 (Confidence-Weighted Soft Voting) ---
                # 使用每个片段的最大概率值（置信度）作为权重
                weights = selected_confs
                # 避免全 0 权重
                if np.sum(weights) < 1e-6:
                    weights = np.ones_like(weights)
                weights = weights / np.sum(weights)
                
                # 计算加权 Top-P 均值
                mean_probs[c] = np.sum(selected_probs * weights)

            # 重新归一化 (因为 Top-P 导致和不为 1)
            if np.sum(mean_probs) > 0:
                mean_probs = mean_probs / np.sum(mean_probs)
                
            soft_pred_class = np.argmax(mean_probs)
            soft_conf = mean_probs[soft_pred_class]

            # --- B. Hard Voting (仅在有效片段上投票) ---
            counts = np.bincount(valid_preds, minlength=args.num_classes)
            hard_pred_class = np.argmax(counts)
            hard_conf = counts[hard_pred_class] / num_valid

            # --- C. Max Probability (仅在有效片段上提取) ---
            # 为了抵抗极值，我们可以取 95 分位数而不是严格的 max
            if num_valid > 10 and args.trim_ratio > 0:
                max_probs = np.percentile(valid_probs, 100 - (args.trim_ratio * 100), axis=0)
            else:
                max_probs = np.max(valid_probs, axis=0)

            # --- 5. 针对多分类的“保守降级”逻辑优化 ---
            if soft_pred_class == hard_pred_class:
                strict_pred_class = soft_pred_class
                strict_status = "Fallback_Low_Conf" if fallback_used else "High_Conf"
            else:
                # 发生分歧
                if soft_pred_class > 0 and hard_pred_class > 0:
                    # 都认为是疾病，只是具体类别有分歧
                    # 取两者中概率更高的类别
                    if mean_probs[soft_pred_class] > (counts[hard_pred_class] / num_valid):
                        strict_pred_class = soft_pred_class
                    else:
                        strict_pred_class = hard_pred_class
                    strict_status = "Disease_Ambiguous"
                else:
                    # 分歧发生在 0 (正常) 和 >0 (疾病) 之间
                    strict_pred_class = 0 # 保守回退到 0
                    strict_status = "Normal_Ambiguous"

        # 默认使用 Soft Voting 作为主要结果
        final_pred_class = soft_pred_class
        confidence = soft_conf
        
        # 构建结果字典
        res_dict = {
            "PatientID": patient_id,
            "Total_Raw": stats["total_raw"],
            "Valid_Final": len(dataset),
            "Std_Threshold_High": round(stats["threshold_high"], 4),
            "Predicted_Class": final_pred_class,
            "Confidence": round(confidence, 4),
            "Hard_Pred": hard_pred_class,
            "Hard_Conf": round(hard_conf, 4),
            "Filtered_Rate": round(filtered_rate, 4),
            "Confidence_Threshold": args.confidence_threshold,
            "Strict_Pred": strict_pred_class,
            "Strict_Status": strict_status
        }
        
        # 记录每个类别的详细统计 (Mean, Max, Count)
        for c in range(args.num_classes):
            res_dict[f"Prob_Mean_{c}"] = round(mean_probs[c], 4)
            res_dict[f"Prob_Max_{c}"] = round(max_probs[c], 4)
            res_dict[f"Count_Hard_{c}"] = counts[c]
            
        results_local.append(res_dict)
        if patient_pbar is not None:
            patient_pbar.update(1)
            patient_pbar.set_postfix_str(f"seg={len(dataset)}|soft={soft_pred_class}|hard={hard_pred_class}")

    if patient_pbar is not None:
        patient_pbar.close()
    if batch_pbar is not None:
        batch_pbar.close()
        
    gathered = all_gather_pyobj(results_local, device=device if device.type == "cuda" else torch.device("cpu"))
    if is_main_process():
        results = []
        for part in gathered:
            results.extend(part)

        df = pd.DataFrame(results)

        cols = ["PatientID", "Predicted_Class", "Confidence", "Strict_Pred", "Strict_Status", "Hard_Pred", "Hard_Conf", "Filtered_Rate", "Confidence_Threshold"] + \
               [f"Prob_Mean_{c}" for c in range(args.num_classes)] + \
               [f"Prob_Max_{c}" for c in range(args.num_classes)] + \
               [f"Count_Hard_{c}" for c in range(args.num_classes)] + \
               ["Total_Raw", "Valid_Final", "Std_Threshold_High"]

        cols = [c for c in cols if c in df.columns]
        df = df[cols]

        df.to_csv(args.output_csv, index=False)
        print(f"\nInference done. Report saved to {args.output_csv}")

    cleanup_distributed()

if __name__ == "__main__":
    main()
