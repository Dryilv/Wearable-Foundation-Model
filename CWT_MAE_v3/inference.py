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

    # 推理参数
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--stride', type=int, default=1500)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--prefetch_factor', type=int, default=2)
    
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
                "Total_Raw": stats["total_raw"],
                "Valid_Final": 0
            }
            # 填充空的概率列
            for c in range(args.num_classes):
                res[f"Prob_Class_{c}"] = 0.0
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
                x = batch
                x = x.to(device)

                amp_ctx = autocast(device_type="cuda", dtype=amp_dtype) if use_amp else contextlib.nullcontext()
                with amp_ctx:
                    logits = model(x)
                probs = F.softmax(logits.float(), dim=1)
                all_probs_list.append(probs.cpu().numpy())
                if batch_pbar is not None:
                    batch_pbar.update(1)

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
            
        results_local.append(res_dict)
        if patient_pbar is not None:
            patient_pbar.update(1)
            patient_pbar.set_postfix_str(f"segments={len(dataset)}")

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

        cols = ["PatientID", "Predicted_Class", "Confidence"] + \
               [f"Prob_Class_{c}" for c in range(args.num_classes)] + \
               ["Total_Raw", "Valid_Final", "Std_Threshold_High"]

        cols = [c for c in cols if c in df.columns]
        df = df[cols]

        df.to_csv(args.output_csv, index=False)
        print(f"\nInference done. Report saved to {args.output_csv}")

    cleanup_distributed()

if __name__ == "__main__":
    main()
