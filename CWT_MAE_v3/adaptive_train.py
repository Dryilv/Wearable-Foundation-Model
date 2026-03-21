import os
import glob
import pickle
import argparse
import contextlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# 导入项目中现有的模块
from model_finetune import TF_MAE_Classifier
from finetune import variable_channel_collate_fn_cls, move_batch_to_device, setup_distributed, cleanup_distributed, is_main_process, reduce_tensor

# ==========================================
# 1. 基础信号检查
# ==========================================
def check_basic_validity(signal):
    if len(signal) == 0: return False
    if not np.isfinite(signal).all(): return False
    if np.std(signal) < 1e-6: return False 
    return True

# ==========================================
# 2. 推理域自适应数据集
# ==========================================
class InferenceAdaptationDataset(Dataset):
    def __init__(self, data_root, signal_len=1000, stride=1000, iqr_scale=1.5):
        self.windows = []
        
        # 递归查找所有 .pkl 文件
        pkl_files = glob.glob(os.path.join(data_root, "**", "*.pkl"), recursive=True)
        if is_main_process():
            print(f"Found {len(pkl_files)} pkl files for adaptation in {data_root}.")
        
        raw_segments = []
        std_values = []
        
        # 使用 tqdm 显示扫描进度
        iterator = tqdm(pkl_files, desc="Scanning files", disable=not is_main_process())
        for fp in iterator:
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
                    segment = raw_data[:, start : start + signal_len]
                    
                    if check_basic_validity(segment):
                        std_val = np.mean(np.std(segment, axis=1))
                        raw_segments.append(segment)
                        std_values.append(std_val)
                        
            except Exception as e:
                pass

        if not std_values:
            if is_main_process():
                print("Warning: No valid segments found.")
            return

        std_array = np.array(std_values)
        
        q1 = np.percentile(std_array, 5)
        q3 = np.percentile(std_array, 95)
        iqr = q3 - q1
        
        lower_bound = max(0.0001, q1 - iqr_scale * iqr)
        upper_bound = q3 + iqr_scale * iqr
        
        for segment, std_val in zip(raw_segments, std_values):
            # 采用和 inference.py 中一致的鲁棒归一化逻辑
            median = np.median(segment, axis=1, keepdims=True)
            q25 = np.percentile(segment, 25, axis=1, keepdims=True)
            q75 = np.percentile(segment, 75, axis=1, keepdims=True)
            iqr_val = q75 - q25
            iqr_val = np.where(iqr_val < 1e-6, 1.0, iqr_val)
            
            segment_norm = (segment - median) / iqr_val
            segment_norm = np.clip(segment_norm, -20.0, 20.0)
            
            self.windows.append(segment_norm)
            
        if is_main_process():
            print(f"Extracted {len(self.windows)} valid segments for adaptation.")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        sig = self.windows[idx]
        sig_tensor = torch.from_numpy(sig) # (M, L)
        M = sig_tensor.shape[0]
        modality_ids = torch.zeros(M, dtype=torch.long)
        # 返回 dummy label -1
        return sig_tensor, modality_ids, torch.tensor(-1, dtype=torch.long)

# ==========================================
# 3. 核心：TENT 自适应配置与熵损失
# ==========================================
def configure_tent_model(model):
    """
    TENT: Test-time Entropy Minimization
    冻结除 Normalization 层之外的所有参数
    """
    model.train()
    model.requires_grad_(False) # 冻结所有
    
    params_to_update = []
    # 遍历所有模块，找到归一化层并解冻
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
            for param in m.parameters():
                params_to_update.append(param)
    
    return model, params_to_update

def entropy_loss(logits):
    """计算香农熵损失，鼓励模型做出更高置信度的预测"""
    probs = F.softmax(logits, dim=-1)
    # H(Y) = - sum_c p_c log(p_c)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
    return entropy.mean()

# ==========================================
# 4. 训练逻辑
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Test-Time Adaptation (TENT) using inference data")
    parser.add_argument('--data_root', type=str, required=True, help="推理数据所在目录")
    parser.add_argument('--checkpoint', type=str, required=True, help="微调后模型的权重路径")
    parser.add_argument('--output_dir', type=str, default="adapted_checkpoints", help="自适应权重保存路径")
    
    # 模型参数
    parser.add_argument('--signal_len', type=int, default=1000)
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--cwt_scales', type=int, default=64)
    parser.add_argument('--patch_size_time', type=int, default=25)
    parser.add_argument('--patch_size_freq', type=int, default=8)

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=1, help="自适应通常只需要1-3个epoch")
    parser.add_argument('--lr', type=float, default=1e-4, help="TENT推荐使用较小的学习率")
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--iqr_scale', type=float, default=1.5)
    parser.add_argument('--stride', type=int, default=1000)

    args = parser.parse_args()

    # 初始化分布式环境
    local_rank, rank, world_size = setup_distributed()
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
        
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Loading model from {args.checkpoint}...")

    # 1. 初始化并加载模型
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
        use_cot=True # 默认开启 CoT
    )
    
    state_dict = torch.load(args.checkpoint, map_location='cpu')
    # 处理 DDP 保存的 'module.' 前缀
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() if 'model' not in k}
    if 'model' in state_dict: # 如果是一个包含了多个字段的字典
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict['model'].items()}
        
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)

    # 2. 配置 TENT (解冻 Normalization 参数)
    model, params_to_update = configure_tent_model(model)
    if is_main_process():
        print(f"TENT mode: updating {len(params_to_update)} parameters (Normalization layers).")

    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # 3. 数据集与 DataLoader
    dataset = InferenceAdaptationDataset(
        data_root=args.data_root,
        signal_len=args.signal_len,
        stride=args.stride,
        iqr_scale=args.iqr_scale
    )
    
    if len(dataset) == 0:
        if is_main_process():
            print("No data available for adaptation. Exiting.")
        cleanup_distributed()
        return

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if dist.is_initialized() else None
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=variable_channel_collate_fn_cls,
        drop_last=True
    )

    # 4. 优化器与 Scaler
    optimizer = optim.Adam(params_to_update, lr=args.lr, weight_decay=0.0)
    scaler = GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    # 5. 适应性训练循环
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
            
        model.train()
        total_loss = 0.0
        count = 0
        
        iterator = tqdm(loader, desc=f"Adaptation Epoch {epoch+1}/{args.epochs}") if is_main_process() else loader
        
        for batch in iterator:
            x, modality_ids, _, channel_mask = move_batch_to_device(batch, device)
            
            optimizer.zero_grad(set_to_none=True)
            
            amp_ctx = autocast(device_type="cuda", dtype=amp_dtype) if use_amp else contextlib.nullcontext()
            with amp_ctx:
                logits = model(x, channel_mask=channel_mask)
                loss = entropy_loss(logits)
            
            if use_amp and amp_dtype == torch.float16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            if dist.is_initialized():
                reduced_loss = reduce_tensor(loss.detach())
                total_loss += reduced_loss.item()
            else:
                total_loss += loss.item()
            count += 1

            if is_main_process():
                iterator.set_postfix({'entropy_loss': total_loss / count})

        avg_loss = total_loss / count if count > 0 else 0
        if is_main_process():
            print(f"Epoch {epoch+1} finished. Avg Entropy Loss: {avg_loss:.4f}")
            
            # 保存权重
            save_path = os.path.join(args.output_dir, f"adapted_model_epoch_{epoch+1}.pth")
            real_model = model.module if hasattr(model, 'module') else model
            torch.save(real_model.state_dict(), save_path)
            print(f"Saved adapted model to {save_path}")

    cleanup_distributed()
    if is_main_process():
        print("Adaptation completed.")

if __name__ == "__main__":
    main()
