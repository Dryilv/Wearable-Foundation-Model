import os
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, classification_report
from dataset import PhysioSignalDataset, DataSplitter, fixed_channel_collate_fn
from patchtst_model import PatchTST_LinearProbeClassifier
from utils import init_distributed_mode, is_main_process


def reduce_tensor(tensor):
    if not dist.is_initialized():
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def gather_tensors(tensor, device):
    if not dist.is_initialized():
        return tensor
    local_size = torch.tensor([tensor.shape[0]], device=device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
    dist.all_gather(all_sizes, local_size)
    max_size = max([x.item() for x in all_sizes])
    size_diff = max_size - local_size.item()
    if size_diff > 0:
        padding = torch.zeros((size_diff, *tensor.shape[1:]), device=device, dtype=tensor.dtype)
        tensor_padded = torch.cat((tensor, padding))
    else:
        tensor_padded = tensor
    gathered_tensors = [torch.zeros_like(tensor_padded) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_tensors, tensor_padded)
    output = []
    for i, gathered_tensor in enumerate(gathered_tensors):
        output.append(gathered_tensor[:all_sizes[i].item()])
    return torch.cat(output)


def get_device(gpu_id):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def mixup_data(x, y, alpha=0.0, device='cuda'):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def train_one_epoch(model, loader, criterion, optimizer, device, use_amp=True, mixup_alpha=0.0):
    model.train()
    total_loss = 0.0
    count = 0
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    scaler = GradScaler(enabled=(use_amp and amp_dtype == torch.float16 and device.type == 'cuda'))
    for batch in loader:
        if len(batch) == 3:
            x, _, y = batch
        else:
            x, y = batch
        x = x.permute(0, 2, 1).to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()
        inputs, targets_a, targets_b, lam = mixup_data(x, y, alpha=mixup_alpha, device=device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp and device.type == 'cuda'):
            logits = model(inputs)
            if lam == 1.0:
                loss = criterion(logits, y)
            else:
                loss = lam * criterion(logits, targets_a) + (1 - lam) * criterion(logits, targets_b)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
        total_loss += reduce_tensor(loss.detach()).item() if dist.is_initialized() else loss.item()
        count += 1
    return total_loss / max(count, 1)


def validate(model, loader, criterion, device, num_classes, total_len, use_amp=True):
    model.eval()
    total_loss = 0.0
    count = 0
    local_labels = []
    local_probs = []
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                x, _, y = batch
            else:
                x, y = batch
            x = x.permute(0, 2, 1).to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp and device.type == 'cuda'):
                logits = model(x)
                loss = criterion(logits, y)
            total_loss += reduce_tensor(loss).item() if dist.is_initialized() else loss.item()
            count += 1
            probs = F.softmax(logits.float(), dim=1)
            local_labels.append(y)
            local_probs.append(probs)
    local_labels = torch.cat(local_labels)
    local_probs = torch.cat(local_probs)
    all_labels = gather_tensors(local_labels, device) if dist.is_initialized() else local_labels
    all_probs = gather_tensors(local_probs, device) if dist.is_initialized() else local_probs
    if not is_main_process():
        return 0, 0, 0, 0, 0, 0, "", 0.5
    if len(all_labels) > total_len:
        all_labels = all_labels[:total_len]
        all_probs = all_probs[:total_len]
    all_labels_np = all_labels.cpu().numpy()
    all_probs_np = all_probs.cpu().numpy()
    row_sums = all_probs_np.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    all_probs_np = all_probs_np / row_sums
    best_threshold = 0.5
    if num_classes == 2:
        y_scores = all_probs_np[:, 1]
        thresholds = np.arange(0.01, 1.00, 0.01)
        best_f1_search = -1.0
        for th in thresholds:
            preds_th = (y_scores >= th).astype(int)
            f1_th = f1_score(all_labels_np, preds_th, average='macro')
            if f1_th > best_f1_search:
                best_f1_search = f1_th
                best_threshold = th
        final_preds = (y_scores >= best_threshold).astype(int)
    else:
        final_preds = np.argmax(all_probs_np, axis=1)
    try:
        if num_classes == 2:
            auroc = roc_auc_score(all_labels_np, all_probs_np[:, 1])
        else:
            auroc = roc_auc_score(all_labels_np, all_probs_np, multi_class='ovr', average='macro')
    except Exception:
        auroc = 0.0
    final_acc = accuracy_score(all_labels_np, final_preds)
    final_f1 = f1_score(all_labels_np, final_preds, average='macro')
    precision = precision_score(all_labels_np, final_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels_np, final_preds, average='macro', zero_division=0)
    report_str = classification_report(all_labels_np, final_preds, digits=4)
    avg_loss = total_loss / max(count, 1)
    return avg_loss, final_acc, precision, recall, final_f1, auroc, report_str, best_threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='finetune_linear_probe_config.yaml', type=str)
    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    train_cfg = config['train']
    data_cfg = config['data']
    model_cfg = config['model']

    gpu_id, rank, world_size = init_distributed_mode()
    device = get_device(gpu_id)
    if is_main_process():
        os.makedirs(train_cfg['save_dir'], exist_ok=True)

    splitter = DataSplitter(
        index_file=data_cfg['index_path'],
        split_ratio=data_cfg.get('split_ratio', 0.1),
        seed=data_cfg.get('split_seed', 42)
    )
    train_indices, val_indices = splitter.get_split()
    train_dataset = PhysioSignalDataset(
        index_file=data_cfg['index_path'],
        indices=train_indices,
        signal_len=data_cfg['signal_len'],
        mode='train',
        data_ratio=model_cfg.get('data_ratio', 1.0),
        use_sliding_window=data_cfg.get('use_sliding_window', False),
        window_stride=data_cfg.get('window_stride', 500),
        expected_channels=model_cfg.get('in_channels', 5)
    )
    val_dataset = PhysioSignalDataset(
        index_file=data_cfg['index_path'],
        indices=val_indices,
        signal_len=data_cfg['signal_len'],
        mode='test',
        data_ratio=model_cfg.get('data_ratio', 1.0),
        use_sliding_window=data_cfg.get('use_sliding_window', False),
        window_stride=data_cfg.get('window_stride', 500),
        expected_channels=model_cfg.get('in_channels', 5)
    )
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg['batch_size'],
        sampler=train_sampler,
        num_workers=data_cfg.get('num_workers', 4),
        pin_memory=(device.type == 'cuda'),
        drop_last=False,
        collate_fn=fixed_channel_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg['batch_size'],
        sampler=val_sampler,
        num_workers=data_cfg.get('num_workers', 4),
        pin_memory=(device.type == 'cuda'),
        drop_last=False,
        collate_fn=fixed_channel_collate_fn
    )

    model = PatchTST_LinearProbeClassifier(
        seq_len=data_cfg['signal_len'],
        patch_len=model_cfg['patch_len'],
        stride=model_cfg['stride'],
        in_channels=model_cfg.get('in_channels', 5),
        d_model=model_cfg.get('embed_dim', 768),
        n_heads=model_cfg.get('num_heads', 12),
        e_layers=model_cfg.get('depth', 12),
        d_ff=model_cfg.get('d_ff', 3072),
        dropout=model_cfg.get('dropout', 0.1),
        use_revin=model_cfg.get('use_revin', True),
        num_classes=data_cfg['num_classes'],
        cls_dropout=model_cfg.get('cls_dropout', 0.0),
        pretrained_path=model_cfg.get('pretrained_path')
    ).to(device)
    model.freeze_backbone()
    model = DDP(model, device_ids=[gpu_id], output_device=gpu_id, find_unused_parameters=False) if device.type == 'cuda' else model

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=float(train_cfg['base_lr']), weight_decay=float(train_cfg['weight_decay']))
    criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg.get('label_smoothing', 0.0))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg['epochs'], eta_min=float(train_cfg['min_lr']))
    best_metric = -1.0
    total_epochs = int(train_cfg['epochs'])

    for epoch in range(total_epochs):
        train_sampler.set_epoch(epoch)
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            use_amp=train_cfg.get('use_amp', True),
            mixup_alpha=train_cfg.get('mixup_alpha', 0.0)
        )
        scheduler.step()
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc, val_report, best_th = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=data_cfg['num_classes'],
            total_len=len(val_dataset),
            use_amp=train_cfg.get('use_amp', True)
        )
        if is_main_process():
            print(f"Epoch {epoch+1}/{total_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val   Loss: {val_loss:.4f}")
            if data_cfg['num_classes'] == 2:
                print(f"Applied Threshold: {best_th:.2f}")
            print(f"Accuracy: {val_acc:.4f} | Precision: {val_prec:.4f} | Recall: {val_rec:.4f} | Macro-F1: {val_f1:.4f} | AUROC: {val_auc:.4f}")
            print(val_report)
        metric_to_track = val_auc if data_cfg['num_classes'] == 2 else val_f1
        if is_main_process() and metric_to_track > best_metric:
            best_metric = metric_to_track
            save_obj = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save(save_obj, os.path.join(train_cfg['save_dir'], 'best_linear_probe.pth'))

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
