import os
import json
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

from dataset_finetune import DownstreamClassificationDataset
from model_finetune import TF_MAE_Classifier
from utils import get_layer_wise_lr
from finetune import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    unwrap_model,
    set_seed,
    setup_logger,
    save_checkpoint,
    load_checkpoint,
    reduce_tensor,
    gather_tensors,
    variable_channel_collate_fn_cls,
    move_batch_to_device,
    FocalLoss
)


def merge_model_cfg(base_cfg, override_cfg):
    merged = dict(base_cfg)
    if override_cfg:
        merged.update(override_cfg)
    return merged


def build_classifier(model_cfg, data_cfg, device):
    model = TF_MAE_Classifier(
        pretrained_path=model_cfg.get('pretrained_path'),
        num_classes=data_cfg['num_classes'],
        signal_len=data_cfg['signal_len'],
        cwt_scales=model_cfg.get('cwt_scales', 64),
        patch_size_time=model_cfg.get('patch_size_time', 25),
        patch_size_freq=model_cfg.get('patch_size_freq', 8),
        embed_dim=model_cfg.get('embed_dim', 768),
        depth=model_cfg.get('depth', 12),
        num_heads=model_cfg.get('num_heads', 12),
        use_diff=model_cfg.get('use_diff', False),
        decoder_embed_dim=model_cfg.get('decoder_embed_dim', 512),
        decoder_depth=model_cfg.get('decoder_depth', 8),
        decoder_num_heads=model_cfg.get('decoder_num_heads', 16),
        use_cot=model_cfg.get('use_cot', True),
        num_reasoning_tokens=model_cfg.get('num_reasoning_tokens', 16),
        use_arcface=model_cfg.get('use_arcface', False),
        arcface_s=model_cfg.get('arcface_s', 30.0),
        arcface_m=model_cfg.get('arcface_m', 0.50)
    )
    model.to(device)
    return model


def load_model_weights(model, ckpt_path, strict=True):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    unwrap_model(model).load_state_dict(state_dict, strict=strict)


def set_dataset_mode(dataset, mode):
    if isinstance(dataset, Subset):
        set_dataset_mode(dataset.dataset, mode)
        return
    if hasattr(dataset, 'mode'):
        dataset.mode = mode


def build_student_batch(x, channel_mask, student_channel_index):
    bsz, _, _ = x.shape
    selected = []
    for i in range(bsz):
        if channel_mask is None:
            idx = 0
        else:
            valid_idx = torch.nonzero(channel_mask[i], as_tuple=False).flatten()
            if valid_idx.numel() == 0:
                idx = 0
            elif 0 <= student_channel_index < channel_mask.shape[1] and bool(channel_mask[i, student_channel_index]):
                idx = int(student_channel_index)
            else:
                idx = int(valid_idx[0].item())
        selected.append(x[i, idx:idx + 1, :])
    student_x = torch.cat(selected, dim=0)
    student_mask = torch.ones((bsz, 1), dtype=torch.bool, device=x.device)
    return student_x, student_mask


def kd_kl_loss(student_logits, teacher_logits, temperature):
    t = float(temperature)
    student_log_prob = F.log_softmax(student_logits / t, dim=1)
    teacher_prob = F.softmax(teacher_logits / t, dim=1)
    return F.kl_div(student_log_prob, teacher_prob, reduction='batchmean') * (t * t)


def evaluate_student(student_model, loader, criterion, device, num_classes, total_len, student_channel_index, use_amp=True):
    student_model.eval()
    total_loss = 0.0
    count = 0
    local_labels = []
    local_probs = []
    amp_dtype = torch.bfloat16 if (device.type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16
    amp_enabled = use_amp and device.type == 'cuda'
    iterator = tqdm(loader, desc="Validating KD Student") if is_main_process() else loader
    with torch.no_grad():
        for batch in iterator:
            x, modality_ids, y, channel_mask = move_batch_to_device(batch, device)
            student_x, student_mask = build_student_batch(x, channel_mask, student_channel_index)
            real_student = unwrap_model(student_model)
            is_arcface = hasattr(real_student, 'use_arcface') and real_student.use_arcface
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                if is_arcface:
                    y_hard = y.argmax(dim=1) if y.dim() > 1 else y
                    logits = student_model(student_x, label=y_hard, channel_mask=student_mask)
                else:
                    logits = student_model(student_x, channel_mask=student_mask)
                loss = criterion(logits, y)
            if dist.is_initialized():
                total_loss += reduce_tensor(loss.detach()).item()
            else:
                total_loss += loss.item()
            count += 1
            y_hard_eval = y.argmax(dim=1) if y.dim() > 1 else y
            local_labels.append(y_hard_eval)
            local_probs.append(F.softmax(logits.float(), dim=1))
    if count == 0 or len(local_labels) == 0:
        return 0.0, 0.0, 0.0, 0.0
    local_labels = torch.cat(local_labels)
    local_probs = torch.cat(local_probs)
    if dist.is_initialized():
        all_labels = gather_tensors(local_labels, device)
        all_probs = gather_tensors(local_probs, device)
    else:
        all_labels = local_labels
        all_probs = local_probs
    if len(all_labels) > total_len:
        all_labels = all_labels[:total_len]
        all_probs = all_probs[:total_len]
    y_true = all_labels.cpu().numpy()
    y_prob = all_probs.cpu().numpy()
    y_pred = np.argmax(y_prob, axis=1)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    try:
        if num_classes == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
    except Exception:
        auc = 0.0
    return total_loss / count, acc, f1, auc


def train_one_epoch_kd(student_model, teacher_model, loader, criterion, optimizer, device, epoch, kd_weight, cls_weight, temperature, student_channel_index, scaler=None, use_amp=True, grad_clip_norm=3.0, scheduler=None):
    student_model.train()
    teacher_model.eval()
    if hasattr(loader.sampler, 'set_epoch'):
        loader.sampler.set_epoch(epoch)
    total_loss = 0.0
    total_cls = 0.0
    total_kd = 0.0
    count = 0
    amp_dtype = torch.bfloat16 if (device.type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16
    amp_enabled = use_amp and device.type == 'cuda'
    iterator = tqdm(loader, desc=f"Epoch {epoch + 1} KD Train") if is_main_process() else loader
    for batch in iterator:
        x, modality_ids, y, channel_mask = move_batch_to_device(batch, device)
        student_x, student_mask = build_student_batch(x, channel_mask, student_channel_index)
        optimizer.zero_grad(set_to_none=True)
        real_student = unwrap_model(student_model)
        student_arcface = hasattr(real_student, 'use_arcface') and real_student.use_arcface
        y_hard = y.argmax(dim=1) if y.dim() > 1 else y
        with torch.no_grad():
            teacher_logits = teacher_model(x, channel_mask=channel_mask)
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            if student_arcface:
                student_logits = student_model(student_x, label=y_hard, channel_mask=student_mask)
            else:
                student_logits = student_model(student_x, channel_mask=student_mask)
            cls_loss = criterion(student_logits, y)
            kd_loss = kd_kl_loss(student_logits, teacher_logits.detach(), temperature=temperature)
            loss = cls_weight * cls_loss + kd_weight * kd_loss
        if use_amp and amp_dtype == torch.float16:
            if scaler is None:
                raise RuntimeError("GradScaler is required for float16 AMP training.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if dist.is_initialized():
            total_loss += reduce_tensor(loss.detach()).item()
            total_cls += reduce_tensor(cls_loss.detach()).item()
            total_kd += reduce_tensor(kd_loss.detach()).item()
        else:
            total_loss += loss.item()
            total_cls += cls_loss.item()
            total_kd += kd_loss.item()
        count += 1
        if is_main_process():
            current_lr = optimizer.param_groups[0]['lr']
            iterator.set_postfix({'loss': total_loss / count, 'cls': total_cls / count, 'kd': total_kd / count, 'lr': f"{current_lr:.2e}"})
    if count == 0:
        return 0.0, 0.0, 0.0
    return total_loss / count, total_cls / count, total_kd / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='finetune_kd_config.yaml')
    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    train_cfg = config['train']
    data_cfg = config['data']
    model_cfg = config['model']
    kd_cfg = config.get('kd', {})
    seed = train_cfg.get('seed', 42)
    deterministic = train_cfg.get('deterministic', False)
    set_seed(seed, deterministic=deterministic)
    if torch.cuda.is_available():
        local_rank, rank, world_size = setup_distributed()
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank, rank, world_size = 0, 0, 1
        device = torch.device("cpu")
    os.makedirs(train_cfg['save_dir'], exist_ok=True)
    logger = setup_logger(train_cfg['save_dir'])
    if is_main_process():
        print(f"World Size: {world_size}, Device: {device}")
    with open(data_cfg['split_file'], 'r', encoding='utf-8') as f:
        split_info = json.load(f)
    val_mode = train_cfg.get('val_mode', 'val')
    if val_mode not in split_info:
        val_mode = 'test' if 'test' in split_info else val_mode
    train_ds = DownstreamClassificationDataset(
        data_cfg['data_root'], data_cfg['split_file'], mode='train',
        signal_len=data_cfg['signal_len'], num_classes=data_cfg['num_classes'],
        on_error=data_cfg.get('on_error', 'raise'), max_error_logs=data_cfg.get('max_error_logs', 20),
        refined_labels_path=data_cfg.get('refined_labels_path', None)
    )
    val_ds = DownstreamClassificationDataset(
        data_cfg['data_root'], data_cfg['split_file'], mode=val_mode,
        signal_len=data_cfg['signal_len'], num_classes=data_cfg['num_classes'],
        on_error=data_cfg.get('on_error', 'raise'), max_error_logs=data_cfg.get('max_error_logs', 20)
    )
    if kd_cfg.get('disable_train_channel_shuffle', True):
        set_dataset_mode(train_ds, "kd_train")
    clean_indices_path = data_cfg.get('clean_indices_path')
    if clean_indices_path and os.path.exists(clean_indices_path):
        idx = np.load(clean_indices_path)
        idx = idx[idx < len(train_ds)]
        train_ds = Subset(train_ds, idx)
    clean_val_indices_path = data_cfg.get('clean_val_indices_path')
    if clean_val_indices_path and os.path.exists(clean_val_indices_path):
        idx = np.load(clean_val_indices_path)
        idx = idx[idx < len(val_ds)]
        val_ds = Subset(val_ds, idx)
    val_dataset_len = len(val_ds)
    if dist.is_initialized():
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        shuffle_train = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle_train = True
        if kd_cfg.get('use_weighted_sampler', False):
            try:
                if isinstance(train_ds, Subset):
                    indices = train_ds.indices
                    base_ds = train_ds.dataset
                else:
                    indices = list(range(len(train_ds)))
                    base_ds = train_ds
                labels = []
                for i in indices:
                    file_path = os.path.join(base_ds.data_root, base_ds.file_list[i])
                    with open(file_path, 'rb') as f:
                        content = torch.load(f) if file_path.endswith('.pt') else None
                    if content is None:
                        import pickle
                        with open(file_path, 'rb') as f:
                            content = pickle.load(f)
                    label = 0
                    if isinstance(content, dict) and 'label' in content:
                        target = content['label']
                        if isinstance(target, list):
                            if base_ds.task_index < len(target):
                                li = target[base_ds.task_index]
                                label = int(li['class']) if isinstance(li, dict) else int(li)
                        else:
                            label = int(target)
                    labels.append(label)
                labels = np.array(labels)
                class_counts = np.bincount(labels, minlength=data_cfg['num_classes'])
                class_counts = np.maximum(class_counts, 1)
                class_weights = 1.0 / class_counts
                sample_weights = class_weights[labels]
                train_sampler = WeightedRandomSampler(
                    weights=torch.from_numpy(sample_weights).double(),
                    num_samples=len(sample_weights),
                    replacement=True
                )
                shuffle_train = False
            except Exception:
                train_sampler = None
                shuffle_train = True
    train_loader = DataLoader(
        train_ds, batch_size=train_cfg['batch_size'], sampler=train_sampler, shuffle=shuffle_train,
        num_workers=data_cfg.get('num_workers', 4), pin_memory=data_cfg.get('pin_memory', True),
        collate_fn=variable_channel_collate_fn_cls
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg['batch_size'], sampler=val_sampler, shuffle=False,
        num_workers=data_cfg.get('num_workers', 4), pin_memory=data_cfg.get('pin_memory', True),
        collate_fn=variable_channel_collate_fn_cls
    )
    teacher_model_cfg = merge_model_cfg(model_cfg, kd_cfg.get('teacher_model', {}))
    student_model_cfg = merge_model_cfg(model_cfg, kd_cfg.get('student_model', {}))
    teacher_ckpt = kd_cfg.get('teacher_checkpoint', None)
    if teacher_ckpt is None:
        raise ValueError("kd.teacher_checkpoint is required.")
    teacher = build_classifier(teacher_model_cfg, data_cfg, device)
    load_model_weights(teacher, teacher_ckpt, strict=kd_cfg.get('teacher_strict', True))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = build_classifier(student_model_cfg, data_cfg, device)
    find_unused_parameters = train_cfg.get('find_unused_parameters', False)
    if dist.is_initialized():
        student = DDP(student, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=find_unused_parameters)
    use_layer_wise_lr = train_cfg.get('use_layer_wise_lr', True)
    if use_layer_wise_lr:
        param_groups = get_layer_wise_lr(unwrap_model(student), base_lr=train_cfg['base_lr'], layer_decay=train_cfg.get('layer_decay', 0.65))
        optimizer = optim.AdamW(param_groups, weight_decay=train_cfg['weight_decay'])
    else:
        optimizer = optim.AdamW(unwrap_model(student).parameters(), lr=train_cfg['base_lr'], weight_decay=train_cfg['weight_decay'])
    steps_per_epoch = len(train_loader)
    total_steps = train_cfg['epochs'] * steps_per_epoch
    warmup_steps = int(train_cfg['warmup_epochs'] * steps_per_epoch)
    if warmup_steps > 0:
        scheduler_warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
        scheduler_cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=train_cfg['min_lr'])
        scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_steps])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=train_cfg['min_lr'])
    use_focal_loss = train_cfg.get('use_focal_loss', False)
    label_smoothing = train_cfg.get('label_smoothing', 0.1)
    if use_focal_loss:
        criterion = FocalLoss(
            gamma=train_cfg.get('focal_gamma', 2.0),
            alpha=train_cfg.get('focal_alpha', None),
            reduction=train_cfg.get('focal_reduction', 'mean'),
            label_smoothing=label_smoothing
        )
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    total_epochs = train_cfg['epochs']
    use_amp = train_cfg.get('use_amp', True)
    grad_clip_norm = train_cfg.get('grad_clip_norm', 3.0)
    amp_dtype = torch.bfloat16 if (device.type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = GradScaler(enabled=(use_amp and device.type == 'cuda' and amp_dtype == torch.float16))
    kd_weight = float(kd_cfg.get('kd_weight', 1.0))
    cls_weight = float(kd_cfg.get('cls_weight', 1.0))
    kd_temperature = float(kd_cfg.get('temperature', 2.0))
    student_channel_index = int(kd_cfg.get('student_channel_index', 1))
    best_metric = float("-inf")
    start_epoch = 0
    resume_path = train_cfg.get('resume_path')
    if (not resume_path) and train_cfg.get('auto_resume', True):
        candidate = os.path.join(train_cfg['save_dir'], "last_checkpoint.pth")
        if os.path.exists(candidate):
            resume_path = candidate
    if resume_path and os.path.exists(resume_path):
        ckpt = load_checkpoint(resume_path, student, optimizer=optimizer, scheduler=scheduler, scaler=scaler)
        start_epoch = int(ckpt.get('epoch', -1)) + 1
        best_metric = float(ckpt.get('best_metric', best_metric))
    if is_main_process():
        logger.info(f"kd_weight={kd_weight} cls_weight={cls_weight} temperature={kd_temperature} student_channel_index={student_channel_index}")
        logger.info(f"teacher_checkpoint={teacher_ckpt}")
    for epoch in range(start_epoch, total_epochs):
        train_loss, cls_loss, distill_loss = train_one_epoch_kd(
            student, teacher, train_loader, criterion, optimizer, device, epoch,
            kd_weight=kd_weight, cls_weight=cls_weight, temperature=kd_temperature,
            student_channel_index=student_channel_index, scaler=scaler, use_amp=use_amp,
            grad_clip_norm=grad_clip_norm, scheduler=scheduler
        )
        val_loss, val_acc, val_f1, val_auc = evaluate_student(
            student, val_loader, criterion, device, data_cfg['num_classes'], val_dataset_len,
            student_channel_index=student_channel_index, use_amp=use_amp
        )
        metric_to_track = val_f1
        if is_main_process():
            print(f"Epoch {epoch+1}/{total_epochs} | train={train_loss:.4f} cls={cls_loss:.4f} kd={distill_loss:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} val_auc={val_auc:.4f}")
            logger.info(f"epoch={epoch+1} train={train_loss:.6f} cls={cls_loss:.6f} kd={distill_loss:.6f} val_loss={val_loss:.6f} val_acc={val_acc:.6f} val_f1={val_f1:.6f} val_auc={val_auc:.6f} lr={optimizer.param_groups[0]['lr']:.8e}")
        if metric_to_track > best_metric:
            best_metric = metric_to_track
            if is_main_process():
                torch.save(unwrap_model(student).state_dict(), os.path.join(train_cfg['save_dir'], "best_student_kd.pth"))
        if is_main_process():
            torch.save(unwrap_model(student).state_dict(), os.path.join(train_cfg['save_dir'], "last_student_kd.pth"))
            save_checkpoint(
                os.path.join(train_cfg['save_dir'], "last_checkpoint.pth"),
                model=student,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_metric,
                best_threshold=0.5,
                scaler=scaler
            )
    cleanup_distributed()


if __name__ == "__main__":
    main()
