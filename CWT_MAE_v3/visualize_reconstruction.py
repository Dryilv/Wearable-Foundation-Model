import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import yaml

from model import CWT_MAE_RoPE


def load_sample(file_path, signal_len):
    with open(file_path, 'rb') as f:
        content = pickle.load(f)
    raw_data = content['data'] if isinstance(content, dict) and 'data' in content else content
    if isinstance(raw_data, list):
        raw_data = np.array(raw_data)
    if raw_data.ndim == 1:
        raw_data = raw_data[np.newaxis, :]
    if raw_data.dtype != np.float32:
        raw_data = raw_data.astype(np.float32)
    if raw_data.shape[1] > signal_len:
        start = (raw_data.shape[1] - signal_len) // 2
        raw_data = raw_data[:, start:start + signal_len]
    elif raw_data.shape[1] < signal_len:
        pad_len = signal_len - raw_data.shape[1]
        raw_data = np.pad(raw_data, ((0, 0), (0, pad_len)), mode='edge')
    return torch.from_numpy(raw_data)


def build_model_from_config(config):
    model_cfg = config['model']
    data_cfg = config['data']
    return CWT_MAE_RoPE(
        signal_len=data_cfg.get('signal_len', 3000),
        cwt_scales=model_cfg.get('cwt_scales', 64),
        patch_size_time=model_cfg.get('patch_size_time', 50),
        patch_size_freq=model_cfg.get('patch_size_freq', 4),
        embed_dim=model_cfg.get('embed_dim', 768),
        depth=model_cfg.get('depth', 12),
        num_heads=model_cfg.get('num_heads', 12),
        decoder_embed_dim=model_cfg.get('decoder_embed_dim', 512),
        decoder_depth=model_cfg.get('decoder_depth', 8),
        decoder_num_heads=model_cfg.get('decoder_num_heads', 16),
        mask_ratio=model_cfg.get('mask_ratio', 0.75),
        time_loss_weight=model_cfg.get('time_loss_weight', 1.0),
        use_diff=model_cfg.get('use_diff', False),
        diff_loss_weight=model_cfg.get('diff_loss_weight', None),
        max_modalities=model_cfg.get('max_modalities', 16),
    )


def load_pretrained_weights(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '').replace('_orig_mod.', '')
        if name.startswith('encoder.'):
            name = name[len('encoder.'):]
        if name.startswith('proj_head.'):
            continue
        cleaned_state_dict[name] = v
    msg = model.load_state_dict(cleaned_state_dict, strict=False)
    return msg


def unpatchify_spec(pred_spec, grid_size, patch_size, channels):
    h_grid, w_grid = grid_size
    p_h, p_w = patch_size
    m = pred_spec.shape[0]
    spec = pred_spec.reshape(m, h_grid, w_grid, channels, p_h, p_w)
    spec = spec.permute(0, 3, 1, 4, 2, 5).contiguous()
    return spec.reshape(m, channels, h_grid * p_h, w_grid * p_w)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output', type=str, default='reconstruction_visualization.png')
    parser.add_argument('--mask_ratio', type=float, default=None)
    parser.add_argument('--channel_idx', type=int, default=0)
    parser.add_argument('--component_idx', type=int, default=0)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    data_cfg = config['data']
    signal_len = data_cfg.get('signal_len', 3000)
    x = load_sample(args.input_path, signal_len=signal_len).unsqueeze(0)

    model = build_model_from_config(config)
    msg = load_pretrained_weights(model, args.checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    x = x.to(device)

    mask_ratio = args.mask_ratio
    if mask_ratio is None:
        mask_ratio = config['model'].get('mask_ratio', 0.75)

    with torch.no_grad():
        loss, loss_dict, pred_spec, pred_time, imgs, mask, latent = model(x, mask_ratio=mask_ratio)

    c = 3 if model.use_diff else 1
    recon_spec = unpatchify_spec(
        pred_spec[0].float().cpu(),
        grid_size=model.grid_size,
        patch_size=(model.patch_embed.patch_size[0], model.patch_embed.patch_size[1]),
        channels=c
    )
    orig_spec = imgs[0].float().cpu()

    channel_idx = max(0, min(args.channel_idx, orig_spec.shape[0] - 1))
    component_idx = max(0, min(args.component_idx, orig_spec.shape[1] - 1))

    orig_img = orig_spec[channel_idx, component_idx].numpy()
    recon_img = recon_spec[channel_idx, component_idx].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    im0 = axes[0].imshow(orig_img, aspect='auto', origin='lower', cmap='jet')
    axes[0].set_title('Original CWT')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Scale')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(recon_img, aspect='auto', origin='lower', cmap='jet')
    axes[1].set_title('Reconstructed CWT')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Scale')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"loss={float(loss):.6f}  loss_spec={float(loss_dict['loss_spec']):.6f}  loss_time={float(loss_dict['loss_time']):.6f}\n"
        f"channel={channel_idx}, component={component_idx}, mask_ratio={mask_ratio}, missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}"
    )
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved visualization to: {os.path.abspath(args.output)}")


if __name__ == '__main__':
    main()
