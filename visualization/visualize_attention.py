import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import sys
import math

# Add path to include model definitions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../CWT_MAE_v3')))

from model_finetune import TF_MAE_Classifier
from dataset_finetune import DownstreamClassificationDataset
from model import RoPEAttention, apply_rotary_pos_emb, cwt_wrap

# Global dictionary to store captured attention weights
attention_store = {}

def get_attention_forward(name):
    """
    Returns a forward function that captures attention weights.
    Replaces F.scaled_dot_product_attention with manual implementation to get weights.
    """
    def forward(self, x, rope_cos=None, rope_sin=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        if rope_cos is not None and rope_sin is not None:
             q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin)
        
        q = q.transpose(1, 2) # (B, H, N, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Manual Attention to capture weights
        scale = (C // self.num_heads) ** -0.5
        attn_scores = (q @ k.transpose(-2, -1)) * scale
        attn_weights = attn_scores.softmax(dim=-1)
        
        # Store weights
        attention_store[name] = attn_weights.detach().cpu()
        
        # Apply dropout and value projection
        x = attn_weights @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    return forward

def get_cross_attention_forward(name):
    """
    For nn.MultiheadAttention in LatentReasoningHead
    """
    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True, is_causal=False):
        # We invoke the original forward but force need_weights=True
        # However, nn.MultiheadAttention.forward returns (output, attn_weights)
        # We need to capture the second return value.
        
        # Calling super().forward is tricky because we are replacing the method on an instance.
        # Instead, we will use the original method which we will save before replacing.
        pass 
    return forward

# Since nn.MultiheadAttention is a standard module, we can just hook into its output 
# IF it returns weights. In LatentReasoningHead, it is called as:
# attn_out, _ = self.cross_attn(query=queries, key=x_encoder, value=x_encoder)
# It explicitly ignores weights. So we definitely need to monkey patch it to save weights 
# before returning.

def capture_cross_attn_forward(original_forward, name):
    def forward(query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True, is_causal=False):
        # Force need_weights=True to get the weights
        output, weights = original_forward(query, key, value, key_padding_mask=key_padding_mask, need_weights=True, attn_mask=attn_mask, average_attn_weights=False)
        attention_store[name] = weights.detach().cpu()
        return output, weights
    return forward


def load_finetune_model(config_path, checkpoint_path, device):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Initializing model...")
    # Infer number of classes from dataset or config (defaulting to 2 or 3 based on typical tasks)
    # Ideally should be in config. Let's assume 3 for now or read from checkpoint?
    # Actually, we can try to guess or just use a safe number, but the weights must match.
    # We'll rely on the user or config. 
    num_classes = config['model'].get('num_classes', 3) # Default fallback

    model = TF_MAE_Classifier(
        pretrained_path=None, # We load weights manually later
        num_classes=num_classes,
        use_cot=config['model'].get('use_cot', True),
        use_arcface=config['model'].get('use_arcface', False),
        num_reasoning_tokens=config['model'].get('num_reasoning_tokens', 16),
        # Encoder args
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
        mask_ratio=0.0, # No masking for inference
    )
    
    if os.path.isfile(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        
        # Clean state dict keys
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k
            if name.startswith('module.'): name = name[7:]
            new_state_dict[name] = v
            
        # Handle shape mismatch for classifier if needed (e.g. if num_classes changed)
        model.load_state_dict(new_state_dict, strict=True)
        print("Weights loaded successfully.")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model.to(device)
    model.eval()
    
    # --- Inject Hooks for Attention Capture ---
    
    # 1. Encoder Last Block Time Attention
    last_block = model.encoder_model.blocks[-1]
    # Bind the method to the instance
    last_block.time_attn.forward = get_attention_forward("encoder_last_time_attn").__get__(last_block.time_attn, RoPEAttention)
    print("Hooked Encoder Last Block Time Attention")

    # 2. CoT Cross Attention (if exists)
    if hasattr(model, 'head') and hasattr(model.head, 'cross_attn'):
        # nn.MultiheadAttention forward replacement
        original_forward = model.head.cross_attn.forward
        model.head.cross_attn.forward = capture_cross_attn_forward(original_forward, "cot_cross_attn")
        print("Hooked CoT Cross Attention")

    return model, config

def visualize_attention(model, dataset, index, device, save_path="attention_vis.png"):
    print(f"Visualizing sample index: {index}")
    
    # Get Data
    sample, modality_id, label = dataset[index]
    # sample: (M, L) or (1, L)
    # Ensure (1, M, L) for model
    if sample.dim() == 2:
        input_tensor = sample.unsqueeze(0).to(device) # (1, M, L)
    else:
        input_tensor = sample.unsqueeze(0).unsqueeze(0).to(device)

    # Forward Pass
    with torch.no_grad():
        logits = model(input_tensor)
        pred_label = torch.argmax(logits, dim=1).item()
    
    print(f"True Label: {label}, Predicted: {pred_label}")
    
    # Retrieve Attention Weights
    # 1. Encoder Attention: (1, H, N, N) -> We care about attention map over time
    # N = M * N_patches. If M=1, N = N_patches.
    # N_patches = L // patch_size_time
    enc_attn = attention_store.get("encoder_last_time_attn") # (1, H, N, N)
    
    # 2. CoT Attention: (1, H, N_reason, N_source)
    cot_attn = attention_store.get("cot_cross_attn") # (1, H, N_reason, N_source)
    
    # --- Visualization ---
    
    # Prepare Plot
    fig_rows = 3
    if cot_attn is not None: fig_rows += 1
    
    plt.figure(figsize=(16, 4 * fig_rows))
    
    # Row 1: Original Signal
    signal = sample[0].cpu().numpy() # Take first channel
    plt.subplot(fig_rows, 1, 1)
    plt.plot(signal, color='black', linewidth=1)
    plt.title(f"Original Signal (Class {label}, Pred {pred_label})")
    plt.xlim(0, len(signal))
    
    # Row 2: CWT Spectrogram (Recompute for visualization)
    with torch.no_grad():
        cwt_img = cwt_wrap(input_tensor, num_scales=model.encoder_model.cwt_scales, lowest_scale=0.1, step=1.0)
        # (1, M, C, H, W) -> (H, W)
        spec = cwt_img[0, 0, 0].cpu().numpy()
        
    plt.subplot(fig_rows, 1, 2)
    plt.imshow(spec, aspect='auto', origin='lower', cmap='jet')
    plt.title("CWT Spectrogram")
    
    # Row 3: Encoder Attention Map (Average over heads)
    if enc_attn is not None:
        # enc_attn: (1, H, N, N). We want to see for each position, where it attends to.
        # Or better: The "importance" of each time step.
        # Since we don't have a CLS token in encoder, we can visualize the diagonal or average attention?
        # A common way for time-series is to show the "Self-Attention Map" as a heatmap.
        # N x N matrix.
        attn_mat = enc_attn[0].mean(dim=0).numpy() # (N, N)
        
        plt.subplot(fig_rows, 1, 3)
        plt.imshow(attn_mat, aspect='auto', origin='lower', cmap='viridis')
        plt.title("Encoder Self-Attention (Avg Heads, Last Layer)")
        plt.xlabel("Key Position (Time Patch)")
        plt.ylabel("Query Position (Time Patch)")
        plt.colorbar()
        
    # Row 4: CoT Attention (Reasoning Tokens -> Encoder Output)
    if cot_attn is not None:
        # cot_attn: (1, H, N_reason, N_source)
        # We want to see which parts of the encoder output (N_source) are attended by the reasoning tokens.
        # N_source corresponds to the time patches.
        # We can average over heads and reasoning tokens to get a "Global Importance" curve over time patches.
        
        # (1, H, N_reason, N_source) -> (N_source,)
        cot_avg = cot_attn[0].mean(dim=0).mean(dim=0).numpy() # (N_source,)
        
        # Resize to signal length for overlay
        # Use torch.nn.functional.interpolate instead of cv2 to avoid extra dependency
        cot_tensor = torch.from_numpy(cot_avg).float().view(1, 1, -1) # (1, 1, N_source)
        cot_overlay = F.interpolate(cot_tensor, size=len(signal), mode='linear', align_corners=False)
        cot_overlay = cot_overlay.squeeze().numpy()
        
        plt.subplot(fig_rows, 1, fig_rows)
        # Plot Signal again
        plt.plot(signal, color='gray', alpha=0.5, label='Signal')
        # Overlay Attention
        # Normalize attention for visibility
        cot_overlay_norm = (cot_overlay - cot_overlay.min()) / (cot_overlay.max() - cot_overlay.min() + 1e-8)
        
        # Scale to signal range
        sig_min, sig_max = signal.min(), signal.max()
        cot_scaled = cot_overlay_norm * (sig_max - sig_min) + sig_min
        
        plt.plot(cot_scaled, color='red', linewidth=2, label='CoT Attention Profile')
        plt.fill_between(range(len(signal)), sig_min, cot_scaled, color='red', alpha=0.2)
        plt.title("CoT Cross-Attention (Avg Reasoning Tokens -> Time)")
        plt.legend()
        plt.xlim(0, len(signal))

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize Attention for Finetuned Model")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to finetuned model checkpoint')
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of dataset')
    parser.add_argument('--split_file', type=str, required=True, help='Path to split json file')
    parser.add_argument('--channel_policy', type=str, default='default_5ch', help='Channel policy for dataset')
    parser.add_argument('--mode', type=str, default='val', help='Dataset split mode (train/val/test)')
    parser.add_argument('--index', type=int, default=None, help='Sample index to visualize')
    parser.add_argument('--output', type=str, default='attention_vis.png', help='Output filename')
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model, config = load_finetune_model(args.config, args.checkpoint, device)
    
    # Load Dataset
    dataset = DownstreamClassificationDataset(
        data_root=args.data_root,
        split_file=args.split_file,
        signal_len=config['data']['signal_len'],
        mode=args.mode,
        num_classes=config['model'].get('num_classes', 3),
        channel_policy=args.channel_policy
    )
    
    if args.index is None:
        import random
        args.index = random.randint(0, len(dataset)-1)
        
    visualize_attention(model, dataset, args.index, device, args.output)

if __name__ == "__main__":
    main()
