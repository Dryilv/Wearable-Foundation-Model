import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import tqdm
import torch.nn as nn
# 假设 model.py 和 dataset.py 在同一目录下
from model_ppg2ecg import CWT_MAE_RoPE, FrozenMAEWrapper, LatentDiffusion1D, get_diffusion_params, train_diffusion_model
from dataset_paired import PairedPhysioDataset

# --- Configuration ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# MAE Pretrained Model Path
MAE_PRETRAINED_PATH = '/home/bml/storage/mnt/v-044d0fb740b04ad3/org/WFM/vit16trans/checkpoint_rope_tensor_768/checkpoint_last.pth' 

# Dataset Configuration
DATA_INDEX_FILE = '/home/bml/storage/mnt/v-044d0fb740b04ad3/org/WFM/model/SharedPhysioTFMAE/train_index.json' # JSON file with {'path': '...', 'row': ...}
SIGNAL_LEN = 3000
ROW_PPG = 4  # Index for PPG signal in the pickle file
ROW_ECG = 0  # Index for ECG signal in the pickle file
BATCH_SIZE = 256 # Adjust based on your GPU memory
NUM_WORKERS = 8 # Adjust based on your CPU cores

# Diffusion Model Configuration
NUM_TIMESTEPS = 1000
BETA_START = 0.0001
BETA_END = 0.02
LEARNING_RATE = 1e-4
EPOCHS = 50 # Number of training epochs

# MAE Model Parameters (Must match your pretrained model)
MAE_EMBED_DIM = 768
MAE_DEPTH = 12
MAE_NUM_HEADS = 12
MAE_DECODER_EMBED_DIM = 512
MAE_DECODER_DEPTH = 8
MAE_DECODER_NUM_HEADS = 16
MAE_CWT_SCALES = 64
MAE_PATCH_SIZE_TIME = 50
MAE_PATCH_SIZE_FREQ = 4

# Latent Diffusion Model Parameters
LDM_TIME_EMB_DIM = 256
LDM_HIDDEN_DIM = 128
LDM_GROUPS = 8

# --- Initialization ---

# 1. Load MAE Wrapper (Frozen)
mae_wrapper = FrozenMAEWrapper(
    pretrained_path=MAE_PRETRAINED_PATH,
    device=DEVICE,
    # Pass MAE config to ensure correct initialization
    signal_len=SIGNAL_LEN,
    cwt_scales=MAE_CWT_SCALES,
    patch_size_time=MAE_PATCH_SIZE_TIME,
    patch_size_freq=MAE_PATCH_SIZE_FREQ,
    embed_dim=MAE_EMBED_DIM,
    depth=MAE_DEPTH,
    num_heads=MAE_NUM_HEADS,
    decoder_embed_dim=MAE_DECODER_EMBED_DIM,
    decoder_depth=MAE_DECODER_DEPTH,
    decoder_num_heads=MAE_DECODER_NUM_HEADS
)

# 2. Initialize Latent Diffusion Model
diffusion_model = LatentDiffusion1D(
    mae_embed_dim=MAE_EMBED_DIM,
    mae_decoder_embed_dim=MAE_DECODER_EMBED_DIM,
    time_emb_dim=LDM_TIME_EMB_DIM,
    hidden_dim=LDM_HIDDEN_DIM,
    num_groups=LDM_GROUPS
).to(DEVICE)

# 3. Initialize Optimizer
optimizer = optim.AdamW(diffusion_model.parameters(), lr=LEARNING_RATE)

# 4. Initialize GradScaler for mixed precision training
scaler = GradScaler(enabled=(DEVICE == 'cuda'))

# 5. Get Diffusion Parameters
betas, alphas, alphas_cumprod = get_diffusion_params(
    num_timesteps=NUM_TIMESTEPS, 
    beta_start=BETA_START, 
    beta_end=BETA_END, 
    device=DEVICE
)

# 6. Load Dataset and DataLoader
train_dataset = PairedPhysioDataset(
    index_file=DATA_INDEX_FILE,
    signal_len=SIGNAL_LEN,
    mode='train',
    row_ppg=ROW_PPG,
    row_ecg=ROW_ECG
)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=True # Drop last batch if it's smaller than batch_size
)

# --- Modified Training Function with AMP ---
def train_diffusion_model_with_amp(
    mae_wrapper: FrozenMAEWrapper, 
    diffusion_model: LatentDiffusion1D, 
    train_loader: torch.utils.data.DataLoader, 
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    num_timesteps: int,
    betas: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    device: str,
    epochs: int = 10
):
    """Trains the Latent Diffusion Model with Automatic Mixed Precision."""
    diffusion_model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for ppg, ecg in progress_bar:
            ppg, ecg = ppg.to(device), ecg.to(device)
            
            # 使用 autocast 进行混合精度训练
            with autocast(enabled=(device == 'cuda')):
                # 1. Get Latent Representations from MAE Encoder
                z_ppg, _ = mae_wrapper.encode(ppg) # Condition (B, N+1, D_encoder)
                z_ecg, _ = mae_wrapper.encode(ecg) # Target (B, N+1, D_encoder)
                
                # 2. Sample random timesteps
                B = ppg.shape[0]
                t = torch.randint(0, num_timesteps, (B,), device=device).long()
                
                # 3. Add noise to ECG latent
                noise = torch.randn_like(z_ecg)
                alpha_bar_t = extract(alphas_cumprod, t, z_ecg.shape)
                z_ecg_noisy = torch.sqrt(alpha_bar_t) * z_ecg + torch.sqrt(1 - alpha_bar_t) * noise
                
                # 4. Predict the noise using the diffusion model
                noise_pred = diffusion_model(z_ecg_noisy, z_ppg, t)
                
                # 5. Calculate MSE loss
                loss = nn.functional.mse_loss(noise_pred, noise)
            
            # 6. Backpropagation and optimization with AMP
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # Optional: Gradient clipping (在 unscaled 之后进行)
            torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.6f}")
        
        # Optional: Save model checkpoint periodically
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"diffusion_model_epoch_{epoch+1}.pth"
            torch.save(diffusion_model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    print("Training finished.")

# 需要在 model_ppg2ecg.py 中添加 extract 函数或导入
from model_ppg2ecg import extract

# --- Training ---
print("Starting diffusion model training with AMP...")
train_diffusion_model_with_amp(
    mae_wrapper=mae_wrapper,
    diffusion_model=diffusion_model,
    train_loader=train_loader,
    optimizer=optimizer,
    scaler=scaler,
    num_timesteps=NUM_TIMESTEPS,
    betas=betas,
    alphas_cumprod=alphas_cumprod,
    device=DEVICE,
    epochs=EPOCHS
)

# --- Save Trained Diffusion Model ---
# Save the diffusion model weights after training
diffusion_model_save_path = 'trained_diffusion_model.pth'
torch.save(diffusion_model.state_dict(), diffusion_model_save_path)
print(f"Trained diffusion model saved to {diffusion_model_save_path}")