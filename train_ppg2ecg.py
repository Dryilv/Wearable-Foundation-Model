import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# 假设 model.py 和 dataset.py 在同一目录下
from model import CWT_MAE_RoPE, FrozenMAEWrapper, LatentDiffusion1D, get_diffusion_params, train_diffusion_model
from dataset import PairedPhysioDataset

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
BATCH_SIZE = 64 # Adjust based on your GPU memory
NUM_WORKERS = 4 # Adjust based on your CPU cores

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

# 4. Get Diffusion Parameters
betas, alphas, alphas_cumprod = get_diffusion_params(
    num_timesteps=NUM_TIMESTEPS, 
    beta_start=BETA_START, 
    beta_end=BETA_END, 
    device=DEVICE
)

# 5. Load Dataset and DataLoader
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

# --- Training ---
print("Starting diffusion model training...")
train_diffusion_model(
    mae_wrapper=mae_wrapper,
    diffusion_model=diffusion_model,
    train_loader=train_loader,
    optimizer=optimizer,
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

# --- Example Inference (Optional) ---
# Load a sample PPG signal for inference
# Make sure you have a way to load a single PPG signal for testing
# For example, load one sample from the dataset or a specific file
# ppg_sample, _ = train_dataset[0] # Load first sample (PPG only)
# ppg_sample = ppg_sample.unsqueeze(0).to(DEVICE) # Add batch dimension

# print("\nStarting inference example...")
# generated_ecg = generate_ecg_from_ppg(
#     mae_wrapper=mae_wrapper,
#     diffusion_model=diffusion_model,
#     ppg_signal=ppg_sample,
#     num_timesteps=NUM_TIMESTEPS,
#     betas=betas,
#     alphas_cumprod=alphas_cumprod,
#     device=DEVICE
# )

# print("Inference complete. Generated ECG shape:", generated_ecg.shape)
# # You can now save or visualize generated_ecg