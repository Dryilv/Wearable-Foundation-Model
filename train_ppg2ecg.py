import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm  # 导入tqdm

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

# --- 修改train_diffusion_model函数以包含tqdm进度条 ---
# 如果train_diffusion_model函数来自model_ppg2ecg，您需要在该文件中修改函数
# 或者在这里重新定义它：

def train_diffusion_model_with_tqdm(
    mae_wrapper,
    diffusion_model,
    train_loader,
    optimizer,
    num_timesteps,
    betas,
    alphas_cumprod,
    device,
    epochs
):
    """训练扩散模型，带有tqdm进度条"""
    diffusion_model.train()
    
    # 保存损失历史
    loss_history = []
    
    for epoch in range(epochs):
        # 初始化epoch的进度条
        epoch_pbar = tqdm(
            enumerate(train_loader), 
            total=len(train_loader), 
            desc=f'Epoch {epoch+1}/{epochs}',
            leave=True
        )
        
        epoch_loss = 0.0
        
        for batch_idx, (ppg_signals, ecg_signals) in epoch_pbar:
            # 移动数据到设备
            ppg_signals = ppg_signals.to(device)
            ecg_signals = ecg_signals.to(device)
            
            # 获取ECG的潜在表示
            with torch.no_grad():
                ecg_latents = mae_wrapper.get_latent_representation(ecg_signals)
            
            # 准备训练扩散模型
            batch_size = ecg_latents.size(0)
            
            # 随机选择时间步
            t = torch.randint(0, num_timesteps, (batch_size,), device=device).long()
            
            # 生成噪声
            noise = torch.randn_like(ecg_latents)
            
            # 计算带噪声的潜在表示
            sqrt_alpha_cumprod_t = torch.sqrt(alphas_cumprod[t]).view(-1, 1, 1)
            sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alphas_cumprod[t]).view(-1, 1, 1)
            noisy_latents = sqrt_alpha_cumprod_t * ecg_latents + sqrt_one_minus_alpha_cumprod_t * noise
            
            # 获取PPG条件信息
            with torch.no_grad():
                ppg_condition = mae_wrapper.get_ppg_condition(ppg_signals)
            
            # 预测噪声
            predicted_noise = diffusion_model(
                x=noisy_latents, 
                time=t, 
                condition=ppg_condition
            )
            
            # 计算损失
            loss = torch.mean((predicted_noise - noise) ** 2)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 更新统计信息
            epoch_loss += loss.item()
            
            # 更新进度条描述
            epoch_pbar.set_postfix({
                'batch_loss': f'{loss.item():.6f}',
                'avg_loss': f'{epoch_loss/(batch_idx+1):.6f}'
            })
        
        # 计算epoch平均损失
        avg_epoch_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_epoch_loss)
        
        # 打印epoch信息
        print(f'Epoch {epoch+1}/{epochs} completed. Average Loss: {avg_epoch_loss:.6f}')
        
        # 每5个epoch保存一次检查点
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f'diffusion_checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': diffusion_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
                'loss_history': loss_history
            }, checkpoint_path)
            print(f'Checkpoint saved to {checkpoint_path}')
    
    return loss_history

# --- Training with tqdm ---
print("Starting diffusion model training with tqdm progress bars...")
loss_history = train_diffusion_model_with_tqdm(
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
diffusion_model_save_path = './ppg2ecg/trained_diffusion_model.pth'
torch.save({
    'model_state_dict': diffusion_model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epochs': EPOCHS,
    'loss_history': loss_history
}, diffusion_model_save_path)
print(f"Trained diffusion model saved to {diffusion_model_save_path}")

# 绘制训练损失曲线
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_history) + 1), loss_history, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Diffusion Model Training Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig('training_loss.png')
    print("Training loss plot saved as 'training_loss.png'")
except ImportError:
    print("Matplotlib not available, skipping loss plot generation")