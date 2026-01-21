import torch
import torch.nn as nn
import torch.nn.functional as F
from model import CWT_MAE_RoPE, cwt_wrap

class PPG2ECG_Translator(nn.Module):
    def __init__(self, pretrained_path=None, **kwargs):
        super().__init__()
        
        # 1. 初始化基础模型
        # 注意：这里 mask_ratio 设为 0，因为翻译任务需要利用 PPG 的全部信息
        self.mae = CWT_MAE_RoPE(mask_ratio=0.0, **kwargs)
        
        # 2. 加载预训练权重
        if pretrained_path:
            self._load_pretrained_weights(pretrained_path)
            
        # 3. 冻结策略 (可选)
        # 如果预训练非常充分，可以冻结 Encoder，只训练 Decoder
        # self._freeze_encoder() 

    def _load_pretrained_weights(self, path):
        print(f"Loading pretrained weights from {path}...")
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        
        # 移除 DDP 前缀
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # 加载权重 (strict=False 以防版本微小差异，但理论上应该完全匹配)
        msg = self.mae.load_state_dict(new_state_dict, strict=False)
        print(f"Weights loaded. {msg}")

    def _freeze_encoder(self):
        print("Freezing Encoder parameters...")
        for param in self.mae.patch_embed.parameters():
            param.requires_grad = False
        for param in self.mae.blocks.parameters():
            param.requires_grad = False
        for param in self.mae.norm.parameters():
            param.requires_grad = False
        # Positional embeddings
        self.mae.pos_embed.requires_grad = False
        self.mae.cls_token.requires_grad = False

    def forward(self, ppg, ecg_target=None):
        """
        Args:
            ppg: [Batch, Length] 输入的 PPG 信号
            ecg_target: [Batch, Length] 目标 ECG 信号 (训练时需要，推理时可为 None)
        Returns:
            如果 ecg_target 不为 None: 返回 (total_loss, pred_ecg_time, pred_ecg_spec)
            如果 ecg_target 为 None: 返回 (pred_ecg_time, pred_ecg_spec)
        """
        # ==========================================
        # 1. 处理输入 (PPG) -> Encoder
        # ==========================================
        if ppg.dim() == 3: ppg = ppg.squeeze(1)
        
        # CWT 变换
        ppg_imgs = cwt_wrap(ppg, num_scales=self.mae.cwt_scales, lowest_scale=0.1, step=1.0)
        
        # 归一化 PPG (Instance Norm)
        dtype_orig = ppg_imgs.dtype
        ppg_imgs = ppg_imgs.float()
        mean = ppg_imgs.mean(dim=(2, 3), keepdim=True)
        std = ppg_imgs.std(dim=(2, 3), keepdim=True)
        ppg_imgs = (ppg_imgs - mean) / (std + 1e-5)
        ppg_imgs = ppg_imgs.to(dtype=dtype_orig)

        # Encoder 前向传播
        # 强制 mask_ratio=0，保留所有 PPG 信息
        self.mae.mask_ratio = 0.0 
        latent, mask, ids_restore = self.mae.forward_encoder(ppg_imgs)

        # ==========================================
        # 2. Latent -> Decoder (生成 ECG)
        # ==========================================
        # Decoder 前向传播
        decoder_features = self.mae.forward_decoder(latent, ids_restore)
        
        # 预测 ECG 频谱
        pred_spec = self.mae.decoder_pred_spec(decoder_features)
        
        # 预测 ECG 时域信号
        B, N, D = decoder_features.shape
        H_grid, W_grid = self.mae.grid_size
        feat_2d = decoder_features.transpose(1, 2).view(B, D, H_grid, W_grid)
        feat_time_agg = self.mae.time_reducer(feat_2d)
        feat_time_agg = feat_time_agg.squeeze(2).transpose(1, 2)
        pred_ecg_time = self.mae.time_pred(feat_time_agg).flatten(1)

        # ==========================================
        # 3. 计算 Loss (如果提供了 Target)
        # ==========================================
        if ecg_target is not None:
            if ecg_target.dim() == 3: ecg_target = ecg_target.squeeze(1)
            
            # --- 3.1 时域 Loss ---
            # 对目标 ECG 进行归一化 (Z-score)，因为模型输出也是归一化的
            target_mean = ecg_target.mean(dim=-1, keepdim=True)
            target_std = ecg_target.std(dim=-1, keepdim=True) + 1e-5
            target_norm = (ecg_target - target_mean) / target_std
            
            loss_time = F.mse_loss(pred_ecg_time, target_norm)
            
            # --- 3.2 频域 Loss (Spectrogram Loss) ---
            # 我们需要计算 Target ECG 的真实 CWT 频谱作为标签
            with torch.no_grad():
                ecg_imgs = cwt_wrap(ecg_target, num_scales=self.mae.cwt_scales, lowest_scale=0.1, step=1.0)
                # 同样的归一化
                ecg_imgs = ecg_imgs.float()
                e_mean = ecg_imgs.mean(dim=(2, 3), keepdim=True)
                e_std = ecg_imgs.std(dim=(2, 3), keepdim=True)
                ecg_imgs = (ecg_imgs - e_mean) / (e_std + 1e-5)
                
                # Reshape 成 Patch 格式: [B, N_patches, Pixels_per_patch]
                p_h, p_w = self.mae.patch_embed.patch_size
                B, C, H, W = ecg_imgs.shape
                target_spec = ecg_imgs.view(B, C, H // p_h, p_h, W // p_w, p_w)
                target_spec = target_spec.permute(0, 2, 4, 1, 3, 5).contiguous()
                target_spec = target_spec.view(B, -1, C * p_h * p_w)

            # 计算频谱 Loss (全量计算，没有 mask)
            loss_spec = F.mse_loss(pred_spec, target_spec)
            
            # 总 Loss
            total_loss = loss_spec + self.mae.time_loss_weight * loss_time
            
            return total_loss, pred_ecg_time, pred_spec
        
        else:
            return pred_ecg_time, pred_spec