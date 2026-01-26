import torch
import torch.nn as nn
import torch.nn.functional as F

# 确保 model.py 在同一目录下，并且包含了 CWT_MAE_RoPE 和 cwt_wrap 的定义
from model import CWT_MAE_RoPE, cwt_wrap

# -------------------------------------------------------------------
# 1. Pearson Correlation Loss 函数
# -------------------------------------------------------------------
def pearson_correlation_loss(x, y):
    """
    计算皮尔逊相关系数损失。
    该损失鼓励模型生成与目标波形形状相似的信号。
    Args:
        x: 预测信号 [Batch, Length]
        y: 真实信号 [Batch, Length]
    Returns:
        Loss value (1 - correlation), 范围在 [0, 2] 之间。
    """
    # Z-Score 归一化 (减均值，除标准差)
    vx = x - torch.mean(x, dim=1, keepdim=True)
    vy = y - torch.mean(y, dim=1, keepdim=True)
    
    # 分子: 协方差
    cost = torch.sum(vx * vy, dim=1)
    
    # 分母: 标准差的乘积
    # 加上 epsilon (1e-8) 防止除以零
    denom = torch.sqrt(torch.sum(vx ** 2, dim=1)) * torch.sqrt(torch.sum(vy ** 2, dim=1)) + 1e-8
    
    # 皮尔逊相关系数
    corr = cost / denom
    
    # 我们希望最大化相关性 (corr -> 1)，所以最小化 (1 - corr)
    return 1 - torch.mean(corr)


# -------------------------------------------------------------------
# 2. PPG 到 ECG 的翻译模型
# -------------------------------------------------------------------
class PPG2ECG_Translator(nn.Module):
    def __init__(self, pretrained_path=None, corr_loss_weight=1.0, **kwargs):
        """
        Args:
            pretrained_path (str): 预训练 CWT-MAE 模型的路径。
            corr_loss_weight (float): Pearson Correlation Loss 相对于 MSE Loss 的权重。
            **kwargs: 传递给 CWT_MAE_RoPE 的模型参数。
        """
        super().__init__()
        
        # 初始化基础的 MAE 模型
        # mask_ratio 在翻译任务中强制为 0.0，因为需要看到完整的 PPG
        self.mae = CWT_MAE_RoPE(mask_ratio=0.0, **kwargs)
        self.corr_loss_weight = corr_loss_weight
        
        # 加载预训练权重
        if pretrained_path:
            self._load_pretrained_weights(pretrained_path)
            
    def _load_pretrained_weights(self, path):
        """加载预训练权重，并处理 DDP 和 torch.compile 引入的前缀。"""
        print(f"Loading pretrained weights from {path}...")
        # 增加 weights_only=True 以消除安全警告
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '')      # 去除 DDP 前缀
            name = name.replace('_orig_mod.', '') # 去除 torch.compile 前缀
            new_state_dict[name] = v
        
        msg = self.mae.load_state_dict(new_state_dict, strict=False)
        print(f"Weights loaded. Missing keys: {msg.missing_keys}")

    def _freeze_encoder(self):
        """(可选) 冻结 Encoder，只训练 Decoder。"""
        print("Freezing Encoder parameters...")
        for param in self.mae.patch_embed.parameters():
            param.requires_grad = False
        for param in self.mae.blocks.parameters():
            param.requires_grad = False
        for param in self.mae.norm.parameters():
            param.requires_grad = False
        self.mae.pos_embed.requires_grad = False
        self.mae.cls_token.requires_grad = False

    def forward(self, ppg, ecg_target=None):
        """
        模型的前向传播。
        - 训练时: 输入 ppg 和 ecg_target, 返回 (total_loss, loss_spec, loss_mse, loss_corr)。
        - 推理时: 输入 ppg, 返回 (pred_ecg_time, pred_ecg_spec)。
        """
        # 1. Encoder: 从 PPG 提取特征
        if ppg.dim() == 3: ppg = ppg.squeeze(1)
        ppg_imgs = cwt_wrap(ppg, num_scales=self.mae.cwt_scales)
        ppg_imgs = (ppg_imgs.float() - ppg_imgs.mean(dim=(2, 3), keepdim=True)) / (ppg_imgs.std(dim=(2, 3), keepdim=True) + 1e-5)
        
        self.mae.mask_ratio = 0.0 
        latent, _, ids_restore = self.mae.forward_encoder(ppg_imgs)

        # 2. Decoder: 从特征生成 ECG
        decoder_features = self.mae.forward_decoder(latent, ids_restore)
        pred_spec = self.mae.decoder_pred_spec(decoder_features)
        
        B, N, D = decoder_features.shape
        H_grid, W_grid = self.mae.grid_size
        feat_2d = decoder_features.transpose(1, 2).view(B, D, H_grid, W_grid)
        feat_time_agg = self.mae.time_reducer(feat_2d).squeeze(2).transpose(1, 2)
        pred_ecg_time = self.mae.time_pred(feat_time_agg).flatten(1)

        # 3. 根据模式返回结果
        if ecg_target is not None:
            # --- 训练模式: 计算 Loss ---
            if ecg_target.dim() == 3: ecg_target = ecg_target.squeeze(1)
            
            # 准备 Target ECG (Z-Score 归一化)
            target_mean = ecg_target.mean(dim=-1, keepdim=True)
            target_std = ecg_target.std(dim=-1, keepdim=True) + 1e-5
            target_norm = (ecg_target - target_mean) / target_std
            
            # --- 3.1 时域 Loss (混合 Loss) ---
            loss_mse = F.mse_loss(pred_ecg_time, target_norm)
            loss_corr = pearson_correlation_loss(pred_ecg_time, target_norm)
            loss_time_combined = loss_mse + self.corr_loss_weight * loss_corr
            
            # --- 3.2 频域 Loss ---
            with torch.no_grad():
                ecg_imgs = cwt_wrap(ecg_target, num_scales=self.mae.cwt_scales)
                ecg_imgs = (ecg_imgs.float() - ecg_imgs.mean(dim=(2, 3), keepdim=True)) / (ecg_imgs.std(dim=(2, 3), keepdim=True) + 1e-5)
                p_h, p_w = self.mae.patch_embed.patch_size
                B, C, H, W = ecg_imgs.shape
                target_spec = ecg_imgs.view(B, C, H // p_h, p_h, W // p_w, p_w).permute(0, 2, 4, 1, 3, 5).contiguous().view(B, -1, C * p_h * p_w)
            loss_spec = F.mse_loss(pred_spec, target_spec)
            
            # --- 总 Loss ---
            total_loss = loss_spec + self.mae.time_loss_weight * loss_time_combined
            
            # 返回所有子 Loss 用于详细日志记录
            return total_loss, loss_spec.item(), loss_mse.item(), loss_corr.item()
        
        else:
            # --- 推理模式: 返回生成结果 ---
            return pred_ecg_time, pred_spec