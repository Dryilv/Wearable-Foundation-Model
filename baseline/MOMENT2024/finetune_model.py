import torch
from torch import nn
from model import MOMENTPretrain

class MOMENTClassification(nn.Module):
    def __init__(self, config_size='base', seq_len=512, patch_len=8, num_classes=2, freeze_encoder=False):
        super().__init__()
        
        # 1. 加载预训练模型架构
        self.pretrain_model = MOMENTPretrain(config_size=config_size, seq_len=seq_len, patch_len=patch_len)
        
        # 获取 d_model
        d_model = self.pretrain_model.encoder.config.d_model
        
        # 2. 冻结 Encoder (可选)
        if freeze_encoder:
            for param in self.pretrain_model.parameters():
                param.requires_grad = False
        
        # 3. 分类头
        # 我们使用简单的 Mean Pooling + MLP
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_classes)
        )
        
    def forward(self, x):
        """
        输入 x: [Batch, Num_Channels, N_Patches, Patch_Len]
        对于双通道数据，Num_Channels = 2
        """
        batch_size, num_channels, n_patches, patch_len = x.shape
        
        # 1. 重塑以通过 Encoder: [B, C, N, P] -> [B*C, N, P]
        x = x.view(batch_size * num_channels, n_patches, patch_len)
        
        # 2. Patch Embedding
        x_embed = self.pretrain_model.patch_embedding(x) # [B*C, N, D]
        
        # 3. T5 Encoder
        outputs = self.pretrain_model.encoder(inputs_embeds=x_embed)
        hidden_states = outputs.last_hidden_state # [B*C, N, D]
        
        # 4. 重塑回通道维度: [B*C, N, D] -> [B, C, N, D]
        hidden_states = hidden_states.view(batch_size, num_channels, n_patches, -1)
        
        # 5. 池化 (Mean Pooling over Channels and Patches)
        # [B, C, N, D] -> [B, D]
        pooled_output = hidden_states.mean(dim=[1, 2])
        
        # 6. 分类
        logits = self.classifier(pooled_output)
        
        return logits

    def load_pretrained_weights(self, checkpoint_path):
        """从预训练检查点加载权重"""
        print(f"Loading pretrained weights from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        
        # 过滤掉不需要的权重（如 head.weight/bias 和 loss_fct）
        # 如果保存的是整个 model 的 state_dict
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
            
        model_dict = self.pretrain_model.state_dict()
        # 匹配 key
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.pretrain_model.load_state_dict(model_dict)
        print(f"Successfully loaded {len(pretrained_dict)} layers.")
