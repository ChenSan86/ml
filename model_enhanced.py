"""
增强版文本匹配模型：融合深度学习特征和手工特征
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config
from model import SiameseEncoder, InteractionLayer
from features import FeatureExtractor


class EnhancedTextMatchModel(nn.Module):
    """
    融合深度学习特征和手工特征的文本匹配模型
    """
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout, feature_dim=19):
        super(EnhancedTextMatchModel, self).__init__()
        
        # 孪生编码器（深度学习特征）
        self.encoder = SiameseEncoder(vocab_size, embed_dim, hidden_dim, num_layers, dropout)
        
        # 交互层
        self.interaction = InteractionLayer(hidden_dim)
        
        # 特征提取器
        self.feature_extractor = FeatureExtractor()
        
        # 手工特征处理层
        self.feature_processor = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 深度学习交互特征维度
        interaction_dim = hidden_dim * 3 + 1
        
        # 融合层：深度学习特征 + 手工特征
        fusion_dim = interaction_dim + 64
        
        # 最终分类器
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
        # 初始化权重
        self._init_weights()
        
        # 用于特征标准化的统计量
        self.register_buffer('feature_mean', None)
        self.register_buffer('feature_std', None)
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.uniform_(module.weight, -0.1, 0.1)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
    
    def forward(self, query1, query2, handcrafted_features=None):
        """
        前向传播
        
        Args:
            query1, query2: [batch_size, seq_len]
            handcrafted_features: [batch_size, feature_dim] 可选的预计算特征
        
        Returns:
            logits: [batch_size]
        """
        batch_size = query1.size(0)
        
        # 1. 深度学习特征
        # 编码两个句子
        repr1, _ = self.encoder(query1)
        repr2, _ = self.encoder(query2)
        
        # 计算交互特征
        dl_features = self.interaction(repr1, repr2)  # [batch_size, interaction_dim]
        
        # 2. 手工特征
        if handcrafted_features is None:
            # 实时提取手工特征
            handcrafted_features = self.feature_extractor.extract_batch_features(
                query1, query2
            ).to(query1.device)
        
        # 标准化手工特征
        if self.training:
            # 训练时计算并更新统计量
            if self.feature_mean is None or self.feature_std is None:
                self.feature_mean = handcrafted_features.mean(dim=0, keepdim=True)
                self.feature_std = handcrafted_features.std(dim=0, keepdim=True)
                self.feature_std = torch.where(
                    self.feature_std == 0, 
                    torch.ones_like(self.feature_std), 
                    self.feature_std
                )
            else:
                # 使用移动平均更新统计量
                momentum = 0.1
                batch_mean = handcrafted_features.mean(dim=0, keepdim=True)
                batch_std = handcrafted_features.std(dim=0, keepdim=True)
                self.feature_mean = (1 - momentum) * self.feature_mean + momentum * batch_mean
                self.feature_std = (1 - momentum) * self.feature_std + momentum * batch_std
        
        if self.feature_mean is not None and self.feature_std is not None:
            handcrafted_features = (handcrafted_features - self.feature_mean) / self.feature_std
        
        # 处理手工特征
        hc_features = self.feature_processor(handcrafted_features)  # [batch_size, 64]
        
        # 3. 特征融合
        fused_features = torch.cat([dl_features, hc_features], dim=1)
        
        # 4. 分类
        logits = self.classifier(fused_features)
        
        return logits.squeeze(-1)  # [batch_size]


def create_enhanced_model():
    """创建增强版模型实例"""
    model = EnhancedTextMatchModel(
        vocab_size=config.VOCAB_SIZE,
        embed_dim=config.EMBED_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        feature_dim=19  # 手工特征维度
    )
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n🏗️  增强版模型创建完成:")
    print(f"   总参数量: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")
    print(f"   手工特征维度: 19")
    
    return model


if __name__ == '__main__':
    # 测试模型
    model = create_enhanced_model()
    model.to(config.DEVICE)
    
    # 测试前向传播
    batch_size = 4
    seq_len = 60
    query1 = torch.randint(1, 1000, (batch_size, seq_len)).to(config.DEVICE)
    query2 = torch.randint(1, 1000, (batch_size, seq_len)).to(config.DEVICE)
    
    outputs = model(query1, query2)
    print(f"\n🔍 模型测试:")
    print(f"   Input shape: [{batch_size}, {seq_len}]")
    print(f"   Output shape: {outputs.shape}")
    print(f"   Sample outputs: {outputs[:3]}")
