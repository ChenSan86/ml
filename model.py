"""
文本匹配模型：双塔架构 + 多层交互
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config


class SiameseEncoder(nn.Module):
    """孪生编码器：使用BiLSTM编码文本"""
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super(SiameseEncoder, self).__init__()
        
        # Embedding层
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # BiLSTM层
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim // 2,  # 双向，所以hidden_dim要除以2
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        x: [batch_size, seq_len]
        return: [batch_size, hidden_dim]
        """
        # Embedding: [batch_size, seq_len, embed_dim]
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        # LSTM: output [batch_size, seq_len, hidden_dim]
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # 使用最后一个时间步的输出 (或者可以用mean pooling)
        # hidden: [num_layers*2, batch_size, hidden_dim//2]
        # 取最后一层的正向和反向hidden state拼接
        forward_hidden = hidden[-2, :, :]  # 最后一层正向
        backward_hidden = hidden[-1, :, :] # 最后一层反向
        
        # 拼接: [batch_size, hidden_dim]
        sentence_repr = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        return sentence_repr, lstm_out


class InteractionLayer(nn.Module):
    """交互层：计算两个句子表示之间的多种交互特征"""
    
    def __init__(self, hidden_dim):
        super(InteractionLayer, self).__init__()
        
    def forward(self, repr1, repr2):
        """
        计算多种交互特征
        repr1, repr2: [batch_size, hidden_dim]
        return: [batch_size, feature_dim]
        """
        # 1. Cosine similarity
        cos_sim = F.cosine_similarity(repr1, repr2, dim=1, eps=1e-8)
        
        # 2. Element-wise product
        element_product = repr1 * repr2
        
        # 3. Element-wise difference
        element_diff = torch.abs(repr1 - repr2)
        
        # 4. Element-wise sum
        element_sum = repr1 + repr2
        
        # 拼接所有特征
        # [batch_size, hidden_dim * 3 + 1]
        interaction_features = torch.cat([
            cos_sim.unsqueeze(1),
            element_product,
            element_diff,
            element_sum
        ], dim=1)
        
        return interaction_features


class TextMatchModel(nn.Module):
    """完整的文本匹配模型"""
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super(TextMatchModel, self).__init__()
        
        # 共享的孪生编码器
        self.encoder = SiameseEncoder(vocab_size, embed_dim, hidden_dim, num_layers, dropout)
        
        # 交互层
        self.interaction = InteractionLayer(hidden_dim)
        
        # 分类层
        interaction_dim = hidden_dim * 3 + 1
        self.classifier = nn.Sequential(
            nn.Linear(interaction_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.uniform_(module.weight, -0.1, 0.1)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
    
    def forward(self, query1, query2):
        """
        query1, query2: [batch_size, seq_len]
        return: [batch_size, 1]
        """
        # 编码两个句子
        repr1, _ = self.encoder(query1)
        repr2, _ = self.encoder(query2)
        
        # 计算交互特征
        interaction_features = self.interaction(repr1, repr2)
        
        # 分类
        logits = self.classifier(interaction_features)
        
        return logits.squeeze(-1)  # [batch_size]


def create_model():
    """创建模型实例"""
    model = TextMatchModel(
        vocab_size=config.VOCAB_SIZE,
        embed_dim=config.EMBED_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT
    )
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n🏗️  模型创建完成:")
    print(f"   总参数量: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")
    
    return model


if __name__ == '__main__':
    # 测试模型
    model = create_model()
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
