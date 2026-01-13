"""
带注意力机制的文本匹配模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config


class AttentionLayer(nn.Module):
    """自注意力层"""
    
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, lstm_output, mask=None):
        """
        Args:
            lstm_output: [batch_size, seq_len, hidden_dim]
            mask: [batch_size, seq_len] padding mask
        
        Returns:
            attended: [batch_size, hidden_dim]
            attention_weights: [batch_size, seq_len, 1]
        """
        # 计算注意力分数
        attention_scores = self.attention(lstm_output)  # [batch, seq_len, 1]
        
        # 应用mask（padding位置设为极小值）
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        
        # Softmax归一化
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # 加权求和
        attended = torch.sum(attention_weights * lstm_output, dim=1)
        
        return attended, attention_weights


class SiameseEncoderWithAttention(nn.Module):
    """带注意力的孪生编码器"""
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super(SiameseEncoderWithAttention, self).__init__()
        
        # Embedding层
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # BiLSTM层
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 注意力层
        self.attention = AttentionLayer(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        x: [batch_size, seq_len]
        return: sentence_repr [batch_size, hidden_dim], attention_weights
        """
        # 创建padding mask
        mask = (x != 0).float()  # [batch_size, seq_len]
        
        # Embedding
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.dropout(lstm_out)
        
        # 注意力池化
        sentence_repr, attention_weights = self.attention(lstm_out, mask)
        
        return sentence_repr, attention_weights


class InteractionLayer(nn.Module):
    """交互层"""
    
    def __init__(self, hidden_dim):
        super(InteractionLayer, self).__init__()
    
    def forward(self, repr1, repr2):
        # Cosine similarity
        cos_sim = F.cosine_similarity(repr1, repr2, dim=1, eps=1e-8)
        
        # Element-wise operations
        element_product = repr1 * repr2
        element_diff = torch.abs(repr1 - repr2)
        element_sum = repr1 + repr2
        
        # 拼接所有特征
        interaction_features = torch.cat([
            cos_sim.unsqueeze(1),
            element_product,
            element_diff,
            element_sum
        ], dim=1)
        
        return interaction_features


class TextMatchModelWithAttention(nn.Module):
    """带注意力机制的文本匹配模型"""
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super(TextMatchModelWithAttention, self).__init__()
        
        # 共享的孪生编码器（带注意力）
        self.encoder = SiameseEncoderWithAttention(
            vocab_size, embed_dim, hidden_dim, num_layers, dropout
        )
        
        # 交互层
        self.interaction = InteractionLayer(hidden_dim)
        
        # 分类层
        interaction_dim = hidden_dim * 3 + 1
        self.classifier = nn.Sequential(
            nn.Linear(interaction_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
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
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.uniform_(module.weight, -0.1, 0.1)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
    
    def forward(self, query1, query2, return_attention=False):
        """
        query1, query2: [batch_size, seq_len]
        return_attention: 是否返回注意力权重
        """
        # 编码两个句子
        repr1, attn1 = self.encoder(query1)
        repr2, attn2 = self.encoder(query2)
        
        # 计算交互特征
        interaction_features = self.interaction(repr1, repr2)
        
        # 分类
        logits = self.classifier(interaction_features)
        
        if return_attention:
            return logits.squeeze(-1), attn1, attn2
        else:
            return logits.squeeze(-1)


def create_attention_model():
    """创建带注意力的模型实例"""
    model = TextMatchModelWithAttention(
        vocab_size=config.VOCAB_SIZE,
        embed_dim=config.EMBED_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT
    )
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n🏗️  注意力模型创建完成:")
    print(f"   总参数量: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")
    print(f"   新增: 注意力机制")
    
    return model


if __name__ == '__main__':
    # 测试模型
    model = create_attention_model()
    model.to(config.DEVICE)
    
    # 测试前向传播
    batch_size = 4
    seq_len = 60
    query1 = torch.randint(1, 1000, (batch_size, seq_len)).to(config.DEVICE)
    query2 = torch.randint(1, 1000, (batch_size, seq_len)).to(config.DEVICE)
    
    # 不返回注意力
    outputs = model(query1, query2)
    print(f"\n🔍 模型测试:")
    print(f"   Input shape: [{batch_size}, {seq_len}]")
    print(f"   Output shape: {outputs.shape}")
    print(f"   Sample outputs: {outputs[:3]}")
    
    # 返回注意力权重
    outputs, attn1, attn2 = model(query1, query2, return_attention=True)
    print(f"\n📊 注意力权重:")
    print(f"   Attention1 shape: {attn1.shape}")
    print(f"   Attention2 shape: {attn2.shape}")
