"""
Simplified text matching model without BERT.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config


def masked_mean_pool(x, mask):
    mask = mask.unsqueeze(-1).type_as(x)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return (x * mask).sum(dim=1) / denom


class SimpleTextMatchModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.input_dropout = nn.Dropout(dropout)
        self.encoder = nn.LSTM(
            embed_dim,
            hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_dropout = nn.Dropout(dropout)

        repr_dim = hidden_dim
        interaction_dim = repr_dim * 4
        self.classifier = nn.Sequential(
            nn.Linear(interaction_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.uniform_(module.weight, -0.1, 0.1)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def _encode(self, input_ids):
        mask = (input_ids != 0).float()
        embeddings = self.embedding(input_ids)
        embeddings = self.input_dropout(embeddings)
        outputs, _ = self.encoder(embeddings)
        outputs = self.output_dropout(outputs)
        repr_vec = masked_mean_pool(outputs, mask)
        return repr_vec

    def forward(self, query1, query2):
        repr1 = self._encode(query1)
        repr2 = self._encode(query2)

        element_product = repr1 * repr2
        element_diff = torch.abs(repr1 - repr2)
        interaction = torch.cat([repr1, repr2, element_product, element_diff], dim=1)
        logits = self.classifier(interaction).squeeze(-1)
        return logits


def create_simple_model():
    model = SimpleTextMatchModel(
        vocab_size=config.VOCAB_SIZE,
        embed_dim=config.EMBED_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n🏗️  简化模型创建完成:")
    print(f"   总参数量: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")
    print("   特性: BiLSTM + mean pooling + 简单交互特征")

    return model


if __name__ == "__main__":
    model = create_simple_model()
    model.to(config.DEVICE)

    batch_size = 4
    seq_len = config.MAX_LEN
    query1 = torch.randint(1, 1000, (batch_size, seq_len)).to(config.DEVICE)
    query2 = torch.randint(1, 1000, (batch_size, seq_len)).to(config.DEVICE)

    logits = model(query1, query2)
    print(f"\n🔍 模型测试:")
    print(f"   Logits shape: {logits.shape}")
