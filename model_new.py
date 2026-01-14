"""
New text matching model with matching matrix CNN, multi-layer fusion,
BERT->BiLSTM encoding, gated feature fusion, and auxiliary contrastive losses.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config
from features import FeatureExtractor


def masked_mean_pool(x, mask):
    mask = mask.unsqueeze(-1).type_as(x)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return (x * mask).sum(dim=1) / denom


def masked_max_pool(x, mask):
    mask = mask.unsqueeze(-1).bool()
    neg_inf = torch.finfo(x.dtype).min
    x = x.masked_fill(~mask, neg_inf)
    return x.max(dim=1).values


class AttentionPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x, mask):
        scores = self.attn(x).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1)
        return (weights.unsqueeze(-1) * x).sum(dim=1)


class StackedBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.layers.append(
                nn.LSTM(
                    in_dim,
                    hidden_dim // 2,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                    dropout=0.0,
                )
            )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        outputs = []
        out = x
        for lstm in self.layers:
            out, _ = lstm(out)
            out = self.dropout(out)
            outputs.append(out)
        return outputs


class MultiLayerFusion(nn.Module):
    def __init__(self, hidden_dim, num_layers, repr_dim, dropout):
        super().__init__()
        self.attn_pool = AttentionPool(hidden_dim)
        concat_dim = num_layers * hidden_dim * 3
        self.proj = nn.Linear(concat_dim, repr_dim)
        self.residual_proj = nn.Linear(hidden_dim, repr_dim)
        self.layer_norm = nn.LayerNorm(repr_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, layer_outputs, mask):
        pooled = []
        for out in layer_outputs:
            mean_pool = masked_mean_pool(out, mask)
            max_pool = masked_max_pool(out, mask)
            attn_pool = self.attn_pool(out, mask)
            pooled.append(torch.cat([mean_pool, max_pool, attn_pool], dim=1))

        concat = torch.cat(pooled, dim=1)
        proj = self.dropout(self.proj(concat))
        residual = self.residual_proj(masked_mean_pool(layer_outputs[-1], mask))
        return self.layer_norm(proj + residual)


class MatchingMatrixCNN(nn.Module):
    def __init__(self, hidden_dim, out_dim=64, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.proj = nn.Linear(16, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, out1, out2, mask1, mask2):
        norm1 = F.normalize(out1, p=2, dim=-1)
        norm2 = F.normalize(out2, p=2, dim=-1)
        sim = torch.bmm(norm1, norm2.transpose(1, 2))
        if mask1 is not None and mask2 is not None:
            sim = sim * (mask1.unsqueeze(2) * mask2.unsqueeze(1))

        x = sim.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.adaptive_max_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.proj(x)
        return self.layer_norm(x)


class ResidualMLP(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or dim * 2
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        return self.layer_norm(x + out)


class InteractionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, repr1, repr2):
        cos_sim = F.cosine_similarity(repr1, repr2, dim=1, eps=1e-8)
        element_product = repr1 * repr2
        element_diff = torch.abs(repr1 - repr2)
        element_sum = repr1 + repr2
        return torch.cat(
            [cos_sim.unsqueeze(1), element_product, element_diff, element_sum],
            dim=1,
        )


class BertEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, dropout, bert_model=None, bert_hidden_size=None):
        super().__init__()
        self.bert_model = bert_model
        if bert_model is None:
            self.bert_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.output_dim = embed_dim
        else:
            if bert_hidden_size is None:
                bert_hidden_size = bert_model.config.hidden_size
            self.output_dim = bert_hidden_size
        self.layer_norm = nn.LayerNorm(self.output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        if self.bert_model is None:
            outputs = self.bert_embeddings(input_ids)
        else:
            outputs = self.bert_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            ).last_hidden_state
        outputs = self.layer_norm(outputs)
        return self.dropout(outputs)


class NewTextMatchModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        num_layers,
        dropout,
        feature_dim=19,
        use_bert=False,
        bert_model=None,
        bert_hidden_size=None,
    ):
        super().__init__()
        self.use_bert = use_bert

        if use_bert:
            self.bert = BertEncoder(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                dropout=dropout,
                bert_model=bert_model,
                bert_hidden_size=bert_hidden_size,
            )
            encoder_input_dim = self.bert.output_dim
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            encoder_input_dim = embed_dim

        self.input_dropout = nn.Dropout(dropout)
        self.encoder = StackedBiLSTM(
            input_dim=encoder_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.fusion = MultiLayerFusion(hidden_dim, num_layers, hidden_dim, dropout)
        self.interaction = InteractionLayer(hidden_dim)
        self.match_cnn = MatchingMatrixCNN(hidden_dim, out_dim=64, dropout=dropout)

        interaction_dim = hidden_dim * 3 + 1
        deep_input_dim = interaction_dim + 64 + hidden_dim * 2
        self.deep_proj = nn.Sequential(
            nn.Linear(deep_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.feature_extractor = FeatureExtractor()
        self.feature_processor = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim)

        self.pre_classifier = ResidualMLP(hidden_dim, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

        self.register_buffer('feature_mean', None)
        self.register_buffer('feature_std', None)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.uniform_(module.weight, -0.1, 0.1)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def _standardize_features(self, features):
        if self.training:
            if self.feature_mean is None or self.feature_std is None:
                self.feature_mean = features.mean(dim=0, keepdim=True)
                self.feature_std = features.std(dim=0, keepdim=True)
                self.feature_std = torch.where(
                    self.feature_std == 0,
                    torch.ones_like(self.feature_std),
                    self.feature_std,
                )
            else:
                momentum = 0.1
                batch_mean = features.mean(dim=0, keepdim=True)
                batch_std = features.std(dim=0, keepdim=True)
                self.feature_mean = (1 - momentum) * self.feature_mean + momentum * batch_mean
                self.feature_std = (1 - momentum) * self.feature_std + momentum * batch_std

        if self.feature_mean is not None and self.feature_std is not None:
            features = (features - self.feature_mean) / self.feature_std
        return features

    def _encode(self, input_ids):
        mask = (input_ids != 0).float()
        if self.use_bert:
            embeddings = self.bert(input_ids, attention_mask=mask)
        else:
            embeddings = self.embedding(input_ids)
        embeddings = self.input_dropout(embeddings)
        layer_outputs = self.encoder(embeddings)
        repr_vec = self.fusion(layer_outputs, mask)
        return layer_outputs, repr_vec, mask

    def forward(self, query1, query2, handcrafted_features=None, return_reprs=False):
        layers1, repr1, mask1 = self._encode(query1)
        layers2, repr2, mask2 = self._encode(query2)

        interaction = self.interaction(repr1, repr2)
        match_features = self.match_cnn(layers1[-1], layers2[-1], mask1, mask2)

        deep_features = torch.cat([repr1, repr2, interaction, match_features], dim=1)
        deep_features = self.deep_proj(deep_features)

        if handcrafted_features is None:
            handcrafted_features = self.feature_extractor.extract_batch_features(
                query1, query2
            ).to(query1.device)
        handcrafted_features = self._standardize_features(handcrafted_features)
        hc_features = self.feature_processor(handcrafted_features)

        gate = self.gate(torch.cat([deep_features, hc_features], dim=1))
        fused = self.fusion_norm(deep_features + gate * hc_features)
        fused = self.pre_classifier(fused)

        logits = self.classifier(fused).squeeze(-1)
        if return_reprs:
            return logits, repr1, repr2
        return logits

    @staticmethod
    def contrastive_loss(repr1, repr2, labels, margin=1.0):
        distance = F.pairwise_distance(repr1, repr2, p=2)
        loss_pos = labels * distance.pow(2)
        loss_neg = (1 - labels) * F.relu(margin - distance).pow(2)
        return (loss_pos + loss_neg).mean()

    @staticmethod
    def triplet_loss(anchor, positive, negative, margin=1.0):
        pos_distance = F.pairwise_distance(anchor, positive, p=2)
        neg_distance = F.pairwise_distance(anchor, negative, p=2)
        return F.relu(pos_distance - neg_distance + margin).mean()


def create_new_model(use_bert=False, bert_model=None, bert_hidden_size=None):
    model = NewTextMatchModel(
        vocab_size=config.VOCAB_SIZE,
        embed_dim=config.EMBED_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        feature_dim=19,
        use_bert=use_bert,
        bert_model=bert_model,
        bert_hidden_size=bert_hidden_size,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n🏗️  新模型创建完成:")
    print(f"   总参数量: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")
    print("   特性: Matching Matrix CNN + 多层融合 + gating特征融合")
    print("   特性: BERT->BiLSTM(可选) + Residual/LayerNorm + 对比学习接口")

    return model


if __name__ == '__main__':
    model = create_new_model()
    model.to(config.DEVICE)

    batch_size = 4
    seq_len = config.MAX_LEN
    query1 = torch.randint(1, 1000, (batch_size, seq_len)).to(config.DEVICE)
    query2 = torch.randint(1, 1000, (batch_size, seq_len)).to(config.DEVICE)

    logits, repr1, repr2 = model(query1, query2, return_reprs=True)
    print(f"\n🔍 模型测试:")
    print(f"   Logits shape: {logits.shape}")
    print(f"   Repr1 shape: {repr1.shape}")
    print(f"   Repr2 shape: {repr2.shape}")
