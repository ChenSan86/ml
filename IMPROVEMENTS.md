# 📈 模型改进方案汇总

## 🎯 当前状态

**已实现**：
- ✅ BiLSTM双塔模型（AUC 0.9718）
- ✅ 特征工程（19个手工特征）
- ✅ K折交叉验证
- ✅ 数据增强（通过dropout）

**性能**：
- 验证集AUC：0.9718
- 准确率：0.9252
- F1分数：0.9029

---

## 🚀 改进方向总览

| 改进方向 | 难度 | 预期提升 | 实现时间 |
|---------|------|---------|---------|
| 1. 注意力机制 | ⭐⭐ | +0.5~1% | 2小时 |
| 2. 对比学习 | ⭐⭐⭐ | +1~2% | 4小时 |
| 3. 数据增强 | ⭐⭐ | +0.5~1% | 3小时 |
| 4. Transformer架构 | ⭐⭐⭐⭐ | +2~3% | 6小时 |
| 5. 模型集成 | ⭐ | +0.5~1% | 1小时 |
| 6. 难样本挖掘 | ⭐⭐ | +0.5~1% | 2小时 |
| 7. 损失函数优化 | ⭐⭐ | +0.3~0.5% | 1小时 |
| 8. 预训练词向量 | ⭐⭐⭐ | +1~2% | 4小时 |

---

## 1️⃣ 注意力机制 ⭐⭐ (推荐)

### 为什么需要？
当前模型使用BiLSTM的最后状态，可能丢失重要信息。注意力机制可以关注关键词。

### 实现方案

```python
class AttentionLayer(nn.Module):
    """自注意力层"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    
    def forward(self, lstm_output):
        # lstm_output: [batch, seq_len, hidden_dim]
        weights = torch.softmax(self.attention(lstm_output), dim=1)
        attended = torch.sum(weights * lstm_output, dim=1)
        return attended

# 在encoder中使用
repr = self.attention(lstm_out)  # 替代取最后时刻
```

### 预期效果
- AUC: 0.9718 → **0.9730+**
- 更好地捕捉关键词信息

---

## 2️⃣ 对比学习 (Contrastive Learning) ⭐⭐⭐

### 原理
让相似的样本在embedding空间靠近，不相似的样本远离。

### 实现方案

```python
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, repr1, repr2, labels):
        # 欧氏距离
        distance = F.pairwise_distance(repr1, repr2)
        
        # 对比损失
        loss = labels * distance.pow(2) + \
               (1 - labels) * F.relu(self.margin - distance).pow(2)
        return loss.mean()

# 结合BCE和对比损失
total_loss = bce_loss + 0.2 * contrastive_loss
```

### 预期效果
- AUC: 0.9718 → **0.9740+**
- 更好的语义表示空间

---

## 3️⃣ 数据增强 ⭐⭐

### 当前问题
训练数据固定，容易过拟合。

### 增强策略

#### 方案A：词级增强
```python
def word_dropout(ids, p=0.1):
    """随机删除词"""
    mask = torch.rand(len(ids)) > p
    return ids[mask]

def word_shuffle(ids, k=3):
    """局部打乱"""
    # 在k范围内随机交换
    for i in range(len(ids) - k):
        if random.random() < 0.5:
            j = random.randint(i, min(i+k, len(ids)-1))
            ids[i], ids[j] = ids[j], ids[i]
    return ids
```

#### 方案B：回译（需要外部模型）
```python
# 正样本对：互换query1和query2
if label == 1:
    augmented_samples.append((query2, query1, 1))
```

#### 方案C：负样本生成
```python
# 随机配对生成困难负样本
def create_hard_negatives(df):
    pos_samples = df[df['label'] == 1]
    # 随机打乱query2
    shuffled = pos_samples.sample(frac=1)
    hard_negs = pd.DataFrame({
        'query1': pos_samples['query1'].values,
        'query2': shuffled['query2'].values,
        'label': 0
    })
    return hard_negs
```

### 预期效果
- AUC: 0.9718 → **0.9735+**
- 更好的泛化能力

---

## 4️⃣ Transformer架构 ⭐⭐⭐⭐

### 为什么更好？
- 并行计算，更快
- 长距离依赖建模
- 预训练模型可用

### 实现方案

```python
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads=8, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.pos_encoding(embedded)
        output = self.transformer(embedded)
        return output.mean(dim=1)  # 平均池化
```

### 预期效果
- AUC: 0.9718 → **0.9750+**
- 训练速度提升2-3倍

---

## 5️⃣ 模型集成 ⭐ (最简单，效果好)

### 方案A：多模型投票
```python
# 训练多个不同的模型
models = [
    BiLSTM_model,
    Transformer_model,
    Enhanced_model
]

# 预测时平均
probs = np.mean([model.predict(x) for model in models], axis=0)
```

### 方案B：Stacking
```python
# 第一层：多个基模型
base_models = [model1, model2, model3]
base_preds = [m.predict(X_val) for m in base_models]

# 第二层：元学习器
meta_model = LogisticRegression()
meta_model.fit(np.column_stack(base_preds), y_val)
```

### 预期效果
- AUC: 0.9718 → **0.9730+**
- 稳定性提升

---

## 6️⃣ 难样本挖掘 (Hard Negative Mining) ⭐⭐

### 原理
重点学习模型容易出错的样本。

### 实现方案

```python
class FocalLoss(nn.Module):
    """Focal Loss: 对难样本赋予更高权重"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        loss = self.alpha * focal_weight * bce_loss
        return loss.mean()

# 使用Focal Loss替代BCE
criterion = FocalLoss()
```

### 预期效果
- AUC: 0.9718 → **0.9728+**
- 减少边界样本错误

---

## 7️⃣ 损失函数优化 ⭐⭐

### 方案A：多任务学习
```python
# 同时预测多个目标
loss = bce_loss + 0.3 * ranking_loss + 0.2 * triplet_loss
```

### 方案B：标签平滑
```python
# 避免过度自信
def label_smoothing(labels, epsilon=0.1):
    return labels * (1 - epsilon) + 0.5 * epsilon
```

### 方案C：类别权重
```python
# 处理类别不平衡
pos_weight = (negative_count / positive_count)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

### 预期效果
- AUC: 0.9718 → **0.9725+**

---

## 8️⃣ 预训练词向量 ⭐⭐⭐

### 方案A：Word2Vec预训练
```python
# 在当前数据上预训练
from gensim.models import Word2Vec

# 准备语料
corpus = []
for text in all_texts:
    corpus.append([str(id) for id in text.split()])

# 训练Word2Vec
model = Word2Vec(corpus, vector_size=300, window=5, min_count=1)

# 初始化embedding
pretrained_weights = np.zeros((vocab_size, 300))
for word_id in range(vocab_size):
    if str(word_id) in model.wv:
        pretrained_weights[word_id] = model.wv[str(word_id)]

# 加载到模型
model.embedding.weight.data.copy_(torch.from_numpy(pretrained_weights))
```

### 方案B：自监督预训练
```python
# Masked Language Model
def mlm_pretrain(texts):
    for text in texts:
        # 随机mask 15%的词
        masked_text, targets = mask_tokens(text)
        # 预测被mask的词
        loss = criterion(model(masked_text), targets)
```

### 预期效果
- AUC: 0.9718 → **0.9740+**
- 更好的词表示

---

## 9️⃣ 训练优化

### A. 混合精度训练
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    logits = model(query1, query2)
    loss = criterion(logits, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```
**效果**：训练速度提升30-50%，显存占用减半

### B. 梯度累积
```python
accumulation_steps = 4

for i, batch in enumerate(train_loader):
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```
**效果**：等效更大batch size，更稳定

### C. 学习率策略
```python
# Warmup + Cosine
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)
```

---

## 🔟 特征工程增强

### 当前19个特征基础上，可以添加：

```python
# 1. TF-IDF特征
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=1000)
tfidf_features = tfidf.fit_transform(texts)

# 2. N-gram特征
def get_ngrams(text, n=2):
    ngrams = zip(*[text[i:] for i in range(n)])
    return [' '.join(gram) for gram in ngrams]

# 3. 模糊匹配分数
from fuzzywuzzy import fuzz

fuzzy_ratio = fuzz.ratio(query1, query2)
fuzzy_partial = fuzz.partial_ratio(query1, query2)

# 4. 序列模式特征
def longest_common_subsequence(s1, s2):
    # LCS长度
    pass

# 5. 位置特征
first_common_pos = # 第一个公共词的位置
last_common_pos = # 最后一个公共词的位置
```

**新增特征数**：+10个  
**预期提升**：+0.3~0.5%

---

## 🎨 可视化与分析

### A. 注意力可视化
```python
import seaborn as sns

def visualize_attention(attention_weights, tokens):
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights, xticklabels=tokens, 
                yticklabels=tokens, cmap='YlOrRd')
    plt.show()
```

### B. 错误案例分析
```python
def analyze_errors(model, val_loader):
    errors = []
    for batch in val_loader:
        preds = model(batch)
        wrong_indices = (preds != batch['labels'])
        errors.extend(batch[wrong_indices])
    
    # 分析错误模式
    print("错误样本特征分析:")
    print(f"平均长度: {np.mean([len(e) for e in errors])}")
    print(f"平均重叠度: ...")
```

### C. 特征重要性
```python
from sklearn.ensemble import RandomForestClassifier

# 用随机森林评估特征重要性
rf = RandomForestClassifier()
rf.fit(handcrafted_features, labels)

importances = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
```

---

## 📊 实施优先级建议

### 🔥 高优先级（快速见效）
1. **模型集成** (1小时) → +0.5~1%
2. **注意力机制** (2小时) → +0.5~1%
3. **Focal Loss** (1小时) → +0.3~0.5%

### ⭐ 中优先级（中等投入）
4. **数据增强** (3小时) → +0.5~1%
5. **对比学习** (4小时) → +1~2%
6. **难样本挖掘** (2小时) → +0.5~1%

### 🎯 长期优化（大幅改进）
7. **Transformer架构** (6小时) → +2~3%
8. **预训练词向量** (4小时) → +1~2%

---

## 🚀 快速实施方案

### 方案1：保守型（2小时，+1~1.5%）
```bash
1. 添加注意力机制
2. 使用Focal Loss
3. K折集成（已有代码）
```

### 方案2：进取型（8小时，+2~3%）
```bash
1. 实现Transformer模型
2. 添加对比学习
3. 数据增强
4. 模型集成
```

### 方案3：全面优化（20小时，+3~5%）
```bash
1. Transformer + 注意力
2. 预训练词向量
3. 对比学习 + Focal Loss
4. 完整数据增强
5. 多模型Stacking集成
6. 特征工程扩展
```

---

## 📝 总结

| 当前性能 | 快速优化后 | 深度优化后 | 理论上限 |
|---------|-----------|-----------|---------|
| **0.9718** | **0.9735** | **0.9760** | **0.9800+** |

**建议路线**：
1. 先实现简单的集成和注意力（2小时）
2. 如果时间充裕，尝试Transformer（6小时）
3. 最后考虑预训练和对比学习（4-8小时）

---

**需要我帮你实现哪个改进方向吗？**
