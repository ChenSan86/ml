# 📊 数据划分说明

## 🎯 统一数据划分策略

**为了公平对比所有模型**，本项目中**所有基线和改进模型**都使用完全相同的数据划分方式。

---

## 📦 数据来源

### 原始数据
```
data/gaiic_track3_round1_train_20210228.tsv  →  100,000 条 (初赛数据)
data/gaiic_track3_round2_train_20210407.tsv  →  300,000 条 (复赛数据)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
合并后总数据                                    400,000 条
```

### 标签分布
- **正样本** (label=1)：149,957 条 (37.49%)
- **负样本** (label=0)：250,043 条 (62.51%)

---

## ✂️ 数据划分方式

### 划分比例：**90 : 10**

```python
训练集: 360,000 条 (90%)
测试集:  40,000 条 (10%)
```

### 关键配置

在 `config.py` 中：
```python
VAL_SPLIT = 0.1      # 测试集比例 10%
SEED = 42            # 固定随机种子，确保可复现
```

在 `dataset.py` 的 `prepare_data()` 函数中：
```python
train_df, val_df = train_test_split(
    df, 
    test_size=config.VAL_SPLIT,      # 0.1 = 10%
    random_state=config.SEED,         # 42 (固定)
    stratify=df['label']              # 保持标签分布一致
)
```

---

## 🔒 一致性保证

### ✅ 所有模型使用相同划分

以下**所有训练脚本**都调用相同的 `prepare_data()` 函数：

| 脚本 | 模型 | 数据划分 |
|------|------|---------|
| `train.py` | 基线模型 | ✅ 统一划分 |
| `train_attention.py` | 注意力模型 | ✅ 统一划分 |
| `train_enhanced.py` | 特征工程模型 | ✅ 统一划分 |
| `train_focal.py` | Focal Loss模型 | ✅ 统一划分 |
| `train_augmented.py` | 数据增强模型 | ✅ 统一划分 |
| `train_contrastive.py` | 对比学习模型 | ✅ 统一划分 |

### ✅ 固定随机种子

所有训练脚本在开始时都调用：
```python
set_seed(config.SEED)  # SEED = 42
```

这确保：
- ✅ 数据划分完全一致
- ✅ 模型初始化一致
- ✅ Dropout等随机操作可复现
- ✅ 结果可以公平对比

---

## 📊 划分后的数据分布

### 训练集 (360,000 条)
- 正样本: ~134,964 条 (37.49%)
- 负样本: ~225,036 条 (62.51%)

### 测试集 (40,000 条)
- 正样本: ~14,993 条 (37.49%)
- 负样本: ~25,007 条 (62.51%)

**注意**：使用 `stratify=df['label']` 确保训练集和测试集的标签分布与原始数据一致。

---

## 🔍 验证数据划分

运行任何训练脚本，都会看到：
```
✅ 加载数据完成: 400,000 条样本
   正样本: 149,957 (37.49%)
   负样本: 250,043

📊 数据划分:
   训练集: 360,000 条
   验证集: 40,000 条
```

---

## 📝 代码实现

### 完整数据加载流程

```python
# 1. 加载并合并数据
def load_data(file_paths):
    dfs = []
    for file_path in file_paths:
        df = pd.read_csv(file_path, sep='\t', header=None, 
                        names=['query1', 'query2', 'label'])
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

# 2. 统一划分数据
def prepare_data():
    # 加载两个文件
    df = load_data([config.TRAIN_FILE_1, config.TRAIN_FILE_2])
    
    # 90:10 划分，保持标签分布
    train_df, val_df = train_test_split(
        df, 
        test_size=0.1,              # 10%作为测试集
        random_state=42,            # 固定种子
        stratify=df['label']        # 分层采样
    )
    
    return train_df, val_df

# 3. 所有训练脚本统一调用
train_df, val_df = prepare_data()
```

---

## 🎯 消融实验的公平性

由于所有模型使用：
- ✅ 相同的训练集 (360,000 条)
- ✅ 相同的测试集 (40,000 条)
- ✅ 相同的随机种子 (42)

因此，各模型的性能对比是**完全公平**的，可以准确反映每个改进方向的贡献。

---

## 📈 示例：查看数据划分

```python
from dataset import prepare_data

# 获取数据
train_df, val_df = prepare_data()

# 查看统计
print(f"训练集大小: {len(train_df)}")
print(f"测试集大小: {len(val_df)}")
print(f"训练集正样本比例: {train_df['label'].mean():.2%}")
print(f"测试集正样本比例: {val_df['label'].mean():.2%}")
```

输出：
```
训练集大小: 360000
测试集大小: 40000
训练集正样本比例: 37.49%
测试集正样本比例: 37.49%
```

---

## ⚠️ K折交叉验证的特殊情况

`train_kfold.py` 使用不同的划分策略：

```python
# K折交叉验证：将全部40万数据分成K折
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 每折轮流作为验证集
# 例如5折: 每折 32万训练，8万验证
```

**这是合理的**，因为K折交叉验证本身就是一种更稳健的评估方法，不需要固定测试集。

---

## 💡 修改数据划分比例

如果需要修改测试集比例，只需修改 `config.py`：

```python
VAL_SPLIT = 0.2  # 改为 80:20
VAL_SPLIT = 0.15 # 改为 85:15
```

**所有模型会自动使用新的划分比例**。

---

## ✅ 总结

| 项目 | 值 |
|------|-----|
| **总数据量** | 400,000 条 |
| **训练集** | 360,000 条 (90%) |
| **测试集** | 40,000 条 (10%) |
| **随机种子** | 42 (固定) |
| **标签分布** | 保持一致 (stratify) |
| **所有模型** | 使用相同划分 ✅ |

**结论**：本项目的实验设计确保了所有模型在相同的数据上进行训练和评估，对比结果完全公平可靠。

---

**需要修改数据划分？** 编辑 `config.py` 中的 `VAL_SPLIT` 参数即可。
