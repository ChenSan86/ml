# 对话短文本语义匹配 - 深度学习解决方案

## 📋 项目简介

本项目实现了基于深度学习的中文对话短文本语义匹配模型，用于判断两条对话文本是否表达相同的语义/意图。

**任务**：给定两条脱敏的中文对话短文本，判断它们是否属于同一语义/意图（二分类）  
**数据规模**：40万条训练数据（初赛10万 + 复赛30万）  
**评估指标**：AUC (ROC曲线下面积)

## 🏗️ 模型架构

采用 **双塔架构 + 多层交互** 的深度学习模型：

```
输入层
  ↓
Embedding层 (可训练)
  ↓
双向LSTM编码器 (参数共享)
  ↓
句子表示向量
  ↓
交互层 (Cosine相似度, Element-wise操作)
  ↓
多层全连接网络
  ↓
二分类输出
```

### 核心特点

1. **孪生编码器**：两个输入句子共享同一个BiLSTM编码器
2. **多种交互特征**：
   - Cosine相似度
   - Element-wise乘积
   - Element-wise差值
   - Element-wise求和
3. **深度分类网络**：512 → 256 → 1 的多层感知机

## 📁 项目结构

```
ml2/
├── data/                           # 数据目录
│   ├── gaiic_track3_round1_train_20210228.tsv  # 初赛训练数据
│   └── gaiic_track3_round2_train_20210407.tsv  # 复赛训练数据
├── config.py                       # 配置文件
├── dataset.py                      # 数据加载和预处理
├── model.py                        # 模型定义
├── train.py                        # 训练流程
├── predict.py                      # 预测和评估
├── main.py                         # 主程序入口
├── requirements.txt                # 依赖包
├── homework.md                     # 赛题说明
└── README.md                       # 本文件
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练模型

```bash
# 训练模型
python main.py --mode train

# 或者直接运行训练脚本
python train.py
```

### 3. 评估模型

```bash
# 评估最佳模型
python main.py --mode eval

# 或者指定模型路径
python main.py --mode eval --model best_model.pth
```

### 4. 一键训练+评估

```bash
python main.py --mode all
```

## ⚙️ 模型配置

主要超参数配置（在 `config.py` 中）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| VOCAB_SIZE | 32800 | 词表大小 |
| EMBED_DIM | 300 | 词嵌入维度 |
| HIDDEN_DIM | 512 | LSTM隐藏层维度 |
| NUM_LAYERS | 2 | LSTM层数 |
| DROPOUT | 0.3 | Dropout比例 |
| MAX_LEN | 60 | 最大序列长度 |
| BATCH_SIZE | 256 | 批次大小 |
| LEARNING_RATE | 0.001 | 学习率 |
| EPOCHS | 10 | 训练轮数 |

## 📊 数据说明

### 数据格式

每行一个样本，使用 `\t` 分隔：

```
query1_ids \t query2_ids \t label
```

- `query1_ids`：第一个句子的词ID序列（空格分隔）
- `query2_ids`：第二个句子的词ID序列（空格分隔）
- `label`：0（不匹配）或 1（匹配）

### 数据统计

| 统计项 | 数值 |
|--------|------|
| 总样本数 | 400,000 |
| 正样本数 | 149,957 (37.49%) |
| 负样本数 | 250,043 (62.51%) |
| 词表大小 | 32,766 |
| 平均词数 | 6-7个词 |
| 平均字符数 | 23个字符 |

## 📈 性能表现

模型在验证集上的预期表现：

- **AUC**: > 0.85
- **准确率**: > 0.80
- **F1-Score**: > 0.75

（实际性能取决于训练过程和数据划分）

## 🛠️ 技术栈

- **深度学习框架**: PyTorch 2.0+
- **数据处理**: Pandas, NumPy
- **评估指标**: Scikit-learn
- **可视化**: Matplotlib, Seaborn

## 📝 输出文件

训练和评估过程会生成以下文件：

1. `best_model.pth` - 最佳模型权重
2. `training_history.png` - 训练曲线（Loss和AUC）
3. `confusion_matrix.png` - 混淆矩阵
4. `prediction_distribution.png` - 预测概率分布
5. `predictions.csv` - 详细预测结果

## 🎯 技术亮点

1. **双塔共享参数**：减少参数量，提高泛化能力
2. **多种交互特征**：充分捕捉两个句子之间的关系
3. **BiLSTM编码**：能够捕捉双向上下文信息
4. **梯度裁剪**：防止梯度爆炸，提高训练稳定性
5. **余弦退火学习率**：动态调整学习率，提高收敛效果
6. **完整的评估体系**：AUC、准确率、精确率、召回率、F1等多维度评估

## 🔧 进一步优化建议

1. **数据增强**：
   - 词级别的随机替换
   - 句子回译（Back Translation）

2. **模型优化**：
   - 使用Transformer替代LSTM
   - 引入注意力机制
   - 尝试对比学习方法（如SimCSE）

3. **特征工程**：
   - 添加统计特征（词重叠度、编辑距离等）
   - TF-IDF特征

4. **模型集成**：
   - 多模型投票
   - Stacking

## 📧 联系方式

如有问题或建议，欢迎交流！

---

**祝训练顺利！🎉**
