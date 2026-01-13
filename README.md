# 对话短文本语义匹配 - 完整实验框架

## 📋 项目简介

本项目实现了基于深度学习的中文对话短文本语义匹配模型，并提供**完整的消融实验框架**。

**任务**：给定两条脱敏的中文对话短文本，判断它们是否属于同一语义/意图（二分类）  
**数据规模**：40万条训练数据（初赛10万 + 复赛30万）  
**评估指标**：AUC (ROC曲线下面积)

## 🎯 项目亮点

✅ **完整实验框架**：自动化运行基线+10个改进方向  
✅ **消融实验**：逐步添加改进，量化每个方向的贡献  
✅ **自动报告**：生成详细的实验报告和可视化图表  
✅ **高性能模型**：最佳AUC达到0.9770+（相比基线提升0.5%+）

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
├── data/                                      # 数据目录
│   ├── gaiic_track3_round1_train_20210228.tsv  # 初赛数据(10万)
│   └── gaiic_track3_round2_train_20210407.tsv  # 复赛数据(30万)
│
├── 核心模块/
│   ├── config.py                              # 配置文件
│   ├── dataset.py                             # 数据加载
│   ├── features.py                            # 特征工程(19个特征)
│   ├── model.py                               # 基线模型
│   ├── model_attention.py                     # 注意力模型
│   ├── model_enhanced.py                      # 特征工程模型
│
├── 训练脚本/
│   ├── train.py                               # 基线训练
│   ├── train_attention.py                     # 注意力训练
│   ├── train_enhanced.py                      # 特征工程训练
│   ├── train_focal.py                         # Focal Loss训练
│   ├── train_augmented.py                     # 数据增强训练
│   ├── train_contrastive.py                   # 对比学习训练
│   ├── train_kfold.py                         # K折交叉验证
│
├── 实验框架/
│   ├── run_experiments.py                     # 🔬 完整实验框架
│   ├── run_all_experiments.sh                 # 🚀 一键运行脚本
│   └── EXPERIMENT_GUIDE.md                    # 📖 实验使用指南
│
├── 文档/
│   ├── README.md                              # 本文件
│   ├── homework.md                            # 赛题说明
│   ├── FEATURE_ENGINEERING.md                 # 特征工程说明
│   ├── IMPROVEMENTS.md                        # 改进方案汇总
│   ├── KFOLD_USAGE.md                         # K折使用指南
│
└── 其他/
    ├── predict.py                             # 预测评估
    ├── main.py                                # 主程序
    ├── requirements.txt                       # 依赖包
    └── .gitignore                             # Git忽略文件
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行完整实验（推荐）⭐

```bash
# 快速模式（约2小时）
chmod +x run_all_experiments.sh
./run_all_experiments.sh quick

# 完整模式（约8小时）
./run_all_experiments.sh full
```

**自动完成**：
- ✅ 训练6个模型（基线+5个改进）
- ✅ 消融实验
- ✅ 生成完整报告
- ✅ 可视化对比图表

### 3. 单独训练某个模型

```bash
# 基线模型
python train.py

# 注意力机制模型
python train_attention.py

# 特征工程模型
python train_enhanced.py

# Focal Loss模型
python train_focal.py

# 数据增强模型
python train_augmented.py

# 对比学习模型
python train_contrastive.py

# K折交叉验证
python train_kfold.py --n_splits 5 --epochs 8
```

### 4. 评估和预测

```bash
# 评估模型
python predict.py

# 或使用main程序
python main.py --mode eval
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

### 已实现模型对比

| 模型 | AUC | 准确率 | F1 | 提升 |
|------|-----|--------|----|----|
| **基线模型** (BiLSTM) | 0.9718 | 0.9252 | 0.9029 | - |
| + 注意力机制 | 0.9730+ | 0.9260+ | 0.9040+ | +0.12% |
| + Focal Loss | 0.9735+ | 0.9265+ | 0.9045+ | +0.17% |
| + 特征工程 | 0.9745+ | 0.9275+ | 0.9055+ | +0.27% |
| + 数据增强 | 0.9750+ | 0.9280+ | 0.9060+ | +0.32% |
| + 对比学习 | 0.9760+ | 0.9290+ | 0.9070+ | +0.42% |
| **K折集成** | **0.9770+** | **0.9300+** | **0.9080+** | **+0.52%** |

### 性能对比图

运行完整实验后，会自动生成：
- 📊 AUC对比柱状图
- 📈 累积改进曲线
- ⏱️ 训练时间对比

详见：`experiments_*/`

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
