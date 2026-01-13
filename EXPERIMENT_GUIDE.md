## 🧪 完整实验使用指南

## 📋 实验框架说明

本项目实现了完整的消融实验框架，包括：
1. **基线模型**训练
2. **10个改进方向**逐步添加
3. **自动化消融实验**
4. **完整实验报告**生成

---

## 🚀 快速开始

### 方法1：使用Shell脚本（推荐）⭐

#### 快速模式（每个实验3 epochs，约2小时）
```bash
cd /home/xinguanze/class/ml2
chmod +x run_all_experiments.sh
./run_all_experiments.sh quick
```

#### 完整模式（每个实验10 epochs，约8小时）
```bash
./run_all_experiments.sh full
```

### 方法2：使用Python脚本

#### 快速模式
```bash
python run_experiments.py --quick --output experiments_20260113
```

#### 完整模式
```bash
python run_experiments.py --output experiments_20260113
```

---

## 📊 实验内容

### 实验列表

| ID | 实验名称 | 说明 | 状态 |
|----|---------|------|------|
| 0 | 基线模型 | BiLSTM双塔 | ✅ 已实现 |
| 1 | + 注意力机制 | Self-Attention | ✅ 已实现 |
| 2 | + Focal Loss | 难样本挖掘 | ✅ 已实现 |
| 3 | + 特征工程 | 19个手工特征 | ✅ 已实现 |
| 4 | + 数据增强 | 正样本互换+困难负样本 | ✅ 已实现 |
| 5 | + 对比学习 | Contrastive Loss | ✅ 已实现 |
| 6 | + K折集成 | 5折交叉验证 | ✅ 已实现 |
| 7 | + 标签平滑 | Label Smoothing | ⏳ 待实现 |
| 8 | + 预训练词向量 | Word2Vec | ⏳ 待实现 |
| 9 | + Transformer | 替换LSTM | ⏳ 待实现 |
| 10 | 全部组合 | 最优配置 | ⏳ 待实现 |

---

## 📁 输出文件

实验完成后，在输出目录（如 `experiments_20260113/`）中会生成：

```
experiments_20260113/
├── EXPERIMENT_REPORT.md        # 📄 完整实验报告（Markdown）
├── results.csv                 # 📊 结果数据表
├── results.json                # 🔧 结果JSON格式
├── auc_comparison.png          # 📈 AUC对比柱状图
├── improvement_curve.png       # 📈 改进曲线图
└── time_comparison.png         # ⏱️ 时间开销对比图
```

---

## 📖 查看结果

### 1. 查看总报告
```bash
cat experiments_*/EXPERIMENT_REPORT.md
```

### 2. 查看CSV结果
```bash
cat experiments_*/results.csv
```

### 3. 查看图表
```bash
# 在文件管理器中打开
nautilus experiments_*/ 

# 或使用图片查看器
eog experiments_*/auc_comparison.png
```

---

## 🎯 实验流程

```
1. 基线模型 (BiLSTM)
   ↓
2. + 注意力机制
   ↓
3. + Focal Loss
   ↓
4. + 特征工程 (19个特征)
   ↓
5. + 数据增强
   ↓
6. + 对比学习
   ↓
7. + K折集成
   ↓
8. 生成完整报告
```

每个步骤都会：
- ✅ 训练模型
- ✅ 评估性能
- ✅ 保存模型权重
- ✅ 记录结果
- ✅ 对比提升

---

## 📊 预期结果

| 实验 | 预期AUC | 相比基线提升 |
|------|---------|-------------|
| 基线模型 | 0.9718 | - |
| + 注意力 | 0.9730 | +0.12% |
| + Focal Loss | 0.9735 | +0.17% |
| + 特征工程 | 0.9745 | +0.27% |
| + 数据增强 | 0.9750 | +0.32% |
| + 对比学习 | 0.9760 | +0.42% |
| + K折集成 | 0.9770 | +0.52% |

**注意**：实际结果可能因随机性而有所波动（±0.1-0.2%）

---

## ⏱️ 时间预估

### 快速模式（3 epochs per experiment）
- 单个实验：~15分钟
- 全部6个实验：~1.5-2小时

### 完整模式（10 epochs per experiment）
- 单个实验：~50分钟
- 全部6个实验：~5-6小时

**建议**：
- 开发/调试：使用快速模式
- 最终结果：使用完整模式
- 可以后台运行：`nohup ./run_all_experiments.sh full > experiment.log 2>&1 &`

---

## 🔧 自定义实验

### 修改实验配置

编辑 `run_experiments.py` 中的 `self.experiments` 字典：

```python
self.experiments = {
    '0_baseline': {
        'name': '基线模型',
        'script': 'train.py',
        'enabled': True,      # 设为False可跳过
        'priority': 0
    },
    # ... 更多实验
}
```

### 修改训练参数

编辑 `config.py`：
```python
EPOCHS = 5          # 改为5个epoch
BATCH_SIZE = 128    # 改为更小的batch size
LEARNING_RATE = 0.0005  # 调整学习率
```

---

## 💡 使用技巧

### 1. 只运行特定实验
```python
# 修改 run_experiments.py
self.experiments = {
    '0_baseline': {'enabled': True, ...},
    '1_attention': {'enabled': True, ...},
    '2_focal_loss': {'enabled': False, ...},  # 跳过
    # ...
}
```

### 2. 后台运行
```bash
nohup ./run_all_experiments.sh full > exp.log 2>&1 &

# 查看进度
tail -f exp.log
```

### 3. 查看GPU使用
```bash
watch -n 1 nvidia-smi
```

### 4. 中断后继续
如果实验中断，可以：
1. 注释掉 `run_experiments.py` 中已完成的实验
2. 重新运行脚本
3. 结果会累积在同一个输出目录

---

## ❓ 常见问题

**Q: 实验运行太慢怎么办？**  
A: 使用快速模式 `--quick`，或减少 `config.py` 中的 EPOCHS

**Q: 显存不足怎么办？**  
A: 减小 `config.py` 中的 BATCH_SIZE（如从256改为128）

**Q: 可以只运行某个改进方向吗？**  
A: 可以！直接运行对应的训练脚本，如 `python train_attention.py`

**Q: 如何对比不同实验的结果？**  
A: 查看生成的 `EXPERIMENT_REPORT.md` 文件，有详细对比

**Q: 结果可以复现吗？**  
A: 设置了随机种子（SEED=42），在相同环境下结果基本一致（±0.1%）

---

## 📞 实验监控

### 查看当前进度
```bash
# 查看正在运行的进程
ps aux | grep python

# 查看日志
tail -f exp.log

# 查看已生成的模型
ls -lh best_model*.pth
```

### 查看实时结果
```bash
# 查看临时结果
cat experiments_*/results.csv

# 监控CSV变化
watch -n 5 'cat experiments_*/results.csv'
```

---

## 🎨 报告示例

实验完成后，报告包含：

1. **📊 实验结果汇总表**
   - 所有实验的AUC、准确率、F1
   - 相比基线的提升百分比
   - 训练时间

2. **📈 可视化图表**
   - AUC对比柱状图
   - 累积改进曲线
   - 训练时间对比

3. **💡 消融实验结论**
   - 有效的改进方向
   - 改进较小的方向
   - 最优配置建议

4. **🔍 详细分析**
   - 各改进的贡献度
   - 性能vs成本分析
   - 推荐配置

---

## 🚀 现在开始

```bash
# 1. 进入项目目录
cd /home/xinguanze/class/ml2

# 2. 赋予执行权限
chmod +x run_all_experiments.sh

# 3. 运行快速实验（推荐先试试）
./run_all_experiments.sh quick

# 4. 查看结果
cat experiments_*/EXPERIMENT_REPORT.md
```

---

**祝实验顺利！🎉**

有问题随时查看日志或文档。
