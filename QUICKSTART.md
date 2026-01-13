# 🚀 快速启动指南

## 一分钟了解项目

这是一个**完整的文本语义匹配实验框架**，包含：
- ✅ 6个不同的深度学习模型
- ✅ 自动化消融实验
- ✅ 详细的实验报告生成

---

## 🎯 三种使用方式

### 方式1：完整实验（推荐用于论文/作业）⭐

```bash
cd /home/xinguanze/class/ml2

# 快速模式（2小时，适合测试）
./run_all_experiments.sh quick

# 完整模式（8小时，最佳结果）
./run_all_experiments.sh full
```

**你会得到**：
- ✅ 6个训练好的模型
- ✅ 完整的性能对比报告（EXPERIMENT_REPORT.md）
- ✅ 可视化图表（AUC对比、改进曲线等）
- ✅ 消融实验结果

---

### 方式2：单独训练一个模型（快速验证）

```bash
# 基线模型（50分钟）
python train.py

# 注意力模型（55分钟）  
python train_attention.py

# K折交叉验证（8折×8epoch，约1小时）
python train_kfold.py --n_splits 5 --epochs 8
```

---

### 方式3：使用已有模型（最快）

如果已经训练过，直接评估：
```bash
python predict.py
```

---

## 📊 查看结果

```bash
# 查看实验报告
cat experiments_*/EXPERIMENT_REPORT.md

# 查看CSV结果
cat experiments_*/results.csv

# 查看图表
ls experiments_*/*.png
```

---

## 💡 推荐流程

### 对于课程作业 📚
```bash
# 1. 快速实验（证明框架可用）
./run_all_experiments.sh quick

# 2. 查看报告
cat experiments_*/EXPERIMENT_REPORT.md

# 3. 提交报告和代码
```

### 对于竞赛/论文 🏆
```bash
# 1. 完整实验
./run_all_experiments.sh full

# 2. K折交叉验证
python train_kfold.py --n_splits 10 --epochs 10

# 3. 集成预测
python train_kfold.py --n_splits 10 --epochs 10 --ensemble
```

---

## ⏱️ 时间安排

| 任务 | 快速模式 | 完整模式 |
|------|---------|---------|
| 基线模型 | 15分钟 | 50分钟 |
| 6个实验 | 1.5小时 | 5小时 |
| K折验证(5折) | 30分钟 | 2小时 |
| **总计** | **2小时** | **8小时** |

**建议**：
- 开发调试：快速模式
- 最终提交：完整模式
- 可后台运行：`nohup ./run_all_experiments.sh full > exp.log 2>&1 &`

---

## 📖 详细文档

- **完整使用说明**：`EXPERIMENT_GUIDE.md`
- **改进方案详解**：`IMPROVEMENTS.md`
- **特征工程说明**：`FEATURE_ENGINEERING.md`
- **K折使用指南**：`KFOLD_USAGE.md`

---

## ❓ 常见问题

**Q: 我只想快速看到一个结果，怎么办？**  
A: 运行 `python train.py`，50分钟得到基线结果（AUC 0.9718）

**Q: 实验太慢了？**  
A: 1) 使用 `quick` 模式；2) 减小 `config.py` 中的 EPOCHS

**Q: 显存不足？**  
A: 减小 `config.py` 中的 BATCH_SIZE（256→128）

**Q: 只想对比几个模型？**  
A: 编辑 `run_experiments.py`，设置某些实验的 `enabled=False`

---

## 🎉 现在开始

```bash
# 1. 进入目录
cd /home/xinguanze/class/ml2

# 2. 运行快速实验
./run_all_experiments.sh quick

# 3. 10分钟后查看进度
tail -f exp.log

# 4. 2小时后查看报告
cat experiments_*/EXPERIMENT_REPORT.md
```

**祝实验顺利！** 🚀
