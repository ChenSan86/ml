# K折交叉验证使用指南

## 📚 什么是K折交叉验证？

K折交叉验证是一种更可靠的模型评估方法：
- 将数据分成K份
- 每次用K-1份训练，1份验证
- 重复K次，每份数据都会作为验证集
- 最后取K次结果的平均值和标准差

**优势**：比单次验证更能反映模型的真实泛化能力，减少过拟合风险。

---

## 🚀 运行K折交叉验证

### 方法1：使用Python命令（推荐）

#### 基本用法（5折，每折8个epoch）
```bash
cd /home/xinguanze/class/ml2
python train_kfold.py --n_splits 5 --epochs 8
```

#### 完整训练（5折，每折10个epoch）
```bash
python train_kfold.py --n_splits 5 --epochs 10
```

#### 10折交叉验证（更稳健，但耗时更长）
```bash
python train_kfold.py --n_splits 10 --epochs 8
```

#### 带集成预测
```bash
python train_kfold.py --n_splits 5 --epochs 8 --ensemble
```

---

### 方法2：使用Shell脚本
```bash
cd /home/xinguanze/class/ml2
chmod +x run_kfold.sh
./run_kfold.sh
```

---

## ⚙️ 参数说明

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--n_splits` | 折数（K值） | 5 | `--n_splits 10` |
| `--epochs` | 每折训练的epoch数 | 10 | `--epochs 8` |
| `--ensemble` | 是否进行集成预测 | False | `--ensemble` |

---

## 📊 输出文件

运行完成后会生成：

1. **模型文件**
   - `best_model_fold1.pth` ~ `best_model_fold5.pth`（各折最佳模型）

2. **结果文件**
   - `kfold_results.csv` - 各折详细指标
   - `kfold_results.png` - 可视化图表

3. **终端输出**
   - 各折的训练进度
   - 最终统计结果（平均值±标准差）

---

## 📈 结果解读

运行结束后，你会看到类似这样的输出：

```
📊 总体结果统计:
================================================================================

各折详细结果:
     auc  accuracy  precision    recall        f1
  0.9720    0.9255     0.8795    0.9275    0.9029
  0.9715    0.9248     0.8781    0.9288    0.9028
  0.9718    0.9252     0.8789    0.9282    0.9029
  0.9722    0.9260     0.8801    0.9290    0.9039
  0.9716    0.9249     0.8785    0.9284    0.9028

平均值和标准差:
--------------------------------------------------------------------------------
Auc         : 0.9718 ± 0.0003
Accuracy    : 0.9253 ± 0.0004
Precision   : 0.8790 ± 0.0007
Recall      : 0.9284 ± 0.0005
F1          : 0.9031 ± 0.0004

🏆 最佳折: Fold 4 (AUC: 0.9722)
```

---

## 🎯 快速命令参考

### 快速测试（3折，5个epoch）
```bash
python train_kfold.py --n_splits 3 --epochs 5
```

### 标准评估（5折，8个epoch）⭐ 推荐
```bash
python train_kfold.py --n_splits 5 --epochs 8
```

### 完整评估（5折，10个epoch）
```bash
python train_kfold.py --n_splits 5 --epochs 10
```

### 高精度评估（10折，10个epoch）
```bash
python train_kfold.py --n_splits 10 --epochs 10
```

### 带集成预测
```bash
python train_kfold.py --n_splits 5 --epochs 8 --ensemble
```

---

## ⏱️ 预计运行时间

基于您的硬件（CUDA GPU），预计时间：

| 配置 | 预计时间 |
|------|----------|
| 3折 × 5 epoch | ~15分钟 |
| 5折 × 8 epoch | ~40分钟 ⭐ |
| 5折 × 10 epoch | ~50分钟 |
| 10折 × 8 epoch | ~80分钟 |
| 10折 × 10 epoch | ~100分钟 |

---

## 💡 使用建议

1. **课程作业/快速验证**：使用 5折 × 8epoch
   ```bash
   python train_kfold.py --n_splits 5 --epochs 8
   ```

2. **论文/竞赛**：使用 5折 × 10epoch 或 10折 × 10epoch
   ```bash
   python train_kfold.py --n_splits 10 --epochs 10
   ```

3. **最终提交**：使用集成预测
   ```bash
   python train_kfold.py --n_splits 5 --epochs 10 --ensemble
   ```

---

## 🆚 对比：单次验证 vs K折交叉验证

| 方法 | 优点 | 缺点 |
|------|------|------|
| **单次验证** | ✅ 快速<br>✅ 简单 | ⚠️ 可能过拟合验证集<br>⚠️ 结果不够稳健 |
| **K折交叉验证** | ✅ 更可靠<br>✅ 提供标准差<br>✅ 减少过拟合 | ⚠️ 耗时K倍<br>⚠️ 占用更多空间 |

---

## 📞 查看进度

如果训练正在进行，可以查看：
```bash
# 查看当前进度
tail -f \home\xinguanze\.cursor\projects\home-xinguanze-class-ml2\terminals\1.txt

# 或者查看已生成的模型文件
ls -lh best_model_fold*.pth
```

---

## 🔍 查看结果

```bash
# 查看CSV结果
cat kfold_results.csv

# 查看图表
# 在文件管理器中打开 kfold_results.png
```

---

## ❓ 常见问题

**Q: 为什么要用K折交叉验证？**  
A: 单次验证可能因为验证集的选择而产生偏差，K折交叉验证使用所有数据进行验证，结果更可靠。

**Q: K应该选多大？**  
A: 通常选5或10。K越大越准确，但耗时越长。

**Q: 可以中断训练吗？**  
A: 可以用 Ctrl+C 中断。已完成的折的模型会被保存。

**Q: 如何使用最佳模型？**  
A: 查看输出中的"最佳折"，使用对应的 `best_model_foldX.pth` 文件。

---

**祝评估顺利！🎉**
