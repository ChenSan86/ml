"""
K折交叉验证训练模块
"""
import os
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

from config import config
from model import create_model
from dataset import load_data, get_dataloaders
from train import set_seed, train_epoch, evaluate


def plot_kfold_results(fold_results, save_path='kfold_results.png'):
    """绘制K折交叉验证结果"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = ['auc', 'accuracy', 'f1', 'precision']
    titles = ['AUC Score', 'Accuracy', 'F1 Score', 'Precision']

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]

        # 提取每一折的指标
        values = [result[metric] for result in fold_results]
        folds = list(range(1, len(values) + 1))

        # 绘制折线图
        ax.plot(folds, values, marker='o', linewidth=2,
                markersize=8, label=f'{title}')

        # 添加平均线
        mean_val = np.mean(values)
        ax.axhline(y=mean_val, color='r', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_val:.4f}')

        # 添加标准差范围
        std_val = np.std(values)
        ax.fill_between(folds, mean_val - std_val, mean_val + std_val,
                        alpha=0.2, color='blue', label=f'±1 std: {std_val:.4f}')

        ax.set_xlabel('Fold', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(f'{title} across Folds', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(folds)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 K折结果图已保存到: {save_path}")


def train_kfold(n_splits=5, epochs_per_fold=10):
    """
    K折交叉验证训练

    Args:
        n_splits: 折数（默认5折）
        epochs_per_fold: 每一折训练的epoch数
    """
    print("=" * 80)
    print(f"🚀 开始 {n_splits} 折交叉验证")
    print("=" * 80)

    # 设置随机种子
    set_seed(config.SEED)
    print(f"\n⚙️  配置信息: {config}")

    # 加载所有数据
    print("\n📂 加载数据...")
    df = load_data([config.TRAIN_FILE_1, config.TRAIN_FILE_2])

    # 创建K折分割器（保持标签分布）
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=config.SEED)

    # 存储每一折的结果
    fold_results = []
    fold_models = []

    print("\n" + "=" * 80)
    print(f"📊 开始 {n_splits} 折训练")
    print("=" * 80)

    # K折交叉验证
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label']), 1):
        print("\n" + "=" * 80)
        print(f"🔄 Fold {fold}/{n_splits}")
        print("=" * 80)

        # 划分数据
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        print(f"\n📊 数据划分:")
        print(f"   训练集: {len(train_df):,} 条")
        print(f"   验证集: {len(val_df):,} 条")
        print(f"   训练集正样本比例: {train_df['label'].mean():.2%}")
        print(f"   验证集正样本比例: {val_df['label'].mean():.2%}")

        # 创建数据加载器
        train_loader, val_loader = get_dataloaders(
            train_df, val_df,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS
        )

        # 创建模型
        print(f"\n🏗️  创建 Fold {fold} 模型...")
        model = create_model()
        model = model.to(config.DEVICE)

        # 损失函数和优化器
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs_per_fold, eta_min=1e-6
        )

        # 记录本折最佳结果
        best_auc = 0.0
        best_metrics = None

        # 训练本折模型
        print(f"\n{'='*80}")
        print(f"🎯 Fold {fold} 训练开始")
        print(f"{'='*80}")

        for epoch in range(1, epochs_per_fold + 1):
            print(f"\n--- Fold {fold}, Epoch {epoch}/{epochs_per_fold} ---")

            # 训练
            train_loss, train_auc = train_epoch(
                model, train_loader, criterion, optimizer, config.DEVICE
            )

            # 验证
            val_loss, val_metrics = evaluate(
                model, val_loader, criterion, config.DEVICE
            )

            # 更新学习率
            scheduler.step()

            # 打印结果
            print(f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_metrics['auc']:.4f}, "
                  f"Val Acc: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1']:.4f}")

            # 保存本折最佳模型
            if val_metrics['auc'] > best_auc:
                best_auc = val_metrics['auc']
                best_metrics = val_metrics.copy()

                # 保存模型
                fold_model_path = f'best_model_fold{fold}.pth'
                torch.save({
                    'fold': fold,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_auc': best_auc,
                    'metrics': best_metrics
                }, fold_model_path)
                print(f"   ✅ Fold {fold} 最佳模型已保存! (AUC: {best_auc:.4f})")

        # 记录本折最佳结果
        print(f"\n{'='*80}")
        print(f"📈 Fold {fold} 最佳结果:")
        print(f"{'='*80}")
        for metric, value in best_metrics.items():
            print(f"   {metric.capitalize()}: {value:.4f}")

        fold_results.append(best_metrics)
        fold_models.append(f'best_model_fold{fold}.pth')

        # 清理内存
        del model, optimizer, scheduler
        torch.cuda.empty_cache()

    # 计算总体统计
    print("\n" + "=" * 80)
    print("🎉 K折交叉验证完成!")
    print("=" * 80)

    # 计算平均和标准差
    print("\n📊 总体结果统计:")
    print("=" * 80)

    results_df = pd.DataFrame(fold_results)

    print("\n各折详细结果:")
    print(results_df.to_string(index=False))

    print("\n\n平均值和标准差:")
    print("-" * 80)
    for metric in results_df.columns:
        mean_val = results_df[metric].mean()
        std_val = results_df[metric].std()
        print(f"{metric.capitalize():12s}: {mean_val:.4f} ± {std_val:.4f}")

    # 找出最佳折
    best_fold = results_df['auc'].idxmax() + 1
    best_fold_auc = results_df['auc'].max()
    print(f"\n🏆 最佳折: Fold {best_fold} (AUC: {best_fold_auc:.4f})")

    # 保存结果
    results_df['fold'] = range(1, n_splits + 1)
    results_df = results_df[['fold'] +
                            [col for col in results_df.columns if col != 'fold']]
    results_df.to_csv('kfold_results.csv', index=False)
    print(f"\n💾 详细结果已保存到: kfold_results.csv")

    # 绘制结果图
    plot_kfold_results(fold_results)

    # 集成预测（可选）
    print("\n" + "=" * 80)
    print("💡 提示:")
    print("=" * 80)
    print(f"1. 各折模型已保存为: best_model_fold1.pth ~ best_model_fold{n_splits}.pth")
    print(f"2. 可以使用最佳折模型 (Fold {best_fold}) 进行预测")
    print(f"3. 或者使用所有模型的集成预测以获得更好的泛化性能")

    return results_df, fold_models


def ensemble_predict(fold_models, val_df):
    """
    使用所有折的模型进行集成预测

    Args:
        fold_models: 各折模型路径列表
        val_df: 验证数据
    """
    print("\n" + "=" * 80)
    print("🔮 集成预测")
    print("=" * 80)

    from torch.utils.data import DataLoader
    from dataset import TextMatchDataset

    # 准备数据
    val_dataset = TextMatchDataset(val_df, max_len=config.MAX_LEN)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    all_probs = []

    # 加载每个模型并预测
    for fold_idx, model_path in enumerate(fold_models, 1):
        print(f"\n加载 Fold {fold_idx} 模型...")

        checkpoint = torch.load(
            model_path, map_location=config.DEVICE, weights_only=False)
        model = create_model()
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(config.DEVICE)
        model.eval()

        fold_probs = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Fold {fold_idx} Predicting'):
                query1 = batch['query1'].to(config.DEVICE)
                query2 = batch['query2'].to(config.DEVICE)

                logits = model(query1, query2)
                probs = torch.sigmoid(logits).cpu().numpy()
                fold_probs.extend(probs)

        all_probs.append(fold_probs)
        del model
        torch.cuda.empty_cache()

    # 平均所有模型的预测
    ensemble_probs = np.mean(all_probs, axis=0)
    ensemble_preds = (ensemble_probs > 0.5).astype(int)

    # 计算集成指标
    labels = val_df['label'].values
    auc = roc_auc_score(labels, ensemble_probs)
    acc = accuracy_score(labels, ensemble_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, ensemble_preds, average='binary', zero_division=0
    )

    print("\n" + "=" * 80)
    print("📈 集成预测结果")
    print("=" * 80)
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")

    return ensemble_probs, ensemble_preds


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='K折交叉验证训练')
    parser.add_argument('--n_splits', type=int, default=5, help='折数')
    parser.add_argument('--epochs', type=int, default=10, help='每折训练的epoch数')
    parser.add_argument('--ensemble', action='store_true', help='是否进行集成预测')

    args = parser.parse_args()

    # K折训练
    results_df, fold_models = train_kfold(
        n_splits=args.n_splits, epochs_per_fold=args.epochs)

    # 集成预测（可选）
    if args.ensemble:
        df = load_data([config.TRAIN_FILE_1, config.TRAIN_FILE_2])
        # 使用全部数据的一个子集作为测试
        from sklearn.model_selection import train_test_split
        _, test_df = train_test_split(
            df, test_size=0.1, random_state=config.SEED, stratify=df['label'])
        ensemble_probs, ensemble_preds = ensemble_predict(fold_models, test_df)
