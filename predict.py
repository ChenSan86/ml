"""
预测和评估模块
"""
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from config import config
from model import create_model
from dataset import TextMatchDataset, get_dataloaders


@torch.no_grad()
def predict(model, dataloader, device):
    """
    对数据进行预测
    返回概率和预测标签
    """
    model.eval()
    all_probs = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc='Predicting')
    for batch in progress_bar:
        query1 = batch['query1'].to(device)
        query2 = batch['query2'].to(device)
        labels = batch['label']

        # 前向传播
        logits = model(query1, query2)
        probs = torch.sigmoid(logits).cpu().numpy()

        all_probs.extend(probs)
        all_labels.extend(labels.numpy())

    return np.array(all_probs), np.array(all_labels)


def evaluate_model(model_path, val_df):
    """
    加载最佳模型并在验证集上评估
    """
    print("\n" + "=" * 70)
    print("📊 模型评估")
    print("=" * 70)

    # 加载模型
    print(f"\n📥 加载模型: {model_path}")
    checkpoint = torch.load(
        model_path, map_location=config.DEVICE, weights_only=False)

    model = create_model()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.DEVICE)
    model.eval()

    print(f"   模型来自 Epoch: {checkpoint['epoch']}")
    print(f"   最佳 AUC: {checkpoint['best_auc']:.4f}")

    # 准备数据
    from torch.utils.data import DataLoader
    val_dataset = TextMatchDataset(val_df, max_len=config.MAX_LEN)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    # 预测
    print("\n🔮 开始预测...")
    probs, labels = predict(model, val_loader, config.DEVICE)
    preds = (probs > 0.5).astype(int)

    # 计算指标
    auc = roc_auc_score(labels, probs)

    print("\n" + "=" * 70)
    print("📈 评估结果")
    print("=" * 70)
    print(f"\n🎯 AUC Score: {auc:.4f}")

    print("\n📋 分类报告:")
    print(classification_report(labels, preds,
          target_names=['不匹配', '匹配'], digits=4))

    # 混淆矩阵
    cm = confusion_matrix(labels, preds)
    print("\n📊 混淆矩阵:")
    print(f"                预测不匹配  预测匹配")
    print(f"实际不匹配:      {cm[0][0]:>8}    {cm[0][1]:>8}")
    print(f"实际匹配:        {cm[1][0]:>8}    {cm[1][1]:>8}")

    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['不匹配', '匹配'],
                yticklabels=['不匹配', '匹配'])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title(f'混淆矩阵 (AUC: {auc:.4f})')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("\n📊 混淆矩阵图已保存到: confusion_matrix.png")

    # 预测分布
    plot_prediction_distribution(probs, labels)

    return auc, probs, preds


def plot_prediction_distribution(probs, labels):
    """绘制预测概率分布"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 按真实标签分组的概率分布
    probs_neg = probs[labels == 0]
    probs_pos = probs[labels == 1]

    axes[0].hist(probs_neg, bins=50, alpha=0.6,
                 label='不匹配 (label=0)', color='blue')
    axes[0].hist(probs_pos, bins=50, alpha=0.6,
                 label='匹配 (label=1)', color='red')
    axes[0].set_xlabel('预测概率')
    axes[0].set_ylabel('样本数')
    axes[0].set_title('预测概率分布（按真实标签）')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 整体概率分布
    axes[1].hist(probs, bins=50, alpha=0.7, color='green')
    axes[1].axvline(x=0.5, color='red', linestyle='--',
                    linewidth=2, label='阈值=0.5')
    axes[1].set_xlabel('预测概率')
    axes[1].set_ylabel('样本数')
    axes[1].set_title('预测概率整体分布')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('prediction_distribution.png', dpi=150, bbox_inches='tight')
    print("📊 预测分布图已保存到: prediction_distribution.png")


def save_predictions(probs, labels, output_file='predictions.csv'):
    """保存预测结果"""
    df = pd.DataFrame({
        'true_label': labels,
        'predicted_prob': probs,
        'predicted_label': (probs > 0.5).astype(int)
    })
    df.to_csv(output_file, index=False)
    print(f"\n💾 预测结果已保存到: {output_file}")


if __name__ == '__main__':
    from dataset import prepare_data

    # 准备数据
    train_df, val_df = prepare_data()

    # 评估模型
    auc, probs, preds = evaluate_model(config.MODEL_SAVE_PATH, val_df)

    # 保存预测结果
    save_predictions(probs, val_df['label'].values)
