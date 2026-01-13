"""
增强版模型训练：使用特征工程
"""
import os
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

from config import config
from model_enhanced import create_enhanced_model
from dataset import prepare_data, get_dataloaders
from train import set_seed, compute_metrics, plot_training_history


def train_epoch_enhanced(model, train_loader, criterion, optimizer, device):
    """训练一个epoch（增强版）"""
    model.train()
    total_loss = 0
    all_labels = []
    all_probs = []
    
    progress_bar = tqdm(train_loader, desc='Training')
    for batch in progress_bar:
        query1 = batch['query1'].to(device)
        query2 = batch['query2'].to(device)
        labels = batch['label'].to(device)
        
        # 前向传播（模型内部会提取手工特征）
        optimizer.zero_grad()
        logits = model(query1, query2)
        
        # 计算损失
        loss = criterion(logits, labels)
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs)
        
        # 更新进度条
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # 计算平均损失和AUC
    avg_loss = total_loss / len(train_loader)
    auc = roc_auc_score(all_labels, all_probs)
    
    return avg_loss, auc


@torch.no_grad()
def evaluate_enhanced(model, val_loader, criterion, device):
    """评估模型（增强版）"""
    model.eval()
    total_loss = 0
    all_labels = []
    all_probs = []
    all_preds = []
    
    progress_bar = tqdm(val_loader, desc='Evaluating')
    for batch in progress_bar:
        query1 = batch['query1'].to(device)
        query2 = batch['query2'].to(device)
        labels = batch['label'].to(device)
        
        # 前向传播
        logits = model(query1, query2)
        loss = criterion(logits, labels)
        
        # 统计
        total_loss += loss.item()
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > 0.5).astype(int)
        
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs)
        all_preds.extend(preds)
    
    # 计算指标
    avg_loss = total_loss / len(val_loader)
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    
    return avg_loss, metrics


def train_enhanced():
    """增强版完整训练流程"""
    print("=" * 80)
    print("🚀 开始训练增强版文本匹配模型（融合特征工程）")
    print("=" * 80)
    
    # 设置随机种子
    set_seed(config.SEED)
    print(f"\n⚙️  配置信息: {config}")
    
    # 准备数据
    print("\n📂 准备数据...")
    train_df, val_df = prepare_data()
    train_loader, val_loader = get_dataloaders(
        train_df, val_df,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )
    
    # 创建增强版模型
    print("\n🏗️  创建增强版模型...")
    model = create_enhanced_model()
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
        optimizer, T_max=config.EPOCHS, eta_min=1e-6
    )
    
    # 训练历史
    history = {
        'train_loss': [],
        'train_auc': [],
        'val_loss': [],
        'val_auc': [],
        'val_metrics': []
    }
    
    best_auc = 0.0
    best_epoch = 0
    
    # 开始训练
    print("\n" + "=" * 80)
    print("🎯 开始训练循环")
    print("=" * 80)
    
    for epoch in range(1, config.EPOCHS + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{config.EPOCHS}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        # 训练
        train_loss, train_auc = train_epoch_enhanced(
            model, train_loader, criterion, optimizer, config.DEVICE
        )
        
        # 验证
        val_loss, val_metrics = evaluate_enhanced(
            model, val_loader, criterion, config.DEVICE
        )
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_auc'].append(train_auc)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_metrics['auc'])
        history['val_metrics'].append(val_metrics)
        
        # 打印结果
        epoch_time = time.time() - start_time
        print(f"\n📈 Epoch {epoch} 结果:")
        print(f"   Time: {epoch_time:.2f}s")
        print(f"   Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
        print(f"   Val Loss: {val_loss:.4f}, Val AUC: {val_metrics['auc']:.4f}")
        print(f"   Val Acc: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1']:.4f}")
        print(f"   Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 保存最佳模型
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auc': best_auc,
                'config': config,
                'feature_mean': model.feature_mean,
                'feature_std': model.feature_std
            }, 'best_model_enhanced.pth')
            print(f"   ✅ 最佳增强模型已保存! (AUC: {best_auc:.4f})")
    
    # 训练完成
    print("\n" + "=" * 80)
    print("🎉 训练完成!")
    print("=" * 80)
    print(f"\n📊 最佳结果:")
    print(f"   Best Epoch: {best_epoch}")
    print(f"   Best Val AUC: {best_auc:.4f}")
    
    # 绘制训练曲线
    plot_training_history(history, save_path='training_history_enhanced.png')
    
    # 打印最终验证集详细指标
    print(f"\n📋 最终验证集指标:")
    final_metrics = history['val_metrics'][-1]
    for metric, value in final_metrics.items():
        print(f"   {metric.capitalize()}: {value:.4f}")
    
    # 与基线模型对比
    print(f"\n📊 性能对比:")
    print(f"   基线模型 AUC: 0.9718 (不含特征工程)")
    print(f"   增强模型 AUC: {best_auc:.4f} (含特征工程)")
    if best_auc > 0.9718:
        improvement = (best_auc - 0.9718) * 100
        print(f"   🎉 提升: +{improvement:.2f}%")
    
    return model, history


if __name__ == '__main__':
    model, history = train_enhanced()
