"""
数据增强训练
"""
import torch
import random
import numpy as np
from config import config
from model import create_model
from dataset import prepare_data, get_dataloaders, TextMatchDataset
from train import set_seed, plot_training_history, train_epoch, evaluate
import torch.nn as nn
import pandas as pd


def augment_data(df, augment_ratio=0.2):
    """数据增强: 正样本互换 + 困难负样本"""
    augmented_samples = []
    
    # 1. 正样本互换query1和query2
    pos_samples = df[df['label'] == 1]
    swapped = pos_samples.copy()
    swapped['query1'], swapped['query2'] = pos_samples['query2'].values, pos_samples['query1'].values
    augmented_samples.append(swapped.sample(frac=augment_ratio))
    
    # 2. 困难负样本: 随机配对正样本
    if len(pos_samples) > 100:
        n_hard_neg = int(len(pos_samples) * augment_ratio * 0.5)
        shuffled = pos_samples.sample(n=n_hard_neg)
        hard_negs = pd.DataFrame({
            'query1': pos_samples.sample(n=n_hard_neg)['query1'].values,
            'query2': shuffled['query2'].values,
            'label': 0
        })
        augmented_samples.append(hard_negs)
    
    # 合并
    augmented_df = pd.concat([df] + augmented_samples, ignore_index=True)
    return augmented_df.sample(frac=1).reset_index(drop=True)  # 打乱


def train():
    """训练数据增强模型"""
    print("=" * 80)
    print("🚀 训练数据增强模型")
    print("=" * 80)
    
    set_seed(config.SEED)
    train_df, val_df = prepare_data()
    
    # 数据增强
    print(f"原始训练集: {len(train_df)}")
    train_df = augment_data(train_df, augment_ratio=0.2)
    print(f"增强后训练集: {len(train_df)}")
    
    train_loader, val_loader = get_dataloaders(train_df, val_df, config.BATCH_SIZE, config.NUM_WORKERS)
    
    model = create_model().to(config.DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=1e-6)
    
    history = {'train_loss': [], 'train_auc': [], 'val_loss': [], 'val_auc': [], 'val_metrics': []}
    best_auc = 0.0
    
    for epoch in range(1, config.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.EPOCHS}")
        train_loss, train_auc = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, config.DEVICE)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_auc'].append(train_auc)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_metrics['auc'])
        history['val_metrics'].append(val_metrics)
        
        print(f"Train: Loss={train_loss:.4f}, AUC={train_auc:.4f}")
        print(f"Val: Loss={val_loss:.4f}, AUC={val_metrics['auc']:.4f}")
        
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            torch.save({'model_state_dict': model.state_dict(), 'best_auc': best_auc}, 
                      'best_model_augmented.pth')
    
    plot_training_history(history, 'training_history_augmented.png')
    return model, history


if __name__ == '__main__':
    train()
