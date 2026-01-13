"""
注意力机制模型训练
"""
import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from config import config
from model_attention import create_attention_model
from dataset import prepare_data, get_dataloaders
from train import set_seed, compute_metrics, plot_training_history, train_epoch, evaluate


def train():
    """训练注意力模型"""
    print("=" * 80)
    print("🚀 训练注意力机制模型")
    print("=" * 80)
    
    set_seed(config.SEED)
    
    # 准备数据
    train_df, val_df = prepare_data()
    train_loader, val_loader = get_dataloaders(train_df, val_df, config.BATCH_SIZE, config.NUM_WORKERS)
    
    # 创建模型
    model = create_attention_model()
    model = model.to(config.DEVICE)
    
    # 训练配置
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=1e-6)
    
    history = {'train_loss': [], 'train_auc': [], 'val_loss': [], 'val_auc': [], 'val_metrics': []}
    best_auc = 0.0
    
    # 训练循环
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
        
        print(f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_metrics['auc']:.4f}")
        
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            torch.save({'model_state_dict': model.state_dict(), 'best_auc': best_auc}, 
                      'best_model_attention.pth')
    
    plot_training_history(history, 'training_history_attention.png')
    return model, history


if __name__ == '__main__':
    train()
