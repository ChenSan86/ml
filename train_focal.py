"""
Focal Loss训练
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config
from model import create_model
from dataset import prepare_data, get_dataloaders
from train import set_seed, compute_metrics, plot_training_history, evaluate
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


class FocalLoss(nn.Module):
    """Focal Loss: 对难样本赋予更高权重"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        loss = self.alpha * focal_weight * bce_loss
        return loss.mean()


def train_epoch_focal(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    all_labels, all_probs = [], []
    
    for batch in tqdm(train_loader, desc='Training'):
        query1, query2, labels = batch['query1'].to(device), batch['query2'].to(device), batch['label'].to(device)
        
        optimizer.zero_grad()
        logits = model(query1, query2)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(train_loader), roc_auc_score(all_labels, all_probs)


def train():
    """训练Focal Loss模型"""
    print("=" * 80)
    print("🚀 训练Focal Loss模型")
    print("=" * 80)
    
    set_seed(config.SEED)
    train_df, val_df = prepare_data()
    train_loader, val_loader = get_dataloaders(train_df, val_df, config.BATCH_SIZE, config.NUM_WORKERS)
    
    model = create_model().to(config.DEVICE)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=1e-6)
    
    history = {'train_loss': [], 'train_auc': [], 'val_loss': [], 'val_auc': [], 'val_metrics': []}
    best_auc = 0.0
    
    for epoch in range(1, config.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.EPOCHS}")
        train_loss, train_auc = train_epoch_focal(model, train_loader, criterion, optimizer, config.DEVICE)
        val_loss, val_metrics = evaluate(model, val_loader, nn.BCEWithLogitsLoss(), config.DEVICE)
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
                      'best_model_focal.pth')
    
    plot_training_history(history, 'training_history_focal.png')
    return model, history


if __name__ == '__main__':
    train()
