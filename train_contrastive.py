"""
对比学习训练
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config
from model import create_model
from dataset import prepare_data, get_dataloaders
from train import set_seed, compute_metrics, plot_training_history
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


class ContrastiveLoss(nn.Module):
    """对比损失"""
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, repr1, repr2, labels):
        distance = F.pairwise_distance(repr1, repr2)
        loss = labels * distance.pow(2) + (1 - labels) * F.relu(self.margin - distance).pow(2)
        return loss.mean()


def train():
    """训练对比学习模型"""
    print("=" * 80)
    print("🚀 训练对比学习模型")
    print("=" * 80)
    
    set_seed(config.SEED)
    train_df, val_df = prepare_data()
    train_loader, val_loader = get_dataloaders(train_df, val_df, config.BATCH_SIZE, config.NUM_WORKERS)
    
    model = create_model().to(config.DEVICE)
    bce_criterion = nn.BCEWithLogitsLoss()
    contrastive_criterion = ContrastiveLoss(margin=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=1e-6)
    
    history = {'train_loss': [], 'train_auc': [], 'val_loss': [], 'val_auc': [], 'val_metrics': []}
    best_auc = 0.0
    
    for epoch in range(1, config.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.EPOCHS}")
        
        # 训练
        model.train()
        total_loss = 0
        all_labels, all_probs = [], []
        
        for batch in tqdm(train_loader, desc='Training'):
            query1, query2, labels = batch['query1'].to(config.DEVICE), batch['query2'].to(config.DEVICE), batch['label'].to(config.DEVICE)
            
            optimizer.zero_grad()
            
            # 获取表示和logits
            repr1, _ = model.encoder(query1)
            repr2, _ = model.encoder(query2)
            logits = model(query1, query2)
            
            # 组合损失
            bce_loss = bce_criterion(logits, labels)
            cont_loss = contrastive_criterion(repr1, repr2, labels)
            loss = bce_loss + 0.2 * cont_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        train_loss = total_loss / len(train_loader)
        train_auc = roc_auc_score(all_labels, all_probs)
        
        # 验证
        model.eval()
        val_total_loss = 0
        val_labels, val_probs, val_preds = [], [], []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Evaluating'):
                query1, query2, labels = batch['query1'].to(config.DEVICE), batch['query2'].to(config.DEVICE), batch['label'].to(config.DEVICE)
                logits = model(query1, query2)
                loss = bce_criterion(logits, labels)
                
                val_total_loss += loss.item()
                probs = torch.sigmoid(logits).cpu().numpy()
                val_probs.extend(probs)
                val_preds.extend((probs > 0.5).astype(int))
                val_labels.extend(labels.cpu().numpy())
        
        val_loss = val_total_loss / len(val_loader)
        val_metrics = compute_metrics(val_labels, val_preds, val_probs)
        
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
                      'best_model_contrastive.pth')
    
    plot_training_history(history, 'training_history_contrastive.png')
    return model, history


if __name__ == '__main__':
    train()
