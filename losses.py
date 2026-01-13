"""
损失函数集合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss: 对难样本赋予更高权重
    
    论文: Focal Loss for Dense Object Detection
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [batch_size] 模型输出（未经sigmoid）
            targets: [batch_size] 目标标签 (0 or 1)
        """
        # BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        # 计算概率
        probs = torch.sigmoid(logits)
        
        # p_t: 正确类别的概率
        pt = targets * probs + (1 - targets) * (1 - probs)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - pt) ** self.gamma
        
        # Alpha weight
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        # Focal Loss
        loss = alpha_weight * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class ContrastiveLoss(nn.Module):
    """
    对比损失：让相似样本靠近，不相似样本远离
    """
    
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, repr1, repr2, labels):
        """
        Args:
            repr1, repr2: [batch_size, hidden_dim] 句子表示
            labels: [batch_size] 标签 (1表示相似，0表示不相似)
        """
        # 欧氏距离
        distance = F.pairwise_distance(repr1, repr2, p=2)
        
        # 对比损失
        loss_positive = labels * distance.pow(2)
        loss_negative = (1 - labels) * F.relu(self.margin - distance).pow(2)
        
        loss = loss_positive + loss_negative
        return loss.mean()


class TripletLoss(nn.Module):
    """
    三元组损失：anchor-positive距离 < anchor-negative距离 + margin
    """
    
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor: [batch_size, hidden_dim] 锚点
            positive: [batch_size, hidden_dim] 正样本
            negative: [batch_size, hidden_dim] 负样本
        """
        pos_distance = F.pairwise_distance(anchor, positive, p=2)
        neg_distance = F.pairwise_distance(anchor, negative, p=2)
        
        loss = F.relu(pos_distance - neg_distance + self.margin)
        return loss.mean()


class LabelSmoothingLoss(nn.Module):
    """
    标签平滑：避免模型过度自信
    """
    
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [batch_size] 模型输出
            targets: [batch_size] 目标标签
        """
        # 标签平滑
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        
        # BCE loss with smoothed labels
        loss = F.binary_cross_entropy_with_logits(logits, targets_smooth)
        return loss


if __name__ == '__main__':
    # 测试Focal Loss
    print("测试 Focal Loss:")
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    
    logits = torch.randn(10)
    targets = torch.randint(0, 2, (10,)).float()
    
    loss = focal_loss(logits, targets)
    print(f"   Focal Loss: {loss.item():.4f}")
    
    # 对比BCE Loss
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets)
    print(f"   BCE Loss: {bce_loss.item():.4f}")
    
    # 测试Contrastive Loss
    print("\n测试 Contrastive Loss:")
    contrastive_loss = ContrastiveLoss(margin=1.0)
    
    repr1 = torch.randn(10, 128)
    repr2 = torch.randn(10, 128)
    labels = torch.randint(0, 2, (10,)).float()
    
    loss = contrastive_loss(repr1, repr2, labels)
    print(f"   Contrastive Loss: {loss.item():.4f}")
