import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=0.5):
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
    def forward(self, pred, target):
        # Binary Cross Entropy Loss
        bce_loss = F.binary_cross_entropy_with_logits(pred, target)
        
        # Dice Loss
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum()
        dice_loss = 1 - (2. * intersection + 1) / (pred_sigmoid.sum() + target.sum() + 1)
        
        # Edge-aware Loss
        edge_mask = torch.abs(F.conv2d(target, torch.ones(1, 1, 3, 3).to(target.device), padding=1) - 9 * target)
        edge_loss = F.binary_cross_entropy_with_logits(pred, target, weight=edge_mask)
        
        # Combined loss
        loss = self.alpha * bce_loss + self.beta * dice_loss + self.gamma * edge_loss
        
        return loss

def calculate_metrics(pred, target):
    pred_sigmoid = torch.sigmoid(pred)
    pred_binary = (pred_sigmoid > 0.5).float()
    
    # True Positives, False Positives, False Negatives
    tp = (pred_binary * target).sum()
    fp = (pred_binary * (1 - target)).sum()
    fn = ((1 - pred_binary) * target).sum()
    
    # Dice Coefficient
    dice = (2. * tp + 1) / (pred_binary.sum() + target.sum() + 1)
    
    # IoU (Jaccard Index)
    iou = (tp + 1) / (tp + fp + fn + 1)
    
    # Precision
    precision = (tp + 1) / (tp + fp + 1)
    
    # Recall
    recall = (tp + 1) / (tp + fn + 1)
    
    return {
        'dice': dice,
        'iou': iou,
        'precision': precision,
        'recall': recall
    }