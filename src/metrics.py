import torch
import numpy as np
from typing import Union

def dice_coefficient(pred, target, smooth=1e-5):
    """
    Calculate Dice coefficient between prediction and target masks.
    
    Args:
        pred (torch.Tensor): Predicted mask
        target (torch.Tensor): Ground truth mask
        smooth (float): Smoothing factor to avoid division by zero
    
    Returns:
        float: Dice coefficient
    """
    pred = pred.float()
    target = target.float()
    
    # Flatten the tensors
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    # Calculate intersection and union
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    # Calculate Dice coefficient
    dice = (2. * intersection + smooth) / (union + smooth)
    
    return dice

def iou(pred: Union[np.ndarray, torch.Tensor], target: Union[np.ndarray, torch.Tensor]) -> float:
    """Calculate Intersection over Union (IoU) score.
    
    Args:
        pred: Predicted mask (numpy array or torch tensor)
        target: Ground truth mask (numpy array or torch tensor)
        
    Returns:
        IoU score
    """
    # Convert inputs to numpy arrays if they are tensors
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    # Ensure binary masks
    pred = (pred > 0.5).astype(np.uint8)
    target = (target > 0.5).astype(np.uint8)
    
    # Calculate intersection and union
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    
    # Avoid division by zero
    if union == 0:
        return 0.0
    
    return float(intersection) / float(union)

def precision(pred, target, smooth=1e-5):
    """
    Calculate precision between prediction and target masks.
    
    Args:
        pred (torch.Tensor): Predicted mask
        target (torch.Tensor): Ground truth mask
        smooth (float): Smoothing factor to avoid division by zero
    
    Returns:
        float: Precision score
    """
    pred = pred.float()
    target = target.float()
    
    # Flatten the tensors
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    # Calculate true positives and predicted positives
    true_positives = (pred_flat * target_flat).sum()
    predicted_positives = pred_flat.sum()
    
    # Calculate precision
    precision_score = (true_positives + smooth) / (predicted_positives + smooth)
    
    return precision_score

def recall(pred, target, smooth=1e-5):
    """
    Calculate recall between prediction and target masks.
    
    Args:
        pred (torch.Tensor): Predicted mask
        target (torch.Tensor): Ground truth mask
        smooth (float): Smoothing factor to avoid division by zero
    
    Returns:
        float: Recall score
    """
    pred = pred.float()
    target = target.float()
    
    # Flatten the tensors
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    # Calculate true positives and actual positives
    true_positives = (pred_flat * target_flat).sum()
    actual_positives = target_flat.sum()
    
    # Calculate recall
    recall_score = (true_positives + smooth) / (actual_positives + smooth)
    
    return recall_score

def f1_score(pred, target, smooth=1e-5):
    """
    Calculate F1 score between prediction and target masks.
    
    Args:
        pred (torch.Tensor): Predicted mask
        target (torch.Tensor): Ground truth mask
        smooth (float): Smoothing factor to avoid division by zero
    
    Returns:
        float: F1 score
    """
    pred = pred.float()
    target = target.float()
    
    # Flatten the tensors
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    # Calculate true positives, false positives, and false negatives
    true_positives = (pred_flat * target_flat).sum()
    false_positives = (pred_flat * (1 - target_flat)).sum()
    false_negatives = ((1 - pred_flat) * target_flat).sum()
    
    # Calculate precision and recall
    precision = (true_positives + smooth) / (true_positives + false_positives + smooth)
    recall = (true_positives + smooth) / (true_positives + false_negatives + smooth)
    
    # Calculate F1 score
    f1 = (2 * precision * recall) / (precision + recall + smooth)
    
    return f1

def calculate_all_metrics(pred, target, smooth=1e-5):
    """
    Calculate all metrics at once.
    
    Args:
        pred (torch.Tensor): Predicted mask
        target (torch.Tensor): Ground truth mask
        smooth (float): Smoothing factor to avoid division by zero
    
    Returns:
        dict: Dictionary containing all metrics
    """
    # Calculate individual metrics
    dice = dice_coefficient(pred, target, smooth)
    iou_score = iou(pred, target)
    f1 = f1_score(pred, target, smooth)
    
    # Calculate precision and recall separately to avoid redundant calculations
    pred_flat = pred.float().view(-1)
    target_flat = target.float().view(-1)
    true_positives = (pred_flat * target_flat).sum()
    false_positives = (pred_flat * (1 - target_flat)).sum()
    false_negatives = ((1 - pred_flat) * target_flat).sum()
    
    precision = (true_positives + smooth) / (true_positives + false_positives + smooth)
    recall = (true_positives + smooth) / (true_positives + false_negatives + smooth)
    
    return {
        'dice': dice,
        'iou': iou_score,
        'precision': precision,
        'recall': recall,
        'f1': f1
    } 