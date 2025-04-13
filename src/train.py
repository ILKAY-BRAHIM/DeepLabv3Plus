import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import os
from tqdm import tqdm
import time
from datetime import datetime
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from inference import predict  # Import the predict function
import cv2
from metrics import dice_coefficient, iou  # Changed from src.metrics to metrics
import logging
from typing import Dict, Any, List, Tuple, Optional
import random
import traceback

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from other modules
from src.model import DeepLabv3Plus
from src.dataset import PhosphateDataset, get_dataloaders
from src.utils import AverageMeter, save_checkpoint, load_checkpoint
from src.losses import HybridLoss, calculate_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

def validate_config(config: Dict[str, Any]) -> None:
    """Validate the configuration dictionary."""
    required_keys = ['model', 'training', 'paths', 'data']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # Validate model config
    if 'num_classes' not in config['model']:
        raise ValueError("Missing num_classes in model config")
    if 'backbone' not in config['model']:
        raise ValueError("Missing backbone in model config")
    
    # Validate paths
    required_paths = ['checkpoints', 'logs', 'data']
    for path in required_paths:
        if path not in config['paths']:
            raise ValueError(f"Missing required path: {path}")
        if not os.path.exists(config['paths'][path]):
            os.makedirs(config['paths'][path])

def save_visualization(epoch: int, image_paths: List[str], pred_masks: np.ndarray, 
                      true_masks: np.ndarray, save_dir: str) -> None:
    """Save visualization of predictions and ground truth for multiple images."""
    try:
        num_images = len(image_paths)
        fig, axes = plt.subplots(num_images, 3, figsize=(15, 5*num_images))
        
        for i, (img_path, pred_mask, true_mask) in enumerate(zip(image_paths, pred_masks, true_masks)):
            if not os.path.exists(img_path):
                logging.warning(f"Image path does not exist: {img_path}")
                continue
                
            # Read original image
            image = cv2.imread(img_path)
            if image is None:
                logging.warning(f"Failed to read image: {img_path}")
                continue
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Plot original image
            axes[i, 0].imshow(image)
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')
            
            # Plot prediction
            axes[i, 1].imshow(pred_mask, cmap='jet')
            axes[i, 1].set_title('Prediction')
            axes[i, 1].axis('off')
            
            # Plot ground truth
            axes[i, 2].imshow(true_mask, cmap='jet')
            axes[i, 2].set_title('Ground Truth')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'epoch_{epoch}_visualization.png'))
        plt.close()
    except Exception as e:
        logging.error(f"Error in visualization: {str(e)}")

def train(config_path: str) -> None:
    """Train the DeepLabv3Plus model.
    
    Args:
        config_path: Path to the configuration file
    """
    try:
        # Load and validate config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        validate_config(config)
        
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {device}")
        
        # Initialize mixed precision training
        scaler = torch.amp.GradScaler() if device.type == 'cuda' else None
        
        # Create model
        model = DeepLabv3Plus(
            num_classes=config['model']['num_classes'],
            backbone=config['model']['backbone'],
            pretrained=config['model'].get('pretrained', True)
        ).to(device)
        
        # Setup optimizer with weight decay
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 1e-4)
        )
        
        # Get image size from config
        image_size = tuple(config['data'].get('image_size', [512, 512]))
        
        # Create datasets and dataloaders
        train_loader, val_loader, test_loader = get_dataloaders(
            root_dir=config['paths']['data'],
            batch_size=config['training']['batch_size'],
            test_size=config['data'].get('test_split', 0.1),
            val_size=config['data'].get('val_split', 0.1),
            image_size=image_size,
            max_train_samples=config['training'].get('max_train_samples', -1),
            max_val_samples=config['training'].get('max_val_samples', -1),
            max_test_samples=config['training'].get('max_test_samples', -1),
            augment=config['data'].get('augmentations', {}).get('enabled', True),
            edge_scale=config['data'].get('edge_scale', 1.0)
        )
        
        # Log dataset sizes
        logging.info(f"Training samples: {len(train_loader.dataset)}")
        logging.info(f"Validation samples: {len(val_loader.dataset)}")
        logging.info(f"Test samples: {len(test_loader.dataset)}")
        
        # Setup learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,  # Reduce LR by half
            patience=5,   # Number of epochs with no improvement
            verbose=True,
            min_lr=1e-6   # Minimum learning rate
        )
        
        # Setup loss function
        criterion = HybridLoss(
            alpha=config['training'].get('loss_alpha', 0.5),
            beta=config['training'].get('loss_beta', 0.5)
        )
        
        # Setup early stopping
        early_stopping = EarlyStopping(
            patience=config['training'].get('early_stopping_patience', 10),
            min_delta=config['training'].get('early_stopping_min_delta', 0.001)
        )
        
        # Setup TensorBoard
        log_dir = os.path.join(config['paths']['logs'], datetime.now().strftime('%Y%m%d-%H%M%S'))
        writer = SummaryWriter(log_dir)
        
        # Get test images directory
        test_images_dir = os.path.join(config['paths']['data'], 'test', 'images')
        test_masks_dir = os.path.join(config['paths']['data'], 'test', 'masks')
        
        # Get list of test images and masks
        test_image_files = sorted([f for f in os.listdir(test_images_dir) if f.endswith(('.png', '.jpg', '.tif'))])
        test_mask_files = sorted([f for f in os.listdir(test_masks_dir) if f.endswith(('.png', '.jpg', '.tif'))])
        
        # Create visualization directory
        vis_dir = os.path.join(config['paths']['results'], 'training_visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Training loop
        best_val_loss = float('inf')
        for epoch in range(config['training']['epochs']):
            # Train for one epoch
            train_loss = train_epoch(
                model, train_loader, optimizer, criterion, 
                device, scaler
            )
            
            # Validate
            val_loss, val_metrics = validate(
                model, val_loader, criterion, device
            )
            
            # Update learning rate based on validation loss
            scheduler.step(val_loss)
            
            # Log metrics to TensorBoard
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Metrics/val_iou', val_metrics['iou'], epoch)
            writer.add_scalar('Metrics/val_dice', val_metrics['dice'], epoch)
            writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
            
            # Visualize predictions on test images
            model.eval()
            with torch.no_grad():
                try:
                    # Randomly select 5 test images for this epoch
                    random.seed(epoch)  # Use epoch as seed for reproducibility but different each epoch
                    selected_indices = random.sample(range(len(test_image_files)), 
                                                   min(5, len(test_image_files)))
                    
                    # Load and prepare sample images
                    sample_test_images = []
                    sample_test_masks = []
                    
                    for idx in selected_indices:
                        # Load and preprocess image
                        img_path = os.path.join(test_images_dir, test_image_files[idx])
                        mask_path = os.path.join(test_masks_dir, test_mask_files[idx])
                        
                        image = cv2.imread(img_path)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image = cv2.resize(image, image_size, interpolation=cv2.INTER_LINEAR)
                        image = image / 255.0
                        
                        # Load and preprocess mask
                        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        mask = cv2.resize(mask, image_size, interpolation=cv2.INTER_NEAREST)
                        
                        # Normalize mask like in dataloader
                        mask = mask.astype(np.float32)
                        mask = mask / 255.0  # Normalize to [0, 1]
                        
                        # Convert to tensor
                        image = torch.from_numpy(image).permute(2, 0, 1).float()
                        mask = torch.from_numpy(mask).unsqueeze(0).float()
                        
                        sample_test_images.append(image)
                        sample_test_masks.append(mask)
                    
                    # Stack tensors
                    sample_test_images = torch.stack(sample_test_images)
                    sample_test_masks = torch.stack(sample_test_masks)
                    
                    # Get predictions
                    pred_masks = model(sample_test_images.to(device))
                    pred_masks = torch.sigmoid(pred_masks)
                    
                    # Create visualization grid
                    fig, axes = plt.subplots(5, 4, figsize=(20, 25))
                    fig.suptitle(f'Epoch {epoch + 1} Predictions', fontsize=16, y=1.02)
                    
                    for i in range(5):
                        # Original image
                        img = sample_test_images[i].permute(1, 2, 0).cpu().numpy()
                        img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
                        img = img.astype(np.uint8)
                        
                        # Ground truth mask (already normalized to [0, 1])
                        gt_mask = sample_test_masks[i][0].cpu().numpy()
                        
                        # Predicted mask
                        pred_mask = pred_masks[i][0].cpu().numpy()
                        binary_pred = (pred_mask > 0.5).astype(np.uint8)
                        
                        # Calculate IoU for this sample
                        iou_score = iou(binary_pred, (gt_mask > 0.5).astype(np.uint8))
                        
                        # Plot
                        axes[i, 0].imshow(img)
                        axes[i, 0].set_title('Original Image')
                        axes[i, 0].axis('off')
                        
                        axes[i, 1].imshow(gt_mask, cmap='gray', vmin=0, vmax=1)
                        axes[i, 1].set_title('Ground Truth')
                        axes[i, 1].axis('off')
                        
                        axes[i, 2].imshow(pred_mask, cmap='viridis', vmin=0, vmax=1)
                        axes[i, 2].set_title(f'Raw Prediction\n(IoU: {iou_score:.3f})')
                        axes[i, 2].axis('off')
                        
                        axes[i, 3].imshow(binary_pred, cmap='gray', vmin=0, vmax=1)
                        axes[i, 3].set_title('Binary Prediction')
                        axes[i, 3].axis('off')
                    
                    plt.tight_layout()
                    
                    # Save visualization with high quality
                    vis_path = os.path.join(vis_dir, f'epoch_{epoch+1:03d}_predictions.png')
                    plt.savefig(vis_path, bbox_inches='tight', dpi=300, facecolor='white')
                    plt.close()
                    
                    # Add to TensorBoard
                    writer.add_figure('Test_Predictions', fig, epoch)
                    
                    # Log visualization path
                    logging.info(f"Saved visualization for epoch {epoch + 1} to {vis_path}")
                    
                except Exception as e:
                    logging.error(f"Error during visualization: {str(e)}")
                    logging.error(traceback.format_exc())
            
            # Log metrics with more detail
            logging.info(
                f"Epoch {epoch + 1}/{config['training']['epochs']} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f} - "
                f"Val IoU: {val_metrics['iou']:.4f} - "
                f"Val Dice: {val_metrics['dice']:.4f} - "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )
            
            # Check early stopping
            if early_stopping(val_loss):
                logging.info("Early stopping triggered")
                break
            
            # Save checkpoint for every epoch
            epoch_checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'best_val_loss': best_val_loss
            }
            
            # Save epoch checkpoint
            epoch_checkpoint_path = os.path.join(
                config['paths']['checkpoints'],
                f'checkpoint_epoch_{epoch+1:03d}.pth'
            )
            torch.save(epoch_checkpoint, epoch_checkpoint_path)
            logging.info(f"Saved checkpoint for epoch {epoch + 1}")
            
            # Save best model if validation loss improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(
                    config['paths']['checkpoints'],
                    'best_model.pth'
                )
                torch.save(epoch_checkpoint, best_model_path)
                logging.info("Saved new best model")
        
        # Close TensorBoard writer
        writer.close()
        
        # Log completion
        logging.info(f"Training completed. Checkpoints saved in {config['paths']['checkpoints']}")
        logging.info(f"Visualizations saved in {vis_dir}")
    
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler]
) -> float:
    """Train for one epoch.
    
    Args:
        model: The model to train
        loader: DataLoader for training data
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        scaler: Gradient scaler for mixed precision
    
    Returns:
        Average training loss
    """
    model.train()
    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    
    # Create progress bar with additional metrics
    pbar = tqdm(loader, desc="Training")
    
    for batch in pbar:
        # Move data to device
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
        
        # Calculate Dice score
        preds = torch.sigmoid(outputs) > 0.5
        dice = dice_coefficient(preds, masks)
        
        # Update meters
        loss_meter.update(loss.item())
        dice_meter.update(dice.item())
        
        # Update progress bar with current metrics
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'dice': f'{dice_meter.avg:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })
    
    return loss_meter.avg

def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, Dict[str, float]]:
    """Validate the model.
    
    Args:
        model: The model to validate
        loader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on
    
    Returns:
        Tuple of (average loss, metrics dictionary)
    """
    model.eval()
    loss_meter = AverageMeter()
    metrics = {
        'dice': 0.0,
        'iou': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0
    }
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            # Move data to device
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Calculate metrics
            preds = torch.sigmoid(outputs) > 0.5
            batch_metrics = calculate_metrics(preds, masks)
            
            # Update meters
            loss_meter.update(loss.item())
            for metric_name, metric_value in batch_metrics.items():
                metrics[metric_name] += metric_value.item()
    
    # Calculate average metrics
    for metric_name in metrics:
        metrics[metric_name] /= len(loader)
    
    return loss_meter.avg, metrics

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    args = parser.parse_args()
    
    train(args.config)