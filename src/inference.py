import torch
import cv2
import numpy as np
from model import DeepLabv3Plus
from torchvision import transforms
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
import yaml
import os
import logging
from typing import Dict, Any, Optional, Tuple, Union, List
from pathlib import Path
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('inference.log'),
        logging.StreamHandler()
    ]
)

def validate_config(config: Dict[str, Any]) -> None:
    """Validate the configuration dictionary for inference."""
    required_keys = ['model', 'paths', 'data']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # Validate model config
    if 'num_classes' not in config['model']:
        raise ValueError("Missing num_classes in model config")
    if 'backbone' not in config['model']:
        raise ValueError("Missing backbone in model config")
    
    # Validate paths
    if 'checkpoints' not in config['paths']:
        raise ValueError("Missing checkpoints path in config")
    checkpoint_path = os.path.join(config['paths']['checkpoints'], 'checkpoint_epoch_048.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

def load_model(config_path: Union[str, Path], device: str = 'cuda') -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Load the model and configuration for inference.
    
    Args:
        config_path: Path to the configuration file
        device: Device to load the model on ('cuda' or 'cpu')
    
    Returns:
        Tuple of (model, config)
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        validate_config(config)
        
        model = DeepLabv3Plus(
            num_classes=config['model']['num_classes'],
            backbone=config['model']['backbone'],
            pretrained=False
        ).to(device)
        
        checkpoint_path = os.path.join(config['paths']['checkpoints'], 'best_model.pth')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load only the model state dict from the checkpoint
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # If the checkpoint is just the model state dict
            model.load_state_dict(checkpoint)
            
        model.eval()
        
        logging.info(f"Successfully loaded model from {checkpoint_path}")
        return model, config
    
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise

def process_image(image_path: Union[str, Path], image_size: Tuple[int, int], edge_scale: float = 1.0) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Process an image for inference.
    
    Args:
        image_path: Path to the input image
        image_size: Target size for resizing (height, width)
        edge_scale: Scale factor for edge detection
    
    Returns:
        Tuple of (processed tensor, original shape)
    """
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        # Read and resize image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_shape = image.shape[:2]
        
        # Resize image
        image = cv2.resize(image, (image_size[1], image_size[0]), interpolation=cv2.INTER_LINEAR)
        
        # Add edge channel
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge = np.sqrt(sobelx**2 + sobely**2)
        edge = (edge - edge.min()) / (edge.max() - edge.min()) * edge_scale
        
        # Normalize image
        image = image.astype(np.float32) / 255.0
        
        # Combine image and edge
        image = np.concatenate([image, edge[..., np.newaxis]], axis=-1)
        
        # Normalize using the same values as training
        mean = np.array([0.485, 0.456, 0.406, 0.5])
        std = np.array([0.229, 0.224, 0.225, 0.5])
        image = (image - mean) / std
        
        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
        return image, original_shape
        
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {str(e)}")
        raise

def postprocess_mask(pred, original_shape, min_size=50, min_distance=7):
    pred = cv2.resize(pred, (original_shape[1], original_shape[0]))
    binary = (pred > 0.5).astype(np.uint8)
    
    # Remove small objects
    cleaned = remove_small_objects(binary.astype(bool), min_size=min_size)
    
    # Watershed separation
    distance = ndi.distance_transform_edt(cleaned)
    coords = peak_local_max(distance, min_distance=min_distance, labels=cleaned)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers = ndi.label(mask)[0]
    labels = watershed(-distance, markers, mask=cleaned)
    
    return labels

def save_visualization(image_paths: List[str], pred_masks: np.ndarray, 
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
        plt.savefig(os.path.join(save_dir, 'inference_visualization.png'))
        plt.close()
    except Exception as e:
        logging.error(f"Error in visualization: {str(e)}")

def predict(config_path, image_path=None, output_dir=None, visualize=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, config = load_model(config_path, device)
    
    # Get image size from config
    image_size = tuple(config['data'].get('image_size', [512, 512]))
    
    # If no specific image is provided, use test set
    if image_path is None:
        test_images_dir = os.path.join(config['paths']['data'], 'test', 'images')
        test_masks_dir = os.path.join(config['paths']['data'], 'test', 'masks')
        
        # Get list of test images and masks
        test_image_files = sorted([f for f in os.listdir(test_images_dir) if f.endswith(('.png', '.jpg', '.tif'))])
        test_mask_files = sorted([f for f in os.listdir(test_masks_dir) if f.endswith(('.png', '.jpg', '.tif'))])
        
        # Randomly select 5 images
        random.seed(42)  # For reproducibility
        selected_indices = random.sample(range(len(test_image_files)), min(5, len(test_image_files)))
        
        # Process selected images
        image_paths = []
        pred_masks = []
        true_masks = []
        
        for idx in selected_indices:
            img_path = os.path.join(test_images_dir, test_image_files[idx])
            mask_path = os.path.join(test_masks_dir, test_mask_files[idx])
            
            # Process image
            image_tensor, original_shape = process_image(
                img_path,
                image_size=image_size,
                edge_scale=config['data'].get('edge_scale', 1.0)
            )
            image_tensor = image_tensor.to(device)
            
            # Predict
            with torch.no_grad():
                pred = model(image_tensor)
                pred = torch.sigmoid(pred).squeeze().cpu().numpy()
            
            # Post-process
            segmented = postprocess_mask(
                pred, 
                original_shape,
                min_size=config['postprocessing'].get('min_size', 50),
                min_distance=config['postprocessing'].get('min_distance', 7)
            )
            
            # Load ground truth mask
            true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            true_mask = cv2.resize(true_mask, (original_shape[1], original_shape[0]))
            
            image_paths.append(img_path)
            pred_masks.append(segmented)
            true_masks.append(true_mask)
        
        # Save visualization
        if output_dir:
            save_visualization(image_paths, pred_masks, true_masks, output_dir)
            logging.info(f"Visualization saved to {output_dir}")
        
        return pred_masks
    
    else:
        # Process single image
        image_tensor, original_shape = process_image(
            image_path,
            image_size=image_size,
            edge_scale=config['data'].get('edge_scale', 1.0)
        )
        image_tensor = image_tensor.to(device)
        
        # Predict
        with torch.no_grad():
            pred = model(image_tensor)
            pred = torch.sigmoid(pred).squeeze().cpu().numpy()
        
        # Post-process
        segmented = postprocess_mask(
            pred, 
            original_shape,
            min_size=config['postprocessing'].get('min_size', 50),
            min_distance=config['postprocessing'].get('min_distance', 7)
        )
        
        # Save results
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.basename(image_path)
            
            # Save raw prediction
            cv2.imwrite(os.path.join(output_dir, f'pred_{base_name}'), (pred*255).astype(np.uint8))
            
            # Save final segmentation
            cv2.imwrite(os.path.join(output_dir, f'seg_{base_name}'), segmented.astype(np.uint8))
            
            # Save visualization
            if visualize:
                original = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                ax[0].imshow(original)
                ax[0].set_title('Original')
                ax[1].imshow(pred, cmap='jet')
                ax[1].set_title('Prediction')
                ax[2].imshow(segmented, cmap='jet')
                ax[2].set_title('Segmentation')
                plt.savefig(os.path.join(output_dir, f'vis_{os.path.splitext(base_name)[0]}.png'))
                plt.close()
        
        return segmented

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--input', type=str, help='Path to input image (optional)')
    parser.add_argument('--output', type=str, default='results')
    args = parser.parse_args()
    
    predict(args.config, args.input, args.output)