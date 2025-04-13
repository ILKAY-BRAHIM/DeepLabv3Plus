import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import random
import logging

class PhosphateDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, augment=False, edge_scale=1.0, 
                 image_size=(512, 512), max_samples=-1):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.augment = augment
        self.edge_scale = edge_scale
        self.image_size = image_size
        
        # Get list of image and mask files
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.tif'))])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.tif'))])
        
        # Verify that we have matching pairs
        assert len(self.image_files) == len(self.mask_files), "Number of images and masks must match"
        for img, mask in zip(self.image_files, self.mask_files):
            assert os.path.splitext(img)[0] == os.path.splitext(mask)[0], f"Image {img} and mask {mask} names don't match"
        
        # Limit number of samples if specified
        if max_samples > 0 and max_samples < len(self.image_files):
            self.image_files = self.image_files[:max_samples]
            self.mask_files = self.mask_files[:max_samples]
            logging.info(f"Limited dataset to {max_samples} samples")
        
        # Augmentation transforms for RGB channels only
        self.rgb_aug_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        ])
        
        # Normalization
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5], 
                                            std=[0.229, 0.224, 0.225, 0.5])
    
    def _normalize_mask(self, mask: np.ndarray) -> np.ndarray:
        """Normalize mask values to handle compression artifacts and format conversion.
        
        Args:
            mask: Input mask array
            
        Returns:
            Normalized mask array with values in [0, 1]
        """
        # Get unique values and sort them
        unique_values = np.unique(mask)
        if len(unique_values) < 2:
            return mask.astype(np.float32) / 255.0
        
        # Find the largest gap between consecutive values
        sorted_values = np.sort(unique_values)
        diffs = np.diff(sorted_values)
        split_idx = np.argmax(diffs)
        
        # Calculate threshold safely to avoid overflow
        threshold = sorted_values[split_idx] + (diffs[split_idx] / 2)
        
        # Create binary mask
        binary_mask = np.zeros_like(mask, dtype=np.float32)
        binary_mask[mask > threshold] = 1.0
        
        return binary_mask
    
    def __len__(self):
        return len(self.image_files)
    
    def _add_edge_channel(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge = np.sqrt(sobelx**2 + sobely**2)
        edge = (edge - edge.min()) / (edge.max() - edge.min() + 1e-6) * self.edge_scale
        return edge
    
    def _apply_augmentation(self, image, mask):
        """Apply augmentation to image and mask."""
        # Random horizontal flip
        if random.random() > 0.5:
            image = torch.flip(image, [2])
            mask = torch.flip(mask, [2])
        
        # Random vertical flip
        if random.random() > 0.5:
            image = torch.flip(image, [1])
            mask = torch.flip(mask, [1])
        
        # Random rotation
        angle = random.uniform(-15, 15)
        image = transforms.functional.rotate(image, angle)
        mask = transforms.functional.rotate(mask, angle)
        
        # Apply color jitter to RGB channels only
        rgb_channels = image[:3]
        rgb_channels = self.rgb_aug_transform(rgb_channels)
        image = torch.cat([rgb_channels, image[3:]], dim=0)
        
        return image, mask
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        
        # Read image and mask
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not read mask: {mask_path}")
        
        # Resize image and mask to the specified size
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
        
        # Normalize mask
        mask = self._normalize_mask(mask)
        
        # Add edge channel
        edge = self._add_edge_channel(image)
        
        # Combine RGB and edge channels
        image = np.concatenate([image/255.0, edge[..., np.newaxis]], axis=-1)
        
        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        # Apply augmentation if needed
        if self.augment:
            image, mask = self._apply_augmentation(image, mask)
        
        # Normalize image
        image = self.normalize(image)
        
        return {
            'image': image,
            'mask': mask,
            'image_path': image_path,
            'mask_path': mask_path
        }

def get_dataloaders(root_dir, batch_size=8, test_size=0.1, val_size=0.1, image_size=(512, 512), 
                   max_train_samples=-1, max_val_samples=-1, max_test_samples=-1, **kwargs):
    """Create train, validation, and test dataloaders."""
    # Create datasets for each split
    train_dataset = PhosphateDataset(
        image_dir=os.path.join(root_dir, 'train', 'images'),
        mask_dir=os.path.join(root_dir, 'train', 'masks'),
        image_size=image_size,
        augment=True,  # Enable augmentation for training
        max_samples=max_train_samples,
        **{k: v for k, v in kwargs.items() if k != 'augment'}  # Remove augment from kwargs
    )
    
    val_dataset = PhosphateDataset(
        image_dir=os.path.join(root_dir, 'val', 'images'),
        mask_dir=os.path.join(root_dir, 'val', 'masks'),
        image_size=image_size,
        augment=False,  # No augmentation for validation
        max_samples=max_val_samples,
        **{k: v for k, v in kwargs.items() if k != 'augment'}  # Remove augment from kwargs
    )
    
    test_dataset = PhosphateDataset(
        image_dir=os.path.join(root_dir, 'test', 'images'),
        mask_dir=os.path.join(root_dir, 'test', 'masks'),
        image_size=image_size,
        augment=False,  # No augmentation for testing
        max_samples=max_test_samples,
        **{k: v for k, v in kwargs.items() if k != 'augment'}  # Remove augment from kwargs
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader