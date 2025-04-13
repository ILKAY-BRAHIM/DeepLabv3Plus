import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import os
import shutil
from datetime import datetime

def visualize_results(image, pred, mask, save_path=None):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(image)
    ax[0].set_title('Original')
    ax[1].imshow(pred, cmap='jet')
    ax[1].set_title('Prediction')
    ax[2].imshow(mask, cmap='jet')
    ax[2].set_title('Ground Truth')
    if save_path:
        plt.savefig(save_path)
    plt.close()

def save_as_geotiff(array, path, reference_image=None):
    """Save segmentation as GeoTIFF if reference geospatial data exists"""
    if reference_image is not None:
        # Implement with rasterio if needed
        pass
    else:
        cv2.imwrite(path, array)

def calculate_grain_statistics(segmented):
    """Calculate grain size statistics"""
    from skimage.measure import regionprops
    props = regionprops(segmented)
    areas = [p.area for p in props]
    return {
        'count': len(props),
        'mean_size': np.mean(areas),
        'std_size': np.std(areas),
        'min_size': np.min(areas),
        'max_size': np.max(areas)
    }

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def load_checkpoint(model, optimizer, filename):
    """Loads checkpoint from disk"""
    if os.path.isfile(filename):
        print(f"=> loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"=> loaded checkpoint '{filename}' (epoch {checkpoint['epoch']})")
        return checkpoint['epoch']
    else:
        print(f"=> no checkpoint found at '{filename}'")
        return 0

def create_experiment_dir(base_dir='experiments'):
    """Creates a new experiment directory with timestamp"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join(base_dir, timestamp)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir

def get_lr(optimizer):
    """Gets the current learning rate from the optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']