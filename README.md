# DeepLabv3Plus for semantic Segmentation

## Implementation of paper: https://arxiv.org/pdf/1802.02611

A PyTorch implementation of DeepLabv3Plus for semantic segmentation . This project provides a robust solution for segmenting objects with high accuracy and includes features for monitoring training progress and model performance.
```
Note: depthwise separable convolutions and the Xception backbone are not included.
```
<img width="2000" height="500" alt="vis_crop_34539" src="https://github.com/user-attachments/assets/40c97124-24ba-4a9d-aa1e-54b3a496d864" />

## Project Structure

```
DeepLabv3Plus/
├── src/
│   ├── model.py          # DeepLabv3Plus model implementation
│   ├── dataset.py        # Dataset and data loading utilities
│   ├── train.py          # Training script with visualization
│   ├── inference.py      # Inference script
│   ├── metrics.py        # Evaluation metrics
│   └── utils.py          # Utility functions
├── configs/
│   └── default.yaml      # Configuration file
├── data/                 # Dataset directory (ignored by git)
│   ├── train/
│   │   ├── images/      # Training images
│   │   └── masks/       # Training masks
│   ├── val/
│   │   ├── images/      # Validation images
│   │   └── masks/       # Validation masks
│   └── test/
│       ├── images/      # Test images
│       └── masks/       # Test masks
├── checkpoints/         # Model checkpoints (ignored by git)
├── logs/               # Training logs (ignored by git)
├── results/            # Inference results (ignored by git)
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Features

- **Robust Mask Handling**: Automatic normalization of masks in various formats (JPG, PNG, TIFF)
- **Training Visualization**: Monitor model progress with visualizations of predictions
- **Checkpoint Management**: Save model checkpoints for every epoch
- **Early Stopping**: Prevent overfitting with configurable early stopping
- **Mixed Precision Training**: Optimize training speed with automatic mixed precision
- **Data Augmentation**: Comprehensive augmentation pipeline for robust training
- **Edge Detection**: Enhanced edge detection for better boundary segmentation

## Requirements

- Python 3.8+
- PyTorch 1.8+
- CUDA 11.0+ (for GPU training)
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DeepLabv3Plus.git
cd DeepLabv3Plus
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

The `configs/default.yaml` file contains all configurable parameters:

```yaml
model:
  num_classes: 1
  backbone: resnet50
  pretrained: true

training:
  epochs: 100
  batch_size: 8
  learning_rate: 0.001
  weight_decay: 0.0001
  early_stopping_patience: 10
  early_stopping_min_delta: 0.001
  max_train_samples: -1  # Set to limit training samples
  max_val_samples: -1    # Set to limit validation samples
  max_test_samples: -1   # Set to limit test samples
  num_visualization_samples: 5

data:
  image_size: [512, 512]
  test_split: 0.1
  val_split: 0.1
  augmentations:
    enabled: true
    horizontal_flip: true
    vertical_flip: true
    rotation: true
    brightness: 0.2
    contrast: 0.2
    saturation: 0.2
    hue: 0.1
  edge_scale: 1.0
```

## Training

1. Prepare your dataset:
   - Place images in `data/train/images/`
   - Place corresponding masks in `data/train/masks/`
   - Repeat for validation and test sets

2. Start training:
```bash
python src/train.py --config configs/default.yaml
```

Training features:
- Automatic checkpoint saving for each epoch
- Visualization of predictions every epoch
- Progress bar with loss and metrics
- TensorBoard integration for monitoring
- Early stopping to prevent overfitting

## Inference

Run inference on a single image:
```bash
python src/inference.py --config configs/default.yaml --image path/to/image
```

## Model Architecture

The model uses DeepLabv3Plus with:
- ResNet50 backbone (pretrained)
- Atrous Spatial Pyramid Pooling (ASPP)
- Decoder module for refined segmentation
- Edge detection enhancement

## Training Process

1. **Data Loading**:
   - Images and masks are automatically normalized
   - Masks are handled robustly regardless of format
   - Data augmentation is applied during training

2. **Training Loop**:
   - Mixed precision training for efficiency
   - Automatic learning rate scheduling
   - Validation after each epoch
   - Metrics calculation (IoU, Dice)

3. **Monitoring**:
   - TensorBoard integration
   - Visualizations of predictions
   - Checkpoint saving
   - Early stopping

## Results

The model achieves:
- High IoU scores on validation set
- Accurate boundary detection
- Robust performance across different image types

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original DeepLabv3Plus paper: [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)
- PyTorch implementation inspiration
- Dataset providers and contributors 
