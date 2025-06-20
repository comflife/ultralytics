# Dual-Stream YOLO Training

This code adds dual-stream (multi-camera) support to the Ultralytics YOLO framework, allowing you to train and run inference using two camera inputs simultaneously.

## Overview

The dual-stream architecture allows you to:
- Process inputs from two cameras (e.g., wide and narrow) simultaneously
- Fuse feature maps from both streams using configurable fusion strategies
- Train with paired image datasets
- Share information between streams for better detection performance

## Getting Started

### Dataset Structure

For dual-stream training, prepare your dataset with the following structure:

```
swm/
├── images/
│   ├── train_wide/      # Wide camera images
│   ├── train_narrow/    # Narrow camera images
│   ├── val_wide/        # Wide camera validation images
│   └── val_narrow/      # Narrow camera validation images
└── labels/
    ├── train/           # Labels for wide camera images only
    └── val/             # Labels for wide camera validation images
```

### Dataset Configuration

Create a dual-stream YAML configuration file:

```yaml
# Example: swm_dual.yaml
train_wide: /path/to/swm/images/train_wide
train_narrow: /path/to/swm/images/train_narrow
val_wide: /path/to/swm/images/val_wide
val_narrow: /path/to/swm/images/val_narrow

# Classes
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: bus
  5: truck
```

### Model Configuration

Use a dual-stream model architecture like this:

```yaml
# yolov8-dual.yaml
backbone:
  [
    [-1, 1, MultiStreamConv, [64, 6, 2, 2]], # Initial dual stream convolution
    [-1, 1, MultiStreamConv, [128, 3, 2]],   # Dual stream convolution with downsampling 
    [-1, 3, MultiStreamC3, [128]],           # Dual stream C3 module
    [-1, 1, Fusion, ['concat', 2]],          # Fusion of dual streams
    # ... rest of the backbone ...
  ]
```

### Training

Use the `train_yolov8_dual_v0.py` script to train the model:

```bash
python train_yolov8_dual_v0.py \
  --cfg models/yolov8-dual.yaml \
  --data ultralytics/cfg/datasets/swm_dual_updated.yaml \
  --epochs 100 \
  --batch-size 16 \
  --imgsz 640 \
  --dual-stream
```

## Modules

### MultiStream Modules

- **MultiStreamConv**: Applies convolution to each input stream separately
- **MultiStreamC3**: Applies C3 (CSP Bottleneck with 3 convolutions) to each input stream
- **Fusion**: Combines features from multiple streams using one of these methods:
  - `concat`: Channel-wise concatenation
  - `add`: Element-wise addition
  - `max`: Element-wise maximum
  - `weighted_sum`: Weighted sum with learnable weights

## Inference

To run inference with a trained dual-stream model:

```python
from ultralytics import YOLO
from ultralytics.nn.modules.dual_model import DualStreamWrapper

# Load the model
model = YOLO('path/to/best.pt')
model.model = DualStreamWrapper(model.model)

# Prepare your wide and narrow camera images
wide_img = 'path/to/wide_image.jpg'
narrow_img = 'path/to/narrow_image.jpg'

# Run inference with the dual-stream model
results = model.predict([wide_img, narrow_img])
```

## Implementation Notes

- Labels are only needed for the wide camera images
- The narrow camera provides supplementary information for feature fusion
- During training, images from wide and narrow cameras are paired and processed together
- Fusion strategies can be adjusted in the model configuration
