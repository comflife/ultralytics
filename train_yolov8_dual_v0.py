# YOLOv8 🚀 by Ultralytics, AGPL-3.0 license
"""
Train a YOLOv8 model on a custom dataset.

Usage - Single-GPU training:
    $ python train_yolov8.py --data swm.yaml --weights yolov8n.pt --cfg yolov8n.yaml --img 640 --epochs 20

"""

import argparse
import os
import random
import sys
import time
import yaml
from pathlib import Path

import numpy as np
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv8 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.callbacks import get_default_callbacks
from ultralytics import YOLO
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.checks import check_file, check_yaml, print_args
from ultralytics.utils.torch_utils import select_device
from ultralytics.utils.files import increment_path
import yaml


def train(cfg, opt, device, callbacks=None):
    """
    Trains YOLOv8 model with given configuration, options, and device.
    
    `cfg` argument is path/to/config.yaml or configuration dictionary.
    """
    callbacks = callbacks or get_default_callbacks()
    save_dir = Path(opt.save_dir)
    
    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save run settings
    with open(save_dir / "opt.yaml", 'w') as f:
        yaml.dump(vars(opt), f)
    
    # Loggers
    LOGGER.info(f"Starting YOLOv8 training in {save_dir}")
    
   # Load model
    if opt.resume:
        LOGGER.info(f"Resuming training from {opt.weights}")
        model = YOLO(opt.weights)
    else:
        if opt.cfg and Path(opt.cfg).exists():
            LOGGER.info(f"Loading model from config {opt.cfg}")
            model = YOLO(opt.cfg)  # Load from yaml config
            if opt.weights and opt.weights.endswith('.pt') and Path(opt.weights).exists():
                LOGGER.info(f"Loading weights from {opt.weights}")
                model.load(opt.weights)  # Load pretrained weights
        else:
            LOGGER.info(f"Loading pretrained model {opt.weights}")
            model = YOLO(opt.weights)
    
    # Check if using a dual-stream model
    is_dual_model = False
    if opt.cfg:
        try:
            with open(opt.cfg, 'r') as f:
                model_yaml = yaml.safe_load(f)
                # Check if 'backbone' contains 'MultiStreamConv' or similar modules
                if model_yaml and isinstance(model_yaml, dict) and 'backbone' in model_yaml:
                    backbone = model_yaml['backbone']
                    if isinstance(backbone, list):
                        modules = [module[2] for module in backbone if len(module) > 2]
                        is_dual_model = any('MultiStream' in str(module) for module in modules)
                        
                        if is_dual_model:
                            LOGGER.info(f"Detected dual-stream model with MultiStream modules")
                            # Mark as dual model but don't wrap - we'll rely on our dual_stream flag
        except Exception as e:
            LOGGER.warning(f"Error checking model configuration: {e}")
            is_dual_model = False
            
    # Set dual stream attribute on model if needed
    if is_dual_model or opt.dual_stream:
        LOGGER.info(f"Enabling dual-stream mode for training")
        # Set dual_stream attribute on the model
        if hasattr(model, 'model'):
            model.model.dual_stream = True
        # Also set on the trainer args
        model.overrides = getattr(model, 'overrides', {})
        model.overrides['dual_stream'] = True
    
    # ✅ Configure training settings with stability improvements
    model_training_args = {
        'data': opt.data,
        'epochs': opt.epochs,
        'patience': opt.patience,
        'batch': opt.batch_size,
        'imgsz': opt.imgsz,
        'save': not opt.nosave,
        'cache': opt.cache,
        'device': device,
        'workers': opt.workers,
        'project': opt.project,
        'name': opt.name,
        'exist_ok': opt.exist_ok,
        'pretrained': opt.weights.endswith('.pt'),
        'optimizer': opt.optimizer,
        'verbose': True,
        'seed': opt.seed,
        'deterministic': True,
        'single_cls': opt.single_cls,
        'rect': opt.rect,
        'cos_lr': opt.cos_lr,
        'close_mosaic': 10,  # last 10 epochs disable mosaic
        'resume': opt.resume,
        'amp': True,  # Automatic Mixed Precision
        'fraction': 1.0,  # dataset fraction to train on
        'profile': False,  # profile ONNX and TensorRT speeds
        'val': not opt.noval,
        'label_smoothing': opt.label_smoothing,
        'save_period': opt.save_period,
        'conf': 0.1,   # confidence threshold
        'iou': 0.3,    # ✅ IoU threshold 낮추기 (기본값 0.7 → 0.3)
        'dual_stream': is_dual_model or opt.dual_stream,
        
        # 🔧 Gradient exploding 해결을 위한 설정들
        'lr0': 0.001,           # ✅ Learning rate 감소 (기본값 0.01 → 0.001)
        'lrf': 0.01,            # ✅ Final learning rate (lr0 * lrf)
        'momentum': 0.937,      # ✅ SGD momentum/Adam beta1
        'weight_decay': 0.0005, # ✅ Weight decay
        'warmup_epochs': 3.0,   # ✅ Warmup epochs (fractions ok)
        'warmup_momentum': 0.8, # ✅ Warmup initial momentum
        'warmup_bias_lr': 0.1,  # ✅ Warmup initial bias lr
        'box': 7.5,             # ✅ Box loss gain
        'cls': 0.5,             # ✅ Classification loss gain (감소)
        'dfl': 1.5,             # ✅ DFL loss gain
        'pose': 12.0,           # ✅ Pose loss gain (only for pose models)
        'kobj': 2.0,            # ✅ Keypoint obj loss gain (only for pose models)
        'hsv_h': 0.015,         # ✅ Image HSV-Hue augmentation (fraction)
        'hsv_s': 0.7,           # ✅ Image HSV-Saturation augmentation (fraction)
        'hsv_v': 0.4,           # ✅ Image HSV-Value augmentation (fraction)
        'degrees': 0.0,         # ✅ Image rotation (+/- deg)
        'translate': 0.1,       # ✅ Image translation (+/- fraction)
        'scale': 0.5,           # ✅ Image scale (+/- gain)
        'shear': 0.0,           # ✅ Image shear (+/- deg)
        'perspective': 0.0,     # ✅ Image perspective (+/- fraction), range 0-0.001
        'flipud': 0.0,          # ✅ Image flip up-down (probability)
        'fliplr': 0.5,          # ✅ Image flip left-right (probability)
        'mosaic': 1.0,          # ✅ Image mosaic (probability)
        'mixup': 0.0,           # ✅ Image mixup (probability)
        'copy_paste': 0.0,      # ✅ Image copy-paste (probability)
        'auto_augment': None,   # ✅ Auto augmentation policy for classification (randaugment, autoaugment, augmix)
        'erasing': 0.4,         # ✅ Random erasing probability during classification training (0-0.9), 0 to disable
        'crop_fraction': 1.0,   # ✅ Image crop fraction for classification (0.1-1.0)
    }
    
    # 🔧 특별히 dual stream 모델의 경우 더욱 안정적인 설정
    if is_dual_model or opt.dual_stream:
        LOGGER.info("🔧 Applying dual-stream specific training stabilization...")
        model_training_args.update({
            'lr0': 0.0005,      # ✅ dual stream의 경우 더욱 낮은 learning rate
            'cls': 0.25,        # ✅ Classification loss 더욱 감소
            'warmup_epochs': 5.0, # ✅ 더 긴 warmup
            'patience': 50,     # ✅ 더 긴 patience
        })
    
    # Start training
    LOGGER.info(f"Starting training for {opt.epochs} epochs...")
    LOGGER.info(f"🔧 Stability settings: lr0={model_training_args['lr0']}, cls_loss={model_training_args['cls']}")
    t0 = time.time()
    
    # Set up dual-stream handling if needed
    if is_dual_model or opt.dual_stream:
        LOGGER.info(f"Enabling dual-stream mode for training")
        # We'll use the dataset's custom attributes to handle dual-stream loading
        # No need to pass 'dual_stream' parameter to model.train() as it's not in the standard config
    
    # Train the model using the Ultralytics YOLO API
    try:
        LOGGER.info(f"Training arguments: {model_training_args}")
        results = model.train(**model_training_args)
        LOGGER.info(f"Training completed in {(time.time() - t0) / 3600:.3f} hours")
        
        # Evaluate on validation set
        if not opt.noval:
            LOGGER.info("Running final validation...")
            val_args = {
                'data': opt.data,
                'batch': opt.batch_size * 2,
            }
            if is_dual_model or opt.dual_stream:
                val_args['dual_stream'] = True
            results = model.val(**val_args)
        
        if is_dual_model or opt.dual_stream:
            LOGGER.info("🔍 Checking model training state...")
            
            total_params = 0
            trainable_params = 0
            frozen_layers = []
            
            for name, param in model.model.named_parameters():
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
                else:
                    frozen_layers.append(name)
            
            LOGGER.info(f"📊 Model Parameter Summary:")
            LOGGER.info(f"  Total parameters: {total_params:,}")
            LOGGER.info(f"  Trainable parameters: {trainable_params:,}")
            LOGGER.info(f"  Frozen parameters: {total_params - trainable_params:,}")
            LOGGER.info(f"  Trainable ratio: {trainable_params/total_params:.2%}")
            
            if frozen_layers:
                LOGGER.warning(f"🧊 Found {len(frozen_layers)} frozen layers:")
                for layer in frozen_layers[:10]:  # 처음 10개만 표시
                    LOGGER.warning(f"  - {layer}")
                if len(frozen_layers) > 10:
                    LOGGER.warning(f"  ... and {len(frozen_layers)-10} more")
            
            # 🔧 모든 레이어를 강제로 trainable로 설정
            LOGGER.info("🔓 Ensuring all parameters are trainable...")
            for param in model.model.parameters():
                param.requires_grad = True
            
            # 재확인
            trainable_after = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
            LOGGER.info(f"✅ After unfreezing: {trainable_after:,} trainable parameters")
        
        return results
    except Exception as e:
        LOGGER.error(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        raise e


def parse_opt(known=False):
    """Parses command-line arguments for YOLOv8 training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=ROOT / "yolov8n.pt", help="initial weights path")
    parser.add_argument("--cfg", type=str, default="", help="model yaml path")
    parser.add_argument("--data", type=str, default=ROOT / "ultralytics/cfg/datasets/swm_dual_updated.yaml", help="dataset.yaml path")
    parser.add_argument("--epochs", type=int, default=100, help="total training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="total batch size for all GPUs")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="train, val image size (pixels)")
    parser.add_argument("--rect", action="store_true", help="rectangular training")
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="resume most recent training")
    parser.add_argument("--nosave", action="store_true", help="only save final checkpoint")
    parser.add_argument("--noval", action="store_true", help="only validate final epoch")
    parser.add_argument("--cache", type=str, nargs="?", const="ram", help="image --cache ram/disk")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--single-cls", action="store_true", help="train multi-class data as single-class")
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW"], default="SGD", help="optimizer")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--project", default=ROOT / "runs/train", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--quad", action="store_true", help="quad dataloader")
    parser.add_argument("--cos-lr", action="store_true", help="cosine LR scheduler")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing epsilon")
    parser.add_argument("--patience", type=int, default=100, help="EarlyStopping patience (epochs without improvement)")
    parser.add_argument("--save-period", type=int, default=-1, help="Save checkpoint every x epochs (disabled if < 1)")
    parser.add_argument("--seed", type=int, default=0, help="Global training seed")
    parser.add_argument("--dual-stream", action="store_true", help="Enable dual-stream training mode")
    
    # ✅ Stability 관련 옵션 추가
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--stable-training", action="store_true", help="Use extra stable training settings for dual-stream")
    
    # Distributed training arguments
    parser.add_argument("--local_rank", type=int, default=-1, help="Automatic DDP Multi-GPU argument")
    
    # Logger arguments
    parser.add_argument("--entity", default=None, help="W&B entity")
    parser.add_argument("--upload_dataset", action="store_true", help="Upload dataset to W&B")
    parser.add_argument("--bbox_interval", type=int, default=-1, help="Set bounding-box image logging interval")
    parser.add_argument("--artifact_alias", type=str, default="latest", help="Version of dataset artifact to use")
    
    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt, callbacks=None):
    """Runs training with specified options and optional callbacks."""
    # Print arguments
    LOGGER.info(colorstr('Arguments: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    
    # Check files
    opt.data, opt.cfg, opt.weights = str(opt.data), str(opt.cfg), str(opt.weights)
    
    # Create save directory
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    
    # Set random seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    
    # Train
    device = select_device(opt.device)
    train(opt.cfg, opt, device, callbacks)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)