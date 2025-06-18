# YOLOv8 🚀 by Ultralytics, AGPL-3.0 license
"""
Train a YOLOv8 model on a custom dataset.

Usage - Single-GPU training:
    $ python train_yolov8.py --data swm.yaml --weights yolov8n.pt --img 640  # from pretrained (recommended)
    $ python train_yolov8.py --data swm.yaml --weights yolov8n.pt --cfg yolov8n.yaml --img 640 --epochs 20

Usage - Multi-GPU DDP training:
    $ torchrun --nproc_per_node 2 train_yolov8.py --data coco.yaml --weights yolov8s.pt --device 0,1
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
        LOGGER.info(f"Loading model {opt.weights}")
        model = YOLO(opt.weights if opt.weights.endswith('.pt') else opt.cfg)
    
    # Configure training settings
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
    }
    
    # Start training
    LOGGER.info(f"Starting training for {opt.epochs} epochs...")
    t0 = time.time()
    
    # Train the model using the Ultralytics YOLO API
    try:
        results = model.train(**model_training_args)
        LOGGER.info(f"Training completed in {(time.time() - t0) / 3600:.3f} hours")
        
        # Evaluate on validation set
        if not opt.noval:
            LOGGER.info("Running final validation...")
            results = model.val(data=opt.data, batch=opt.batch_size * 2)
            # LOGGER.info(f"Validation results: {results}")
        
        return results
    except Exception as e:
        LOGGER.error(f"Training error: {e}")
        raise e


def parse_opt(known=False):
    """Parses command-line arguments for YOLOv8 training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=ROOT / "yolov8s.pt", help="initial weights path")
    parser.add_argument("--cfg", type=str, default="", help="model yaml path")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path")
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