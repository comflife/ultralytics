# YOLOv8 🚀 by Ultralytics, AGPL-3.0 license

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

# 🔴 Import tasks.py modules for CustomDetectionModel
from ultralytics.nn.tasks import DetectionModel

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
    
    # Load model
    if opt.resume:
        model = YOLO(opt.weights)
    else:
        if opt.cfg and Path(opt.cfg).exists():
            with open(opt.cfg, 'r') as f:
                cfg_dict = yaml.safe_load(f)
            
            cfg_dict['ch'] = 6
            
            from ultralytics.nn.tasks import DetectionModel
            custom_model_instance = DetectionModel(cfg=cfg_dict, verbose=False)

            if opt.weights and opt.weights.endswith('.pt') and Path(opt.weights).exists():
                print(f"⚠️  Skipping pretrained weights for dual-stream model: {opt.weights}")
                print("💡 Dual-stream architecture is incompatible with standard YOLOv8 weights")
                print("🚀 Training from scratch with random initialization")
            
            model = YOLO(task='detect')
            model.model = custom_model_instance
        else:
            model = YOLO(opt.weights)

    # Check and unfreeze parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if trainable_params < total_params:
        for param in model.parameters():
            param.requires_grad = True

    # Check if using a dual-stream model
    is_dual_model = False
    if opt.cfg:
        try:
            with open(opt.cfg, 'r') as f:
                model_yaml = yaml.safe_load(f)
                if model_yaml and isinstance(model_yaml, dict) and 'backbone' in model_yaml:
                    backbone = model_yaml['backbone']
                    if isinstance(backbone, list):
                        modules = [module[2] for module in backbone if len(module) > 2]
                        is_dual_model = any('MultiStream' in str(module) for module in modules)
        except Exception as e:
            is_dual_model = False
            
    if is_dual_model or opt.dual_stream:
        if hasattr(model, 'model'):
            model.model.dual_stream = True
        model.overrides = getattr(model, 'overrides', {})
        model.overrides['dual_stream'] = True
    
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
        'close_mosaic': 10,
        'resume': opt.resume,
        'amp': True,
        'fraction': 1.0,
        'profile': False,
        'val': True,
        'label_smoothing': opt.label_smoothing,
        'save_period': opt.save_period,
        'dual_stream': is_dual_model or opt.dual_stream,
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 12.0,
        'kobj': 1.0,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 0.3,
        'mixup': 0.0,
        'copy_paste': 0.0,
    }
    
    if is_dual_model or opt.dual_stream:
        base_lr = 0.001
        batch_scale_factor = opt.batch_size / 16
        scaled_lr = base_lr * batch_scale_factor
        
        model_training_args.update({
            'optimizer': 'AdamW',
            'lr0': scaled_lr,
            'lrf': 0.001,
            'cos_lr': True,
            'momentum': 0.9,
            'weight_decay': 0.001,
            'warmup_epochs': 10.0,
            'warmup_momentum': 0.1,
            'warmup_bias_lr': 0.01,
            'cls': 0.5,
            'box': 7.5,
            'dfl': 1.5,
            'label_smoothing': 0.1,
            'close_mosaic': 20,
            'mosaic': 0.8,
            'mixup': 0.1,
            'copy_paste': 0.05,
            'translate': 0.05,
            'scale': 0.3,
            'shear': 0.02,
            'degrees': 5.0,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'save_period': 5,
            'patience': 30,
            'rect': False,
            'multi_scale': False,
        })
    
    # Train the model using the Ultralytics YOLO API
    try:
        # ✅ This is the final, correct way to call the training function.
        results = model.train(**model_training_args)
        
        # Validation process removed for faster training
        print("🚀 Training completed without validation")
        
        return results
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e


def parse_opt(known=False):
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
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--stable-training", action="store_true", help="Use extra stable training settings for dual-stream")
    parser.add_argument("--local_rank", type=int, default=-1, help="Automatic DDP Multi-GPU argument")
    parser.add_argument("--entity", default=None, help="W&B entity")
    parser.add_argument("--upload_dataset", action="store_true", help="Upload dataset to W&B")
    parser.add_argument("--bbox_interval", type=int, default=-1, help="Set bounding-box image logging interval")
    parser.add_argument("--artifact_alias", type=str, default="latest", help="Version of dataset artifact to use")
    
    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt, callbacks=None):
    opt.data, opt.cfg, opt.weights = str(opt.data), str(opt.cfg), str(opt.weights)
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    
    device = select_device(opt.device)
    train(opt.cfg, opt, device, callbacks)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)