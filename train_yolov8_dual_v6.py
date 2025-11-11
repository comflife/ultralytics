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
    Trains Dual-Stream YOLOv8 model with given configuration, options, and device.
    
    `cfg` argument is path/to/config.yaml or configuration dictionary.
    """
    callbacks = callbacks or get_default_callbacks()
    save_dir = Path(opt.save_dir)
    
    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save run settings
    with open(save_dir / "opt.yaml", 'w') as f:
        yaml.dump(vars(opt), f)
    
    # Load dual-stream model
    if opt.resume:
        model = YOLO(opt.weights)
    else:
        with open(opt.cfg, 'r') as f:
            cfg_dict = yaml.safe_load(f)
        
        cfg_dict['ch'] = 6  # Dual-stream: RGB (3) + Depth (3)
        
        # Pre-process custom module args: remove explicit c1 from SpatialAlignedMultiStreamConv args
        # Let parse_model prepend the correct ch[f] (input channels from previous layer)
        for layer in cfg_dict['backbone'] + cfg_dict['head']:
            if len(layer) > 2 and layer[2] == 'SpatialAlignedMultiStreamConv':
                args = layer[3]
                if isinstance(args, list) and len(args) > 1:
                    # Remove first arg (explicit c1), parse_model will prepend actual ch[f]
                    layer[3] = args[1:]
                    print(f"✅ Adjusted SpatialAlignedMultiStreamConv args: {layer[3]} (c1 will be prepended)")
        
        from ultralytics.nn.tasks import DetectionModel
        custom_model_instance = DetectionModel(cfg=cfg_dict, verbose=False)
        
        model = YOLO(task='detect')
        model.model = custom_model_instance
        
        print("🚀 Training Dual-Stream model from scratch with random initialization")

    # Unfreeze all parameters
    for param in model.parameters():
        param.requires_grad = True
    
    # Set dual-stream flags
    if hasattr(model, 'model'):
        model.model.dual_stream = True
    model.overrides = getattr(model, 'overrides', {})
    model.overrides['dual_stream'] = True
    
    # Configure dual-stream training settings
    base_lr = 0.001
    batch_scale_factor = opt.batch_size / 16
    scaled_lr = base_lr * batch_scale_factor
    
    model_training_args = {
        'data': opt.data,
        'epochs': opt.epochs,
        'batch': opt.batch_size,
        'imgsz': opt.imgsz,
        'save': not opt.nosave,
        'cache': opt.cache,
        'device': device,
        'workers': opt.workers,
        'project': opt.project,
        'name': opt.name,
        'exist_ok': opt.exist_ok,
        'pretrained': False,
        'verbose': True,
        'seed': opt.seed,
        'deterministic': True,
        'single_cls': opt.single_cls,
        'resume': opt.resume,
        'amp': True,
        'fraction': 1.0,
        'profile': False,
        'val': True,
        'dual_stream': True,
        # Dual-stream optimized hyperparameters
        'optimizer': 'AdamW',
        'lr0': scaled_lr,
        'lrf': 0.001,
        'cos_lr': True,
        'momentum': 0.9,
        'weight_decay': 0.001,
        'warmup_epochs': 10.0,
        'warmup_momentum': 0.1,
        'warmup_bias_lr': 0.01,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'label_smoothing': 0.1,
        'close_mosaic': 20,
        'patience': 30,
        'save_period': 5,
        'rect': False,
        # Augmentation settings
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 5.0,
        'translate': 0.05,
        'scale': 0.3,
        'shear': 0.02,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        # ⚠️ Mosaic/MixUp/CopyPaste disabled for dual-stream (need special handling)
        'mosaic': 0.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
    }
    
    # Train the dual-stream model
    try:
        results = model.train(**model_training_args)
        print("🚀 Dual-Stream training completed")
        return results
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e


def parse_opt(known=False):
    parser = argparse.ArgumentParser(description="Train Dual-Stream YOLOv8")
    parser.add_argument("--weights", type=str, default="", help="resume weights path (optional)")
    parser.add_argument("--cfg", type=str, required=True, help="dual-stream model yaml path")
    parser.add_argument("--data", type=str, default=ROOT / "ultralytics/cfg/datasets/swm_dual_updated.yaml", help="dataset.yaml path")
    parser.add_argument("--epochs", type=int, default=100, help="total training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="total batch size for all GPUs")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="train, val image size (pixels)")
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="resume most recent training")
    parser.add_argument("--nosave", action="store_true", help="only save final checkpoint")
    parser.add_argument("--cache", type=str, nargs="?", const="ram", help="image --cache ram/disk")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--single-cls", action="store_true", help="train multi-class data as single-class")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers")
    parser.add_argument("--project", default=ROOT / "runs/train", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--seed", type=int, default=0, help="Global training seed")
    parser.add_argument("--local_rank", type=int, default=-1, help="Automatic DDP Multi-GPU argument")
    
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