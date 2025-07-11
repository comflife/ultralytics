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
    # LOGGER.info(f"Starting YOLOv8 training in {save_dir}")
    
    # Load model
    if opt.resume:
        # LOGGER.info(f"Resuming training from {opt.weights}")
        model = YOLO(opt.weights)
    else:
        if opt.cfg and Path(opt.cfg).exists():
            # LOGGER.info(f"Loading model from config {opt.cfg}")
            model = YOLO(opt.cfg)  # Load from yaml config
            if opt.weights and opt.weights.endswith('.pt') and Path(opt.weights).exists():
                # LOGGER.info(f"Loading weights from {opt.weights}")
                model.load(opt.weights)  # Load pretrained weights
        else:
            # LOGGER.info(f"Loading pretrained model {opt.weights}")
            model = YOLO(opt.weights)

    # ✅ 모델 로딩 직후에 freeze 상태 체크 및 해제
    # LOGGER.info("🔍 Checking model parameters after loading...")
    
    # 모델을 training mode로 설정
    model.model.train()
    
    # Freeze 상태 체크
    total_params = 0
    trainable_params = 0
    frozen_layers = []
    
    for name, param in model.model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        else:
            frozen_layers.append(name)
    
    # LOGGER.info(f"📊 Initial Model Parameter Summary:")
    # LOGGER.info(f"  Total parameters: {total_params:,}")
    # LOGGER.info(f"  Trainable parameters: {trainable_params:,}")
    # LOGGER.info(f"  Frozen parameters: {total_params - trainable_params:,}")
    # LOGGER.info(f"  Trainable ratio: {trainable_params/total_params:.2%}")
    
    # ✅ 모든 파라미터를 trainable로 강제 설정 (학습 전에!)
    if trainable_params == 0 or len(frozen_layers) > 0:
        # LOGGER.warning(f"🧊 Found {len(frozen_layers)} frozen layers. Unfreezing all parameters...")
        
        for param in model.model.parameters():
            param.requires_grad = True
        
        # 재확인
        trainable_after = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    #     LOGGER.info(f"✅ After unfreezing: {trainable_after:,} trainable parameters ({trainable_after/total_params:.2%})")
    # else:
    #     LOGGER.info("✅ All parameters are already trainable")

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
                        
                        # if is_dual_model:
                        #     LOGGER.info(f"Detected dual-stream model with MultiStream modules")
                            # Mark as dual model but don't wrap - we'll rely on our dual_stream flag
        except Exception as e:
            # LOGGER.warning(f"Error checking model configuration: {e}")
            is_dual_model = False
            
    # Set dual stream attribute on model if needed
    if is_dual_model or opt.dual_stream:
        # LOGGER.info(f"Enabling dual-stream mode for training")
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
        'dual_stream': is_dual_model or opt.dual_stream,
        
        # 🔧 기본 Ultralytics YOLO CLI와 동일한 설정들
        'lr0': 0.01,            # ✅ 기본값 복원 (0.001 → 0.01)
        'lrf': 0.01,            # ✅ Final learning rate (lr0 * lrf)
        'momentum': 0.937,      # ✅ SGD momentum/Adam beta1 (기본값)
        'weight_decay': 0.0005, # ✅ Weight decay (기본값)
        'warmup_epochs': 3.0,   # ✅ Warmup epochs (기본값)
        'warmup_momentum': 0.8, # ✅ Warmup initial momentum (기본값)
        'warmup_bias_lr': 0.1,  # ✅ Warmup initial bias lr (기본값)
        'box': 7.5,             # ✅ Box loss gain (기본값)
        'cls': 0.5,             # ✅ Classification loss gain (기본값)
        'dfl': 1.5,             # ✅ DFL loss gain (기본값)
        'pose': 12.0,           # ✅ Pose loss gain (기본값)
        'kobj': 1.0,            # ✅ Keypoint obj loss gain (기본값으로 수정)
        'hsv_h': 0.015,         # ✅ Image HSV-Hue augmentation (기본값)
        'hsv_s': 0.7,           # ✅ Image HSV-Saturation augmentation (기본값)
        'hsv_v': 0.4,           # ✅ Image HSV-Value augmentation (기본값)
        'degrees': 0.0,         # ✅ Image rotation (기본값)
        'translate': 0.1,       # ✅ Image translation (기본값)
        'scale': 0.5,           # ✅ Image scale (기본값)
        'shear': 0.0,           # ✅ Image shear (기본값)
        'perspective': 0.0,     # ✅ Image perspective (기본값)
        'flipud': 0.0,          # ✅ Image flip up-down (기본값)
        'fliplr': 0.5,          # ✅ Image flip left-right (기본값)
        'mosaic': 1.0,          # ✅ Image mosaic (기본값)
        'mixup': 0.0,           # ✅ Image mixup (기본값)
        'copy_paste': 0.0,      # ✅ Image copy-paste (기본값)
        # 기본 CLI에서 지원하지 않는 파라미터들 제거
        # 'auto_augment': 'randaugment',  # ✅ 기본값 (하지만 detection에서는 None)
        # 'erasing': 0.4,         # ✅ Classification 전용이므로 제거
        # 'crop_fraction': 1.0,   # ✅ Classification 전용이므로 제거
        # 'conf': 0.25,          # ✅ validation 기본값이지만 train에서는 설정 안함
        # 'iou': 0.7,            # ✅ validation 기본값이지만 train에서는 설정 안함
    }
    
    # 🔧 특별히 dual stream 모델의 경우 더욱 안정적인 설정
    if is_dual_model or opt.dual_stream:
        # LOGGER.info("🔧 Applying dual-stream specific training stabilization...")
        
        # 🚀 Batch Size에 따른 Learning Rate Scaling
        base_lr = 0.002
        batch_scale_factor = opt.batch_size / 16  # 기준 batch_size=16
        scaled_lr = base_lr * batch_scale_factor
        
        model_training_args.update({
            # 🚀 Advanced Optimizer & Learning Rate Optimization
            'optimizer': 'AdamW',                    # ✅ AdamW가 dual-stream에 더 효과적
            'lr0': scaled_lr,                        # ✅ Batch size에 따른 scaled learning rate
            'lrf': 0.01,                            # ✅ Final learning rate factor
            'cos_lr': True,                         # ✅ Cosine learning rate scheduler 활성화
            'momentum': 0.9,                        # ✅ AdamW beta1 (momentum 역할)
            'weight_decay': 0.0001,                 # ✅ AdamW에 적합한 weight decay
            
            # 🚀 Advanced Warmup Strategy - Dual-stream 안정성
            'warmup_epochs': 5.0,                   # ✅ 긴 warmup (dual-stream feature alignment)
            'warmup_momentum': 0.5,                 # ✅ 낮은 초기 momentum
            'warmup_bias_lr': 0.05,                 # ✅ 낮은 bias learning rate
            
            # 🚀 Dual-Stream Loss Optimization
            'cls': 0.3,                             # ✅ Classification loss 미세 조정
            'box': 8.0,                             # ✅ Box loss 약간 증가 (정확도 향상)
            'dfl': 1.8,                             # ✅ DFL loss 약간 증가
            
            # 🚀 Advanced Training Techniques
            'label_smoothing': 0.05,                # ✅ Label smoothing으로 일반화 성능 향상
            'close_mosaic': 15,                     # ✅ 마지막 15 epochs는 mosaic 비활성화
            
            # 🚀 Dual-Stream Specific Augmentation
            'mosaic': 0.8,                          # ✅ Mosaic 약간 감소 (안정성)
            'mixup': 0.1,                           # ✅ Mixup 약간 활성화 (feature mixing)
            'copy_paste': 0.1,                      # ✅ Copy-paste 약간 활성화
            'translate': 0.05,                      # ✅ Translation 감소 (feature alignment)
            'scale': 0.3,                           # ✅ Scale 감소 (안정성)
            'hsv_h': 0.01,                          # ✅ HSV augmentation 감소 (dual-stream 안정성)
            'hsv_s': 0.5,                           # ✅ Saturation 감소
            'hsv_v': 0.2,                           # ✅ Value 감소
            
            # 🚀 Training Efficiency & Stability
            'save_period': 10,                      # ✅ 정기적 저장 (10 epochs마다)
            'patience': 50,                         # ✅ Early stopping patience
            'amp': True,                            # ✅ Mixed Precision 유지
            
            # 🚀 Advanced Training Features
            'rect': False,                          # ✅ Rectangular training 비활성화 (dual-stream 안정성)
            'multi_scale': True,                    # ✅ Multi-scale training 활성화
            
            # 🚀 Validation 관련 - dual-stream에 맞는 임계값
            # 참고: 이 값들은 validation 시에만 적용됨
        })
        
        # LOGGER.info(f"🚀 Applied dual-stream optimizations:")
        # LOGGER.info(f"  Optimizer: AdamW with scaled LR: {scaled_lr:.4f} (batch_size={opt.batch_size})")
        # LOGGER.info(f"  Cosine LR scheduler enabled with 5-epoch warmup")
        # LOGGER.info(f"  Enhanced augmentation and loss balancing for dual-stream")
    
    # Start training
    # LOGGER.info(f"Starting training for {opt.epochs} epochs...")
    # LOGGER.info(f"🔧 Stability settings: lr0={model_training_args['lr0']}, cls_loss={model_training_args['cls']}")
    t0 = time.time()
    
    # Set up dual-stream handling if needed
    if is_dual_model or opt.dual_stream:
        # LOGGER.info(f"Enabling dual-stream mode for training")
        # We'll use the dataset's custom attributes to handle dual-stream loading
        # No need to pass 'dual_stream' parameter to model.train() as it's not in the standard config
        pass
    
    # Train the model using the Ultralytics YOLO API
    try:
        # LOGGER.info(f"Training arguments: {model_training_args}")
        results = model.train(**model_training_args)
        # LOGGER.info(f"Training completed in {(time.time() - t0) / 3600:.3f} hours")
        
        # Evaluate on validation set
        if not opt.noval:
            # LOGGER.info("Running final validation...")
            val_args = {
                'data': opt.data,
                'batch': opt.batch_size * 2,
                'dual_stream': is_dual_model or opt.dual_stream,
            }
            
            # 🔥 Dual-stream 모델의 경우 더 관대한 validation 설정
            if is_dual_model or opt.dual_stream:
                val_args.update({
                    'conf': 0.1,   # ✅ 낮은 confidence threshold (0.25 → 0.1)
                    'iou': 0.3,    # ✅ 낮은 IoU threshold (0.7 → 0.3)
                })
            
            results = model.val(**val_args)
        
        # ✅ 학습 후 상태 체크 (inference mode 해제 없이)
        if is_dual_model or opt.dual_stream:
            # LOGGER.info("🔍 Final model training state check...")
            
            final_trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
            # LOGGER.info(f"📊 Final trainable parameters: {final_trainable:,} ({final_trainable/total_params:.2%})")
        
        return results
    except Exception as e:
        # LOGGER.error(f"Training error: {e}")
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
    # LOGGER.info(colorstr('Arguments: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    
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