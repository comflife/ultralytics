# train_siamese_v2.py
# Siamese loss 없이 wide detection만 학습하는 버전

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import yaml
import shutil # For saving opt file

# Custom modules - ensure they are in PYTHONPATH or same directory
from yolov8_siamese import SiameseYOLOv8s
from dataload_siamese import SiameseDataset, get_siamese_train_transforms, get_siamese_val_transforms, siamese_collate_fn

from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.checks import check_amp, print_args
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.utils.loss import v8DetectionLoss
import time
from types import SimpleNamespace

DEFAULT_HYPERPARAMS_FILE = 'hyp_siamese_scratch.yaml'

def save_opt_yaml(opt, save_dir):
    """Saves the training options (argparse Namespace) to a YAML file."""
    opt_dict = vars(opt)
    opt_path = os.path.join(save_dir, 'options.yaml')
    try:
        with open(opt_path, 'w') as f:
            yaml.dump(opt_dict, f, sort_keys=False)
        LOGGER.info(f"Training options saved to {opt_path}")
    except Exception as e:
        LOGGER.warning(f"Could not save options.yaml: {e}")


def train(opt):
    # Setup device
    if opt.device:
        device = torch.device(opt.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LOGGER.info(f"Using device: {device}")
    if str(device) == 'cpu':
        LOGGER.warning("Training on CPU, this might be very slow. Consider using a GPU.")

    amp = device.type != 'cpu'
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # Create save directory and save options
    save_dir = os.path.join(opt.project, opt.name)
    os.makedirs(save_dir, exist_ok=True)
    save_opt_yaml(opt, save_dir)

    # DataLoaders
    LOGGER.info("Loading training data...")
    train_dataset = SiameseDataset(
        wide_img_dir=opt.wide_img_dir_train,
        narrow_img_dir=opt.narrow_img_dir_train,
        label_dir=opt.label_dir_train,
        img_size=opt.imgsz,
        transform=get_siamese_train_transforms(opt.imgsz)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.workers,
        pin_memory=True,
        collate_fn=siamese_collate_fn
    )

    # Model
    LOGGER.info(f"Initializing model with weights: {opt.weights}")
    import yaml
    if isinstance(opt.hyp, dict):
        model_args = opt.hyp
    elif hasattr(opt.hyp, '__dict__'):
        model_args = vars(opt.hyp)
    elif isinstance(opt.hyp, str):
        with open(opt.hyp, 'r') as f:
            model_args = yaml.safe_load(f)
    else:
        raise TypeError(f"opt.hyp must be a dict, str (YAML path), or have __dict__, got {type(opt.hyp)}")
    # --- Hardcoded dataset yaml path for nc (number of classes) extraction ---
    dataset_yaml = '/home/byounggun/ultralytics/ultralytics/cfg/datasets/coco.yaml'
    if os.path.exists(dataset_yaml):
        with open(dataset_yaml, 'r') as f:
            data_yaml = yaml.safe_load(f)
        names = data_yaml.get('names', None)
        if isinstance(names, dict):
            nc = len(names)
        elif isinstance(names, list):
            nc = len(names)
        else:
            nc = None
        if nc is not None:
            model_args['nc'] = nc
            LOGGER.info(f"[train_siamese_v2.py] Set number of classes (nc) to {nc} from dataset yaml: {dataset_yaml}")
        else:
            LOGGER.warning(f"[train_siamese_v2.py] Could not determine nc from dataset yaml: {dataset_yaml}")
    else:
        LOGGER.warning(f"[train_siamese_v2.py] Hardcoded dataset yaml not found: {dataset_yaml}, using nc from hyp or default")
    model = SiameseYOLOv8s(
        yolo_weights_path=opt.weights,
        siamese_lambda=0.0,  # disable siamese loss
        feature_dim=opt.siamese_feature_dim,
        args=model_args
    ).to(device)

    # --- Backbone freeze (전이학습 최적화, detection head만 학습) ---
    backbone_end = model.yolo_model.save[-1] + 1
    for param in model.yolo_model.model[:backbone_end].parameters():
        param.requires_grad = False  # backbone freeze
    for param in model.yolo_model.model[backbone_end:].parameters():
        param.requires_grad = True   # detection head는 반드시 풀어줌
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = optim.AdamW(trainable_params, lr=opt.lr0, weight_decay=opt.weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=opt.lr0, total_steps=opt.epochs * len(train_loader))

    # Detection loss criterion
    detection_criterion = v8DetectionLoss(model.yolo_model)
    box_gain, cls_gain, dfl_gain = opt.box_gain, opt.cls_gain, opt.dfl_gain

    try:
        import wandb
    except ImportError:
        wandb = None
        LOGGER.warning("Wandb is not installed. Wandb logging is disabled.")

    if wandb is not None:
        # wandb project name must not contain '/', etc. Use last part only
        wandb_project = os.path.basename(opt.project.rstrip('/')) if hasattr(opt, 'project') else 'siamese-yolov8'
        wandb.init(project=wandb_project, name=opt.name, config=vars(opt))
        LOGGER.info(f"Initialized wandb run: {wandb.run.id}")

    for epoch in range(opt.epochs):
        model.train()
        epoch_loss_total_sum = 0.0
        epoch_loss_detect_sum = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{opt.epochs}", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for i, (wide_imgs, narrow_imgs, wide_targets) in progress_bar:
            wide_imgs = wide_imgs.to(device, non_blocking=True)
            wide_targets = wide_targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp):
                # Wide 이미지만 사용, siamese branch는 무시
                detection_preds_wide, _ = model(wide_imgs, wide_imgs)  # narrow_imgs=wide_imgs로 전달
                # DEBUG: check requires_grad
                if isinstance(detection_preds_wide, (list, tuple)):
                    debug_tensor = detection_preds_wide[0]
                else:
                    debug_tensor = detection_preds_wide
                # print('[DEBUG] detection_preds_wide[0] requires_grad:', debug_tensor.requires_grad)
                if not debug_tensor.requires_grad:
                    raise RuntimeError('Detection head output does not require grad. Check optimizer and requires_grad settings.')
                loss_detect_val, loss_items_detect = detection_criterion(
                    detection_preds_wide,
                    {
                        'cls': wide_targets[:, 1:2],
                        'bboxes': wide_targets[:, 2:6],
                        'batch_idx': wide_targets[:, 0].long()
                    }
                )
                loss_box, loss_cls, loss_dfl = loss_items_detect[0], loss_items_detect[1], loss_items_detect[2]
                total_loss = loss_detect_val.sum()  # grad 연결된 scalar 텐서만 backward

            if not torch.all(torch.isfinite(total_loss)):
                LOGGER.warning(f"WARNING: Non-finite loss detected: {total_loss} at epoch {epoch+1}, batch {i}. Skipping this batch.")
                optimizer.zero_grad()
                continue

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss_total_sum += total_loss.item()
            epoch_loss_detect_sum += loss_detect_val.sum().item()

            if wandb is not None:
                wandb.log({
                    'step': epoch * len(train_loader) + i,
                    'epoch': epoch + 1,
                    'batch': i,
                    'loss/total': total_loss.item(),
                    'loss/detect': loss_detect_val.tolist(),
                    'loss/box': loss_box.item(),
                    'loss/cls': loss_cls.item(),
                    'loss/dfl': loss_dfl.item(),
                    'lr': optimizer.param_groups[0]['lr']
                })

            progress_bar.set_postfix({
                'total': f"{total_loss.item():.3f}",
                'detect': f"{loss_detect_val.tolist()}",
                'box': f"{loss_box:.3f}",
                'cls': f"{loss_cls:.3f}",
                'dfl': f"{loss_dfl:.3f}"
            })
            LOGGER.info(f"Epoch {epoch+1}/{opt.epochs}, batch {i}/{len(train_loader)}, lr={optimizer.param_groups[0]['lr']:.6f}, total={total_loss.item():.3f}, detect={loss_detect_val.tolist()}, box={loss_box:.3f}, cls={loss_cls:.3f}, dfl={loss_dfl:.3f}")

        avg_epoch_loss_total = epoch_loss_total_sum / len(train_loader)
        avg_epoch_loss_detect = epoch_loss_detect_sum / len(train_loader)
        LOGGER.info(f"Epoch {epoch + 1} Summary: Avg Total Loss: {avg_epoch_loss_total:.4f}, Avg Detect Loss: {avg_epoch_loss_detect:.4f}")
        if wandb is not None:
            wandb.log({
                'epoch': epoch + 1,
                'avg_total_loss': avg_epoch_loss_total,
                'avg_detect_loss': avg_epoch_loss_detect,
            })
        # Save checkpoint
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'loss': avg_epoch_loss_total,
        }
        torch.save(checkpoint, os.path.join(save_dir, f"ckpt_epoch_{epoch+1}.pth"))

    if opt.wandb and wandb is not None:
        wandb.finish()


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8s.pt', help='initial weights path')
    parser.add_argument('--hyp', type=str, default=DEFAULT_HYPERPARAMS_FILE, help='hyperparameters yaml')
    parser.add_argument('--project', type=str, default='runs/my_siamese_training', help='project dir')
    parser.add_argument('--name', type=str, default='exp', help='experiment name')
    parser.add_argument('--device', type=str, default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=4, help='dataloader workers')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--lr0', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--grad-accum-steps', type=int, default=2)
    parser.add_argument('--siamese-feature-dim', type=int, default=128)
    parser.add_argument('--detect-loss-weight', type=float, default=1.0)
    parser.add_argument('--box-gain', type=float, default=7.5)
    parser.add_argument('--cls-gain', type=float, default=0.5)
    parser.add_argument('--dfl-gain', type=float, default=1.5)
    parser.add_argument('--wide-img-dir-train', type=str, required=True)
    parser.add_argument('--narrow-img-dir-train', type=str, required=True)
    parser.add_argument('--label-dir-train', type=str, required=True)
    parser.add_argument('--wandb', action='store_true', help='use Weights & Biases logging')
    opt = parser.parse_args() if not known else parser.parse_known_args()[0]

    # YAML hyperparameter override
    if opt.hyp and os.path.isfile(opt.hyp):
        try:
            with open(opt.hyp, 'r') as f:
                hyp_yaml = yaml.safe_load(f)
            for k, v in hyp_yaml.items():
                if hasattr(opt, k):
                    setattr(opt, k, v)
                else:
                    LOGGER.warning(f"Hyperparameter '{k}' from YAML '{opt.hyp}' not found in ArgumentParser args.")
        except Exception as e:
            LOGGER.error(f"Error loading hyperparameters from {opt.hyp}: {e}")
    elif opt.hyp:
        LOGGER.warning(f"Hyperparameter YAML file not found: {opt.hyp}. Using defaults and CLI args.")
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    print_args(vars(opt))
    train(opt)
