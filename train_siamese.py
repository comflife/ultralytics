# train_siamese.py
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

###
# python train_siamese.py --hyp hyp_siamese_scratch.yaml \
#     --weights /home/byounggun/ultralytics/yolov8s.pt \
#     --wide-img-dir-train /home/byounggun/cococo/train_resized/images \
#     --narrow-img-dir-train /home/byounggun/cococo/train_short/images \
#     --label-dir-train /home/byounggun/cococo/train_resized/labels \
#     --project runs/my_siamese_training \
#     --name experiment2 \
#     --workers 4
###

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
    if str(device) == 'cpu': # Log a warning if no GPU is used.
        LOGGER.warning("Training on CPU, this might be very slow. Consider using a GPU.")

    # AMP (Automatic Mixed Precision)
    amp = device.type != 'cpu'  # Use AMP for CUDA, but not for CPU
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # Detection loss criterion (official YOLOv8 loss)
    # detection_criterion = None


    # Create save directory and save options
    save_dir = os.path.join(opt.project, opt.name)
    os.makedirs(save_dir, exist_ok=True)
    save_opt_yaml(opt, save_dir) # Save the run options

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
        collate_fn=siamese_collate_fn,
        pin_memory=True,
        drop_last=True # Might be useful if batch norm layers are sensitive to small last batches
    )
    
    # TODO: Validation Loader (if opt.do_val is True)
    # val_loader = None
    # if opt.do_val:
    #    ... setup val_loader ...

    # Model
    LOGGER.info(f"Initializing model with weights: {opt.weights}")
    # Always pass hyp as dict to SiameseYOLOv8s
    import yaml
    if isinstance(opt.hyp, dict):
        model_args = opt.hyp
    elif hasattr(opt.hyp, '__dict__'):
        model_args = vars(opt.hyp)
    elif isinstance(opt.hyp, str):
        # opt.hyp가 파일명인 경우 YAML로 로드
        with open(opt.hyp, 'r') as f:
            model_args = yaml.safe_load(f)
    else:
        raise TypeError(f"opt.hyp must be a dict, str (YAML path), or have __dict__, got {type(opt.hyp)}")
    # --- Hardcoded dataset yaml path for nc (number of classes) extraction ---
    dataset_yaml = '/home/byounggun/ultralytics/ultralytics/cfg/datasets/coco.yaml'
    if os.path.exists(dataset_yaml):
        with open(dataset_yaml, 'r') as f:
            data_yaml = yaml.safe_load(f)
        # names can be list or dict
        names = data_yaml.get('names', None)
        if isinstance(names, dict):
            nc = len(names)
        elif isinstance(names, list):
            nc = len(names)
        else:
            nc = None
        if nc is not None:
            model_args['nc'] = nc
            LOGGER.info(f"[train_siamese.py] Set number of classes (nc) to {nc} from dataset yaml: {dataset_yaml}")
        else:
            LOGGER.warning(f"[train_siamese.py] Could not determine nc from dataset yaml: {dataset_yaml}")
    else:
        LOGGER.warning(f"[train_siamese.py] Hardcoded dataset yaml not found: {dataset_yaml}, using nc from hyp or default")
    model = SiameseYOLOv8s(
        yolo_weights_path=opt.weights,
        siamese_lambda=opt.siamese_lambda,
        feature_dim=opt.siamese_feature_dim,
        args=model_args
    ).to(device)

    # Detection loss 생성 (model.yolo_model.args is always dict)
    detection_criterion = v8DetectionLoss(model.yolo_model)

    # wandb init
    try:
        import wandb
    except ImportError:
        wandb = None
        print("[WARNING] wandb가 설치되어 있지 않습니다. 'pip install wandb'로 설치 후 사용하세요.")
    if wandb is not None:
        wandb.init(project="siamese-yolov8", name=f"run_{int(time.time())}", config=vars(opt))
        wandb.watch(model, log="all")


    # Optimizer
    # 백본 동결 (전이 학습 최적화)
    LOGGER.info("Freezing backbone for transfer learning optimization...")
    for param in model.yolo_model.model[:model.yolo_model.save[-1]+1].parameters():
        param.requires_grad = False
    
    # 훈련 가능한 파라미터만 필터링
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    LOGGER.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,} / {sum(p.numel() for p in model.parameters()):,}")
    
    # AdamW 옵티마이저 사용 (SGD보다 일반적으로 더 효과적)
    optimizer = optim.AdamW(trainable_params, lr=opt.lr0, weight_decay=opt.weight_decay, betas=(0.9, 0.999))
    # optimizer = optim.SGD(trainable_params, lr=opt.lr0, weight_decay=opt.weight_decay, momentum=0.9)
    
    # OneCycleLR 스케줄러 (학습률을 동적으로 조정하여 수렴 속도 향상)
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=opt.lr0, 
        epochs=opt.epochs, 
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,  # 최대 학습률까지 상승하는 시간 비율
        div_factor=10,  # 초기 학습률 = max_lr/div_factor
        final_div_factor=10000  # 최종 학습률 = max_lr/(div_factor*final_div_factor)
    )

    LOGGER.info(f"{colorstr('Starting training:')} for {opt.epochs} epochs on {device} device...")
    LOGGER.info(f"Hyperparameters: {vars(opt)}") # Log all options
    best_total_loss = float('inf')
    last_opt_step = -1 # For EMA if implemented
    
    # Set model in training mode
    best_total_loss = float('inf')
    unfreeze_epoch = 1  # 5에폭 이후 백본 동결 해제
    # log_batch_interval 옵션이 없으면 기본값 설정
    if not hasattr(opt, 'log_batch_interval'):
        opt.log_batch_interval = 50

    for epoch in range(opt.epochs):
        if wandb is not None:
            wandb.log({"epoch": epoch}, step=epoch)

        # 백본 동결 해제 로직: 초기 에폭에서는 백본을 고정하고, 이후에 전체 모델을 미세 조정
        if epoch == unfreeze_epoch:
            LOGGER.info(f"Unfreezing backbone at epoch {epoch+1} for fine-tuning...")
            for param in model.yolo_model.model[:model.yolo_model.save[-1]+1].parameters():
                param.requires_grad = True
            
            # 모든 파라미터를 학습 가능하게 설정한 후 옵티마이저 재설정
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            LOGGER.info(f"Trainable parameters after unfreezing: {sum(p.numel() for p in trainable_params):,}")
            
            # 백본 동결 해제 후 학습률 조정 (더 낮은 학습률로 미세 조정)
            for g in optimizer.param_groups:
                g['lr'] = opt.lr0 / 1  # 백본 미세 조정을 위해 더 낮은 학습률 사용
        model.train()
        epoch_loss_total_sum = 0.0
        epoch_loss_detect_sum = 0.0
        epoch_loss_siamese_sum = 0.0
        
        # tqdm progress bar
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{opt.epochs}", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        
        optimizer.zero_grad() # Zero gradients once per epoch or N iterations
        for i, (wide_imgs, narrow_imgs, wide_targets) in progress_bar:
            # Data to device
            wide_imgs = wide_imgs.to(device, non_blocking=True)
            narrow_imgs = narrow_imgs.to(device, non_blocking=True)
            wide_targets = wide_targets.to(device, non_blocking=True) # Format: (total_objs, 6) [batch_idx, cls, x, y, w, h]

            # Forward pass
            with torch.cuda.amp.autocast(enabled=amp):
                # Forward pass through the model
                detection_preds_wide, siamese_loss = model(wide_imgs, narrow_imgs)

                # detection_preds_wide가 list/tuple이면 전체를 loss에 넘김
                if isinstance(detection_preds_wide, (list, tuple)):
                    pred = detection_preds_wide
                else:
                    pred = detection_preds_wide

                
                # 손실 가중치 설정
                box_gain = opt.box_gain if hasattr(opt, 'box_gain') else 7.5
                cls_gain = opt.cls_gain if hasattr(opt, 'cls_gain') else 0.5
                dfl_gain = opt.dfl_gain if hasattr(opt, 'dfl_gain') else 1.5
                
                # pred가 list면 pred[0].device, 아니면 pred.device
                if isinstance(pred, list):
                    device = pred[0].device
                else:
                    device = pred.device
                h = model.yolo_model.model[-1]
                nc = model.nc  # 클래스 수
                
                # 공식 YOLOv8 detection loss 사용
                if len(wide_targets):
                    loss_detect_val, loss_items_detect = detection_criterion(
                        detection_preds_wide,  # preds가 첫 번째 인자
                        {
                            'img': wide_imgs,
                            'cls': wide_targets[:, 1:2],
                            'bboxes': wide_targets[:, 2:6],
                            'batch_idx': wide_targets[:, 0].long()
                        }
                    )
                    loss_box, loss_cls, loss_dfl = loss_items_detect[0], loss_items_detect[1], loss_items_detect[2]
                # else:
                #     loss_detect_val = torch.tensor(0.0, device=device)
                #     loss_box = torch.tensor(0.0, device=device)
                #     loss_cls = torch.tensor(0.0, device=device)
                #     loss_dfl = torch.tensor(0.0, device=device)
            
            # Combine losses using weights from opt (both should be scalars now)
            total_loss = opt.detect_loss_weight * (box_gain * loss_box + cls_gain * loss_cls + dfl_gain * loss_dfl) + opt.siamese_loss_weight * siamese_loss

            # 디버깅: 첫 배치에서 feature, label 등 shape/log 찍기
            if epoch == 0 and i == 0:
                print("[DEBUG] wide_imgs:", wide_imgs.shape, wide_imgs.min().item(), wide_imgs.max().item())
                print("[DEBUG] wide_targets:", wide_targets[:5])
                if 'detection_preds_wide' in locals():
                    if isinstance(detection_preds_wide, (list, tuple)):
                        print("[DEBUG] detection_preds_wide[0].shape:", detection_preds_wide[0].shape)
                    else:
                        print("[DEBUG] detection_preds_wide.shape:", detection_preds_wide.shape)
                print("[DEBUG] loss_box:", loss_box.item(), "loss_cls:", loss_cls.item(), "loss_dfl:", loss_dfl.item())
                print("[DEBUG] siamese_loss:", siamese_loss.item())
                if wandb is not None:
                    wandb.log({
                        "debug/wide_imgs_min": wide_imgs.min().item(),
                        "debug/wide_imgs_max": wide_imgs.max().item(),
                        "debug/loss_box": loss_box.item(),
                        "debug/loss_cls": loss_cls.item(),
                        "debug/loss_dfl": loss_dfl.item(),
                        "debug/siamese_loss": siamese_loss.item(),
                    }, step=epoch*len(progress_bar)+i)

            if not torch.all(torch.isfinite(total_loss)):
                LOGGER.warning(f"WARNING: Non-finite loss detected: {total_loss} at epoch {epoch+1}, batch {i}. Skipping this batch.")
                if wandb is not None:
                    wandb.log({"warning/nonfinite_loss": total_loss.item()}, step=epoch*len(progress_bar)+i)
                optimizer.zero_grad() # Clear gradients for this problematic batch
                continue

            # AMP Backward pass
            scaler.scale(total_loss).backward()

            # 배치 동안의 손실 저장
            epoch_loss_total_sum += total_loss.item()
            epoch_loss_detect_sum += loss_detect_val.sum().item()
            epoch_loss_siamese_sum += siamese_loss.item()

            # wandb 로깅 (매 배치)
            if wandb is not None:
                wandb.log({
                    "loss/total": total_loss.item(),
                    "loss/detect": loss_detect_val.sum().item() if hasattr(loss_detect_val, 'sum') else float(loss_detect_val),
                    "loss/box": loss_box.item() if hasattr(loss_box, 'item') else float(loss_box),
                    "loss/cls": loss_cls.item() if hasattr(loss_cls, 'item') else float(loss_cls),
                    "loss/dfl": loss_dfl.item() if hasattr(loss_dfl, 'item') else float(loss_dfl),
                    "loss/siamese": siamese_loss.item() if hasattr(siamese_loss, 'item') else float(siamese_loss),
                    "lr": optimizer.param_groups[0]['lr'],
                    "batch": i,
                    "epoch": epoch
                }, step=epoch*len(progress_bar)+i)

            # 그래디언트 업데이트 및 옵티마이저 스텝
            if (i + 1) % opt.grad_accum_steps == 0 or i == len(progress_bar) - 1:
                scaler.step(optimizer) # 모델 파라미터 업데이트
                scaler.update() # 추론 스케일 업데이트
                optimizer.zero_grad() # 그래디언트 초기화
                
                # OneCycleLR 스케줄러 스텝
                scheduler.step()
            
            # 로깅
            if i % opt.log_batch_interval == 0:
                current_lr = [param_group['lr'] for param_group in optimizer.param_groups][0] # 첫 단 학습률 가져오기
                
                if isinstance(loss_items_detect, torch.Tensor) and loss_items_detect is not None:
                    # YOLOv8 손실 구성요소를 각각 분리하여 로깅
                    box_loss, cls_loss, dfl_loss = loss_items_detect.detach().cpu().numpy()
                    
                    # 프로그레스 바 업데이트
                    progress_bar.set_postfix({
                        'lr': f"{current_lr:.6f}", 
                        'total': f"{total_loss.item():.3f}",
                        'box': f"{box_loss:.3f}", 
                        'cls': f"{cls_loss:.3f}", 
                        'dfl': f"{dfl_loss:.3f}",
                        'siamese': f"{siamese_loss.item():.3f}"
                    })
                    
                    # 상세 로그 출력
                    LOGGER.info(f"Epoch {epoch+1}/{opt.epochs}, batch {i}/{len(train_loader)}, "
                              f"lr={current_lr:.6f}, total={total_loss.item():.3f}, siamese={siamese_loss.item():.3f}, "
                              f"box={box_loss:.3f}, cls={cls_loss:.3f}, dfl={dfl_loss:.3f}, "
                              f"siamese={siamese_loss.item():.3f}")
                else:
                    # 단순 로그 (상세 손실이 없는 경우)
                    progress_bar.set_postfix({
                        'lr': f"{current_lr:.6f}", 
                        'total': f"{total_loss.item():.3f}",
                        'detect': f"{loss_detect_val.item():.3f}", 
                        'siamese': f"{siamese_loss.item():.3f}"
                    })
                    
                    LOGGER.info(f"Epoch {epoch+1}/{opt.epochs}, batch {i}/{len(train_loader)}, "
                              f"lr={current_lr:.6f}, total_loss={total_loss.item():.3f}, "
                              f"detect_loss={loss_detect_val.item():.3f}, siamese_loss={siamese_loss_val.item():.3f}")
            
        # End of epoch
        # if scheduler: scheduler.step() # Step LR scheduler

        avg_epoch_loss_total = epoch_loss_total_sum / len(train_loader)
        avg_epoch_loss_detect = epoch_loss_detect_sum / len(train_loader)
        avg_epoch_loss_siamese = epoch_loss_siamese_sum / len(train_loader)

        LOGGER.info(f"Epoch {epoch + 1} Summary: Avg Total Loss: {avg_epoch_loss_total:.4f}, "
                    f"Avg Detect Loss: {avg_epoch_loss_detect:.4f}, Avg Siamese Loss: {avg_epoch_loss_siamese:.4f}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': de_parallel(model).state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_epoch_loss_total,
            'opt': opt # Save run options
        }
        torch.save(checkpoint, os.path.join(save_dir, f'epoch_{epoch+1}.pt'))
        if avg_epoch_loss_total < best_total_loss:
            best_total_loss = avg_epoch_loss_total
            torch.save(checkpoint, os.path.join(save_dir, 'best.pt'))
            LOGGER.info(f"New best model saved to {os.path.join(save_dir, 'best.pt')} with total loss {best_total_loss:.4f}")
        torch.save(checkpoint, os.path.join(save_dir, 'last.pt')) # Save last epoch

        # TODO: Validation (run_validation function)
        # if opt.do_val and val_loader:
        #    run_validation(...)

    LOGGER.info(f"{colorstr('Finished training.')} Best model saved at {os.path.join(save_dir, 'best.pt')}")
    return os.path.join(save_dir, 'best.pt')


def parse_opt(known=False):
    parser = argparse.ArgumentParser(description="Train Siamese YOLOv8s Model")
    
    # Paths and Basic Config
    parser.add_argument('--weights', type=str, default='yolov8s.pt', help='Initial weights path for YOLOv8 backbone/head')
    parser.add_argument('--hyp', type=str, default=DEFAULT_HYPERPARAMS_FILE, help='Path to hyperparameters YAML file')
    parser.add_argument('--project', default='runs/train_siamese', help='Save to project/name')
    parser.add_argument('--name', default='exp', help='Save to project/name')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=4, help='Max dataloader workers (per RANK in DDP)')

    # Training Hyperparameters (will be overridden by --hyp file if specified)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--lr0', type=float, default=0.001, help='initial learning rate') # AdamW typically uses smaller LR
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum/Adam beta1') # AdamW beta1
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--grad-accum-steps', type=int, default=1, help='gradient accumulation steps')


    # Siamese Specific Hyperparameters
    parser.add_argument('--siamese-lambda', type=float, default=0.1, help='DEPRECATED: internal model lambda, use --siamese-loss-weight')
    parser.add_argument('--siamese-feature-dim', type=int, default=128, help='Dimension of Siamese embedding vector')
    parser.add_argument('--siamese-loss-weight', type=float, default=0.1, help='Weight for the Siamese loss component')
    parser.add_argument('--detect-loss-weight', type=float, default=1.0, help='Weight for the detection loss component')

    # Detection Loss Gains (from Ultralytics defaults for scratch training)
    parser.add_argument('--box-gain', type=float, default=7.5, help='box loss gain')
    parser.add_argument('--cls-gain', type=float, default=0.5, help='cls loss gain (scale with num_classes)')
    parser.add_argument('--dfl-gain', type=float, default=1.5, help='dfl loss gain')

    # Data paths
    parser.add_argument('--wide-img-dir-train', type=str, default='dataset/train/wide/images')
    parser.add_argument('--narrow-img-dir-train', type=str, default='dataset/train/narrow/images')
    parser.add_argument('--label-dir-train', type=str, default='dataset/train/wide/labels')
    # parser.add_argument('--wide-img-dir-val', type=str, default='dataset/val/wide/images') # Add if validation needed
    # parser.add_argument('--narrow-img-dir-val', type=str, default='dataset/val/narrow/images')
    # parser.add_argument('--label-dir-val', type=str, default='dataset/val/wide/labels')
    # parser.add_argument('--do-val', action='store_true', help='Perform validation during training')


    args = parser.parse_args()

    # Load hyperparameters from YAML if specified
    if args.hyp and os.path.exists(args.hyp):
        with open(args.hyp, 'r') as f:
            try:
                hyp_yaml = yaml.safe_load(f)
                # Override argparse defaults with YAML values
                # CLI arguments will take precedence over YAML if both are specified for the same arg
                # This merge logic can be tricky. A common way: args has defaults, YAML overrides defaults, CLI overrides YAML.
                # Current argparse behavior: CLI overrides default. If YAML is loaded, it can override the (default or CLI-set) arg.
                # To make CLI override YAML: load YAML first, then parse_args with updated defaults from YAML.
                # Simpler: update args Namespace with YAML values if the arg wasn't explicitly set via CLI.
                # For now, YAML values will override argparse defaults. CLI args will override both.
                # This means we'd need to parse known_args, load yaml, then re-parse or update.
                # Easiest: just update the args Namespace after parsing.
                for k, v in hyp_yaml.items():
                    if hasattr(args, k):
                        setattr(args, k, v)
                    else:
                        LOGGER.warning(f"Hyperparameter '{k}' from YAML '{args.hyp}' not found in ArgumentParser args.")
            except Exception as e:
                LOGGER.error(f"Error loading hyperparameters from {args.hyp}: {e}")
    elif args.hyp:
        LOGGER.warning(f"Hyperparameter YAML file not found: {args.hyp}. Using defaults and CLI args.")
    
    # If siamese_lambda (deprecated) is used and loss weights are default, give siamese_lambda precedence
    if args.siamese_lambda != 0.1 and args.siamese_loss_weight == 0.1 and args.detect_loss_weight == 1.0:
        args.siamese_loss_weight = args.siamese_lambda
        LOGGER.warning(f"'--siamese-lambda' is deprecated. Its value ({args.siamese_lambda}) has been used for '--siamese-loss-weight'. "
                       f"Please use '--siamese-loss-weight' and '--detect-loss-weight' in the future.")


    return args

if __name__ == '__main__':
    opt = parse_opt()
    print_args(vars(opt)) # Log arguments
    train(opt)