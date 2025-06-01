# -*- coding: utf-8 -*-
# yolov8_dual.py
"""
Dual YOLO 학습 스크립트
- wide/narrow 이미지를 각각 backbone에 통과시켜 feature를 concat한 뒤 detection head에 입력
- 라벨은 wide에만 존재
- dataload_dual.py와 연동


python yolov8_dual.py   --wide_img_dir_train /home/byounggun/ultralytics/traffic_train/wide/images   --narrow_img_dir_train /home/byounggun/ultralytics/traffic_train/narrow/images   --label_dir_train /home/byounggun/ultralytics/traffic_train/wide/labels   --weights yolov8s.pt   --img_size 640   --batch_size 12   --epochs 150   --wandb
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.checks import check_amp, print_args
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.utils.loss import v8DetectionLoss
# from dataload_dual import DualImageDataset, get_dual_train_transforms, get_dual_val_transforms, dual_collate_fn
from dataload_dual import DualImageDataset, dual_collate_fn
import numpy as np

# wandb import with fallback
try:
    import wandb
except ImportError:
    wandb = None
    LOGGER.warning("Wandb is not installed. Wandb logging is disabled.")

# --- 모델 정의 ---
def get_default_fusion_bbox(xw_out):
    """
    xw_out: (B, C, H, W) or (C, H, W)
    Returns (fx_min, fx_max, fy_min, fy_max) for the central square region.
    """
    if isinstance(xw_out, torch.Tensor):
        shape = xw_out.shape
        if len(shape) == 4:
            _, C, H, W = shape
        elif len(shape) == 3:
            C, H, W = shape
        else:
            raise ValueError("xw_out must be (B,C,H,W) or (C,H,W)")
    else:
        raise ValueError("xw_out must be a torch.Tensor")
    region_size = min(H, W) // 2
    fx_min = (W - region_size) // 2
    fx_max = fx_min + region_size
    fy_min = (H - region_size) // 2
    fy_max = fy_min + region_size
    return fx_min, fx_max, fy_min, fy_max

__all__ = ["DualYOLOv8", "get_default_fusion_bbox"]

class DualYOLOv8(nn.Module):
    def __init__(self, yolo_weights_path='yolov8s.pt', args=None,
                 wide_K=None, wide_P=None, narrow_K=None, narrow_P=None, img_w=1920, img_h=1080, img_size=640):
        super().__init__()
        from ultralytics import YOLO
        # 수정: pt 파일이면 가중치까지 로드, 아니면 구조만 생성
        if yolo_weights_path and yolo_weights_path.endswith('.pt') and os.path.exists(yolo_weights_path):
            yolo_model_full = YOLO(yolo_weights_path)
        # else:
        #     yolo_model_full = YOLO('/home/byounggun/ultralytics/ultralytics/cfg/models/v8/yolov8.yaml')
        self.yolo_model = yolo_model_full.model  # DetectionModel
        self.model = self.yolo_model.model  # for v8DetectionLoss compatibility (nn.ModuleList)
        self.stride = self.yolo_model.stride
        import yaml
        dataset_yaml = 'ultralytics/cfg/datasets/traffic.yaml'
        self.nc = None
        self.names = None
        if os.path.exists(dataset_yaml):
            with open(dataset_yaml, 'r') as f:
                data_yaml = yaml.safe_load(f)
            if isinstance(data_yaml, dict):
                self.nc = data_yaml.get('nc', None)
                self.names = data_yaml.get('names', None)
        detect_head = self.model[-1]
        if self.nc is None:
            self.nc = getattr(detect_head, 'nc', getattr(self.yolo_model, 'nc', None))
        if self.names is None:
            self.names = getattr(detect_head, 'names', getattr(self.yolo_model, 'names', None))
        self.save = self.yolo_model.save  # backbone output indices for multi-scale features
        self.args = args
        # --- 카메라 파라미터 저장 ---
        self.wide_K = wide_K
        self.wide_P = wide_P
        self.narrow_K = narrow_K
        self.narrow_P = narrow_P
        self.img_w = img_w
        self.img_h = img_h
        self.img_size = img_size if img_size is not None else (args.img_size if args and hasattr(args, 'img_size') else 640)

        # --- Feature Fusion Normalization & Learnable Weights (Multi-Scale) ---
        # Use separate BN for wide/narrow at each fusion scale (YOLOv8 official style)
        fusion_channels = []
        for i in self.save:
            layer = self.model[i]
            if hasattr(layer, 'out_channels'):
                fusion_channels.append(layer.out_channels)
            elif hasattr(layer, 'cv2') and hasattr(layer.cv2, 'out_channels'):
                fusion_channels.append(layer.cv2.out_channels)

        self.fusion_weight = nn.Parameter(torch.ones(len(self.save), 2) * 0.5, requires_grad=True)
        # [Manual normalization (min-max, F.normalize, etc) is NOT used. Only BN as in YOLOv8.]

    def project_narrow_to_wide(self):
        """
        narrow 이미지의 4개 코너를 wide 이미지 평면에 투영하여 wide_corners(img_size 기준, float32, (4,2)) 반환
        (이미지와 feature는 dataloader에서 img_size로 resize되어 들어오므로, 추가 스케일링 불필요)
        """
        narrow_corners = np.array([
            [0, 0],
            [self.img_w-1, 0],
            [self.img_w-1, self.img_h-1],
            [0, self.img_h-1]
        ], dtype=np.float32)
        narrow_K_inv = np.linalg.inv(self.narrow_K)
        rays = []
        for u, v in narrow_corners:
            pixel = np.array([u, v, 1.0])
            ray = narrow_K_inv @ pixel
            ray = ray / ray[2]
            rays.append(ray)
        rays = np.stack(rays, axis=0)  # (4, 3)
        wide_corners = []
        for ray in rays:
            X, Y, Z = ray[0], ray[1], 1.0
            pt3d_wide = np.array([X, Y + 0.2, Z, 1.0])
            proj = self.wide_P @ pt3d_wide
            proj = proj / proj[2]
            wide_corners.append([proj[0], proj[1]])
        wide_corners = np.array(wide_corners, dtype=np.float32)  # (4,2)
        return wide_corners

    def extract_single_backbone_feature(self, x, save_idx):
        # x를 backbone에 통과시켜 save_idx에 해당하는 feature만 추출
        y = []
        for i, m in enumerate(self.model):
            if hasattr(m, 'f') and m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [y[j] for j in m.f]
            x = m(x)
            y.append(x)
            if i == save_idx:
                return x

    def forward(self, wide_img, narrow_img):
        """
        wide_img, narrow_img: (B, 3, 640, 640)
        카메라 파라미터 기반으로 narrow FOV가 wide feature map에서 어디에 위치하는지 계산 후 sum
        """
        # 1. narrow의 4개 코너를 wide로 투영 (이미 img_size 기준으로 반환)
        wide_corners = self.project_narrow_to_wide()  # (4,2) img_size 기준

        y_wide, y_narrow = [], []
        y = []
        save_outputs = dict()  # model idx -> y idx
        x_wide, x_narrow = wide_img, narrow_img
        for i, m in enumerate(self.model[:self.save[-1]+1]):
            if not hasattr(m, 'f') or m.f == -1:
                xw_in = y_wide[-1] if len(y_wide) > 0 else x_wide
                xn_in = y_narrow[-1] if len(y_narrow) > 0 else x_narrow
            else:
                xw_in = y_wide[m.f] if isinstance(m.f, int) else [y_wide[j] for j in m.f]
                xn_in = y_narrow[m.f] if isinstance(m.f, int) else [y_narrow[j] for j in m.f]
            xw_out = m(xw_in)
            xn_out = m(xn_in)
            y_wide.append(xw_out)
            y_narrow.append(xn_out)
            # --- 마지막 backbone output에서 spatial sum 적용 ---
            if i == self.save[-1]:
                B, C, H, W = xw_out.shape
                # === YOLO label 비율 기반 bbox 계산 ===
                # 원하는 YOLO label 비율 (중앙, 1/4.494, 1/4.552)
                x_center_norm = 0.5
                y_center_norm = 0.5
                width_norm = 1 / 4.494  # ≈ 0.2225
                height_norm = 1 / 4.552 # ≈ 0.2197
                cx = x_center_norm * W
                cy = y_center_norm * H
                w = width_norm * W
                h = height_norm * H
                fx_min = int(round(cx - w / 2))
                fx_max = int(round(cx + w / 2))
                fy_min = int(round(cy - h / 2))
                fy_max = int(round(cy + h / 2))
                fx_min = max(fx_min, 0)
                fy_min = max(fy_min, 0)
                fx_max = min(fx_max, W)
                fy_max = min(fy_max, H)
                region_w = fx_max - fx_min
                region_h = fy_max - fy_min
                print(f"[DEBUG] [YOLO BBOX] fusion bbox: fx_min={fx_min}, fx_max={fx_max}, fy_min={fy_min}, fy_max={fy_max}, width={region_w}, height={region_h}")
                # narrow feature를 해당 영역 크기로 resize
                narrow_region = torch.nn.functional.interpolate(
                    xn_out, size=(region_h, region_w), mode='bilinear', align_corners=False
                )
                narrow_full = torch.zeros_like(xw_out)
                narrow_full[..., fy_min:fy_max, fx_min:fx_max] = narrow_region
                # wide feature와 합성 (Multi-Scale BN, Learnable Weighted Sum)
                fusion_idx = list(self.save).index(i)
                xw_out_bn = xw_out
                narrow_full_bn = narrow_full
                # fusion_weight = torch.softmax(self.fusion_weight[fusion_idx], dim=0)
                # xw_out_spatial = fusion_weight[0] * xw_out_bn + fusion_weight[1] * narrow_full_bn
                xw_out_spatial = xw_out_bn + narrow_full_bn
                print(f"[DEBUG] xw_out_spatial shape after sum: {xw_out_spatial.shape}")
                y.append(xw_out_spatial)
                save_outputs[i] = len(y) - 1
            elif i in self.save:
                y.append(xw_out)
                save_outputs[i] = len(y) - 1
        # neck/head 들어가기 전에 NoneType 체크
        for idx, item in enumerate(y):
            if item is None:
                print(f"[ERROR] y[{idx}] is None before neck/head! y: {[type(x) for x in y]}")
            # elif hasattr(item, 'shape'):
            #     print(f"[DEBUG] y[{idx}] shape: {item.shape}")
            # else:
            #     print(f"[DEBUG] y[{idx}] type: {type(item)})")
        # 2. neck/head: summed_features부터 시작, y에 계속 append
        for i in range(self.save[-1]+1, len(self.model)):
            m = self.model[i]
            if not hasattr(m, 'f') or m.f == -1:
                x_in = y[-1]
            else:
                if isinstance(m.f, int):
                    x_in = y[save_outputs[m.f]]
                else:
                    x_in = [y[save_outputs[j]] for j in m.f]
            out = m(x_in)
            y.append(out)
        return y[-1]

# --- 학습 함수 ---
def train(opt):
    device = torch.device(opt.device if opt.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    LOGGER.info("Using device: {}".format(device))

    amp = device.type != 'cpu'
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    save_dir = os.path.join(opt.project, opt.name)
    os.makedirs(save_dir, exist_ok=True)

    # 자동 train/val split: val 인자가 없으면 train set에서 90/10 분할
    import random
    def split_dual_dataset(wide_dir, narrow_dir, label_dir, split_ratio=0.1, seed=42):
        # 파일 리스트 생성 (DualImageDataset 방식)
        import os
        possible_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        files = []
        for f_name_ext in os.listdir(wide_dir):
            f_name, f_ext = os.path.splitext(f_name_ext)
            if f_ext.lower() not in possible_extensions:
                continue
            wide_path = os.path.join(wide_dir, f_name_ext)
            narrow_path_found = None
            for ext_option in possible_extensions:
                potential_narrow_path = os.path.join(narrow_dir, f_name + ext_option)
                if os.path.exists(potential_narrow_path):
                    narrow_path_found = potential_narrow_path
                    break
            label_path = os.path.join(label_dir, f_name + '.txt')
            if narrow_path_found and os.path.exists(label_path):
                files.append((wide_path, narrow_path_found, label_path))
        random.seed(seed)
        random.shuffle(files)
        n_val = int(len(files) * split_ratio)
        val_files = files[:n_val]
        train_files = files[n_val:]
        def unzip(files):
            if not files:
                return [], [], []
            wide, narrow, label = zip(*files)
            return list(wide), list(narrow), list(label)
        return unzip(train_files), unzip(val_files)

    # --- 카메라 파라미터 직접 선언 ---
    import numpy as np
    # Wide 카메라
    wide_K = np.array([
        [559.258761, 0, 928.108242],
        [0, 565.348774, 518.787048],
        [0, 0, 1]
    ], dtype=np.float32)
    wide_P = np.array([
        [535.711792, 0, 924.086569, 0],
        [0, 558.997375, 510.222325, 0],
        [0, 0, 1, 0]
    ], dtype=np.float32)
    # Narrow 카메라
    narrow_K = np.array([
        [2651.127798, 0, 819.397071],
        [0, 2635.360938, 896.163803],
        [0, 0, 1]
    ], dtype=np.float32)
    narrow_P = np.array([
        [2407.709780, 0, 801.603047, 0],
        [0, 2544.697607, 897.250521, 0],
        [0, 0, 1, 0]
    ], dtype=np.float32)

    # val 인자가 없으면 자동 분할, 있으면 기존 방식
    if (not hasattr(opt, 'wide_img_dir_val') or not opt.wide_img_dir_val) and \
       (not hasattr(opt, 'narrow_img_dir_val') or not opt.narrow_img_dir_val) and \
       (not hasattr(opt, 'label_dir_val') or not opt.label_dir_val):
        # 자동 분할
        (train_wide, train_narrow, train_label), (val_wide, val_narrow, val_label) = split_dual_dataset(
            opt.wide_img_dir_train, opt.narrow_img_dir_train, opt.label_dir_train, split_ratio=0.1, seed=42)
        train_dataset = DualImageDataset(
            wide_image_paths=train_wide, narrow_image_paths=train_narrow, label_paths=train_label,
            img_size=opt.img_size, transform=None
        )
        val_dataset = DualImageDataset(
            wide_image_paths=val_wide, narrow_image_paths=val_narrow, label_paths=val_label,
            img_size=opt.img_size, transform=None
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.workers,
            collate_fn=dual_collate_fn,
            pin_memory=True
        )
        val_loader_for_train = DataLoader(
            val_dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.workers,
            collate_fn=dual_collate_fn,
            pin_memory=True
        )
        # val 인자처럼 opt에 임시로 세팅 (validate_dual에서 사용)
        opt.val_img_dir = None
        opt.val_label_dir = None
        opt._internal_val_dataset = val_dataset
    else:
        train_dataset = DualImageDataset(
            wide_img_dir=opt.wide_img_dir_train,
            narrow_img_dir=opt.narrow_img_dir_train,
            label_dir=opt.label_dir_train,
            img_size=opt.img_size,
            transform=None
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.workers,
            collate_fn=dual_collate_fn,
            pin_memory=True
        )

    # 하이퍼파라미터 로딩 및 모델 준비
    import yaml
    model_args = {}
    if hasattr(opt, 'hyp') and opt.hyp:
        if isinstance(opt.hyp, dict):
            model_args = opt.hyp
        elif isinstance(opt.hyp, str):
            with open(opt.hyp, 'r') as f:
                model_args = yaml.safe_load(f)
    # nc 자동 세팅 (옵션)
    dataset_yaml = '/home/byounggun/ultralytics/ultralytics/cfg/datasets/coco.yaml'
    if os.path.exists(dataset_yaml):
        with open(dataset_yaml, 'r') as f:
            data_yaml = yaml.safe_load(f)
        names = data_yaml.get('names', None)
        if isinstance(names, (dict, list)):
            model_args['nc'] = len(names)
    model = DualYOLOv8(
        yolo_weights_path=opt.weights,
        args=opt,
        wide_K=wide_K, wide_P=wide_P,
        narrow_K=narrow_K, narrow_P=narrow_P,
        img_w=1920, img_h=1080
    ).to(device)
    model.train()

    # backbone freeze (선택)
    backbone_end = model.yolo_model.save[-1] + 1 if hasattr(model.yolo_model, 'save') else None
    if backbone_end:
        for param in model.yolo_model.model[:backbone_end].parameters():
            param.requires_grad = False
        for param in model.yolo_model.model[backbone_end:].parameters():
            param.requires_grad = True
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    else:
        trainable_params = model.parameters()

    optimizer = optim.AdamW(trainable_params, lr=1e-4, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=opt.epochs, pct_start=0.2, anneal_strategy='cos', final_div_factor=1e4)

    import yaml
    from types import SimpleNamespace
    hyp_path = "hyp_siamese_scratch.yaml"  # 필요시 opt.hyp 등으로 바꿔도 됨
    with open(hyp_path, 'r') as f:
        hyp_dict = yaml.safe_load(f)

    defaults = {
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "obj": 1.0,
        "label_smoothing": 0.0,
        "fl_gamma": 0.0,
        "iou_type": "ciou",
        "anchor_t": 4.0,
    }
    for k, v in defaults.items():
        if k not in hyp_dict or hyp_dict[k] is None:
            hyp_dict[k] = v

    hyp = SimpleNamespace(**hyp_dict)

    for k, v in {"box": 7.5, "cls": 0.5, "dfl": 1.5}.items():
        if not hasattr(hyp, k):
            alt_key = k + "_gain"
            if hasattr(hyp, alt_key):
                setattr(hyp, k, getattr(hyp, alt_key))
            else:
                setattr(hyp, k, v)

    model.hyp = hyp

    loss_fn = v8DetectionLoss(model)

    if isinstance(loss_fn.hyp, dict):
        loss_fn.hyp = SimpleNamespace(**loss_fn.hyp)

    for k, v in {"box": 7.5, "cls": 0.5, "dfl": 1.5}.items():
        if not hasattr(loss_fn.hyp, k):
            alt_key = k + "_gain"
            if hasattr(loss_fn.hyp, alt_key):
                setattr(loss_fn.hyp, k, getattr(loss_fn.hyp, alt_key))
            else:
                setattr(loss_fn.hyp, k, v)

    # ... (학습 루프 및 기타 코드) ...
    # 학습 종료 후 wandb 종료
    if wandb is not None and opt.wandb:
        wandb.finish()

    # wandb experiment logging (학습 시작 직전!)
    if wandb is not None and opt.wandb:
        wandb_project = os.path.basename(opt.project.rstrip('/')) if hasattr(opt, 'project') else 'yolov8-dual'
        wandb.init(
            project=wandb_project,
            name=opt.name if hasattr(opt, "name") else "exp",
            config=vars(opt)
        )
        LOGGER.info(f"Initialized wandb run: {wandb.run.id}")
    else:
        LOGGER.warning("Wandb logging is disabled (either wandb is not installed or --wandb not set).")

    for epoch in range(opt.epochs):
        model.train()
        epoch_loss_sum = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{opt.epochs}")
        for batch_idx, batch in pbar:
            wide_img = batch["wide_img"].to(device, non_blocking=True)
            narrow_img = batch["narrow_img"].to(device, non_blocking=True)
            labels = batch["labels"]
            labels = [l.to(device) for l in labels]
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp):
                preds = model(wide_img, narrow_img)
                batch_idx_list, cls_list, bbox_list = [], [], []
                for i, label in enumerate(labels):
                    if label.numel() == 0:
                        continue
                    n = label.shape[0]
                    batch_idx_list.append(torch.full((n, 1), i, dtype=label.dtype, device=label.device))
                    cls_list.append(label[:, 0:1])
                    bbox_list.append(label[:, 1:5])
                batch_idx_tensor = torch.cat(batch_idx_list, 0) if batch_idx_list else torch.zeros((0, 1), device=labels[0].device)
                cls_tensor = torch.cat(cls_list, 0) if cls_list else torch.zeros((0, 1), device=labels[0].device)
                bboxes_tensor = torch.cat(bbox_list, 0) if bbox_list else torch.zeros((0, 4), device=labels[0].device)
                label_dict = {"batch_idx": batch_idx_tensor, "cls": cls_tensor, "bboxes": bboxes_tensor}
                loss, loss_items = loss_fn(preds, label_dict)
            if hasattr(loss, 'dim') and loss.dim() > 0:
                loss = loss.mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss_sum += loss.item()
            log_dict = {
                "train/loss": loss.item(),
                "epoch": epoch + 1,
                "step": epoch * len(train_loader) + batch_idx + 1,
                "lr": optimizer.param_groups[0]['lr'],
            }
            if isinstance(loss_items, (list, tuple)):
                log_dict["train/box_loss"] = loss_items[0]
                log_dict["train/cls_loss"] = loss_items[1]
                log_dict["train/dfl_loss"] = loss_items[2]
            if wandb is not None and opt.wandb:
                wandb.log(log_dict)
            pbar.set_postfix({"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})
        # lr scheduler step
        if hasattr(optimizer, 'param_groups') and hasattr(opt, 'epochs'):
            pass  # OneCycleLR 등은 step마다 호출됨
        avg_loss = epoch_loss_sum / len(train_loader)
        LOGGER.info(f"[Epoch {epoch+1}] Avg Train Loss: {avg_loss:.4f}")
        # ===================== Validation =====================
        val_metrics = None
        # 자동 분할 모드: opt._internal_val_dataset가 있으면 그걸로 validation 진행
        if hasattr(opt, '_internal_val_dataset') and opt._internal_val_dataset is not None:
            val_metrics = validate_dual(model, opt, device, sample_ratio=1.0, custom_val_dataset=opt._internal_val_dataset)
        elif hasattr(opt, 'val_img_dir') and hasattr(opt, 'val_label_dir') and opt.val_img_dir and opt.val_label_dir:
            val_metrics = validate_dual(model, opt, device, sample_ratio=0.1)
        # Print all metrics clearly to terminal
        if val_metrics:
            print("\n[Validation Metrics]")
            for k, v in val_metrics.items():
                print(f"  {k}: {v}")
        # Aggressively log all available metrics to wandb
        if wandb is not None and opt.wandb and val_metrics:
            wandb.log({f"val/{k}": v for k, v in val_metrics.items() if isinstance(v, (float, int))})
        # ===================== Save checkpoint =====================
        if (epoch + 1) % 10 == 0 or (epoch + 1) == opt.epochs:
            ckpt_path = os.path.join(save_dir, f"dual_yolov8_epoch{epoch+1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            LOGGER.info(f"Model checkpoint saved to {ckpt_path}")
    # 마지막 모델 저장
    final_path = os.path.join(save_dir, "dual_yolov8_final.pt")
    torch.save(model.state_dict(), final_path)
    LOGGER.info(f"Final model saved to {final_path}")

# ======= Validation 함수 추가 =======
def validate_dual(model, opt, device, wide_img_dir=None, narrow_img_dir=None, label_dir=None, sample_ratio=1.0, custom_val_dataset=None):
    """
    Custom validation for DualYOLO: computes loss, mAP, precision, recall (ultralytics style).
    If custom_val_dataset is given, use it directly. Otherwise, build from dirs.
    """
    # from dataload_dual import DualImageDataset, get_dual_val_transforms, dual_collate_fn
    from dataload_dual import DualImageDataset, dual_collate_fn
    from ultralytics.utils.metrics import ap_per_class
    import torch
    import numpy as np

    if custom_val_dataset is not None:
        val_dataset = custom_val_dataset
    else:
        wide_img_dir = wide_img_dir or getattr(opt, 'wide_img_dir_val', None) or getattr(opt, 'val_img_dir', None)
        narrow_img_dir = narrow_img_dir or getattr(opt, 'narrow_img_dir_val', None) or wide_img_dir
        label_dir = label_dir or getattr(opt, 'label_dir_val', None) or getattr(opt, 'val_label_dir', None)
        val_dataset = DualImageDataset(
            wide_img_dir=wide_img_dir,
            narrow_img_dir=narrow_img_dir,
            label_dir=label_dir,
            img_size=opt.img_size,
            # transform=get_dual_val_transforms(opt.img_size),
            transform=None,
            sample_ratio=sample_ratio
        )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.workers,
        collate_fn=dual_collate_fn,
        pin_memory=True
    )

    model.eval()
    total_loss = 0.0
    total_batches = 0
    # Patch model.args for validation if missing YOLOv8 hyperparameters
    import types
    if not hasattr(model, 'args') or not (hasattr(model.args, 'box') and hasattr(model.args, 'cls') and hasattr(model.args, 'dfl')):
        model.args = types.SimpleNamespace(box=0.05, cls=0.5, dfl=1.5)
    loss_fn = v8DetectionLoss(model)
    stats = []
    for batch_idx, batch in enumerate(val_loader):
        imgs_wide, imgs_narrow, labels_list = batch['wide_img'].to(device), batch['narrow_img'].to(device), batch['labels']
        # labels_list: list of [num_targets, 5] (class, x, y, w, h)
        # Convert to [all_targets, 6]: (batch_idx, class, x, y, w, h)
        labels = []
        for i, l in enumerate(labels_list):
            if l.numel() == 0:
                continue
            l = l if l.ndim == 2 else l.unsqueeze(0)
            batch_idx = torch.full((l.shape[0], 1), i, dtype=l.dtype, device=l.device)
            labels.append(torch.cat((batch_idx, l), dim=1))
        labels = torch.cat(labels, dim=0) if labels else torch.zeros((0, 6), device=imgs_wide.device)
        # YOLOv8 validation loss expects dict with keys: batch_idx, cls, bboxes
        val_targets = {
            "batch_idx": labels[:, 0].long() if labels.numel() else torch.zeros((0,), dtype=torch.long, device=imgs_wide.device),
            "cls": labels[:, 1].long() if labels.numel() else torch.zeros((0,), dtype=torch.long, device=imgs_wide.device),
            "bboxes": labels[:, 2:] if labels.numel() else torch.zeros((0, 4), device=imgs_wide.device)
        }
        with torch.no_grad():
            outputs = model(imgs_wide, imgs_narrow)
            loss, loss_items = loss_fn(outputs, val_targets)
            # loss: tensor of shape [3] (box, cls, dfl), sum for total
            total_loss += loss.sum().item()
            total_batches += 1
            # --- metric 계산 ---
            # outputs: (batch, num_boxes, 6) [x1, y1, x2, y2, conf, cls]
            # labels: (num_targets, 6) [batch_idx, cls, x, y, w, h]
            metric_out = outputs[0] if isinstance(outputs, tuple) else outputs
            for si in range(metric_out.shape[0]):
                pred = metric_out[si]
                pred = pred[pred[:, 4] > 0.001]  # conf threshold
                if pred.shape[0] == 0:
                    # pred_boxes: (0, 4), pred_conf: (0,), pred_cls: (0,), tcls: (0,)
                    stats.append((np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,)), np.zeros((0,))) )
                    continue
                # GT
                tcls = labels[labels[:, 0] == si][:, 1].cpu().numpy() if labels.numel() else np.array([])
                tbox = labels[labels[:, 0] == si][:, 2:6].cpu().numpy() if labels.numel() else np.array([])
                # Prediction
                pred_cls = pred[:, 5].cpu().numpy() if pred.numel() else np.array([])
                pred_boxes = pred[:, :4].cpu().numpy() if pred.numel() else np.array([])
                pred_conf = pred[:, 4].cpu().numpy() if pred.numel() else np.array([])
                # Ensure all are np.ndarray, not tuple
                pred_boxes = np.array(pred_boxes)
                pred_conf = np.array(pred_conf)
                pred_cls = np.array(pred_cls)
                tcls = np.array(tcls)
                stats.append((pred_boxes, pred_conf, pred_cls, tcls))
    model.train()
    avg_loss = total_loss / max(1, total_batches)
    # mAP/precision/recall 계산
    if len(stats) and all(isinstance(x, np.ndarray) and x.size > 0 for x in stats[0]):
        p, r, ap, f1, ap_class = ap_per_class(*zip(*stats))
        metrics = {
            "val/loss": avg_loss,
            "val/precision": float(np.mean(p)) if len(p) else 0.0,
            "val/recall": float(np.mean(r)) if len(r) else 0.0,
            "val/mAP50": float(np.mean(ap[:, 0])) if ap.ndim > 1 else float(np.mean(ap)),
            "val/mAP50-95": float(np.mean(ap[:, 1])) if ap.ndim > 1 and ap.shape[1] > 1 else 0.0,
        }
    else:
        p = r = ap = f1 = ap_class = np.array([])
        metrics = {"val/loss": avg_loss, "val/precision": 0.0, "val/recall": 0.0, "val/mAP50": 0.0, "val/mAP50-95": 0.0}
    return metrics



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wide_img_dir_train', type=str, required=True)
    parser.add_argument('--narrow_img_dir_train', type=str, required=True)
    parser.add_argument('--label_dir_train', type=str, required=True)
    parser.add_argument('--wide_img_dir_val', type=str, default=None, help='Wide 이미지 validation set 경로')
    parser.add_argument('--narrow_img_dir_val', type=str, default=None, help='Narrow 이미지 validation set 경로')
    parser.add_argument('--label_dir_val', type=str, default=None, help='Validation label 경로')
    parser.add_argument('--weights', type=str, default='yolov8s.pt')
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--project', type=str, default='runs/dual')
    parser.add_argument('--name', type=str, default='exp')
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--wandb', action='store_true', help='use Weights & Biases logging')
    return parser.parse_args()

if __name__ == '__main__':
    opt = parse_opt()
    print_args(vars(opt))
    train(opt)
