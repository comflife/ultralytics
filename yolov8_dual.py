# -*- coding: utf-8 -*-
# yolov8_dual.py
"""
Dual YOLO 학습 스크립트
- wide/narrow 이미지를 각각 backbone에 통과시켜 feature를 concat한 뒤 detection head에 입력
- 라벨은 wide에만 존재
- dataload_dual.py와 연동


python yolov8_dual.py   --wide_img_dir_train /home/byounggun/ultralytics/traffic_train/wide/images   --narrow_img_dir_train /home/byounggun/ultralytics/traffic_train/narrow/images   --label_dir_train /home/byounggun/ultralytics/traffic_train/wide/labels   --weights yolov8s.pt   --img_size 640   --batch_size 12   --epochs 150   --wandb
python yolov8_dual.py   --wide_img_dir_train /home/byounggun/cococo/train_resized/images   --narrow_img_dir_train /home/byounggun/cococo/train_short/images   --label_dir_train /home/byounggun/cococo/train_resized/labels   --weights yolov8s.pt   --img_size 640   --batch_size 12   --epochs 150   --wandb
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


__all__ = ["DualYOLOv8"]

class DualYOLOv8(nn.Module):
    def __init__(self, yolo_weights_path='yolov8s.pt', args=None, img_size=640):
        super().__init__()
        from ultralytics import YOLO
        # 수정: pt 파일이면 가중치까지 로드, 아니면 구조만 생성
        # if yolo_weights_path is not None and yolo_weights_path.endswith('.pt') and os.path.exists(yolo_weights_path):
        yolo_model_full = YOLO(yolo_weights_path)

        self.yolo_model = yolo_model_full.model  # DetectionModel
        self.model = self.yolo_model.model  # for v8DetectionLoss compatibility (nn.ModuleList)
        self.stride = self.yolo_model.stride
        import yaml
        # dataset_yaml = 'ultralytics/cfg/datasets/traffic.yaml'
        dataset_yaml = '/home/byounggun/cococo/data.yaml'
        self.nc = 80
        # self.nc = 10
        self.names = ['aeroplane', 'apple', 'backpack', 'banana', 'baseball bat', 'baseball glove', 'bear', 'bed', 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle', 'bowl', 'broccoli', 'bus', 'cake', 'car', 'carrot', 'cat', 'cell phone', 'chair', 'clock', 'cow', 'cup', 'diningtable', 'dog', 'donut', 'elephant', 'fire hydrant', 'fork', 'frisbee', 'giraffe', 'hair drier', 'handbag', 'horse', 'hot dog', 'keyboard', 'kite', 'knife', 'laptop', 'microwave', 'motorbike', 'mouse', 'orange', 'oven', 'parking meter', 'person', 'pizza', 'pottedplant', 'refrigerator', 'remote', 'sandwich', 'scissors', 'sheep', 'sink', 'skateboard', 'skis', 'snowboard', 'sofa', 'spoon', 'sports ball', 'stop sign', 'suitcase', 'surfboard', 'teddy bear', 'tennis racket', 'tie', 'toaster', 'toilet', 'toothbrush', 'traffic light', 'train', 'truck', 'tvmonitor', 'umbrella', 'vase', 'wine glass', 'zebra']
        # self.names = ['Green', 'Red', 'Green-up', 'Empty-count-down', 'Count-down', 'Yellow', 'Empty', 'Green-right', 'Green-left', 'Red-yellow']
        if os.path.exists(dataset_yaml):
            with open(dataset_yaml, 'r') as f:
                data_yaml = yaml.safe_load(f)
            if isinstance(data_yaml, dict):
                self.nc = data_yaml.get('nc', None)
                self.names = data_yaml.get('names', None)
        # detect_head = self.model[-1]
        # if self.nc is None:
        #     self.nc = getattr(detect_head, 'nc', getattr(self.yolo_model, 'nc', None))
        # if self.names is None:
        #     self.names = getattr(detect_head, 'names', getattr(self.yolo_model, 'names', None))
        # self.save = self.yolo_model.save  # backbone output indices for multi-scale features
        # self.save = [4,6,9]
        self.save = [0,1,3,5,7]
        self.args = args



        fusion_channels = []
        for i in self.save:
            layer = self.model[i]
            if hasattr(layer, 'out_channels'):
                fusion_channels.append(layer.out_channels)
            elif hasattr(layer, 'cv2') and hasattr(layer.cv2, 'out_channels'):
                fusion_channels.append(layer.cv2.out_channels)

        self.fusion_weight = nn.Parameter(torch.ones(len(self.save), 2) * 0.5, requires_grad=True)
        # [Manual normalization (min-max, F.normalize, etc) is NOT used. Only BN as in YOLOv8.]



    # def extract_single_backbone_feature(self, x, save_idx):
    #     # x를 backbone에 통과시켜 save_idx에 해당하는 feature만 추출
    #     y = []
    #     for i, m in enumerate(self.model):
    #         if hasattr(m, 'f') and m.f != -1:
    #             x = y[m.f] if isinstance(m.f, int) else [y[j] for j in m.f]
    #         x = m(x)
    #         y.append(x)
    #         if i == save_idx:
    #             return x

    def forward(self, wide_img, narrow_img):

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

            if i in self.save:
                B, C, H, W = xw_out.shape
                # === YOLO label 비율 기반 bbox 계산 ===
                # 원하는 YOLO label 비율 (중앙, 1/4.494, 1/4.552)
                x_center_norm = 0.5
                y_center_norm = 0.5
                # width_norm = 1 / 4.494  # ≈ 0.2225
                width_norm = 0.5
                # height_norm = 1 / 4.552 # ≈ 0.2197
                height_norm = 0.5
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
                # print(f"[DEBUG] [YOLO BBOX] fusion bbox: fx_min={fx_min}, fx_max={fx_max}, fy_min={fy_min}, fy_max={fy_max}, width={region_w}, height={region_h}")
                # fusion
                narrow_region = torch.nn.functional.interpolate(
                    xn_out, size=(region_h, region_w), mode='bilinear', align_corners=False
                )
                narrow_full = torch.zeros_like(xw_out)
                narrow_full[..., fy_min:fy_max, fx_min:fx_max] = narrow_region
                # wide feature와 합성 (Multi-Scale BN, Learnable Weighted Sum)
                fusion_idx = list(self.save).index(i)
                xw_out_bn = xw_out
                narrow_full_bn = narrow_full
                fusion_weight = torch.softmax(self.fusion_weight[fusion_idx], dim=0)
                xw_out_spatial = fusion_weight[0] * xw_out_bn + fusion_weight[1] * narrow_full_bn
                # xw_out_spatial = xw_out_bn + narrow_full_bn
                # print(f"[DEBUG][fusion] Layer idx={i} | xw_out shape={xw_out_bn.shape}, narrow_full_bn shape={narrow_full_bn.shape}, fused shape={xw_out_spatial.shape}")
                y.append(xw_out_spatial)
                save_outputs[i] = len(y) - 1
            else:
                y.append(xw_out)
                save_outputs[i] = len(y) - 1
        # neck/head 들어가기 전에 NoneType 체크
        for idx, item in enumerate(y):
            if item is None:
                print(f"[ERROR] y[{idx}] is None before neck/head! y: {[type(x) for x in y]}")

        for i in range(self.save[-1]+1, len(self.model)):
            m = self.model[i]
            if not hasattr(m, 'f') or m.f == -1:
                x_in = y[-1]
            else:
                if isinstance(m.f, int):
                    x_in = y[m.f] if m.f != -1 else y[-1]
                else:
                    x_in = [y[j] if j != -1 else y[-1] for j in m.f]
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
    dataset_yaml = '/home/byounggun/cococo/data.yaml'
    if os.path.exists(dataset_yaml):
        with open(dataset_yaml, 'r') as f:
            data_yaml = yaml.safe_load(f)
        names = data_yaml.get('names', None)
        if isinstance(names, (dict, list)):
            model_args['nc'] = len(names)
    model = DualYOLOv8(
        yolo_weights_path=opt.weights,
        args=opt
    ).to(device)
    model.train()

    backbone_end = model.yolo_model.save[-1] + 1 if hasattr(model.yolo_model, 'save') else None
    if backbone_end:
        for param in model.yolo_model.model[:backbone_end].parameters():
            param.requires_grad = True
        for param in model.yolo_model.model[backbone_end:].parameters():
            param.requires_grad = True
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    else:
        trainable_params = model.parameters()

    optimizer = optim.AdamW(trainable_params, lr=1e-4, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=opt.epochs, pct_start=0.2, anneal_strategy='cos', final_div_factor=1e4)

    import yaml
    from types import SimpleNamespace
    hyp_path = "hyp_scratch.yaml"  # 필요시 opt.hyp 등으로 바꿔도 됨
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
            scheduler.step()  # OneCycleLR 스케줄러 실제 적용 (step마다 호출)
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
    from ultralytics.utils.metrics import DetMetrics
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
    from ultralytics.utils.ops import non_max_suppression
    det_metrics = DetMetrics(names=getattr(model, 'names', None) or {})
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
            # Apply YOLOv8 NMS/postprocess to outputs
            preds = non_max_suppression(outputs, conf_thres=0.001, iou_thres=0.6)
            loss, loss_items = loss_fn(outputs, val_targets)
            # loss: tensor of shape [3] (box, cls, dfl), sum for total
            total_loss += loss.sum().item()
            total_batches += 1
            # --- metric 계산 ---
            for si, pred in enumerate(preds):
                # pred: (N, 6) [x1, y1, x2, y2, conf, cls]
                # gt: (M, 6) [batch_idx, class, x, y, w, h] (labels)
                # Convert GT to [cls, x1, y1, x2, y2] in pixels
                tcls = labels[labels[:, 0] == si][:, 1].cpu().numpy() if labels.numel() else np.array([])
                tbox_norm = labels[labels[:, 0] == si][:, 2:6].cpu().numpy() if labels.numel() else np.array([])
                img_h, img_w = imgs_wide.shape[2:]
                if len(tbox_norm):
                    x_c, y_c, w, h = tbox_norm[:, 0], tbox_norm[:, 1], tbox_norm[:, 2], tbox_norm[:, 3]
                    x1 = (x_c - w / 2) * img_w
                    y1 = (y_c - h / 2) * img_h
                    x2 = (x_c + w / 2) * img_w
                    y2 = (y_c + h / 2) * img_h
                    tbox_pixel = np.stack([x1, y1, x2, y2], axis=1)
                else:
                    tbox_pixel = np.zeros((0, 4), dtype=np.float32)
                gt = np.concatenate([tcls[:, None], tbox_pixel], axis=1) if len(tcls) else np.zeros((0, 5), dtype=np.float32)
                # pred: (N, 6) [x1, y1, x2, y2, conf, cls]
                pred_np = pred.cpu().numpy() if pred is not None and pred.numel() else np.zeros((0, 6), dtype=np.float32)
                # --- Official YOLOv8-style stats matching logic ---
                import numpy as np
                import torch
                from ultralytics.utils.metrics import box_iou

                # ===== DEBUG: Show prediction/GT stats for this image =====
                print(f"[DEBUG] pred_np shape: {pred_np.shape}, gt shape: {gt.shape}")
                if pred_np.shape[0] == 0:
                    print(f"[DEBUG] No predictions for this image")
                if gt.shape[0] == 0:
                    print(f"[DEBUG] No ground-truth for this image")
                if pred_np.shape[0]:
                    print(f"[DEBUG] pred_cls unique: {np.unique(pred_np[:, 5])}")
                if gt.shape[0]:
                    print(f"[DEBUG] gt_cls unique: {np.unique(gt[:, 0])}")
                # ======================================

                if 'stats' not in locals():
                    stats = []
                iouv = np.linspace(0.5, 0.95, 10)
                n_iouv = len(iouv)

                # pred_np: (N, 6) [x1, y1, x2, y2, conf, cls]
                # gt: (M, 5) [cls, x1, y1, x2, y2]
                if pred_np.shape[0] == 0:
                    continue
                # Compute IoU between each prediction and GT
                ious = box_iou(torch.from_numpy(pred_np[:, :4]), torch.from_numpy(gt[:, 1:5])).numpy()  # (num_preds, num_gts)
                pred_cls = pred_np[:, 5]
                gt_cls = gt[:, 0]
                n_preds, n_gts = pred_np.shape[0], gt.shape[0]
                # For each prediction, find the best matching GT (by IoU and class)
                correct = np.zeros((n_preds, 1), dtype=bool)
                detected = []  # indices of GTs already assigned
                for pred_idx, (p_cls, ious_row) in enumerate(zip(pred_cls, ious)):
                    # Only match to GTs of the same class
                    candidates = np.where(gt_cls == p_cls)[0]
                    if candidates.size == 0:
                        continue
                    # For each IoU threshold, check if there is a match
                    ious_cand = ious_row[candidates]
                    best_iou_idx = ious_cand.argmax() if ious_cand.size else -1
                    if ious_cand.size and ious_cand[best_iou_idx] >= 0.5 and candidates[best_iou_idx] not in detected:
                        correct[pred_idx, 0] = True
                        detected.append(candidates[best_iou_idx])
                # --- DEBUG: 클래스 인덱스, 박스 스케일, true positive ---
                # GT와 pred 모두 있을 때만 stats에 append (ap_per_class 오류 방지)
                if gt.shape[0] > 0 and pred_np.shape[0] > 0:
                    correct = np.zeros((pred_np.shape[0], 1), dtype=bool)
                    detected = []
                    for pred_idx, (p_cls, ious_row) in enumerate(zip(pred_np[:, 5], box_iou(torch.from_numpy(pred_np[:, :4]), torch.from_numpy(gt[:, 1:5])).numpy())):
                        candidates = np.where(gt[:, 0] == p_cls)[0]
                        if candidates.size == 0:
                            continue
                        ious_cand = ious_row[candidates]
                        best_iou_idx = ious_cand.argmax() if ious_cand.size else -1
                        if ious_cand.size and ious_cand[best_iou_idx] >= 0.5 and candidates[best_iou_idx] not in detected:
                            correct[pred_idx, 0] = True
                            detected.append(candidates[best_iou_idx])
                    # correct 전체(2D)를 저장해야 ap_per_class가 정상 동작함
                    if 'stats' in locals():
                        stats.append((correct, pred_np[:, 4], pred_np[:, 5], gt[:, 0]))
                        print(f"[DEBUG][STATS] stats[-1] shapes: {[arr.shape for arr in stats[-1]]}, GT count: {gt.shape[0]}, pred count: {pred_np.shape[0]}")
                else:
                    print(f"[DEBUG][STATS] Skipped stats append. GT count: {gt.shape[0]}, pred count: {pred_np.shape[0]}")

    model.train()
    # Detection Head가 10개 클래스인지 확인
    print(model.model[-1].nc)  # 10이어야 함
    print(model.names)         # 10개여야 함
    avg_loss = total_loss / max(1, total_batches)
    # Aggregate stats for metric calculation using ap_per_class (legacy Ultralytics style)
    from ultralytics.utils.metrics import ap_per_class
    mean_p, mean_r, mean_ap50, mean_map = 0.0, 0.0, 0.0, 0.0  # defaults in case of empty stats
    if 'stats' in locals() and len(stats):
        stats_np = [np.concatenate(x, 0) if len(x) and len(x[0]) else np.array([]) for x in zip(*stats)]
        # --- DEBUG: stats_np concat shape, dtype, 값, GT 존재 여부 ---
        for idx, arr in enumerate(stats_np):
            print(f"[DEBUG][STATS] stats_np[{idx}] shape: {arr.shape}, dtype: {arr.dtype}, first 10: {arr[:10] if len(arr) else 'EMPTY'}")
        print(f"[DEBUG][STATS] stats_np empty check: {[len(arr) for arr in stats_np]}")
        print(f"[DEBUG][STATS] target_cls(GT) shape: {stats_np[3].shape}, GT sample: {stats_np[3][:10] if len(stats_np[3]) else 'EMPTY'}")
        if len(stats_np) == 4 and len(stats_np[3]):
            tp, conf, pred_cls, target_cls = stats_np
            # tp: (N, num_iou_thrs) or (N,)
            if tp.ndim == 1:
                tp = tp[:, None]  # (N, 1)로 변환 (예외 방지)
            pred_cls = pred_cls.astype(np.int32)
            target_cls = target_cls.astype(np.int32)
            try:
                print(f"[DEBUG][AP] ap_per_class input shapes: {[arr.shape for arr in [tp, conf, pred_cls, target_cls]]}")
                print(f"[DEBUG][AP] ap_per_class input dtypes: {[arr.dtype for arr in [tp, conf, pred_cls, target_cls]]}")
                print(f"[DEBUG][AP] ap_per_class input samples: {[arr[:10] if len(arr) else 'EMPTY' for arr in [tp, conf, pred_cls, target_cls]]}")
                p, r, ap, f1, ap_class, *_ = ap_per_class(tp, conf, pred_cls, target_cls)
                print(f"[DEBUG][AP] ap_per_class output: p({p.shape}), r({r.shape}), ap({ap.shape})")
                mean_p, mean_r = p.mean(), r.mean()
                if ap.ndim == 2 and ap.shape[1] > 0:
                    mean_ap50 = ap[:, 0].mean()
                else:
                    mean_ap50 = ap.mean()
                mean_map = ap.mean()
            except Exception as e:
                print(f"[ERROR][AP] ap_per_class exception: {e}")
                mean_p = mean_r = mean_ap50 = mean_map = 0.0
        else:
            print(f"[DEBUG][STATS] ap_per_class skipped: GT(target_cls) is empty (shape={stats_np[3].shape}). All metrics set to 0.0.")
            mean_p = mean_r = mean_ap50 = mean_map = 0.0
    else:
        print("[DEBUG][STATS] stats is empty or not found, setting all metrics to 0.0")
        mean_p = mean_r = mean_ap50 = mean_map = 0.0

    print(f"[DEBUG][METRICS] Final metrics: precision={mean_p}, recall={mean_r}, mAP50={mean_ap50}, mAP50-95={mean_map}")

    metrics = {
        "val/loss": avg_loss,
        "val/precision": float(mean_p),
        "val/recall": float(mean_r),
        "val/mAP50": float(mean_ap50),
        "val/mAP50-95": float(mean_map),
    }
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
