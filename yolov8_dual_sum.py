# -*- coding: utf-8 -*-
# yolov8_dual.py
"""
Dual YOLO 학습 스크립트
- wide/narrow 이미지를 각각 backbone에 통과시켜 feature를 concat한 뒤 detection head에 입력
- 라벨은 wide에만 존재
- dataload_dual.py와 연동
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
from dataload_dual import DualImageDataset, get_dual_train_transforms, get_dual_val_transforms, dual_collate_fn

# wandb import with fallback
try:
    import wandb
except ImportError:
    wandb = None
    LOGGER.warning("Wandb is not installed. Wandb logging is disabled.")

# --- 모델 정의 ---
class DualYOLOv8(nn.Module):
    def __init__(self, yolo_weights_path='yolov8s.pt', args=None):
        super().__init__()
        from ultralytics import YOLO
        # 커스텀 yaml 사용 (dual 구조)
        yolo_model_full = YOLO('/home/byounggun/ultralytics/ultralytics/cfg/models/v8/yolov8.yaml')
        self.yolo_model = yolo_model_full.model  # DetectionModel
        self.model = self.yolo_model.model  # for v8DetectionLoss compatibility (nn.ModuleList)
        self.stride = self.yolo_model.stride
        # --- 클래스 수 및 클래스명 traffic.yaml에서 로드 ---
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
        # fallback: Detect head/YOLO wrapper에서 가져오기
        detect_head = self.model[-1]
        if self.nc is None:
            self.nc = getattr(detect_head, 'nc', getattr(self.yolo_model, 'nc', None))
        if self.names is None:
            self.names = getattr(detect_head, 'names', getattr(self.yolo_model, 'names', None))
        self.save = self.yolo_model.save  # backbone output indices for multi-scale features
        self.args = args
        # Detect/neck 채널 자동 교체 코드 제거 (yaml에서 이미 구조 반영)

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
        # 1. backbone만 wide/narrow 각각 forward
        y_wide, y_narrow = [], []
        y = [None] * len(self.model)
        x_wide, x_narrow = wide_img, narrow_img
        for i, m in enumerate(self.model[:self.save[-1]+1]):
            # YOLOv8 공식 from(f) 해석: -1이면 직전 output, 아니면 y_wide[m.f]
            if not hasattr(m, 'f') or m.f == -1:
                xw_in = y_wide[-1] if len(y_wide) > 0 else x_wide
                xn_in = y_narrow[-1] if len(y_narrow) > 0 else x_narrow
            else:
                xw_in = y_wide[m.f] if isinstance(m.f, int) else [y_wide[j] for j in m.f]
                xn_in = y_narrow[m.f] if isinstance(m.f, int) else [y_narrow[j] for j in m.f]
            # print(f"[DUAL][WIDE] Layer {i} ({type(m).__name__}): input shape {xw_in.shape if isinstance(xw_in, torch.Tensor) else [t.shape for t in xw_in]}")
            # print(f"[DUAL][NARROW] Layer {i} ({type(m).__name__}): input shape {xn_in.shape if isinstance(xn_in, torch.Tensor) else [t.shape for t in xn_in]}")
            xw_out = m(xw_in)
            xn_out = m(xn_in)
            y_wide.append(xw_out)
            y_narrow.append(xn_out)
            if i in self.save:
                y[i] = xw_out + xn_out
        # 2. neck/head: summed_features부터 시작, y에 계속 append
        for i in range(self.save[-1]+1, len(self.model)):
            m = self.model[i]
            if not hasattr(m, 'f') or m.f == -1:
                x_in = y[-1]
            else:
                x_in = y[m.f] if isinstance(m.f, int) else [y[j] for j in m.f]
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
            img_size=opt.img_size, transform=get_dual_train_transforms(opt.img_size)
        )
        val_dataset = DualImageDataset(
            wide_image_paths=val_wide, narrow_image_paths=val_narrow, label_paths=val_label,
            img_size=opt.img_size, transform=get_dual_val_transforms(opt.img_size)
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
            transform=get_dual_train_transforms(opt.img_size)
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
    model = DualYOLOv8(yolo_weights_path=opt.weights, args=model_args).to(device)
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

    optimizer = optim.AdamW(trainable_params, lr=opt.lr)
    import yaml
    from types import SimpleNamespace
    hyp_path = "hyp_siamese_scratch.yaml"  # 필요시 opt.hyp 등으로 바꿔도 됨
    with open(hyp_path, 'r') as f:
        hyp_dict = yaml.safe_load(f)
    # 필수 키 없으면 기본값 보충
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
    # print("[DEBUG] Final hyp_dict:", hyp_dict)
    hyp = SimpleNamespace(**hyp_dict)
    # box, cls, dfl이 없으면 box_gain, cls_gain, dfl_gain에서 속성으로 할당
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
    # loss_fn.hyp에도 box/cls/dfl 보장 (box_gain 등에서 매핑)
    for k, v in {"box": 7.5, "cls": 0.5, "dfl": 1.5}.items():
        if not hasattr(loss_fn.hyp, k):
            alt_key = k + "_gain"
            if hasattr(loss_fn.hyp, alt_key):
                setattr(loss_fn.hyp, k, getattr(loss_fn.hyp, alt_key))
            else:
                setattr(loss_fn.hyp, k, v)
    # print("[DEBUG] patched loss_fn.hyp:", loss_fn.hyp)

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
    from dataload_dual import DualImageDataset, get_dual_val_transforms, dual_collate_fn
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
