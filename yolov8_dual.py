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
            print(f"[DUAL][WIDE] Layer {i} ({type(m).__name__}): input shape {xw_in.shape if isinstance(xw_in, torch.Tensor) else [t.shape for t in xw_in]}")
            print(f"[DUAL][NARROW] Layer {i} ({type(m).__name__}): input shape {xn_in.shape if isinstance(xn_in, torch.Tensor) else [t.shape for t in xn_in]}")
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
    print("[DEBUG] Final hyp_dict:", hyp_dict)
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
    print("[DEBUG] patched loss_fn.hyp:", loss_fn.hyp)

    for epoch in range(opt.epochs):
        pbar = tqdm(train_loader, desc="Epoch {} / {}".format(epoch+1, opt.epochs))
        for batch_idx, batch in enumerate(train_loader):
            wide_img = batch["wide_img"].to(device, non_blocking=True)
            narrow_img = batch["narrow_img"].to(device, non_blocking=True)
            labels = batch["labels"]
            labels = [l.to(device) for l in labels]
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp):
                preds = model(wide_img, narrow_img)
                # YOLOv8 loss expects a dict with 'batch_idx', 'cls', 'bboxes'
                batch_idx_list = []
                cls_list = []
                bbox_list = []
                for i, label in enumerate(labels):
                    if label.numel() == 0:
                        continue
                    n = label.shape[0]
                    batch_idx_list.append(torch.full((n, 1), i, dtype=label.dtype, device=label.device))
                    cls_list.append(label[:, 0:1])
                    bbox_list.append(label[:, 1:5])
                batch_idx = torch.cat(batch_idx_list, 0) if batch_idx_list else torch.zeros((0, 1), device=labels[0].device)
                cls = torch.cat(cls_list, 0) if cls_list else torch.zeros((0, 1), device=labels[0].device)
                bboxes = torch.cat(bbox_list, 0) if bbox_list else torch.zeros((0, 4), device=labels[0].device)
                label_dict = {"batch_idx": batch_idx, "cls": cls, "bboxes": bboxes}
                loss, loss_items = loss_fn(preds, label_dict)
            # loss가 scalar가 아니면 mean 처리
            if hasattr(loss, 'dim') and loss.dim() > 0:
                loss = loss.mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix({"loss": loss.item()})
        LOGGER.info("Epoch {} completed.".format(epoch+1))
    torch.save(model.state_dict(), os.path.join(save_dir, "dual_yolov8_final.pth"))
    LOGGER.info("Model saved to {}".format(os.path.join(save_dir, 'dual_yolov8_final.pth')))

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wide_img_dir_train', type=str, required=True)
    parser.add_argument('--narrow_img_dir_train', type=str, required=True)
    parser.add_argument('--label_dir_train', type=str, required=True)
    parser.add_argument('--weights', type=str, default='yolov8s.pt')
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--project', type=str, default='runs/dual')
    parser.add_argument('--name', type=str, default='exp')
    parser.add_argument('--device', type=str, default='')
    return parser.parse_args()

if __name__ == '__main__':
    opt = parse_opt()
    print_args(vars(opt))
    train(opt)
