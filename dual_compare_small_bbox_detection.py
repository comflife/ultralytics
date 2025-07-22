import os
from pathlib import Path
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from dual_stream_inference import preprocess_image, create_dual_stream_input
import csv
import matplotlib.pyplot as plt

# =====================
# 설정
# =====================
MODEL_PATH = "/home/byounggun/ultralytics/runs/train/exp108/weights/best.pt"  # dual 모델만 사용
WIDE_DIR = "/home/byounggun/ultralytics/swm_dual_split/val/images"
NARROW_DIR = "/home/byounggun/ultralytics/swm_dual_split/val/val_narrow_images"
LABEL_DIR = "/home/byounggun/ultralytics/swm_dual_split/val/labels"
OUTPUT_DIR = "compare_results"
SMALL_BBOX_THRESH = 20  # 20으로 변경
IOU_THRESH = 0.4
CONFIDENCE_THRESHOLD = 0.15
IMAGE_SIZE = 640

# =====================
# 유틸 함수
# =====================
def load_labels(label_path):
    bboxes = []
    if not os.path.exists(label_path):
        return bboxes
    with open(label_path, 'r') as f:
        for line in f:
            vals = line.strip().split()
            if len(vals) != 5:
                continue
            cls, xc, yc, w, h = map(float, vals)
            bboxes.append({'class': int(cls), 'xc': xc, 'yc': yc, 'w': w, 'h': h})
    return bboxes

def yolo_to_xyxy(bbox, img_w, img_h):
    xc, yc, w, h = bbox['xc'], bbox['yc'], bbox['w'], bbox['h']
    x1 = int((xc - w/2) * img_w)
    y1 = int((yc - h/2) * img_h)
    x2 = int((xc + w/2) * img_w)
    y2 = int((yc + h/2) * img_h)
    return [x1, y1, x2, y2]

def is_small_bbox(bbox, img_w, img_h, threshold=SMALL_BBOX_THRESH):
    x1, y1, x2, y2 = yolo_to_xyxy(bbox, img_w, img_h)
    w = x2 - x1
    h = y2 - y1
    return w <= threshold or h <= threshold  # 가로 또는 세로 중 하나라도 threshold 이하

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
    boxBArea = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))
    denom = float(boxAArea + boxBArea - interArea + 1e-6)
    if denom == 0:
        return 0.0
    iou = interArea / denom
    return iou

def match_bboxes(preds, gts, iou_thr=IOU_THRESH):
    matches = []  # (gt_idx, pred_idx, iou)
    used_pred = set()
    for i, gt in enumerate(gts):
        best_iou = 0
        best_j = -1
        for j, pred in enumerate(preds):
            if j in used_pred:
                continue
            iou = compute_iou(gt['xyxy'], pred['xyxy'])
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= iou_thr:
            matches.append((i, best_j, best_iou))
            used_pred.add(best_j)
        else:
            matches.append((i, None, 0))
    return matches, used_pred

def get_image_pairs(wide_dir, narrow_dir):
    wide_files = sorted([f for f in os.listdir(wide_dir) if f.lower().endswith(('.jpg','.png'))])
    pairs = []
    for wide_name in wide_files:
        base = os.path.splitext(wide_name)[0]
        wide_path = os.path.join(wide_dir, wide_name)
        narrow_path = os.path.join(narrow_dir, wide_name)
        if os.path.exists(narrow_path):
            pairs.append((base, wide_path, narrow_path))
    return pairs

def run_dual_inference(model, wide_path, narrow_path):
    wide_tensor, wide_size, wide_img = preprocess_image(wide_path, IMAGE_SIZE)
    narrow_tensor, narrow_size, _ = preprocess_image(narrow_path, IMAGE_SIZE)
    dual_input = create_dual_stream_input(wide_tensor, narrow_tensor)
    pytorch_model = model.model if hasattr(model, 'model') else model.predictor.model
    pytorch_model.eval()
    with torch.no_grad():
        preds = pytorch_model(dual_input)
        from ultralytics.utils.ops import non_max_suppression
        nc = getattr(pytorch_model, 'nc', 80)
        preds = non_max_suppression(preds, conf_thres=CONFIDENCE_THRESHOLD, iou_thres=IOU_THRESH, classes=None, agnostic=False, max_det=300, nc=nc)
    pred_bboxes = []
    for det in preds[0]:
        x1, y1, x2, y2, conf, cls = det.cpu().numpy()
        pred_bboxes.append({'xyxy':[int(x1),int(y1),int(x2),int(y2)], 'conf':float(conf), 'class':int(cls)})
    return pred_bboxes, wide_size, wide_img

def scale_coords(img1_shape, coords, img0_shape):
    # img1_shape: (h, w) 모델 입력 크기 (640, 640)
    # coords: [N, 4] (x1, y1, x2, y2) 모델 입력 기준
    # img0_shape: (h, w) 원본 이미지 크기
    coords = coords.copy().astype(np.float32)
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad_w = (img1_shape[1] - img0_shape[1] * gain) / 2
    pad_h = (img1_shape[0] - img0_shape[0] * gain) / 2
    coords[:, [0, 2]] -= pad_w
    coords[:, [1, 3]] -= pad_h
    coords[:, :4] /= gain
    coords[:, 0] = np.clip(coords[:, 0], 0, img0_shape[1])
    coords[:, 1] = np.clip(coords[:, 1], 0, img0_shape[0])
    coords[:, 2] = np.clip(coords[:, 2], 0, img0_shape[1])
    coords[:, 3] = np.clip(coords[:, 3], 0, img0_shape[0])
    return coords

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    image_pairs = get_image_pairs(WIDE_DIR, NARROW_DIR)
    summary_rows = []
    total_small = total_tp = total_fn = total_fp = 0
    model = YOLO(MODEL_PATH)
    model_name = Path(MODEL_PATH).stem
    debug_save_count = 0  # 디버그 이미지 저장 개수
    for base, wide_path, narrow_path in image_pairs:
        label_path = os.path.join(LABEL_DIR, base + ".txt")
        gt_bboxes_raw = load_labels(label_path)
        img = cv2.imread(wide_path)
        img_h, img_w = img.shape[:2]
        gt_bboxes = []
        for bbox in gt_bboxes_raw:
            if is_small_bbox(bbox, img_w, img_h):  # 클래스 제한 없이 작은 박스만
                xyxy = yolo_to_xyxy(bbox, img_w, img_h)
                bbox['xyxy'] = xyxy
                gt_bboxes.append(bbox)
        total_small += len(gt_bboxes)
        pred_bboxes, _, wide_img = run_dual_inference(model, wide_path, narrow_path)
        # === 예측 bbox를 원본 이미지 크기로 변환 ===
        if len(pred_bboxes) > 0:
            pred_xyxy = np.array([p['xyxy'] for p in pred_bboxes])
            pred_xyxy_scaled = scale_coords((IMAGE_SIZE, IMAGE_SIZE), pred_xyxy, (img_h, img_w))
            for i, p in enumerate(pred_bboxes):
                p['xyxy'] = [int(x) for x in pred_xyxy_scaled[i]]
        # =========================================
        # 안전한 TP/FN/FP 계산 (인덱스 범위 체크)
        matches, used_pred = match_bboxes(gt_bboxes, pred_bboxes, IOU_THRESH)
        tp = sum(
            1 for m in matches
            if m[1] is not None
            and 0 <= m[0] < len(gt_bboxes)
            and 0 <= m[1] < len(pred_bboxes)
            and gt_bboxes[m[0]]['class'] == pred_bboxes[m[1]]['class']
        )
        fn = sum(
            1 for m in matches
            if m[1] is None and 0 <= m[0] < len(gt_bboxes)
        )
        # === FP: 20픽셀 이하, 클래스 제한 없이 ===
        fp = 0
        for j, pred in enumerate(pred_bboxes):
            if j not in used_pred:
                x1, y1, x2, y2 = pred['xyxy']
                w, h = x2 - x1, y2 - y1
                if w <= SMALL_BBOX_THRESH or h <= SMALL_BBOX_THRESH:
                    fp += 1
        # =============================================
        # === 디버깅: TP가 1개 이상이면서 조건에 맞는 predict 박스가 1개 이상인 이미지 2개만 wide 이미지에 bbox 시각화해서 저장 ===
        pred_box_count = sum(1 for pred in pred_bboxes if pred['class'] == 9 and max(pred['xyxy'][2] - pred['xyxy'][0], pred['xyxy'][3] - pred['xyxy'][1]) <= SMALL_BBOX_THRESH)
        if tp > 0 and pred_box_count > 0 and debug_save_count < 2:
            debug_img = img.copy()
            # GT bbox (녹색)
            for bbox in gt_bboxes:
                x1, y1, x2, y2 = bbox['xyxy']
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0,255,0), 2)
            # Predict bbox (빨간색, class 9, 30픽셀 이하만)
            for pred in pred_bboxes:
                x1, y1, x2, y2 = pred['xyxy']
                w, h = x2 - x1, y2 - y1
                if pred['class'] == 9 and max(w, h) <= SMALL_BBOX_THRESH:
                    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0,0,255), 2)
            debug_path = os.path.join(OUTPUT_DIR, f"debug_pred_{debug_save_count+1}.jpg")
            cv2.imwrite(debug_path, debug_img)
            debug_save_count += 1
        # =============================================
        total_tp += tp
        total_fn += fn
        total_fp += fp
        summary_rows.append({
            'model': model_name,
            'image': base,
            'small_gt': len(gt_bboxes),
            'tp': tp,
            'fn': fn,
            'fp': fp
        })
    # 모델별 통계 저장
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0
    print(f"[SUMMARY] {model_name}: small_gt={total_small}, tp={total_tp}, fn={total_fn}, fp={total_fp}, recall={recall:.3f}, precision={precision:.3f}, f1={f1:.3f}")
    # CSV 저장
    csv_path = os.path.join(OUTPUT_DIR, "summary.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model','image','small_gt','tp','fn','fp'])
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    # Bar chart로 성능 비교 (모델 1개지만 그래프 유지)
    plt.figure(figsize=(6,5))
    plt.bar(['Recall'], [recall], width=0.3, label='Recall')
    plt.bar(['Precision'], [precision], width=0.3, label='Precision')
    plt.bar(['F1-score'], [f1], width=0.3, label='F1-score')
    plt.ylim(0, 1.05)
    plt.ylabel('Score')
    plt.title(f'Small BBox Detection Performance ({model_name})')
    plt.legend()
    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, "compare_summary.png")
    plt.savefig(chart_path)
    print(f"\n[RESULT] Summary saved to {csv_path}")
    print(f"[RESULT] Comparison chart saved to {chart_path}")

if __name__ == "__main__":
    main() 