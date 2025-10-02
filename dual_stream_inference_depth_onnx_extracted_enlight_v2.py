#!/usr/bin/env python3
"""
Dual Stream YOLO ONNX Inference Script with Depth Visualization (Extracted Model Version)
Performs inference with extracted dual-stream YOLO ONNX model (headless) and implements custom post-processing in Python.
Shows depth prediction + ground truth.
Modified to allow arbitrary reg tensor for additional bbox visualization.
Supports dynamic channel size for reg (e.g., C=64 for both ONNX and arbitrary).
Replicates the provided 64 values across all grid cells for arbitrary reg tensors.
Supports optional wide and narrow image paths via command-line arguments; falls back to random if not provided.
Usage: python script.py [wide_image_path] [narrow_image_path]
"""
import sys
import os
from pathlib import Path
import numpy as np
import cv2
import time
import random
import onnxruntime as ort
import math
import argparse

# 🔧 로컬 ultralytics 모듈을 우선적으로 사용하도록 설정
SCRIPT_DIR = Path(__file__).parent.absolute()
ULTRALYTICS_ROOT = SCRIPT_DIR
sys.path.insert(0, str(ULTRALYTICS_ROOT))
print(f"🔧 Using local ultralytics from: {ULTRALYTICS_ROOT}")

# ============================================================
# 🛠️ 설정 부분 - 여기를 수정하세요!
# ============================================================
# ONNX 모델 파일 경로 (추출된 모델)
ONNX_PATH = "/home/byounggun/ultralytics/runs/train/exp350/weights/best_dual_input_depth_extracted.onnx"
# 입력 이미지 디렉토리
WIDE_DIR = "/home/byounggun/ultralytics/swm_dual_split/val/images/"
NARROW_DIR = "/home/byounggun/ultralytics/swm_dual_split/val/val_narrow_images/"
# 라벨 디렉토리 (GT depth 정보)
LABEL_DIR = "/home/byounggun/ultralytics/swm_dual_split/val/labels/"
# 출력 설정
OUTPUT_DIR = "inference_results_depth_onnx_extracted_enlight"
# 추론 설정
CONFIDENCE_THRESHOLD = 0.25  # 높여서 약한 detection 필터 (기존 0.1 -> 0.25)
IOU_THRESHOLD = 0.35
IMAGE_SIZE = 640
MIN_BOX_SIZE = 5.0  # 최소 bbox 크기 (너무 작은 box 스킵)
# Depth 정규화 정보 (C 코드와 맞춤)
DEP_MIN = 0.1
DEP_MAX = 419.1
# Reverse letterbox 적용 여부 (C 코드의 CUSTOM_REVERSE_LETTERBOX)
REVERSE_LETTERBOX = True  # 필요시 True로 변경
# ============================================================

try:
    from ultralytics import YOLO
    import ultralytics
    print(f"✅ Using ultralytics from: {ultralytics.__file__}")
except ImportError as e:
    print(f"❌ Failed to import ultralytics: {e}")
    print("Please ensure you're in the ultralytics directory with custom modules")
    sys.exit(1)

class DepthDenormalizer:
    """Depth 값을 원본 스케일로 복구하는 클래스"""
   
    def __init__(self, min_depth=DEP_MIN, max_depth=DEP_MAX):
        self.min_depth = min_depth
        self.max_depth = max_depth
   
    def denormalize_depth(self, normalized_depth):
        return normalized_depth * (self.max_depth - self.min_depth) + self.min_depth

def load_ground_truth_labels(label_path):
    """
    라벨 파일에서 GT 정보를 로드
   
    Args:
        label_path (str): 라벨 파일 경로
       
    Returns:
        list: GT 정보 리스트 [class_id, x, y, w, h, depth]
    """
    if not os.path.exists(label_path):
        return []
   
    gt_labels = []
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
       
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 6:  # class x y w h depth
                class_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                depth = float(parts[5])  # 정규화된 depth 값
                gt_labels.append([class_id, x, y, w, h, depth])
    except Exception as e:
        print(f"❌ 라벨 파일 읽기 오류: {e}")
   
    return gt_labels

def convert_normalized_to_pixel_coords(bbox, img_width, img_height):
    """
    정규화된 좌표를 픽셀 좌표로 변환
   
    Args:
        bbox: [x_center, y_center, width, height] (정규화된 좌표)
        img_width, img_height: 이미지 크기
       
    Returns:
        [x1, y1, x2, y2] (픽셀 좌표)
    """
    x_center, y_center, width, height = bbox
   
    # 픽셀 좌표로 변환
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    width_px = width * img_width
    height_px = height * img_height
   
    # 좌상단, 우하단 좌표 계산
    x1 = int(x_center_px - width_px / 2)
    y1 = int(y_center_px - height_px / 2)
    x2 = int(x_center_px + width_px / 2)
    y2 = int(y_center_px + height_px / 2)
   
    return [x1, y1, x2, y2]

def preprocess_image(image_path, target_size=640):
    """
    이미지를 모델 입력에 맞게 전처리
   
    Args:
        image_path (str): 이미지 파일 경로
        target_size (int): 타겟 크기
   
    Returns:
        np.ndarray: 전처리된 이미지 [1, 3, H, W] (numpy, for ONNX input)
        tuple: 원본 이미지 크기 (height, width)
        np.ndarray: 원본 이미지 (BGR)
    """
    # 이미지 로드
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
   
    original_height, original_width = image.shape[:2]
    print(f"📸 Loaded image {Path(image_path).name}: {original_width}x{original_height}")
   
    # RGB로 변환 (YOLO는 RGB 사용)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
   
    # 리사이즈 (letterbox 적용)
    # 비율 유지하면서 패딩 추가
    scale = min(target_size / original_width, target_size / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
   
    # 리사이즈
    resized = cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
   
    # 패딩 추가 (640x640 만들기)
    delta_w = target_size - new_width
    delta_h = target_size - new_height
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
   
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                               cv2.BORDER_CONSTANT, value=[114, 114, 114])
   
    # 정규화 및 배열 변환
    normalized = padded.astype(np.float32) / 255.0
    tensor = normalized.transpose(2, 0, 1)[np.newaxis, :]  # [1, 3, H, W]
   
    return tensor, (original_height, original_width), image

def create_dual_stream_inputs(wide_tensor, narrow_tensor):
    """
    두 이미지를 별도의 입력으로 준비 (ONNX 모델이 두 입력을 기대할 경우)
   
    Args:
        wide_tensor (np.ndarray): Wide stream 이미지 [1, 3, H, W]
        narrow_tensor (np.ndarray): Narrow stream 이미지 [1, 3, H, W]
   
    Returns:
        dict: {'images_wide': np.ndarray [1, 3, H, W], 'images_narrow': np.ndarray [1, 3, H, W]}
    """
    print(f"🔗 Created wide input: {wide_tensor.shape}")
    print(f"🔗 Created narrow input: {narrow_tensor.shape}")
   
    return {
        'images_wide': wide_tensor,
        'images_narrow': narrow_tensor
    }

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def stabilized_compute_dfl_dir(reg_output, y, x, start, num_bins=16):
    """
    Stabilized DFL decode for one direction: sum k * softmax(v_k)
    """
    vals = reg_output[0, start:start + num_bins, y, x]  # [16]
    max_val = np.max(vals)
    exp_vals = np.exp(vals - max_val)
    sum_exp = np.sum(exp_vals)
    if sum_exp <= 0.0:
        sum_exp = 1e-6
    softmax = exp_vals / sum_exp
    d = np.sum(np.arange(num_bins, dtype=np.float32) * softmax)
    return d

def apply_reverse_letterbox_coords(x1, y1, x2, y2, orig_w, orig_h, model_size):
    """
    Undo centered letterbox with color=114: remove padding, divide by gain, then clip
    """
    gain_w = model_size / orig_w
    gain_h = model_size / orig_h
    gain = min(gain_w, gain_h)
    pad_w = (model_size - orig_w * gain) * 0.5
    pad_h = (model_size - orig_h * gain) * 0.5
    x1 = (x1 - pad_w) / gain
    x2 = (x2 - pad_w) / gain
    y1 = (y1 - pad_h) / gain
    y2 = (y2 - pad_h) / gain
    x1 = max(0.0, min(x1, orig_w))
    x2 = max(0.0, min(x2, orig_w))
    y1 = max(0.0, min(y1, orig_h))
    y2 = max(0.0, min(y2, orig_h))
    return x1, y1, x2, y2

def box_iou(x1, y1, x2, y2, ax1, ay1, ax2, ay2):
    xx1 = max(x1, ax1)
    yy1 = max(y1, ay1)
    xx2 = min(x2, ax2)
    yy2 = min(y2, ay2)
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    area1 = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area2 = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    denom = area1 + area2 - inter + 1e-6
    return inter / denom if denom > 0.0 else 0.0

def custom_postprocess(outputs, orig_size, img_size=640.0, conf_thres=CONFIDENCE_THRESHOLD, iou_thres=IOU_THRESHOLD, reverse_letterbox=REVERSE_LETTERBOX, dfl_order="ltrb"):
    """
    Python implementation of the C custom_postproc_run function.
    
    Args:
        outputs: List of 9 np.ndarray from ONNX: [reg0, cls0, dep0, reg1, cls1, dep1, reg2, cls2, dep2]
                 Each: [1, C, H, W] where reg C=64, cls C=28, dep C=1
        orig_size: (orig_h, orig_w)
        img_size: Model input size (640)
   
    Returns:
        list: detections [[x1, y1, x2, y2, conf, class_id, depth_normalized]]
    """
    orig_h, orig_w = orig_size
    num_levels = len(outputs) // 3 # Assuming 3 levels
    detections = []
    # Group by level (assuming outputs ordered as P3 reg,cls,dep; P4; P5)
    for li in range(num_levels):
        reg = outputs[li * 3] # [1, 64, H, W]
        cls = outputs[li * 3 + 1] # [1, 28, H, W]
        dep = outputs[li * 3 + 2] # [1, 1, H, W]
        _, reg_C, H, W = reg.shape
        num_bins = reg_C // 4 # 64/4=16
        stride = img_size / H
        print(f"[Level {li}] HxW = {H}x{W}, stride={stride}")
        # Determine DFL group order indices (default assumes channel groups 0: left,1: top,2: right,3: bottom)
        order = dfl_order.lower()
        # Ensure order contains exactly l,t,r,b
        if set(order) != set("ltrb") or len(order) != 4:
            order = "ltrb"  # fallback
        idx_l = order.index('l')
        idx_t = order.index('t')
        idx_r = order.index('r')
        idx_b = order.index('b')
        for y in range(H):
            for x in range(W):
                # DFL decoding (stabilized)
                d_l = stabilized_compute_dfl_dir(reg, y, x, idx_l * num_bins, num_bins)
                d_t = stabilized_compute_dfl_dir(reg, y, x, idx_t * num_bins, num_bins)
                d_r = stabilized_compute_dfl_dir(reg, y, x, idx_r * num_bins, num_bins)
                d_b = stabilized_compute_dfl_dir(reg, y, x, idx_b * num_bins, num_bins)
                anchor_x = x + 0.5
                anchor_y = y + 0.5
                x1 = (anchor_x - d_l) * stride
                y1 = (anchor_y - d_t) * stride
                x2 = (anchor_x + d_r) * stride
                y2 = (anchor_y + d_b) * stride
                # Clip and validity check
                # Do NOT early-clip to 0..img_size; allow negative/overflow then reverse letterbox and final clip.
                if (x2 <= x1 or y2 <= y1 or (x2 - x1) < MIN_BOX_SIZE or (y2 - y1) < MIN_BOX_SIZE):
                    continue # Skip invalid or too small boxes
                if reverse_letterbox:
                    x1, y1, x2, y2 = apply_reverse_letterbox_coords(x1, y1, x2, y2, orig_w, orig_h, img_size)
                    # Final clip to original image bounds
                    x1 = max(0.0, min(x1, orig_w))
                    x2 = max(0.0, min(x2, orig_w))
                    y1 = max(0.0, min(y1, orig_h))
                    y2 = max(0.0, min(y2, orig_h))
                else:
                    # If not reversing letterbox, then clip to model size
                    x1 = max(0.0, min(x1, img_size))
                    x2 = max(0.0, min(x2, img_size))
                    y1 = max(0.0, min(y1, img_size))
                    y2 = max(0.0, min(y2, img_size))
                # Class probs
                max_conf = 0.0
                max_class = -1
                for k in range(28): # cls_C=28
                    logit = cls[0, k, y, x]
                    prob = sigmoid(logit)
                    if prob > max_conf:
                        max_conf = prob
                        max_class = k
                if max_conf > conf_thres:
                    dep_norm = sigmoid(dep[0, 0, y, x]) if dep is not None else 0.0
                    detections.append([x1, y1, x2, y2, max_conf, max_class, dep_norm])
    # Sort by conf descending
    detections = sorted(detections, key=lambda d: d[4], reverse=True)
    # NMS
    keep = []
    for i in range(len(detections)):
        if detections[i][4] == 0.0: continue
        keep.append(detections[i])
        for j in range(i + 1, len(detections)):
            if detections[j][4] == 0.0: continue
            if detections[i][5] != detections[j][5]: continue
            iou = box_iou(*detections[i][:4], *detections[j][:4])
            if iou > iou_thres:
                detections[j][4] = 0.0
    detections = [d for d in detections if d[4] > 0.0]
    print(f"Total detections after NMS: {len(detections)}")
    return detections

def draw_detections_with_depth(image, detections, gt_labels, depth_denormalizer, class_names=None, prefix="PRED"):
    """
    이미지에 검출 결과와 depth 정보를 그리기
   
    Args:
        image (np.ndarray): 원본 이미지 (BGR)
        detections (list): 검출 결과
        gt_labels (list): Ground Truth 라벨
        depth_denormalizer: Depth 역정규화 객체
        class_names (list): 클래스 이름 리스트
        prefix (str): 라벨 prefix (e.g., "PRED" or "ARBITRARY")
   
    Returns:
        np.ndarray: 검출 결과가 그려진 이미지
    """
    if class_names is None:
        class_names = [f"class_{i}" for i in range(28)] # 클래스 수 28로 변경
   
    result_image = image.copy()
    img_height, img_width = image.shape[:2]
   
    # 색상 팔레트
    pred_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)] # 예측용
    gt_color = (128, 128, 128) # GT용 회색
   
    # Ground Truth 박스 그리기 (먼저 그려서 뒤에 위치)
    print(f"📊 Drawing {len(gt_labels)} GT labels")
    for i, gt_label in enumerate(gt_labels):
        class_id, x_center, y_center, width, height, normalized_depth = gt_label
       
        # 정규화된 좌표를 픽셀 좌표로 변환
        bbox_pixel = convert_normalized_to_pixel_coords([x_center, y_center, width, height],
                                                      img_width, img_height)
        x1, y1, x2, y2 = bbox_pixel
       
        # GT depth 역정규화
        gt_depth_original = depth_denormalizer.denormalize_depth(normalized_depth)
       
        # GT 박스 그리기 (점선)
        cv2.rectangle(result_image, (x1, y1), (x2, y2), gt_color, 2, lineType=cv2.LINE_AA)
       
        # GT 라벨 텍스트
        class_name = class_names[int(class_id)] if int(class_id) < len(class_names) else f"class_{int(class_id)}"
        gt_label_text = f"GT-{class_name}: {gt_depth_original:.2f}m"
       
        # GT 텍스트 배경
        (text_width, text_height), _ = cv2.getTextSize(gt_label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(result_image, (x1, y1 - text_height - 10), (x1 + text_width, y1), gt_color, -1)
       
        # GT 텍스트 그리기
        cv2.putText(result_image, gt_label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
   
    # Prediction 박스 그리기 (위에 그려서 앞에 위치)
    print(f"🎯 Drawing {len(detections)} {prefix} predictions")
    for i, detection in enumerate(detections):
        x1, y1, x2, y2, confidence, class_id, pred_depth_normalized = detection
       
        # 예측 depth 역정규화
        pred_depth_original = depth_denormalizer.denormalize_depth(pred_depth_normalized)
       
        # 박스 그리기
        color = pred_colors[int(class_id) % len(pred_colors)]
        cv2.rectangle(result_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 3) # 두꺼운 선
       
        # 라벨 텍스트
        class_name = class_names[int(class_id)] if int(class_id) < len(class_names) else f"class_{int(class_id)}"
        pred_label_text = f"{prefix}-{class_name}: {confidence:.2f}, D:{pred_depth_original:.2f}m"
       
        # 텍스트 배경 위치 조정 (GT와 겹치지 않도록)
        text_y = int(y2) + 25
        if text_y > img_height - 30:
            text_y = int(y1) - 10
       
        # 텍스트 배경
        (text_width, text_height), _ = cv2.getTextSize(pred_label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(result_image, (int(x1), text_y - text_height), (int(x1) + text_width, text_y + 5), color, -1)
       
        # 텍스트 그리기
        cv2.putText(result_image, pred_label_text, (int(x1), text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
   
    return result_image

def create_arbitrary_outputs(original_outputs, arbitrary_regs):
    """
    임의의 reg 텐서를 사용하여 새로운 outputs 리스트 생성.
    cls와 dep는 원본을 유지, reg만 교체.
   
    Args:
        original_outputs: 원본 ONNX outputs (9 tensors)
        arbitrary_regs: list of 3 np.ndarray [1, C, H, W] for each level
   
    Returns:
        list: 새로운 outputs [arbitrary_reg0, cls0, dep0, arbitrary_reg1, cls1, dep1, ...]
    """
    if not len(arbitrary_regs) == 3:
        raise ValueError("arbitrary_regs must contain exactly 3 regression tensors for 3 levels.")
   
    new_outputs = []
    for li in range(3):
        new_outputs.append(arbitrary_regs[li])
        new_outputs.append(original_outputs[li * 3 + 1]) # cls
        new_outputs.append(original_outputs[li * 3 + 2]) # dep
    return new_outputs

def load_head_text(path: str, reg_shape, cls_shape, dep_shape):
    """Load one head text file (head0/head1/head2) into dense tensors.
    Each line: ROLE c y x value  (ROLE in {DFL, CLS, DEP})
    Returns: reg(np.ndarray), cls(np.ndarray), dep(np.ndarray), coverage dict
    Missing entries remain 0.0.
    """
    reg = np.zeros(reg_shape, dtype=np.float32)
    cls = np.zeros(cls_shape, dtype=np.float32)
    dep = np.zeros(dep_shape, dtype=np.float32)
    reg_filled = 0
    cls_filled = 0
    dep_filled = 0
    try:
        with open(path, 'r') as f:
            for line in f:
                if not line.strip() or line.startswith('#'):  # skip comments/blank
                    continue
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                role, c, y, x, val = parts
                c = int(c); y = int(y); x = int(x); v = float(val)
                if role == 'DFL':
                    if 0 <= c < reg_shape[1] and 0 <= y < reg_shape[2] and 0 <= x < reg_shape[3]:
                        reg[0, c, y, x] = v; reg_filled += 1
                elif role == 'CLS':
                    if 0 <= c < cls_shape[1] and 0 <= y < cls_shape[2] and 0 <= x < cls_shape[3]:
                        cls[0, c, y, x] = v; cls_filled += 1
                elif role == 'DEP':
                    # depth channel assumed single (c should be 0)
                    if 0 <= y < dep_shape[2] and 0 <= x < dep_shape[3]:
                        dep[0, 0, y, x] = v; dep_filled += 1
    except FileNotFoundError:
        return None, None, None, {
            'error': f'file_not_found',
        }
    total_reg = reg_shape[1] * reg_shape[2] * reg_shape[3]
    total_cls = cls_shape[1] * cls_shape[2] * cls_shape[3]
    total_dep = dep_shape[2] * dep_shape[3]
    coverage = {
        'reg_filled': reg_filled,
        'reg_total': total_reg,
        'reg_pct': reg_filled / total_reg * 100.0 if total_reg else 0.0,
        'cls_filled': cls_filled,
        'cls_total': total_cls,
        'cls_pct': cls_filled / total_cls * 100.0 if total_cls else 0.0,
        'dep_filled': dep_filled,
        'dep_total': total_dep,
        'dep_pct': dep_filled / total_dep * 100.0 if total_dep else 0.0,
    }
    return reg, cls, dep, coverage

def main():
    """메인 inference 함수 (head 텍스트 로더 + quiet 지원)"""
    parser = argparse.ArgumentParser()
    # Positional (optional) image paths: wide then narrow
    parser.add_argument('wide_image', nargs='?', help='Positional wide image path')
    parser.add_argument('narrow_image', nargs='?', help='Positional narrow image path')
    # Flag alternatives (backwards compatible)
    parser.add_argument('--wide', type=str, help='Wide image path (flag)')
    parser.add_argument('--narrow', type=str, help='Narrow image path (flag)')
    # Head loading modes
    parser.add_argument('--head-base', type=str, help='Head text file base path prefix (e.g., /path/to/head ) or even /path/to/head0.txt')
    parser.add_argument('--head-files', nargs=3, metavar=('HEAD0','HEAD1','HEAD2'), help='Explicit three head text files for P3,P4,P5 in order')
    parser.add_argument('--quiet', action='store_true', help='Suppress console prints; write only summary file')
    parser.add_argument('--skip-onnx', action='store_true', help='Skip ONNX inference (use only head text if provided)')
    parser.add_argument('--head-scale', type=float, default=1.0, help='Scale factor applied to logits loaded from head text files')
    parser.add_argument('--dfl-order', type=str, default='ltrb', help='Order of DFL channel groups (per 16-bin group). Combination of l,t,r,b e.g. ltrb, tlrb, lbrt etc.')
    args, _ = parser.parse_known_args()

    QUIET = args.quiet
    if QUIET:
        # Monkey-patch print to no-op
        import builtins as _b
        _b.print = lambda *a, **k: None

    print("🚀 Starting Dual Stream YOLO ONNX Inference with Depth (Extracted Model)...")

    depth_denormalizer = DepthDenormalizer()
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    run_onnx = (not args.skip_onnx) and (args.head_base is None) and (args.head_files is None)
    ort_session = None
    if run_onnx:
        try:
            providers = ['CPUExecutionProvider']
            ort_session = ort.InferenceSession(ONNX_PATH, providers=providers)
            print("✅ ONNX Runtime session loaded successfully!")
            input_names = [inp.name for inp in ort_session.get_inputs()]
            print(f"📥 Model input names: {input_names}")
        except Exception as e:
            print(f"❌ Failed to load ONNX model: {e}")
            if args.head_base is None:
                return
            else:
                run_onnx = False

    # 이미지 경로 결정
    # Resolve image paths priority: positional > flags > random
    candidate_wide = args.wide_image or args.wide
    candidate_narrow = args.narrow_image or args.narrow
    if candidate_wide and candidate_narrow and os.path.exists(candidate_wide) and os.path.exists(candidate_narrow):
        WIDE_IMAGE_PATH = candidate_wide
        NARROW_IMAGE_PATH = candidate_narrow
        filename = Path(WIDE_IMAGE_PATH).stem
    else:
        # Fallback to random paired selection
        wide_files = [f for f in os.listdir(WIDE_DIR) if f.lower().endswith(('.jpg', '.png'))]
        if not wide_files:
            print("❌ No image files found in wide dir")
            return
        picked = random.choice(wide_files)
        WIDE_IMAGE_PATH = os.path.join(WIDE_DIR, picked)
        NARROW_IMAGE_PATH = os.path.join(NARROW_DIR, picked)
        if not os.path.exists(NARROW_IMAGE_PATH):
            print(f"❌ Matching narrow image not found: {NARROW_IMAGE_PATH}")
            return
        filename = Path(picked).stem

    label_filename = filename.replace('.jpg', '.txt')
    LABEL_PATH = os.path.join(LABEL_DIR, label_filename)
    gt_labels = load_ground_truth_labels(LABEL_PATH)

    OUTPUT_IMAGE_NAME = f"dual_stream_depth_result_onnx_extracted_{filename}"
    HEAD_OUTPUT_IMAGE_NAME = f"dual_stream_depth_result_head_{filename}"

    try:
        wide_tensor, wide_original_size, wide_image = preprocess_image(WIDE_IMAGE_PATH, IMAGE_SIZE)
        narrow_tensor, narrow_original_size, narrow_image = preprocess_image(NARROW_IMAGE_PATH, IMAGE_SIZE)
    except Exception:
        return

    dual_inputs = {'images_wide': wide_tensor, 'images_narrow': narrow_tensor}

    results = None
    inference_time = 0.0
    if run_onnx:
        start_time = time.time()
        try:
            results = ort_session.run(None, dual_inputs)
        except Exception:
            results = None
        inference_time = time.time() - start_time

    original_detections = []
    if results is not None:
        original_detections = custom_postprocess(results, wide_original_size, IMAGE_SIZE, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, REVERSE_LETTERBOX, dfl_order=args.dfl_order)
        result_image = draw_detections_with_depth(wide_image, original_detections, gt_labels, depth_denormalizer, prefix="PRED")
        (output_dir / f"{OUTPUT_IMAGE_NAME}.jpg").parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(output_dir / f"{OUTPUT_IMAGE_NAME}.jpg"), result_image)

    # Head-base 로딩 처리
    head_detections = []
    head_coverages = []
    head_mode = None
    head_identifier = None
    if args.head_files:
        head_mode = 'files'
        head_identifier = ','.join(args.head_files)
        head_paths = list(args.head_files)
    elif args.head_base:
        head_mode = 'base'
        # Normalize base (allow passing head0.txt or head0, or base path w/o index)
        base = args.head_base
        if base.endswith('head0.txt'):
            base = base[:-len('0.txt')]  # drop the trailing 0.txt
        else:
            # If ends with something like 0.txt /1.txt /2.txt remove that pattern
            if base.endswith('.txt'):
                # strip digit + .txt if present
                for digit in ('0','1','2'):
                    suffix = digit + '.txt'
                    if base.endswith(suffix):
                        base = base[:-len(suffix)]
                        break
        head_identifier = base
        head_paths = [f"{base}{i}.txt" for i in range(3)]
    else:
        head_paths = []

    if head_paths:
        head_shapes = [
            {'reg': (1,64,80,80), 'cls': (1,28,80,80), 'dep': (1,1,80,80)},
            {'reg': (1,64,40,40), 'cls': (1,28,40,40), 'dep': (1,1,40,40)},
            {'reg': (1,64,20,20), 'cls': (1,28,20,20), 'dep': (1,1,20,20)},
        ]
        loaded = []
        for i, shp in enumerate(head_shapes):
            path = head_paths[i]
            reg, cls, dep, cov = load_head_text(path, shp['reg'], shp['cls'], shp['dep'])
            cov_entry = {'file': path}
            if 'error' in cov:
                cov_entry.update(cov)
            else:
                cov_entry.update(cov)
            head_coverages.append(cov_entry)
            if reg is None:
                continue
            if args.head_scale != 1.0:
                reg *= args.head_scale
                cls *= args.head_scale
                dep *= args.head_scale
            loaded.extend([reg, cls, dep])
        if len(loaded) == 9:
            head_detections = custom_postprocess(loaded, wide_original_size, IMAGE_SIZE, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, REVERSE_LETTERBOX, dfl_order=args.dfl_order)
            head_image = draw_detections_with_depth(wide_image, head_detections, gt_labels, depth_denormalizer, prefix="HEAD")
            cv2.imwrite(str(output_dir / f"{HEAD_OUTPUT_IMAGE_NAME}.jpg"), head_image)

    # Summary 저장 (prints suppressed if quiet)
    summary_path = output_dir / f"summary_{filename.replace('.jpg','.txt')}"
    with open(summary_path, 'w') as f:
        f.write("Dual Stream YOLO Summary\n")
        f.write("=======================\n")
        f.write(f"Images: wide={WIDE_IMAGE_PATH} narrow={NARROW_IMAGE_PATH}\n")
        f.write(f"GT labels: {len(gt_labels)}\n")
        if run_onnx and results is not None:
            f.write(f"ONNX inference time: {inference_time:.4f}s\n")
            f.write(f"Original detections: {len(original_detections)}\n")
        else:
            f.write("Original detections: SKIPPED\n")
        if head_paths:
            if head_mode == 'base':
                f.write(f"Head detections: {len(head_detections)} (base={head_identifier})\n")
            else:
                f.write(f"Head detections: {len(head_detections)} (files={head_identifier})\n")
            for cov in head_coverages:
                if 'error' in cov:
                    f.write(f" {cov['file']}: ERROR {cov['error']}\n")
                else:
                    f.write(
                        f" {cov['file']}: reg {cov['reg_pct']:.2f}% ({cov['reg_filled']}/{cov['reg_total']}), "
                        f"cls {cov['cls_pct']:.2f}% ({cov['cls_filled']}/{cov['cls_total']}), "
                        f"dep {cov['dep_pct']:.2f}% ({cov['dep_filled']}/{cov['dep_total']})\n"
                    )
        f.write("\nDetections (Original):\n")
        for d in original_detections:
            x1,y1,x2,y2,conf,cls,depn = d
            f.write(f" PRED cls={int(cls)} conf={conf:.3f} depth_norm={depn:.4f} box=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})\n")
        if head_paths:
            f.write("\nDetections (Head Files):\n")
            for d in head_detections:
                x1,y1,x2,y2,conf,cls,depn = d
                f.write(f" HEAD cls={int(cls)} conf={conf:.3f} depth_norm={depn:.4f} box=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})\n")
    if not QUIET:
        print(f"Summary saved: {summary_path}")
        if results is not None:
            print(f"Original image saved: {output_dir / f'{OUTPUT_IMAGE_NAME}.jpg'}")
        if head_paths:
            print(f"Head image saved: {output_dir / f'{HEAD_OUTPUT_IMAGE_NAME}.jpg'}")

    # Done
    return

if __name__ == "__main__":
    main()