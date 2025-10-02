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

def custom_postprocess(outputs, orig_size, img_size=640.0, conf_thres=CONFIDENCE_THRESHOLD, iou_thres=IOU_THRESHOLD, reverse_letterbox=REVERSE_LETTERBOX):
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
        for y in range(H):
            for x in range(W):
                # DFL decoding (stabilized)
                d_l = stabilized_compute_dfl_dir(reg, y, x, 0, num_bins)
                d_t = stabilized_compute_dfl_dir(reg, y, x, num_bins, num_bins)
                d_r = stabilized_compute_dfl_dir(reg, y, x, 2 * num_bins, num_bins)
                d_b = stabilized_compute_dfl_dir(reg, y, x, 3 * num_bins, num_bins)
                anchor_x = x + 0.5
                anchor_y = y + 0.5
                x1 = (anchor_x - d_l) * stride
                y1 = (anchor_y - d_t) * stride
                x2 = (anchor_x + d_r) * stride
                y2 = (anchor_y + d_b) * stride
                # Clip and validity check
                x1 = max(0.0, x1)
                y1 = max(0.0, y1)
                x2 = min(orig_w if reverse_letterbox else img_size, x2)
                y2 = min(orig_h if reverse_letterbox else img_size, y2)
                if (x2 <= x1 or y2 <= y1 or (x2 - x1) < MIN_BOX_SIZE or (y2 - y1) < MIN_BOX_SIZE):
                    continue # Skip invalid or too small boxes
                if reverse_letterbox:
                    x1, y1, x2, y2 = apply_reverse_letterbox_coords(x1, y1, x2, y2, orig_w, orig_h, img_size)
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

def main():
    """메인 inference 함수"""
   
    print("🚀 Starting Dual Stream YOLO ONNX Inference with Depth (Extracted Model)...")
   
    # Depth 역정규화 객체 생성
    depth_denormalizer = DepthDenormalizer()
   
    # 출력 디렉토리 생성
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
   
    # ONNX Runtime 세션 로드
    try:
        providers = ['CPUExecutionProvider'] # 필요시 'CUDAExecutionProvider' 등 추가
        ort_session = ort.InferenceSession(ONNX_PATH, providers=providers)
        print("✅ ONNX Runtime session loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load ONNX model: {e}")
        return
   
    # 모델 입력 이름 확인
    input_names = [input.name for input in ort_session.get_inputs()]
    print(f"📥 Model input names: {input_names}")
   
    # Command-line arguments for optional wide and narrow paths
    wide_path = None
    narrow_path = None
    if len(sys.argv) >= 3:
        wide_path = sys.argv[1]
        narrow_path = sys.argv[2]
        if not os.path.exists(wide_path) or not os.path.exists(narrow_path):
            print(f"❌ Specified paths do not exist: {wide_path}, {narrow_path}. Falling back to random.")
            wide_path = None
            narrow_path = None
   
    if wide_path and narrow_path:
        WIDE_IMAGE_PATH = wide_path
        NARROW_IMAGE_PATH = narrow_path
        filename = Path(wide_path).stem # Use wide filename as base
        print(f"🖼️ Using specified image set: {filename}")
    else:
        # 랜덤 이미지 세트 선택
        wide_files = [f for f in os.listdir(WIDE_DIR) if f.lower().endswith('.jpg')]
        if not wide_files:
            print(f"❌ No JPG files found in {WIDE_DIR}")
            return
        
        filename = random.choice(wide_files)
        WIDE_IMAGE_PATH = os.path.join(WIDE_DIR, filename)
        NARROW_IMAGE_PATH = os.path.join(NARROW_DIR, filename)
        
        if not os.path.exists(NARROW_IMAGE_PATH):
            print(f"❌ Matching narrow image not found: {NARROW_IMAGE_PATH}")
            return
        
        print(f"🖼️ Selected random image set: {filename}")
   
    # 라벨 파일 경로
    label_filename = filename.replace('.jpg', '.txt')
    LABEL_PATH = os.path.join(LABEL_DIR, label_filename)
   
    print(f"📄 Label file: {LABEL_PATH}")
   
    # Ground Truth 라벨 로드
    gt_labels = load_ground_truth_labels(LABEL_PATH)
    print(f"🎯 Loaded {len(gt_labels)} GT labels")
   
    # 출력 파일 이름 동적 설정
    OUTPUT_IMAGE_NAME = f"dual_stream_depth_result_onnx_extracted_{filename}"
    ARBITRARY_OUTPUT_IMAGE_NAME = f"dual_stream_depth_result_arbitrary_{filename}"
   
    # 2. 이미지 전처리
    print("🖼️ Preprocessing images...")
    try:
        wide_tensor, wide_original_size, wide_image = preprocess_image(WIDE_IMAGE_PATH, IMAGE_SIZE)
        narrow_tensor, narrow_original_size, narrow_image = preprocess_image(NARROW_IMAGE_PATH, IMAGE_SIZE)
    except Exception as e:
        print(f"❌ Failed to preprocess images: {e}")
        return
   
    # 3. 듀얼 스트림 입력 생성
    dual_inputs = create_dual_stream_inputs(wide_tensor, narrow_tensor)
   
    # 4. ONNX 추론 실행
    print("🔮 Running ONNX inference...")
    start_time = time.time()
   
    try:
        # ONNX Runtime 추론 (raw outputs)
        onnx_outputs = ort_session.run(None, dual_inputs)
       
        # onnx_outputs: list of 9 tensors
        results = onnx_outputs
       
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return
   
    inference_time = time.time() - start_time
    print(f"⚡ Inference completed in {inference_time:.3f}s")
   
    # 5. 커스텀 후처리 (Python 구현) - 원본 (16 bins)
    print("📊 Processing original results with custom postprocess...")
    original_detections = custom_postprocess(results, wide_original_size, IMAGE_SIZE, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, REVERSE_LETTERBOX)
   
    print(f"🎯 Found {len(original_detections)} original detections")
    for i, detection in enumerate(original_detections):
        x1, y1, x2, y2, conf, cls, depth_normalized = detection
        depth_original = depth_denormalizer.denormalize_depth(depth_normalized)
        print(f" Detection {i+1}: class={int(cls)}, conf={conf:.3f}")
        print(f" Depth: {depth_normalized:.6f} (normalized) -> {depth_original:.2f}m (original)")
        print(f" BBox: ({x1},{y1},{x2},{y2})")
   
    # GT 정보도 출력
    print(f"🎯 Ground Truth labels:")
    for i, gt_label in enumerate(gt_labels):
        class_id, x_center, y_center, width, height, depth_normalized = gt_label
        depth_original = depth_denormalizer.denormalize_depth(depth_normalized)
        print(f" GT {i+1}: class={int(class_id)}")
        print(f" Depth: {depth_normalized:.6f} (normalized) -> {depth_original:.2f}m (original)")
        print(f" BBox: center=({x_center:.3f},{y_center:.3f}), size=({width:.3f},{height:.3f})")
   
    # 6. 결과 시각화 및 저장 (original)
    print("🎨 Drawing original detections with depth...")
    result_image = draw_detections_with_depth(wide_image, original_detections, gt_labels, depth_denormalizer, prefix="PRED")
   
    # 결과 이미지 저장
    output_path = output_dir / f"{OUTPUT_IMAGE_NAME}.jpg"
    cv2.imwrite(str(output_path), result_image)
   
    # === 추가: 임의 reg 텐서 처리 (16 bins) ===
    # 주의: 이제는 각 레벨(P3/P4/P5)에서 '단 하나의 그리드 셀'에만 64값을 주입하고,
    #       나머지 HxW 그리드 값들은 ONNX 원본 출력을 그대로 사용합니다.
    #       기본 주입 위치는 각 맵의 중앙 셀입니다.
    # P3 (80x80x64) 64 values
    p3_values_list = [5120, 5120, 0, -1024, -1024, -2048, -3072, -3072, -3072, -5120, -6144, -6144, -5120, -8192, -6144, -6144, 2048, 2048, 0, -1024, -2048, -2048, -3072, -3072, -3072, -4096, -4096, -4096, -5120, -6144, -4096, -4096, 0, 0, 4096, 5120, 3072, 0, -1024, -2048, -4096, -5120, -5120, -6144, -3072, -8192, -6144, -4096, -2048, 0, 1024, 3072, 2048, 1024, 0, -1024, -1024, -3072, -5120, -5120, -3072, 0, -7168, -5120]

    # P4 (40x40x64) 64 values
    p4_values_list = [1024, 2048, 2048, 1024, 1024, 1024, 1024, 0, 0, 0, 0, -1024, -1024, -1024, -1024, -1024, 0, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 0, 0, 0, 0, 0, -1024, -1024, 0, 1024, 1024, 2048, 1024, 1024, 1024, 1024, 1024, 0, -1024, -1024, -1024, -1024, -1024, -1024, 1024, 1024, 2048, 1024, 1024, 1024, 0, 0, 0, 1024, 0, 0, -1024, -1024, -1024, 0]

    # P5 (20x20x64) 64 values
    p5_values_list = [0, 1024, 1024, 2048, 2048, 2048, 1024, 0, -2048, -1024, -2048, -2048, -1024, -2048, -1024, -1024, -1024, 0, 0, 0, 2048, 3072, 3072, 1024, 0, -2048, -1024, -2048, -2048, -1024, 0, 0, 1024, 2048, 3072, 2048, 0, 0, -1024, -2048, -2048, -2048, -2048, -2048, -2048, -2048, -1024, -2048, 0, 0, 1024, 2048, 2048, 1024, 0, -2048, -3072, -2048, -2048, -2048, -1024, -2048, -1024, -2048]

    # helper: 원본 reg에서 단 하나의 (y, x) 그리드에만 64값을 주입
    def inject_single_cell(original_reg: np.ndarray, values, cell=None):
        new_reg = original_reg.copy()
        _, C, H, W = new_reg.shape
        if len(values) != C:
            raise ValueError(f"Value count must be exactly {C} for channels, got {len(values)}")
        if cell is None:
            cell = (H // 2, W // 2)  # 중앙 셀
        y, x = cell
        y = max(0, min(H - 1, int(y)))
        x = max(0, min(W - 1, int(x)))
        new_reg[0, :, y, x] = np.asarray(values, dtype=new_reg.dtype)
        return new_reg, (y, x)

    # === 추가: cls(28), dep(1)도 동일하게 단일 셀 주입 ===
    # 사용자 제공 예시 값을 기본으로 사용 (원하는 값으로 교체 가능)
    # P3 cls 28개
    p3_cls_values_list = [-8192, -8192, -12288, -12288, -17408, -20480, -20480, -12288, -18432, -13312, -17408, -16384, -18432, -20480, -20480, -17408, -21504, -18432, -21504, -23552, -20480, -18432, -17408, -21504, -15360, -18432, -19456, -13312]
    # P4 cls 28개
    p4_cls_values_list = [-19456, -19456, -20480, -19456, -21504, -22528, -20480, -21504, -20480, -18432, -18432, -20480, -20480, -20480, -20480, -21504, -21504, -21504, -21504, -20480, -20480, -20480, -20480, -20480, -21504, -21504, -21504, -20480]
    # P5 cls 28개
    p5_cls_values_list = [-19456, -20480, -19456, -21504, -18432, -20480, -19456, -20480, -20480, -19456, -19456, -19456, -19456, -19456, -19456, -19456, -19456, -19456, -19456, -19456, -19456, -20480, -19456, -19456, -19456, -19456, -20480, -19456]

    # depth(1) logits 예시
    p3_dep_value = -3072
    p4_dep_value = -3072
    p5_dep_value = -4096

    # ONNX 원본 reg 출력에 주입 (나머지 셀은 그대로 유지)
    reg_p3_injected, pos_p3 = inject_single_cell(results[0], p3_values_list, None)
    reg_p4_injected, pos_p4 = inject_single_cell(results[3], p4_values_list, None)
    reg_p5_injected, pos_p5 = inject_single_cell(results[6], p5_values_list, None)

    # cls/dep도 동일 위치에 주입
    cls_p3_injected, _ = inject_single_cell(results[1], p3_cls_values_list, pos_p3)
    dep_p3_injected, _ = inject_single_cell(results[2], [p3_dep_value], pos_p3)

    cls_p4_injected, _ = inject_single_cell(results[4], p4_cls_values_list, pos_p4)
    dep_p4_injected, _ = inject_single_cell(results[5], [p4_dep_value], pos_p4)

    cls_p5_injected, _ = inject_single_cell(results[7], p5_cls_values_list, pos_p5)
    dep_p5_injected, _ = inject_single_cell(results[8], [p5_dep_value], pos_p5)

    print(f"🧩 Injected custom values at positions -> P3:{pos_p3}, P4:{pos_p4}, P5:{pos_p5}")

    # 새로운 outputs 생성 (reg/cls/dep 모두 단일 셀만 변경된 버전)
    arbitrary_outputs = [
        reg_p3_injected, cls_p3_injected, dep_p3_injected,
        reg_p4_injected, cls_p4_injected, dep_p4_injected,
        reg_p5_injected, cls_p5_injected, dep_p5_injected,
    ]
    
    # 커스텀 후처리 - 임의 (16 bins)
    print("📊 Processing arbitrary reg results with custom postprocess...")
    arbitrary_detections = custom_postprocess(arbitrary_outputs, wide_original_size, IMAGE_SIZE, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, REVERSE_LETTERBOX)
    
    print(f"🎯 Found {len(arbitrary_detections)} arbitrary detections")
    for i, detection in enumerate(arbitrary_detections):
        x1, y1, x2, y2, conf, cls, depth_normalized = detection
        depth_original = depth_denormalizer.denormalize_depth(depth_normalized)
        print(f" Arbitrary Detection {i+1}: class={int(cls)}, conf={conf:.3f}")
        print(f" Depth: {depth_normalized:.6f} (normalized) -> {depth_original:.2f}m (original)")
        print(f" BBox: ({x1},{y1},{x2},{y2})")
    
    # 결과 시각화 및 저장 (arbitrary)
    print("🎨 Drawing arbitrary detections with depth...")
    arbitrary_result_image = draw_detections_with_depth(wide_image, arbitrary_detections, gt_labels, depth_denormalizer, prefix="ARBITRARY")
    
    # 임의 결과 이미지 저장
    arbitrary_output_path = output_dir / f"{ARBITRARY_OUTPUT_IMAGE_NAME}.jpg"
    cv2.imwrite(str(arbitrary_output_path), arbitrary_result_image)
    
    # 요약 정보도 함께 저장 (original + arbitrary)
    summary_filename = f"inference_depth_summary_onnx_extracted_{filename.replace('.jpg', '.txt')}"
    summary_path = output_dir / summary_filename
    with open(summary_path, 'w') as f:
        f.write(f"Dual Stream YOLO ONNX Inference Results with Depth (Extracted Model)\n")
        f.write(f"===============================================\n")
        f.write(f"ONNX Model: {ONNX_PATH}\n")
        f.write(f"Wide Image: {WIDE_IMAGE_PATH}\n")
        f.write(f"Narrow Image: {NARROW_IMAGE_PATH}\n")
        f.write(f"Label File: {LABEL_PATH}\n")
        f.write(f"Inference Time: {inference_time:.3f}s\n")
        f.write(f"Original Detections: {len(original_detections)}\n")
        f.write(f"Arbitrary Detections: {len(arbitrary_detections)}\n")
        f.write(f"GT Labels: {len(gt_labels)}\n")
        f.write(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}\n")
        f.write(f"IoU Threshold: {IOU_THRESHOLD}\n\n")
        
        f.write("Original Prediction Details:\n")
        for i, detection in enumerate(original_detections):
            x1, y1, x2, y2, conf, cls, depth = detection
            depth_original = depth_denormalizer.denormalize_depth(depth)
            f.write(f" PRED {i+1}. Class: {int(cls)}, Confidence: {conf:.3f}, Depth: {depth_original:.2f}m, BBox: ({x1},{y1},{x2},{y2})\n")
        
        f.write("\nArbitrary Prediction Details:\n")
        for i, detection in enumerate(arbitrary_detections):
            x1, y1, x2, y2, conf, cls, depth = detection
            depth_original = depth_denormalizer.denormalize_depth(depth)
            f.write(f" ARBITRARY {i+1}. Class: {int(cls)}, Confidence: {conf:.3f}, Depth: {depth_original:.2f}m, BBox: ({x1},{y1},{x2},{y2})\n")
        
        f.write(f"\nGround Truth Details:\n")
        for i, gt_label in enumerate(gt_labels):
            class_id, x_center, y_center, width, height, depth = gt_label
            depth_original = depth_denormalizer.denormalize_depth(depth)
            f.write(f" GT {i+1}. Class: {int(class_id)}, Depth: {depth_original:.2f}m, BBox: ({x_center:.3f},{y_center:.3f},{width:.3f},{height:.3f})\n")
    
    print(f"✅ Results saved to:")
    print(f" 📸 Original Image: {output_path}")
    print(f" 📸 Arbitrary Image: {arbitrary_output_path}")
    print(f" 📄 Summary: {summary_path}")
    print("🎉 Depth-enhanced ONNX inference (extracted model) with arbitrary reg completed successfully!")

if __name__ == "__main__":
    main()