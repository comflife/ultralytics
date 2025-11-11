#!/usr/bin/env python3
"""
Dual Stream YOLO Inference Script with Depth Visualization
Performs inference with dual-stream YOLO model and shows depth prediction + ground truth
"""

import sys
import os
from pathlib import Path
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import time
import random
import json

# 🔧 로컬 ultralytics 모듈을 우선적으로 사용하도록 설정
SCRIPT_DIR = Path(__file__).parent.absolute()
ULTRALYTICS_ROOT = SCRIPT_DIR
sys.path.insert(0, str(ULTRALYTICS_ROOT))

print(f"🔧 Using local ultralytics from: {ULTRALYTICS_ROOT}")

# ============================================================
# 🛠️ 설정 부분 - 여기를 수정하세요!
# ============================================================

# 모델 파일 경로
# MODEL_PATH = "/home/byounggun/ultralytics/runs/train/exp361/weights/best.pt"
MODEL_PATH = "/home/byounggun/ultralytics/runs/finetune/katri_overfit2/weights/epoch30.pt"

# 입력 이미지 디렉토리
# WIDE_DIR = "/home/byounggun/ultralytics/swm_dual_split/val/images/"
WIDE_DIR = "/home/byounggun/ultralytics/finetune_katri_name/images"
# NARROW_DIR = "/home/byounggun/ultralytics/swm_dual_split/val/val_narrow_images/"
NARROW_DIR = "/home/byounggun/ultralytics/finetune_katri_name/narrow_images"

# 라벨 디렉토리 (GT depth 정보)
# LABEL_DIR = "/home/byounggun/ultralytics/swm_dual_split/val/labels/"
LABEL_DIR = "/home/byounggun/ultralytics/finetune_katri_name/labels"

# 출력 설정
OUTPUT_DIR = "katri_inference_results_depth"

# 추론 설정
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.55
IMAGE_SIZE = 640

# Depth 정규화 정보 파일
DEPTH_NORM_INFO_PATH = "/home/byounggun/ultralytics/depth_normalization_info.json"

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
    
    def __init__(self, norm_info_path=DEPTH_NORM_INFO_PATH):
        """
        Args:
            norm_info_path: 정규화 정보가 저장된 JSON 파일 경로
        """
        self.norm_info_path = norm_info_path
        self.norm_info = None
        self.load_normalization_info()
    
    def load_normalization_info(self):
        """정규화 정보 로드"""
        try:
            with open(self.norm_info_path, 'r') as f:
                self.norm_info = json.load(f)
            print(f"✅ 정규화 정보 로드 완료: {self.norm_info_path}")
            print(f"   원본 범위: [{self.norm_info['min_depth']:.6f}, {self.norm_info['max_depth']:.6f}]")
        except FileNotFoundError:
            print(f"❌ 정규화 정보 파일을 찾을 수 없습니다: {self.norm_info_path}")
            self.norm_info = None
        except Exception as e:
            print(f"❌ 정규화 정보 로드 오류: {e}")
            self.norm_info = None
    
    def denormalize_depth(self, normalized_depth):
        """
        정규화된 depth 값을 원본 스케일로 복구
        
        Args:
            normalized_depth: 정규화된 depth 값 (0~1 범위)
            
        Returns:
            원본 스케일의 depth 값
        """
        if self.norm_info is None:
            return normalized_depth
        
        min_depth = self.norm_info['min_depth']
        max_depth = self.norm_info['max_depth']
        
        # Min-Max 역정규화
        original_depth = normalized_depth * (max_depth - min_depth) + min_depth
        return original_depth

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
        torch.Tensor: 전처리된 이미지 [3, H, W]
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
    
    # 정규화 및 텐서 변환
    normalized = padded.astype(np.float32) / 255.0
    tensor = torch.from_numpy(normalized).permute(2, 0, 1)  # HWC -> CHW
    
    return tensor, (original_height, original_width), image

def create_dual_stream_input(wide_tensor, narrow_tensor):
    """
    두 이미지를 듀얼 스트림 형태로 결합 (NPU 호환 4차원)
    
    Args:
        wide_tensor (torch.Tensor): Wide stream 이미지 [3, H, W]
        narrow_tensor (torch.Tensor): Narrow stream 이미지 [3, H, W]
    
    Returns:
        torch.Tensor: 듀얼 스트림 입력 [1, 6, H, W] - NPU 호환 4차원
    """
    # 🔧 NPU 호환: 4차원 입력으로 변경 [1, 6, H, W]
    # 두 이미지를 채널 차원에서 concat
    dual_stream = torch.cat([wide_tensor, narrow_tensor], dim=0)  # [6, H, W]
    dual_stream = dual_stream.unsqueeze(0)  # [1, 6, H, W]
    
    print(f"🔗 Created dual stream input: {dual_stream.shape}")
    return dual_stream

def postprocess_results_with_depth(predictions, original_size, target_size=640):
    """
    Depth를 포함한 결과 후처리
    
    Args:
        predictions: NMS 후처리된 결과 (리스트)
        original_size (tuple): 원본 이미지 크기 (height, width)
        target_size (int): 모델 입력 크기
    
    Returns:
        list: 검출된 객체 리스트 [x1, y1, x2, y2, confidence, class_id, depth]
    """
    if not predictions or len(predictions) == 0:
        return []
    
    # 첫 번째 배치 결과 사용
    pred = predictions[0]
    
    if pred is None or len(pred) == 0:
        return []
    
    # 원본 이미지 크기로 스케일링
    original_height, original_width = original_size
    scale = min(target_size / original_width, target_size / original_height)
    
    # 패딩 계산
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    delta_w = target_size - new_width
    delta_h = target_size - new_height
    left = delta_w // 2
    top = delta_h // 2
    
    detections = []
    
    for detection in pred:
        # detection shape 확인
        if len(detection) >= 7:  # x1, y1, x2, y2, confidence, class_id, depth
            x1, y1, x2, y2, confidence, class_id, depth = detection.cpu().numpy()[:7]
        elif len(detection) == 6:  # depth 정보가 없는 경우
            x1, y1, x2, y2, confidence, class_id = detection.cpu().numpy()
            depth = 0.0  # 기본값
        else:
            continue
        
        # 패딩 제거
        x1 = max(0, x1 - left)
        y1 = max(0, y1 - top)
        x2 = max(0, x2 - left)
        y2 = max(0, y2 - top)
        
        # 원본 크기로 스케일링
        x1 = int(x1 / scale)
        y1 = int(y1 / scale)
        x2 = int(x2 / scale)
        y2 = int(y2 / scale)
        
        # 원본 이미지 범위 내로 클리핑
        x1 = max(0, min(x1, original_width))
        y1 = max(0, min(y1, original_height))
        x2 = max(0, min(x2, original_width))
        y2 = max(0, min(y2, original_height))
        
        # depth 값 처리 (원래 모델 출력을 그대로 사용)
        if len(detection) >= 7:
            # 모델 출력이 이미 정규화되어 있다고 가정하고 직접 사용
            # sigmoid 제거 - 모델 출력을 그대로 사용
            depth = float(depth)
            
            # depth 값이 음수이거나 1을 초과하는 경우 클리핑
            depth = max(0.0, min(1.0, depth))
        
        detections.append([x1, y1, x2, y2, float(confidence), int(class_id), float(depth)])
    
    return detections

def draw_detections_with_depth(image, detections, gt_labels, depth_denormalizer, class_names=None):
    """
    이미지에 검출 결과와 depth 정보를 그리기
    
    Args:
        image (np.ndarray): 원본 이미지 (BGR)
        detections (list): 검출 결과
        gt_labels (list): Ground Truth 라벨
        depth_denormalizer: Depth 역정규화 객체
        class_names (list): 클래스 이름 리스트
    
    Returns:
        np.ndarray: 검출 결과가 그려진 이미지
    """
    if class_names is None:
        class_names = [f"class_{i}" for i in range(80)]  # COCO 클래스 수
    
    result_image = image.copy()
    img_height, img_width = image.shape[:2]
    
    # 색상 팔레트
    pred_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]  # 예측용
    gt_color = (128, 128, 128)  # GT용 회색
    
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
    print(f"🎯 Drawing {len(detections)} predictions")
    for i, detection in enumerate(detections):
        x1, y1, x2, y2, confidence, class_id, pred_depth_normalized = detection
        
        # 예측 depth 역정규화
        pred_depth_original = depth_denormalizer.denormalize_depth(pred_depth_normalized)
        
        # 박스 그리기
        color = pred_colors[int(class_id) % len(pred_colors)]
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 3)  # 두꺼운 선
        
        # 라벨 텍스트
        class_name = class_names[int(class_id)] if int(class_id) < len(class_names) else f"class_{int(class_id)}"
        pred_label_text = f"PRED-{class_name}: {confidence:.2f}, D:{pred_depth_original:.2f}m"
        
        # 텍스트 배경 위치 조정 (GT와 겹치지 않도록)
        text_y = y2 + 25
        if text_y > img_height - 30:
            text_y = y1 - 10
        
        # 텍스트 배경
        (text_width, text_height), _ = cv2.getTextSize(pred_label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(result_image, (x1, text_y - text_height), (x1 + text_width, text_y + 5), color, -1)
        
        # 텍스트 그리기
        cv2.putText(result_image, pred_label_text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return result_image

def main():
    """메인 inference 함수"""
    
    print("🚀 Starting Dual Stream YOLO Inference with Depth...")
    
    # Depth 역정규화 객체 생성
    depth_denormalizer = DepthDenormalizer()
    if depth_denormalizer.norm_info is None:
        print("⚠️ Depth 정규화 정보가 없습니다. 정규화된 값을 그대로 표시합니다.")
    
    # 출력 디렉토리 생성
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    
    # 1. 모델 로드
    print(f"📦 Loading model from: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # 랜덤 이미지 세트 선택
    wide_files = [f for f in os.listdir(WIDE_DIR) if f.lower().endswith(('.jpg', '.jpeg'))]
    if not wide_files:
        print(f"❌ No JPG/JPEG files found in {WIDE_DIR}")
        return
    
    filename = random.choice(wide_files)
    WIDE_IMAGE_PATH = os.path.join(WIDE_DIR, filename)
    NARROW_IMAGE_PATH = os.path.join(NARROW_DIR, filename)
    
    # 라벨 파일 경로
    label_filename = filename.replace('.jpg', '.txt').replace('.jpeg', '.txt')
    LABEL_PATH = os.path.join(LABEL_DIR, label_filename)
    
    if not os.path.exists(NARROW_IMAGE_PATH):
        print(f"❌ Matching narrow image not found: {NARROW_IMAGE_PATH}")
        return
    
    print(f"🖼️ Selected random image set: {filename}")
    print(f"📄 Label file: {LABEL_PATH}")
    
    # Ground Truth 라벨 로드
    gt_labels = load_ground_truth_labels(LABEL_PATH)
    print(f"🎯 Loaded {len(gt_labels)} GT labels")
    
    # 출력 파일 이름 동적 설정
    OUTPUT_IMAGE_NAME = f"dual_stream_depth_result_{filename}"
    
    # 2. 이미지 전처리
    print("🖼️ Preprocessing images...")
    try:
        wide_tensor, wide_original_size, wide_image = preprocess_image(WIDE_IMAGE_PATH, IMAGE_SIZE)
        narrow_tensor, narrow_original_size, narrow_image = preprocess_image(NARROW_IMAGE_PATH, IMAGE_SIZE)
    except Exception as e:
        print(f"❌ Failed to preprocess images: {e}")
        return
    
    # 3. 듀얼 스트림 입력 생성
    dual_input = create_dual_stream_input(wide_tensor, narrow_tensor)
    
    # 4. 추론 실행
    print("🔮 Running inference...")
    start_time = time.time()
    
    try:
        # 듀얼 스트림 모델의 경우 직접 forward pass 사용
        pytorch_model = model.model if hasattr(model, 'model') else model.predictor.model
        pytorch_model.eval()
        
        with torch.no_grad():
            # 직접 모델 forward pass 수행
            predictions = pytorch_model(dual_input)
            
            # NMS 후처리 적용
            from ultralytics.utils.ops import non_max_suppression
            
            # 모델의 클래스 수 가져오기
            nc = getattr(pytorch_model, 'nc', 80)
            
            # NMS 적용 (depth 정보도 포함)
            predictions = non_max_suppression(
                predictions,
                conf_thres=CONFIDENCE_THRESHOLD,
                iou_thres=IOU_THRESHOLD,
                classes=None,
                agnostic=False,
                max_det=300,
                nc=nc
            )
            
            # 결과를 리스트로 저장
            results = predictions
            
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    inference_time = time.time() - start_time
    print(f"⚡ Inference completed in {inference_time:.3f}s")
    
    # 5. 결과 후처리 (depth 포함)
    print("📊 Processing results with depth...")
    detections = postprocess_results_with_depth(results, wide_original_size, IMAGE_SIZE)
    
    print(f"🎯 Found {len(detections)} detections")
    for i, detection in enumerate(detections):
        x1, y1, x2, y2, conf, cls = detection[:6]
        depth_normalized = detection[6] if len(detection) > 6 else 0.0
        depth_original = depth_denormalizer.denormalize_depth(depth_normalized)
        print(f"  Detection {i+1}: class={int(cls)}, conf={conf:.3f}")
        print(f"    Depth: {depth_normalized:.6f} (normalized) -> {depth_original:.2f}m (original)")
        print(f"    BBox: ({x1},{y1},{x2},{y2})")
    
    # GT 정보도 출력
    print(f"🎯 Ground Truth labels:")
    for i, gt_label in enumerate(gt_labels):
        class_id, x_center, y_center, width, height, depth_normalized = gt_label
        depth_original = depth_denormalizer.denormalize_depth(depth_normalized)
        print(f"  GT {i+1}: class={int(class_id)}")
        print(f"    Depth: {depth_normalized:.6f} (normalized) -> {depth_original:.2f}m (original)")
        print(f"    BBox: center=({x_center:.3f},{y_center:.3f}), size=({width:.3f},{height:.3f})")
    
    # 6. 결과 시각화 및 저장 (depth 포함)
    print("🎨 Drawing detections with depth...")
    
    # Wide 이미지에 검출 결과와 GT 그리기
    result_image = draw_detections_with_depth(wide_image, detections, gt_labels, depth_denormalizer)
    
    # 결과 이미지 저장
    output_path = output_dir / OUTPUT_IMAGE_NAME
    cv2.imwrite(str(output_path), result_image)
    
    # 요약 정보도 함께 저장
    summary_filename = f"inference_depth_summary_{filename.replace('.jpg', '.txt').replace('.jpeg', '.txt')}"
    summary_path = output_dir / summary_filename
    with open(summary_path, 'w') as f:
        f.write(f"Dual Stream YOLO Inference Results with Depth\n")
        f.write(f"===============================================\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Wide Image: {WIDE_IMAGE_PATH}\n")
        f.write(f"Narrow Image: {NARROW_IMAGE_PATH}\n")
        f.write(f"Label File: {LABEL_PATH}\n")
        f.write(f"Inference Time: {inference_time:.3f}s\n")
        f.write(f"Detections: {len(detections)}\n")
        f.write(f"GT Labels: {len(gt_labels)}\n")
        f.write(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}\n")
        f.write(f"IoU Threshold: {IOU_THRESHOLD}\n\n")
        
        f.write("Prediction Details:\n")
        for i, detection in enumerate(detections):
            x1, y1, x2, y2, conf, cls = detection[:6]
            depth = detection[6] if len(detection) > 6 else 0.0
            depth_original = depth_denormalizer.denormalize_depth(depth)
            f.write(f"  PRED {i+1}. Class: {int(cls)}, Confidence: {conf:.3f}, Depth: {depth_original:.2f}m, BBox: ({x1},{y1},{x2},{y2})\n")
        
        f.write(f"\nGround Truth Details:\n")
        for i, gt_label in enumerate(gt_labels):
            class_id, x_center, y_center, width, height, depth = gt_label
            depth_original = depth_denormalizer.denormalize_depth(depth)
            f.write(f"  GT {i+1}. Class: {int(class_id)}, Depth: {depth_original:.2f}m, BBox: ({x_center:.3f},{y_center:.3f},{width:.3f},{height:.3f})\n")
    
    print(f"✅ Results saved to:")
    print(f"  📸 Image: {output_path}")
    print(f"  📄 Summary: {summary_path}")
    print("🎉 Depth-enhanced inference completed successfully!")

if __name__ == "__main__":
    main()
