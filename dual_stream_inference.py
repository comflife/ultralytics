#!/usr/bin/env python3
"""
Dual Stream YOLO Inference Script
Performs inference with dual-stream YOLO model and saves detection results
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

# 🔧 로컬 ultralytics 모듈을 우선적으로 사용하도록 설정
SCRIPT_DIR = Path(__file__).parent.absolute()
ULTRALYTICS_ROOT = SCRIPT_DIR
sys.path.insert(0, str(ULTRALYTICS_ROOT))

print(f"🔧 Using local ultralytics from: {ULTRALYTICS_ROOT}")

# ============================================================
# 🛠️ 설정 부분 - 여기를 수정하세요!
# ============================================================

# 모델 파일 경로
MODEL_PATH = "runs/train/exp225/weights/best.pt"

# 입력 이미지 경로
WIDE_IMAGE_PATH = "swm/images/scene0002_20250422_07540461.jpg"    # Wide stream 이미지
NARROW_IMAGE_PATH = "swm/images/scene0002_20250422_07540461.jpg"  # Narrow stream 이미지

# 출력 설정
OUTPUT_DIR = "inference_results"
OUTPUT_IMAGE_NAME = "dual_stream_result.jpg"

# 추론 설정
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
IMAGE_SIZE = 640

# ============================================================

try:
    from ultralytics import YOLO
    import ultralytics
    print(f"✅ Using ultralytics from: {ultralytics.__file__}")
except ImportError as e:
    print(f"❌ Failed to import ultralytics: {e}")
    print("Please ensure you're in the ultralytics directory with custom modules")
    sys.exit(1)

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
    두 이미지를 듀얼 스트림 형태로 결합
    
    Args:
        wide_tensor (torch.Tensor): Wide stream 이미지 [3, H, W]
        narrow_tensor (torch.Tensor): Narrow stream 이미지 [3, H, W]
    
    Returns:
        torch.Tensor: 듀얼 스트림 입력 [1, 2, 3, H, W]
    """
    # 두 이미지를 스택하여 듀얼 스트림 생성
    dual_stream = torch.stack([wide_tensor, narrow_tensor], dim=0)  # [2, 3, H, W]
    dual_stream = dual_stream.unsqueeze(0)  # [1, 2, 3, H, W]
    
    print(f"🔗 Created dual stream input: {dual_stream.shape}")
    return dual_stream

def postprocess_results(results, original_size, target_size=640):
    """
    모델 결과를 후처리하여 원본 이미지 좌표로 변환
    
    Args:
        results: YOLO 모델 출력
        original_size (tuple): 원본 이미지 크기 (height, width)
        target_size (int): 모델 입력 크기
    
    Returns:
        list: 검출된 객체 리스트 [x1, y1, x2, y2, confidence, class_id]
    """
    if not results or len(results) == 0:
        return []
    
    result = results[0]  # 첫 번째 결과
    
    if not hasattr(result, 'boxes') or result.boxes is None:
        return []
    
    boxes = result.boxes
    
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
    
    for box in boxes:
        # 좌표 추출
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        confidence = box.conf[0].cpu().numpy()
        class_id = int(box.cls[0].cpu().numpy())
        
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
        
        detections.append([x1, y1, x2, y2, confidence, class_id])
    
    return detections

def draw_detections(image, detections, class_names=None):
    """
    이미지에 검출 결과를 그리기
    
    Args:
        image (np.ndarray): 원본 이미지 (BGR)
        detections (list): 검출 결과
        class_names (list): 클래스 이름 리스트
    
    Returns:
        np.ndarray: 검출 결과가 그려진 이미지
    """
    if class_names is None:
        class_names = [f"class_{i}" for i in range(80)]  # COCO 클래스 수
    
    result_image = image.copy()
    
    # 색상 팔레트
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (128, 0, 128), (255, 165, 0), (0, 128, 128), (128, 128, 0)
    ]
    
    for i, detection in enumerate(detections):
        x1, y1, x2, y2, confidence, class_id = detection
        
        # 박스 그리기
        color = colors[int(class_id) % len(colors)]
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
        
        # 라벨 텍스트
        class_name = class_names[int(class_id)] if int(class_id) < len(class_names) else f"class_{int(class_id)}"
        label = f"{class_name}: {confidence:.2f}"
        
        # 텍스트 배경
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(result_image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
        
        # 텍스트 그리기
        cv2.putText(result_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return result_image

def main():
    """메인 inference 함수"""
    
    print("🚀 Starting Dual Stream YOLO Inference...")
    
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
        # 직접 model.model로 접근하여 듀얼 스트림 입력 사용
        with torch.no_grad():
            results = model.predict(
                source=dual_input,
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                verbose=False,
                save=False
            )
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return
    
    inference_time = time.time() - start_time
    print(f"⚡ Inference completed in {inference_time:.3f}s")
    
    # 5. 결과 후처리 (wide 이미지 기준)
    print("📊 Processing results...")
    detections = postprocess_results(results, wide_original_size, IMAGE_SIZE)
    
    print(f"🎯 Found {len(detections)} detections")
    for i, detection in enumerate(detections):
        x1, y1, x2, y2, conf, cls = detection
        print(f"  Detection {i+1}: class={int(cls)}, conf={conf:.3f}, bbox=({x1},{y1},{x2},{y2})")
    
    # 6. 결과 시각화 및 저장
    print("🎨 Drawing detections...")
    
    # Wide 이미지에 검출 결과 그리기
    result_image = draw_detections(wide_image, detections)
    
    # 결과 이미지 저장
    output_path = output_dir / OUTPUT_IMAGE_NAME
    cv2.imwrite(str(output_path), result_image)
    
    # 요약 정보도 함께 저장
    summary_path = output_dir / "inference_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Dual Stream YOLO Inference Results\n")
        f.write(f"=====================================\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Wide Image: {WIDE_IMAGE_PATH}\n")
        f.write(f"Narrow Image: {NARROW_IMAGE_PATH}\n")
        f.write(f"Inference Time: {inference_time:.3f}s\n")
        f.write(f"Detections: {len(detections)}\n")
        f.write(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}\n")
        f.write(f"IoU Threshold: {IOU_THRESHOLD}\n\n")
        
        f.write("Detection Details:\n")
        for i, detection in enumerate(detections):
            x1, y1, x2, y2, conf, cls = detection
            f.write(f"  {i+1}. Class: {int(cls)}, Confidence: {conf:.3f}, BBox: ({x1},{y1},{x2},{y2})\n")
    
    print(f"✅ Results saved to:")
    print(f"  📸 Image: {output_path}")
    print(f"  📄 Summary: {summary_path}")
    print("🎉 Inference completed successfully!")

if __name__ == "__main__":
    main() 