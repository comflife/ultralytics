#!/usr/bin/env python3
"""
Katri Images Auto-Labeling Script
Performs auto-labeling on Katri images using trained dual-stream YOLO model
and organizes them into the swm_dual_split/train structure
"""

import sys
import os
from pathlib import Path
import numpy as np
import cv2
import torch
import shutil
from tqdm import tqdm
import json
import re

# 🔧 로컬 ultralytics 모듈을 우선적으로 사용하도록 설정
SCRIPT_DIR = Path(__file__).parent.absolute()
ULTRALYTICS_ROOT = SCRIPT_DIR
sys.path.insert(0, str(ULTRALYTICS_ROOT))

print(f"🔧 Using local ultralytics from: {ULTRALYTICS_ROOT}")

# ============================================================
# 🛠️ 설정 부분
# ============================================================

# 모델 파일 경로
MODEL_PATH = "/home/byounggun/ultralytics/runs/train/exp385/weights/best.pt"

# Katri 이미지 루트 디렉토리
KATRI_ROOT = "/home/byounggun/ultralytics/katri_images"

# 출력 디렉토리 (finetune_katri 폴더에 구성)
OUTPUT_ROOT = "/home/byounggun/ultralytics/finetune_katri"
OUTPUT_WIDE_DIR = os.path.join(OUTPUT_ROOT, "images")
OUTPUT_NARROW_DIR = os.path.join(OUTPUT_ROOT, "narrow_images")
OUTPUT_LABEL_DIR = os.path.join(OUTPUT_ROOT, "labels")

# 추론 설정
CONFIDENCE_THRESHOLD = 0.3  # confidence threshold
IOU_THRESHOLD = 0.45  # NMS IOU threshold (낮을수록 중복 제거 강화)
IMAGE_SIZE = 640
MAX_DETECTIONS = 300  # 최대 검출 수

# Depth 정규화 정보 파일
DEPTH_NORM_INFO_PATH = "/home/byounggun/ultralytics/depth_normalization_info.json"

# Depth 기본값 (라벨링에서는 depth를 0으로 설정하거나 평균값 사용)
DEFAULT_DEPTH = 0.5  # 정규화된 기본 depth 값

# ============================================================

try:
    from ultralytics import YOLO
    import ultralytics
    print(f"✅ Using ultralytics from: {ultralytics.__file__}")
except ImportError as e:
    print(f"❌ Failed to import ultralytics: {e}")
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
    """
    # 이미지 로드
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    original_height, original_width = image.shape[:2]
    
    # RGB로 변환
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 리사이즈 (letterbox 적용)
    scale = min(target_size / original_width, target_size / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # 리사이즈
    resized = cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # 패딩 추가
    delta_w = target_size - new_width
    delta_h = target_size - new_height
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, 
                               cv2.BORDER_CONSTANT, value=[114, 114, 114])
    
    # 정규화 및 텐서 변환
    normalized = padded.astype(np.float32) / 255.0
    tensor = torch.from_numpy(normalized).permute(2, 0, 1)  # HWC -> CHW
    
    return tensor, (original_height, original_width)


def create_dual_stream_input(wide_tensor, narrow_tensor):
    """
    두 이미지를 듀얼 스트림 형태로 결합
    
    Args:
        wide_tensor (torch.Tensor): Wide stream 이미지 [3, H, W]
        narrow_tensor (torch.Tensor): Narrow stream 이미지 [3, H, W]
    
    Returns:
        torch.Tensor: 듀얼 스트림 입력 [1, 6, H, W]
    """
    dual_stream = torch.cat([wide_tensor, narrow_tensor], dim=0)  # [6, H, W]
    dual_stream = dual_stream.unsqueeze(0)  # [1, 6, H, W]
    return dual_stream


def postprocess_results_to_yolo_format(predictions, original_size, target_size=640):
    """
    모델 출력을 YOLO 라벨 형식으로 변환
    
    Args:
        predictions: NMS 후처리된 결과
        original_size (tuple): 원본 이미지 크기 (height, width)
        target_size (int): 모델 입력 크기
    
    Returns:
        list: YOLO 형식 라벨 [class_id, x_center, y_center, width, height, depth]
    """
    if not predictions or len(predictions) == 0:
        return []
    
    pred = predictions[0]
    
    if pred is None or len(pred) == 0:
        return []
    
    original_height, original_width = original_size
    
    # 스케일과 패딩 계산 (letterbox 적용 시)
    scale = min(target_size / original_width, target_size / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    delta_w = target_size - new_width
    delta_h = target_size - new_height
    left = delta_w // 2
    top = delta_h // 2
    
    labels = []
    
    for detection in pred:
        # detection: [x1, y1, x2, y2, confidence, class_id, depth]
        if len(detection) >= 6:
            x1, y1, x2, y2, confidence, class_id = detection.cpu().numpy()[:6]
            depth = detection[6].cpu().numpy() if len(detection) > 6 else DEFAULT_DEPTH
        else:
            continue
        
        # 패딩 제거 (640x640 -> letterbox 크기)
        x1 = max(0, x1 - left)
        y1 = max(0, y1 - top)
        x2 = max(0, x2 - left)
        y2 = max(0, y2 - top)
        
        # 원본 크기로 스케일 역변환
        x1 = x1 / scale
        y1 = y1 / scale
        x2 = x2 / scale
        y2 = y2 / scale
        
        # 원본 이미지 범위 내로 클리핑
        x1 = max(0, min(x1, original_width))
        y1 = max(0, min(y1, original_height))
        x2 = max(0, min(x2, original_width))
        y2 = max(0, min(y2, original_height))
        
        # 픽셀 좌표를 정규화된 YOLO 형식으로 변환
        x_center = (x1 + x2) / 2 / original_width
        y_center = (y1 + y2) / 2 / original_height
        width = (x2 - x1) / original_width
        height = (y2 - y1) / original_height
        
        # 값 범위 제한 [0, 1]
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        width = max(0.0, min(1.0, width))
        height = max(0.0, min(1.0, height))
        depth = max(0.0, min(1.0, float(depth)))
        
        labels.append([int(class_id), x_center, y_center, width, height, depth])
    
    return labels


def save_yolo_labels(labels, output_path):
    """
    YOLO 형식으로 라벨 저장
    
    Args:
        labels (list): 라벨 리스트
        output_path (str): 출력 파일 경로
    """
    with open(output_path, 'w') as f:
        for label in labels:
            class_id, x_center, y_center, width, height, depth = label
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {depth:.6f}\n")


def extract_timestamp_from_filename(filename):
    """
    파일명에서 타임스탬프 추출
    예: projection_144149_268938.jpeg -> (144149, 268938)
    
    Args:
        filename (str): 파일명
    
    Returns:
        tuple: (시분초, 마이크로초) 또는 None
    """
    # projection_HHMMSS_MICROSEC.jpeg 패턴
    match = re.match(r'projection_(\d+)_(\d+)\.jpe?g', filename, re.IGNORECASE)
    if match:
        time_part = int(match.group(1))  # HHMMSS
        microsec = int(match.group(2))   # microseconds
        return (time_part, microsec)
    return None


def find_closest_match(target_timestamp, candidate_timestamps, threshold_ms=150):
    """
    가장 가까운 타임스탬프 매칭 찾기
    
    Args:
        target_timestamp (tuple): (시분초, 마이크로초)
        candidate_timestamps (dict): {filename: (시분초, 마이크로초)}
        threshold_ms (int): 허용 가능한 최대 시간차 (밀리초)
    
    Returns:
        tuple: (매칭된 파일명, 시간차) 또는 (None, None)
    """
    target_time, target_micro = target_timestamp
    
    min_diff = float('inf')
    best_match = None
    
    for filename, (cand_time, cand_micro) in candidate_timestamps.items():
        # 시간 차이 계산 (마이크로초 단위)
        time_diff_sec = abs(target_time - cand_time)
        
        # 시분초가 다르면 건너뛰기 (같은 초 내에서만 매칭)
        if time_diff_sec > 1:
            continue
        
        # 마이크로초 차이 계산
        total_diff_micro = abs(target_micro - cand_micro)
        
        if total_diff_micro < min_diff:
            min_diff = total_diff_micro
            best_match = filename
    
    # threshold 체크 (마이크로초를 밀리초로 변환)
    threshold_micro = threshold_ms * 1000
    if min_diff <= threshold_micro:
        return best_match, min_diff
    
    return None, None


def process_katri_dataset(model):
    """
    Katri 데이터셋 전체를 처리하여 자동 라벨링 및 복사
    
    Args:
        model: 로드된 YOLO 모델
    """
    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_WIDE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_NARROW_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)
    
    print(f"📁 Output directories:")
    print(f"   Wide images: {OUTPUT_WIDE_DIR}")
    print(f"   Narrow images: {OUTPUT_NARROW_DIR}")
    print(f"   Labels: {OUTPUT_LABEL_DIR}")
    
    # Katri 폴더 목록 가져오기 (cam0, cam1 쌍)
    katri_folders = sorted([f for f in os.listdir(KATRI_ROOT) if os.path.isdir(os.path.join(KATRI_ROOT, f))])
    
    # cam0와 cam1 쌍으로 그룹화
    folder_pairs = {}
    for folder in katri_folders:
        # 예: 20250930_144149_cam0 -> 20250930_144149
        base_name = folder.rsplit('_', 1)[0]
        cam_type = folder.rsplit('_', 1)[1]
        
        if base_name not in folder_pairs:
            folder_pairs[base_name] = {}
        folder_pairs[base_name][cam_type] = folder
    
    print(f"🎯 Found {len(folder_pairs)} folder pairs (cam0 + cam1)")
    
    total_images = 0
    total_labels = 0
    skipped_images = 0
    
    # GPU 사용 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # 모델을 평가 모드로 설정 및 GPU로 이동
    pytorch_model = model.model if hasattr(model, 'model') else model.predictor.model
    pytorch_model.to(device)
    pytorch_model.eval()
    
    from ultralytics.utils.ops import non_max_suppression
    nc = getattr(pytorch_model, 'nc', 80)
    
    # 각 폴더 쌍 처리
    for base_name, cams in tqdm(folder_pairs.items(), desc="Processing folders"):
        if 'cam0' not in cams or 'cam1' not in cams:
            print(f"⚠️ Skipping {base_name}: missing cam0 or cam1")
            continue
        
        cam0_folder = os.path.join(KATRI_ROOT, cams['cam0'])
        cam1_folder = os.path.join(KATRI_ROOT, cams['cam1'])
        
        # cam0 이미지 목록 (wide로 사용) - 정렬
        cam0_images = sorted([f for f in os.listdir(cam0_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        # cam1 이미지 목록 (narrow로 사용) - 정렬
        cam1_images = sorted([f for f in os.listdir(cam1_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        # 매칭 가능한 최소 개수 사용
        min_count = min(len(cam0_images), len(cam1_images))
        
        print(f"\n📊 {base_name}: cam0={len(cam0_images)}, cam1={len(cam1_images)}, pairs={min_count}")
        
        # 인덱스 기반 매칭
        for i in tqdm(range(min_count), desc=f"  {base_name}", leave=False):
            cam0_img_name = cam0_images[i]
            cam1_img_name = cam1_images[i]
            
            cam0_img_path = os.path.join(cam0_folder, cam0_img_name)
            cam1_img_path = os.path.join(cam1_folder, cam1_img_name)
            
            try:
                # 이미지 전처리
                wide_tensor, wide_original_size = preprocess_image(cam0_img_path, IMAGE_SIZE)
                narrow_tensor, _ = preprocess_image(cam1_img_path, IMAGE_SIZE)
                
                # 듀얼 스트림 입력 생성 및 GPU로 이동
                dual_input = create_dual_stream_input(wide_tensor, narrow_tensor)
                dual_input = dual_input.to(device)
                
                # 추론
                with torch.no_grad():
                    predictions = pytorch_model(dual_input)
                    
                    # NMS 적용 (agnostic=True로 변경: 클래스 무관하게 중복 제거)
                    predictions = non_max_suppression(
                        predictions,
                        conf_thres=CONFIDENCE_THRESHOLD,
                        iou_thres=IOU_THRESHOLD,
                        classes=None,
                        agnostic=True,  # 클래스 무관하게 NMS 적용 (같은 위치 다른 클래스도 제거)
                        max_det=MAX_DETECTIONS,
                        nc=nc
                    )
                
                # YOLO 형식으로 변환
                labels = postprocess_results_to_yolo_format(predictions, wide_original_size, IMAGE_SIZE)
                
                # 라벨이 없으면 스킵 (빈 이미지 저장 방지)
                if len(labels) == 0:
                    skipped_images += 1
                    continue
                
                # 새 파일명 생성 (충돌 방지)
                new_img_name = f"{base_name}_{cam0_img_name}"
                
                # 이미지 복사 (cam0=광각, cam1=협각)
                shutil.copy2(cam0_img_path, os.path.join(OUTPUT_WIDE_DIR, new_img_name))
                shutil.copy2(cam1_img_path, os.path.join(OUTPUT_NARROW_DIR, new_img_name))
                
                # 라벨 저장
                label_name = os.path.splitext(new_img_name)[0] + '.txt'
                label_path = os.path.join(OUTPUT_LABEL_DIR, label_name)
                save_yolo_labels(labels, label_path)
                
                total_images += 1
                total_labels += len(labels)
                
            except Exception as e:
                print(f"\n❌ Error processing {cam0_img_name}: {e}")
                import traceback
                traceback.print_exc()
                skipped_images += 1
                continue
    
    print(f"\n✅ Auto-labeling completed!")
    print(f"   Total images processed: {total_images}")
    print(f"   Total labels created: {total_labels}")
    print(f"   Skipped images: {skipped_images}")
    print(f"   Average labels per image: {total_labels / total_images if total_images > 0 else 0:.2f}")


def main():
    """메인 함수"""
    
    print("🚀 Starting Katri Dataset Auto-Labeling...")
    print(f"📂 Input: {KATRI_ROOT}")
    print(f"📂 Output: {OUTPUT_ROOT}")
    
    # 1. 모델 로드
    print(f"\n📦 Loading model from: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # 2. Katri 데이터셋 처리
    print(f"\n🔄 Processing Katri dataset...")
    process_katri_dataset(model)
    
    print("\n🎉 All done!")


if __name__ == "__main__":
    main()
