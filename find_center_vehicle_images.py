#!/usr/bin/env python3
"""
Find images with vehicles in the center (front) position
Searches train/val datasets and saves visualizations with GT labels
"""

import os
import sys
from pathlib import Path
import numpy as np
import cv2
import json
import shutil
from tqdm import tqdm

# ============================================================
# 🛠️ 설정 부분
# ============================================================

# 데이터셋 루트 디렉토리
DATASET_ROOT = "/home/byounggun/ultralytics/swm_dual_split"

# 출력 디렉토리 (finetune_katri_name 폴더 구조에 복사)
OUTPUT_DIR = "/home/byounggun/ultralytics/finetune_katri_name"

# 차량 클래스 ID
VEHICLE_CLASS_IDS = [0, 1, 2, 3, 4, 5]  # car, suv, van, regular_truck, large_truck, bus

# 신호등 및 표지판 클래스 ID (중앙 외 영역에서 카운트하지 않음)
TRAFFIC_LIGHT_CLASS_IDS = [21, 22, 23, 24, 25]  # green_on, yellow_on, red_on, green_left_on, red_left_on
SIGN_CLASS_IDS = [13, 14, 15, 16, 17, 18, 19, 20, 21]  # speed signs

# 중앙 영역 정의 (normalized coordinates)
CENTER_X_MIN = 0.35  # 이미지 중앙 35%~65% 영역
CENTER_X_MAX = 0.65
CENTER_Y_MIN = 0.30  # 이미지 중앙 30%~70% 영역 (세로)
CENTER_Y_MAX = 0.70

# 최소 bbox 크기 (normalized) - 너무 작은 객체 제외
MIN_BBOX_WIDTH = 0.05
MIN_BBOX_HEIGHT = 0.05

# 중앙 차량의 최소 크기 (더 큰 차량만)
MIN_CENTER_VEHICLE_WIDTH = 0.12  # 중앙 차량은 최소 12% 이상
MIN_CENTER_VEHICLE_HEIGHT = 0.15  # 중앙 차량은 최소 15% 이상

# 중앙 외 영역의 최대 객체 수 제한 (신호등/표지판 제외)
MAX_NON_CENTER_OBJECTS = 0  # 중앙 외 영역에 신호등/표지판 제외하고 아무것도 없어야 함

# Depth 정규화 정보 파일
DEPTH_NORM_INFO_PATH = "/home/byounggun/ultralytics/depth_normalization_info.json"

# 클래스 이름
CLASS_NAMES = [
    "car", "suv", "van", "regular_truck", "large_truck", "bus",
    "bicyclist", "motorcyclist", "scooter", "pedestrian", "stroller",
    "rubber_cone", "traffic_drum",
    "speed_30", "speed_40", "speed_50", "speed_60", "speed_70", 
    "speed_80", "speed_90", "speed_100", "speed_110",
    "green_on", "yellow_on", "red_on", "green_left_on", "red_left_on",
    "unknown"
]

# ============================================================


class DepthDenormalizer:
    """Depth 값을 원본 스케일로 복구하는 클래스"""
    
    def __init__(self, norm_info_path=DEPTH_NORM_INFO_PATH):
        self.norm_info_path = norm_info_path
        self.norm_info = None
        self.load_normalization_info()
    
    def load_normalization_info(self):
        """정규화 정보 로드"""
        try:
            with open(self.norm_info_path, 'r') as f:
                self.norm_info = json.load(f)
        except:
            self.norm_info = None
    
    def denormalize_depth(self, normalized_depth):
        """정규화된 depth 값을 원본 스케일로 복구"""
        if self.norm_info is None:
            return normalized_depth
        
        min_depth = self.norm_info['min_depth']
        max_depth = self.norm_info['max_depth']
        original_depth = normalized_depth * (max_depth - min_depth) + min_depth
        return original_depth


def load_ground_truth_labels(label_path):
    """라벨 파일에서 GT 정보를 로드"""
    if not os.path.exists(label_path):
        return []
    
    gt_labels = []
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 6:
                class_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                depth = float(parts[5])
                gt_labels.append([class_id, x, y, w, h, depth])
    except Exception as e:
        print(f"❌ 라벨 파일 읽기 오류: {e}")
    
    return gt_labels


def is_vehicle_in_center(gt_labels):
    """
    차량이 이미지 중앙에 있는지 확인하고, 중앙 외 영역에 객체가 없는지 확인
    (신호등 및 표지판은 제외)
    
    Args:
        gt_labels: GT 라벨 리스트
    
    Returns:
        bool: 중앙에 충분히 큰 차량이 있고 중앙 외 영역에 객체가 없으면 True
        list: 중앙에 있는 차량 정보 리스트
    """
    center_vehicles = []
    non_center_objects = []
    
    for label in gt_labels:
        class_id, x_center, y_center, width, height, depth = label
        
        # bbox 크기 확인 (너무 작은 객체는 카운트하지 않음)
        if width < MIN_BBOX_WIDTH or height < MIN_BBOX_HEIGHT:
            continue
        
        # 중앙 영역에 있는지 확인
        is_in_center = (CENTER_X_MIN <= x_center <= CENTER_X_MAX and 
                       CENTER_Y_MIN <= y_center <= CENTER_Y_MAX)
        
        if is_in_center:
            # 차량 클래스이고 크기가 충분히 큰지 확인
            if class_id in VEHICLE_CLASS_IDS:
                # 중앙 차량은 더 큰 크기 요구
                if width >= MIN_CENTER_VEHICLE_WIDTH and height >= MIN_CENTER_VEHICLE_HEIGHT:
                    center_vehicles.append(label)
        else:
            # 중앙 외 영역의 객체
            # 신호등 및 표지판만 제외하고 모든 객체 카운트 (차량, 보행자 등 모두 포함)
            if class_id not in TRAFFIC_LIGHT_CLASS_IDS and class_id not in SIGN_CLASS_IDS:
                non_center_objects.append(label)
    
    # 조건 확인:
    # 1. 중앙에 충분히 큰 차량이 있어야 함
    # 2. 중앙 외 영역에 신호등/표지판을 제외한 객체가 없어야 함
    has_center_vehicle = len(center_vehicles) > 0
    has_no_non_center_objects = len(non_center_objects) <= MAX_NON_CENTER_OBJECTS
    
    return (has_center_vehicle and has_no_non_center_objects), center_vehicles


def convert_normalized_to_pixel_coords(bbox, img_width, img_height):
    """정규화된 좌표를 픽셀 좌표로 변환"""
    x_center, y_center, width, height = bbox
    
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    width_px = width * img_width
    height_px = height * img_height
    
    x1 = int(x_center_px - width_px / 2)
    y1 = int(y_center_px - height_px / 2)
    x2 = int(x_center_px + width_px / 2)
    y2 = int(y_center_px + height_px / 2)
    
    return [x1, y1, x2, y2]


def draw_gt_labels_on_image(image, gt_labels, center_vehicles, depth_denormalizer):
    """
    이미지에 GT 라벨을 그리기
    
    Args:
        image: 원본 이미지 (BGR)
        gt_labels: 전체 GT 라벨 리스트
        center_vehicles: 중앙에 있는 차량 리스트
        depth_denormalizer: Depth 역정규화 객체
    
    Returns:
        np.ndarray: 라벨이 그려진 이미지
    """
    result_image = image.copy()
    img_height, img_width = image.shape[:2]
    
    # 중앙 영역 표시 (반투명 사각형)
    overlay = result_image.copy()
    center_x1 = int(CENTER_X_MIN * img_width)
    center_x2 = int(CENTER_X_MAX * img_width)
    center_y1 = int(CENTER_Y_MIN * img_height)
    center_y2 = int(CENTER_Y_MAX * img_height)
    cv2.rectangle(overlay, (center_x1, center_y1), (center_x2, center_y2), 
                  (255, 255, 0), -1)
    cv2.addWeighted(overlay, 0.1, result_image, 0.9, 0, result_image)
    cv2.rectangle(result_image, (center_x1, center_y1), (center_x2, center_y2), 
                  (255, 255, 0), 2)
    
    # 중앙 차량 ID 수집
    center_vehicle_ids = [tuple(cv) for cv in center_vehicles]
    
    # 모든 GT 라벨 그리기
    for i, gt_label in enumerate(gt_labels):
        class_id, x_center, y_center, width, height, normalized_depth = gt_label
        
        # 픽셀 좌표로 변환
        bbox_pixel = convert_normalized_to_pixel_coords(
            [x_center, y_center, width, height], 
            img_width, img_height
        )
        x1, y1, x2, y2 = bbox_pixel
        
        # depth 역정규화
        depth_original = depth_denormalizer.denormalize_depth(normalized_depth)
        
        # 중앙 차량인지 확인
        is_center = tuple(gt_label) in center_vehicle_ids
        
        # 색상 선택 (중앙 차량은 빨강, 나머지는 초록)
        if is_center:
            color = (0, 0, 255)  # 빨강 (중앙 차량)
            thickness = 3
        else:
            color = (0, 255, 0)  # 초록 (일반)
            thickness = 2
        
        # 박스 그리기
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, thickness)
        
        # 라벨 텍스트
        class_name = CLASS_NAMES[int(class_id)] if int(class_id) < len(CLASS_NAMES) else f"class_{int(class_id)}"
        label_text = f"{class_name}: {depth_original:.1f}m"
        if is_center:
            label_text = f"[CENTER] {label_text}"
        
        # 텍스트 배경
        font_scale = 0.5
        font_thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        
        # 텍스트 위치 (박스 위)
        text_x = x1
        text_y = y1 - 5
        if text_y < text_height + 5:
            text_y = y2 + text_height + 5
        
        # 텍스트 배경 그리기
        cv2.rectangle(
            result_image, 
            (text_x, text_y - text_height - baseline),
            (text_x + text_width, text_y + baseline),
            color, -1
        )
        
        # 텍스트 그리기
        cv2.putText(
            result_image, label_text, (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness
        )
    
    return result_image


def visualize_image_pair(wide_path, narrow_path, label_path, output_path, depth_denormalizer):
    """
    광각-협각 이미지 페어를 GT 라벨과 함께 시각화
    
    Args:
        wide_path: 광각 이미지 경로
        narrow_path: 협각 이미지 경로
        label_path: 라벨 파일 경로
        output_path: 출력 이미지 경로
        depth_denormalizer: Depth 역정규화 객체
    """
    # 이미지 로드
    wide_img = cv2.imread(wide_path)
    narrow_img = cv2.imread(narrow_path)
    
    if wide_img is None or narrow_img is None:
        return False
    
    # GT 라벨 로드
    gt_labels = load_ground_truth_labels(label_path)
    
    # 중앙 차량 확인
    has_center_vehicle, center_vehicles = is_vehicle_in_center(gt_labels)
    
    if not has_center_vehicle:
        return False
    
    # 광각 이미지에 GT 라벨 그리기
    wide_with_labels = draw_gt_labels_on_image(
        wide_img, gt_labels, center_vehicles, depth_denormalizer
    )
    
    # 두 이미지를 나란히 배치
    # 높이를 맞추기
    h1, w1 = wide_with_labels.shape[:2]
    h2, w2 = narrow_img.shape[:2]
    
    if h1 != h2:
        # 높이를 더 작은 쪽에 맞춤
        target_h = min(h1, h2)
        if h1 > target_h:
            scale = target_h / h1
            wide_with_labels = cv2.resize(
                wide_with_labels, 
                (int(w1 * scale), target_h)
            )
        if h2 > target_h:
            scale = target_h / h2
            narrow_img = cv2.resize(
                narrow_img, 
                (int(w2 * scale), target_h)
            )
    
    # 나란히 배치
    combined = np.hstack([wide_with_labels, narrow_img])
    
    # 제목 추가
    title_height = 40
    title_img = np.zeros((title_height, combined.shape[1], 3), dtype=np.uint8)
    title_text = f"Wide (with GT labels) | Narrow - {len(center_vehicles)} center vehicle(s) detected"
    cv2.putText(
        title_img, title_text, (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )
    
    # 제목과 이미지 결합
    final_img = np.vstack([title_img, combined])
    
    # 저장
    cv2.imwrite(output_path, final_img)
    
    return True


def search_dataset(dataset_type, depth_denormalizer, output_dir):
    """
    데이터셋 탐색 (train 또는 val)
    
    Args:
        dataset_type: 'train' 또는 'val'
        depth_denormalizer: Depth 역정규화 객체
        output_dir: 출력 디렉토리
    
    Returns:
        int: 찾은 이미지 수
    """
    print(f"\n{'='*80}")
    print(f"📂 Searching {dataset_type.upper()} dataset...")
    print(f"{'='*80}")
    
    # 디렉토리 경로 설정
    if dataset_type == 'train':
        wide_dir = os.path.join(DATASET_ROOT, 'train', 'images')
        narrow_dir = os.path.join(DATASET_ROOT, 'train', 'train_narrow_images')
        label_dir = os.path.join(DATASET_ROOT, 'train', 'labels')
    else:  # val
        wide_dir = os.path.join(DATASET_ROOT, 'val', 'images')
        narrow_dir = os.path.join(DATASET_ROOT, 'val', 'val_narrow_images')
        label_dir = os.path.join(DATASET_ROOT, 'val', 'labels')
    
    # 이미지 파일 수집
    if not os.path.exists(wide_dir):
        print(f"❌ Directory not found: {wide_dir}")
        return 0
    
    image_files = [f for f in os.listdir(wide_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"📸 Found {len(image_files)} images in {dataset_type}")
    
    # 출력 디렉토리 생성 (finetune_katri_name 구조에 맞춤)
    output_wide_dir = Path(output_dir) / "images"
    output_narrow_dir = Path(output_dir) / "narrow_images"
    output_label_dir = Path(output_dir) / "labels"
    
    output_wide_dir.mkdir(parents=True, exist_ok=True)
    output_narrow_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)
    
    found_count = 0
    
    # 각 이미지에 대해 확인
    for filename in tqdm(image_files, desc=f"Processing {dataset_type}"):
        wide_path = os.path.join(wide_dir, filename)
        narrow_path = os.path.join(narrow_dir, filename)
        label_filename = filename.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
        label_path = os.path.join(label_dir, label_filename)
        
        # 협각 이미지 존재 확인
        if not os.path.exists(narrow_path):
            continue
        
        # 라벨 로드
        gt_labels = load_ground_truth_labels(label_path)
        
        # 중앙 차량 확인
        has_center_vehicle, center_vehicles = is_vehicle_in_center(gt_labels)
        
        if has_center_vehicle:
            # 중앙 외 영역 객체 수 확인 (신호등/표지판 제외)
            non_center_count = len([
                label for label in gt_labels
                if not (CENTER_X_MIN <= label[1] <= CENTER_X_MAX and 
                       CENTER_Y_MIN <= label[2] <= CENTER_Y_MAX) and
                   (label[3] >= MIN_BBOX_WIDTH and label[4] >= MIN_BBOX_HEIGHT) and
                   (label[0] not in TRAFFIC_LIGHT_CLASS_IDS and label[0] not in SIGN_CLASS_IDS)
            ])
            
            # 파일 복사 (finetune_katri_name 구조에 맞게)
            try:
                shutil.copy2(wide_path, output_wide_dir / filename)
                shutil.copy2(narrow_path, output_narrow_dir / filename)
                shutil.copy2(label_path, output_label_dir / label_filename)
                
                found_count += 1
                print(f"  ✅ Copied: {filename} (Center: {len(center_vehicles)} vehicle(s), Non-center: {non_center_count} objects)")
            except Exception as e:
                print(f"  ❌ Failed to copy {filename}: {e}")
    
    print(f"\n📊 {dataset_type.upper()} Summary: Found {found_count} images with center vehicles")
    
    return found_count


def main():
    """메인 함수"""
    
    print("🚀 Starting Center Vehicle Image Search & Copy...")
    print(f"📁 Source Dataset: {DATASET_ROOT}")
    print(f"📁 Destination: {OUTPUT_DIR}")
    print(f"\n🎯 Search Criteria:")
    print(f"  - Vehicle classes: {VEHICLE_CLASS_IDS}")
    print(f"  - Center region X: {CENTER_X_MIN} ~ {CENTER_X_MAX}")
    print(f"  - Center region Y: {CENTER_Y_MIN} ~ {CENTER_Y_MAX}")
    print(f"  - Min center vehicle size: {MIN_CENTER_VEHICLE_WIDTH} x {MIN_CENTER_VEHICLE_HEIGHT}")
    print(f"  - Min general bbox size: {MIN_BBOX_WIDTH} x {MIN_BBOX_HEIGHT}")
    print(f"  - Max non-center objects (excluding traffic lights/signs): {MAX_NON_CENTER_OBJECTS}")
    print(f"  - Excluded from counting: Traffic lights {TRAFFIC_LIGHT_CLASS_IDS}, Signs {SIGN_CLASS_IDS}")
    
    # 출력 폴더 구조 확인
    print(f"\n📂 Destination Folder Structure:")
    print(f"  - Wide images: {OUTPUT_DIR}/images/")
    print(f"  - Narrow images: {OUTPUT_DIR}/narrow_images/")
    print(f"  - Labels: {OUTPUT_DIR}/labels/")
    
    # Depth denormalizer
    depth_denormalizer = DepthDenormalizer()
    
    # Train 데이터셋 탐색
    train_count = search_dataset('train', depth_denormalizer, OUTPUT_DIR)
    
    # Val 데이터셋 탐색
    val_count = search_dataset('val', depth_denormalizer, OUTPUT_DIR)
    
    # 최종 요약
    print("\n" + "="*80)
    print("📊 FINAL SUMMARY")
    print("="*80)
    print(f"Train images with center vehicles: {train_count}")
    print(f"Val images with center vehicles: {val_count}")
    print(f"Total images copied: {train_count + val_count}")
    print(f"\n✅ Files copied to: {OUTPUT_DIR}")
    print(f"   - {OUTPUT_DIR}/images/ (wide)")
    print(f"   - {OUTPUT_DIR}/narrow_images/ (narrow)")
    print(f"   - {OUTPUT_DIR}/labels/ (labels)")
    print("🎉 Copy completed!")


if __name__ == "__main__":
    main()
