#!/usr/bin/env python3
"""
Visualize Auto-labeling Results
Shows wide (left) and narrow (right) images with bounding boxes from labels
Randomly selects one image pair each time
"""

import os
import random
import cv2
import numpy as np
import json
from pathlib import Path

# ============================================================
# 🛠️ 설정 부분
# ============================================================

# 자동 라벨링 결과 디렉토리
LABELED_ROOT = "/home/byounggun/ultralytics/finetune_katri"
# LABELED_ROOT = "/home/byounggun/ultralytics/swm_dual_split/train"
WIDE_DIR = os.path.join(LABELED_ROOT, "images")
NARROW_DIR = os.path.join(LABELED_ROOT, "narrow_images")
# NARROW_DIR = os.path.join(LABELED_ROOT, "train_narrow_images")
LABEL_DIR = os.path.join(LABELED_ROOT, "labels")

# 출력 디렉토리
OUTPUT_DIR = "/home/byounggun/ultralytics/autolabel_verification"

# Depth 정규화 정보 파일
DEPTH_NORM_INFO_PATH = "/home/byounggun/ultralytics/depth_normalization_info.json"

# 클래스 이름 (전체 28개)
CLASS_NAMES = {
    0: "car",
    1: "suv",
    2: "van",
    3: "regular_truck",
    4: "large_truck",
    5: "bus",
    6: "bicyclist",
    7: "motorcyclist",
    8: "scooter",
    9: "pedestrian",
    10: "stroller",
    11: "rubber_cone",
    12: "traffic_drum",
    13: "speed_30",
    14: "speed_40",
    15: "speed_50",
    16: "speed_60",
    17: "speed_70",
    18: "speed_80",
    19: "speed_90",
    20: "speed_100",
    21: "speed_110",
    22: "green_on",
    23: "yellow_on",
    24: "red_on",
    25: "green_left_on",
    26: "red_left_on",
    27: "unknown",
}

# 색상 팔레트 (BGR)
COLORS = [
    (255, 0, 0),      # 파란색
    (0, 255, 0),      # 초록색
    (0, 0, 255),      # 빨간색
    (255, 255, 0),    # 시안
    (255, 0, 255),    # 마젠타
    (0, 255, 255),    # 노란색
    (128, 0, 128),    # 보라색
    (255, 128, 0),    # 주황색
    (0, 128, 255),    # 하늘색
    (128, 255, 0),    # 연두색
]

# ============================================================


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
            print(f"✅ Depth normalization info loaded")
            print(f"   Range: [{self.norm_info['min_depth']:.6f}, {self.norm_info['max_depth']:.6f}]")
        except FileNotFoundError:
            print(f"⚠️  Depth normalization info not found: {self.norm_info_path}")
            print(f"   Will display normalized depth values (0~1)")
            self.norm_info = None
        except Exception as e:
            print(f"⚠️  Error loading depth normalization info: {e}")
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


def load_yolo_labels(label_path):
    """
    YOLO 형식의 라벨 파일 로드
    
    Args:
        label_path (str): 라벨 파일 경로
    
    Returns:
        list: 라벨 리스트 [class_id, x_center, y_center, width, height, depth]
    """
    if not os.path.exists(label_path):
        return []
    
    labels = []
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:  # 최소한 class x y w h
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                depth = float(parts[5]) if len(parts) > 5 else 0.0
                labels.append([class_id, x_center, y_center, width, height, depth])
    except Exception as e:
        print(f"❌ Error loading labels from {label_path}: {e}")
    
    return labels


def convert_yolo_to_bbox(label, img_width, img_height):
    """
    YOLO 형식을 픽셀 좌표로 변환
    
    Args:
        label: [class_id, x_center, y_center, width, height, depth]
        img_width, img_height: 이미지 크기
    
    Returns:
        tuple: (x1, y1, x2, y2, class_id, depth)
    """
    class_id, x_center, y_center, width, height, depth = label
    
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
    
    return (x1, y1, x2, y2, class_id, depth)


def draw_bboxes_on_image(image, labels, depth_denormalizer, class_names=None):
    """
    이미지에 바운딩 박스 그리기
    
    Args:
        image (np.ndarray): 이미지 (BGR)
        labels (list): 라벨 리스트
        depth_denormalizer: Depth 역정규화 객체
        class_names (dict): 클래스 이름 딕셔너리
    
    Returns:
        np.ndarray: 바운딩 박스가 그려진 이미지
    """
    if class_names is None:
        class_names = CLASS_NAMES
    
    result_image = image.copy()
    img_height, img_width = image.shape[:2]
    
    print(f"   Image size for drawing: {img_width}x{img_height}")
    print(f"   Number of labels to draw: {len(labels)}")
    
    # 각 라벨에 대해 바운딩 박스 그리기
    for i, label in enumerate(labels):
        class_id, x_center, y_center, width, height, depth = label
        
        # YOLO 정규화된 좌표를 픽셀 좌표로 변환
        x_center_px = x_center * img_width
        y_center_px = y_center * img_height
        width_px = width * img_width
        height_px = height * img_height
        
        # 좌상단, 우하단 좌표 계산
        x1 = int(x_center_px - width_px / 2)
        y1 = int(y_center_px - height_px / 2)
        x2 = int(x_center_px + width_px / 2)
        y2 = int(y_center_px + height_px / 2)
        
        # 좌표가 이미지 범위 내에 있도록 클리핑
        x1 = max(0, min(x1, img_width))
        y1 = max(0, min(y1, img_height))
        x2 = max(0, min(x2, img_width))
        y2 = max(0, min(y2, img_height))
        
        # Depth 역정규화
        depth_original = depth_denormalizer.denormalize_depth(depth)
        
        print(f"   Label {i+1}: class={int(class_id)}, bbox=({x1},{y1},{x2},{y2}), depth={depth:.3f} -> {depth_original:.2f}m")
        
        # 색상 선택
        color = COLORS[int(class_id) % len(COLORS)]
        
        # 바운딩 박스 그리기 (두꺼운 선)
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 3)
        
        # 라벨 텍스트 생성 (역정규화된 depth 사용)
        class_name = class_names.get(int(class_id), f"class_{int(class_id)}")
        label_text = f"{class_name}: {depth_original:.2f}m"
        
        # 텍스트 배경
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        # 텍스트 위치 조정 (박스 위에 표시)
        text_y = y1 - 10
        if text_y < text_height + 10:
            text_y = y2 + text_height + 10
        
        # 텍스트 배경 그리기
        cv2.rectangle(
            result_image,
            (x1, text_y - text_height - 5),
            (x1 + text_width + 5, text_y + 5),
            color,
            -1
        )
        
        # 텍스트 그리기
        cv2.putText(
            result_image,
            label_text,
            (x1 + 2, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
    
    return result_image


def create_side_by_side_visualization(wide_img, narrow_img, labels, depth_denormalizer, image_name):
    """
    광각(좌측)과 협각(우측) 이미지를 나란히 배치하고 라벨 시각화
    
    Args:
        wide_img (np.ndarray): 광각 이미지
        narrow_img (np.ndarray): 협각 이미지
        labels (list): 라벨 리스트
        depth_denormalizer: Depth 역정규화 객체
        image_name (str): 이미지 이름
    
    Returns:
        np.ndarray: 합쳐진 이미지
    """
    print(f"\n🎨 Creating visualization for {image_name}")
    print(f"   Wide image shape: {wide_img.shape}")
    print(f"   Narrow image shape: {narrow_img.shape}")
    print(f"   Number of labels: {len(labels)}")
    
    # 두 이미지의 높이를 먼저 맞추기
    h_wide, w_wide = wide_img.shape[:2]
    h_narrow, w_narrow = narrow_img.shape[:2]
    
    # 높이가 다르면 작은 쪽을 큰 쪽에 맞춤 (비율 유지)
    if h_wide != h_narrow:
        if h_wide > h_narrow:
            # narrow를 wide 높이에 맞춤
            scale = h_wide / h_narrow
            new_w_narrow = int(w_narrow * scale)
            narrow_img = cv2.resize(narrow_img, (new_w_narrow, h_wide))
            print(f"   Resized narrow: {narrow_img.shape}")
        else:
            # wide를 narrow 높이에 맞춤
            scale = h_narrow / h_wide
            new_w_wide = int(w_wide * scale)
            wide_img = cv2.resize(wide_img, (new_w_wide, h_narrow))
            print(f"   Resized wide: {wide_img.shape}")
    
    # 광각 이미지에 바운딩 박스 그리기 (리사이즈 후에 그림)
    wide_with_bbox = draw_bboxes_on_image(wide_img, labels, depth_denormalizer)
    
    h_final, w_wide = wide_with_bbox.shape[:2]
    h_final, w_narrow = narrow_img.shape[:2]
    
    # 텍스트 헤더 추가
    header_height = 40
    total_width = w_wide + w_narrow
    
    # 헤더 생성
    header = np.ones((header_height, total_width, 3), dtype=np.uint8) * 50
    
    # 텍스트 추가
    title_text = f"Auto-labeling Verification: {image_name}"
    cv2.putText(header, title_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    wide_text = "Wide (with labels)"
    narrow_text = "Narrow"
    
    cv2.putText(header, wide_text, (w_wide//2 - 80, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(header, narrow_text, (w_wide + w_narrow//2 - 40, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # 이미지 합치기 (좌: 광각, 우: 협각)
    combined = np.hstack([wide_with_bbox, narrow_img])
    
    # 헤더와 합치기
    final_image = np.vstack([header, combined])
    
    # 통계 정보 추가
    stats_height = 30
    stats_bar = np.ones((stats_height, total_width, 3), dtype=np.uint8) * 40
    
    stats_text = f"Total Labels: {len(labels)} | Classes: {len(set([l[0] for l in labels]))}"
    cv2.putText(stats_bar, stats_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    final_image = np.vstack([final_image, stats_bar])
    
    return final_image


def main():
    """메인 함수"""
    
    print("🔍 Starting Auto-labeling Verification Visualization")
    print(f"📂 Labeled data: {LABELED_ROOT}")
    
    # Depth 역정규화 객체 생성
    depth_denormalizer = DepthDenormalizer()
    
    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 광각 이미지 목록 가져오기
    wide_images = [f for f in os.listdir(WIDE_DIR) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not wide_images:
        print(f"❌ No images found in {WIDE_DIR}")
        return
    
    print(f"📊 Total images available: {len(wide_images)}")
    
    # 랜덤으로 하나 선택
    selected_image = random.choice(wide_images)
    print(f"🎲 Randomly selected: {selected_image}")
    
    # 파일 경로 구성
    wide_path = os.path.join(WIDE_DIR, selected_image)
    narrow_path = os.path.join(NARROW_DIR, selected_image)
    label_path = os.path.join(LABEL_DIR, os.path.splitext(selected_image)[0] + '.txt')
    
    # 파일 존재 확인
    if not os.path.exists(narrow_path):
        print(f"❌ Narrow image not found: {narrow_path}")
        return
    
    if not os.path.exists(label_path):
        print(f"⚠️  Label file not found: {label_path}")
        print("   Visualizing without labels...")
    
    # 이미지 로드
    print("📸 Loading images...")
    wide_img = cv2.imread(wide_path)
    narrow_img = cv2.imread(narrow_path)
    
    if wide_img is None:
        print(f"❌ Failed to load wide image: {wide_path}")
        return
    
    if narrow_img is None:
        print(f"❌ Failed to load narrow image: {narrow_path}")
        return
    
    print(f"   Wide image size: {wide_img.shape[:2]}")
    print(f"   Narrow image size: {narrow_img.shape[:2]}")
    
    # 라벨 로드
    labels = load_yolo_labels(label_path)
    print(f"📋 Loaded {len(labels)} labels")
    
    if labels:
        # 클래스별 통계
        class_counts = {}
        for label in labels:
            class_id = int(label[0])
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
        
        print("   Class distribution:")
        for class_id, count in sorted(class_counts.items()):
            class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
            print(f"      {class_name} (ID {class_id}): {count}")
    
    # 시각화 생성
    print("🎨 Creating visualization...")
    visualization = create_side_by_side_visualization(wide_img, narrow_img, labels, depth_denormalizer, selected_image)
    
    # 저장
    output_filename = f"verification_{os.path.splitext(selected_image)[0]}.jpg"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    cv2.imwrite(output_path, visualization)
    
    print(f"✅ Visualization saved to: {output_path}")
    print(f"📐 Output size: {visualization.shape[:2]}")
    print("🎉 Done!")


if __name__ == "__main__":
    main()
