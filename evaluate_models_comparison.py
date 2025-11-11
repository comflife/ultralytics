#!/usr/bin/env python3
"""
Multi-Model Evaluation Script for Dual Stream YOLO
Evaluates multiple models on datasets and compares performance metrics
"""

import sys
import os
from pathlib import Path
import numpy as np
import cv2
import torch
import json
import time
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

# 🔧 로컬 ultralytics 모듈을 우선적으로 사용하도록 설정
SCRIPT_DIR = Path(__file__).parent.absolute()
ULTRALYTICS_ROOT = SCRIPT_DIR
sys.path.insert(0, str(ULTRALYTICS_ROOT))

print(f"🔧 Using local ultralytics from: {ULTRALYTICS_ROOT}")

# ============================================================
# 🛠️ 설정 부분
# ============================================================

# 평가할 모델 리스트
MODEL_PATHS = [
    "/home/byounggun/ultralytics/runs/train/exp361/weights/best.pt",
    # "/home/byounggun/ultralytics/runs/train/exp354/weights/best.pt",
]

# katri_overfit2 디렉토리의 모든 가중치 추가
KATRI_OVERFIT_DIR = "/home/byounggun/ultralytics/runs/finetune/katri_overfit2/weights"
if os.path.exists(KATRI_OVERFIT_DIR):
    for weight_file in sorted(os.listdir(KATRI_OVERFIT_DIR)):
        if weight_file.endswith('.pt'):
            MODEL_PATHS.append(os.path.join(KATRI_OVERFIT_DIR, weight_file))

# 평가할 데이터셋 리스트
DATASETS = [
    {
        "name": "KATRI Dataset",
        "wide_dir": "/home/byounggun/ultralytics/finetune_katri_name/images",
        "narrow_dir": "/home/byounggun/ultralytics/finetune_katri_name/narrow_images",
        "label_dir": "/home/byounggun/ultralytics/finetune_katri_name/labels",
    },
    {
        "name": "SWM Validation Dataset",
        "wide_dir": "/home/byounggun/ultralytics/swm_dual_split/val/images/",
        "narrow_dir": "/home/byounggun/ultralytics/swm_dual_split/val/val_narrow_images/",
        "label_dir": "/home/byounggun/ultralytics/swm_dual_split/val/labels/",
    },
]

# 추론 설정
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.55
IMAGE_SIZE = 640

# 신호등 클래스 ID (커스텀 데이터셋)
# Class indices: green_on(21), yellow_on(22), red_on(23), green_left_on(24), red_left_on(25)
TRAFFIC_LIGHT_CLASS_IDS = [21, 22, 23, 24, 25]

# Depth 정규화 정보 파일
DEPTH_NORM_INFO_PATH = "/home/byounggun/ultralytics/depth_normalization_info.json"

# 출력 디렉토리
OUTPUT_DIR = "model_evaluation_results"

# ============================================================

try:
    from ultralytics import YOLO
    import ultralytics
    print(f"✅ Using ultralytics from: {ultralytics.__file__}")
except ImportError as e:
    print(f"❌ Failed to import ultralytics: {e}")
    sys.exit(1)


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


def preprocess_image(image_path, target_size=640):
    """이미지를 모델 입력에 맞게 전처리"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    original_height, original_width = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 리사이즈 (letterbox)
    scale = min(target_size / original_width, target_size / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    resized = cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # 패딩 추가
    delta_w = target_size - new_width
    delta_h = target_size - new_height
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, 
                               cv2.BORDER_CONSTANT, value=[114, 114, 114])
    
    normalized = padded.astype(np.float32) / 255.0
    tensor = torch.from_numpy(normalized).permute(2, 0, 1)
    
    return tensor, (original_height, original_width)


def create_dual_stream_input(wide_tensor, narrow_tensor):
    """두 이미지를 듀얼 스트림 형태로 결합"""
    dual_stream = torch.cat([wide_tensor, narrow_tensor], dim=0)
    dual_stream = dual_stream.unsqueeze(0)
    return dual_stream


def postprocess_results_with_depth(predictions, original_size, target_size=640):
    """Depth를 포함한 결과 후처리"""
    if not predictions or len(predictions) == 0:
        return []
    
    pred = predictions[0]
    
    if pred is None or len(pred) == 0:
        return []
    
    original_height, original_width = original_size
    scale = min(target_size / original_width, target_size / original_height)
    
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    delta_w = target_size - new_width
    delta_h = target_size - new_height
    left = delta_w // 2
    top = delta_h // 2
    
    detections = []
    
    for detection in pred:
        if len(detection) >= 7:
            x1, y1, x2, y2, confidence, class_id, depth = detection.cpu().numpy()[:7]
        elif len(detection) == 6:
            x1, y1, x2, y2, confidence, class_id = detection.cpu().numpy()
            depth = 0.0
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
        
        # 클리핑
        x1 = max(0, min(x1, original_width))
        y1 = max(0, min(y1, original_height))
        x2 = max(0, min(x2, original_width))
        y2 = max(0, min(y2, original_height))
        
        if len(detection) >= 7:
            depth = float(depth)
            depth = max(0.0, min(1.0, depth))
        
        detections.append([x1, y1, x2, y2, float(confidence), int(class_id), float(depth)])
    
    return detections


def calculate_iou(box1, box2):
    """두 박스 간의 IoU 계산 (normalized coordinates)"""
    # box format: [x_center, y_center, width, height]
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2
    
    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2
    
    # 교집합 영역
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height
    
    # 합집합 영역
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def evaluate_detections(detections, gt_labels, img_width, img_height, iou_threshold=0.5):
    """검출 결과 평가 (precision, recall 등)"""
    
    if len(gt_labels) == 0 and len(detections) == 0:
        return {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'traffic_light_detections': 0,
            'traffic_light_gt': 0,
            'traffic_light_matched': 0,
        }
    
    # GT를 픽셀 좌표로 변환
    gt_boxes_pixel = []
    for gt in gt_labels:
        class_id, x_center, y_center, width, height, depth = gt
        x_center_px = x_center * img_width
        y_center_px = y_center * img_height
        width_px = width * img_width
        height_px = height * img_height
        
        x1 = int(x_center_px - width_px / 2)
        y1 = int(y_center_px - height_px / 2)
        x2 = int(x_center_px + width_px / 2)
        y2 = int(y_center_px + height_px / 2)
        
        gt_boxes_pixel.append({
            'class_id': class_id,
            'box': [x1, y1, x2, y2],
            'box_norm': [x_center, y_center, width, height],
            'matched': False
        })
    
    # Detection 박스 (이미 픽셀 좌표)
    det_boxes = []
    for det in detections:
        x1, y1, x2, y2, confidence, class_id, depth = det
        
        # 픽셀 좌표를 normalized 좌표로 변환
        x_center = ((x1 + x2) / 2) / img_width
        y_center = ((y1 + y2) / 2) / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        
        det_boxes.append({
            'class_id': class_id,
            'confidence': confidence,
            'box': [x1, y1, x2, y2],
            'box_norm': [x_center, y_center, width, height],
            'matched': False
        })
    
    # TP, FP, FN 계산
    true_positives = 0
    false_positives = 0
    
    # 각 detection에 대해 매칭되는 GT 찾기
    for det in det_boxes:
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(gt_boxes_pixel):
            if gt['matched']:
                continue
            
            # 같은 클래스인 경우만 매칭
            if det['class_id'] == gt['class_id']:
                iou = calculate_iou(det['box_norm'], gt['box_norm'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            # True Positive
            true_positives += 1
            det['matched'] = True
            gt_boxes_pixel[best_gt_idx]['matched'] = True
        else:
            # False Positive
            false_positives += 1
    
    # False Negatives (매칭되지 않은 GT)
    false_negatives = sum(1 for gt in gt_boxes_pixel if not gt['matched'])
    
    # 신호등 관련 통계
    traffic_light_gt = sum(1 for gt in gt_boxes_pixel if gt['class_id'] in TRAFFIC_LIGHT_CLASS_IDS)
    traffic_light_detections = sum(1 for det in det_boxes if det['class_id'] in TRAFFIC_LIGHT_CLASS_IDS)
    traffic_light_matched = sum(1 for gt in gt_boxes_pixel 
                               if gt['class_id'] in TRAFFIC_LIGHT_CLASS_IDS and gt['matched'])
    
    return {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'traffic_light_detections': traffic_light_detections,
        'traffic_light_gt': traffic_light_gt,
        'traffic_light_matched': traffic_light_matched,
    }


def evaluate_model_on_dataset(model_path, dataset_config, device='cuda'):
    """단일 모델을 데이터셋에 대해 평가"""
    
    print(f"\n{'='*80}")
    print(f"📦 Evaluating: {Path(model_path).name}")
    print(f"📊 Dataset: {dataset_config['name']}")
    print(f"{'='*80}\n")
    
    # 모델 로드
    try:
        model = YOLO(model_path)
        pytorch_model = model.model if hasattr(model, 'model') else model.predictor.model
        pytorch_model.eval()
        pytorch_model.to(device)
        print(f"✅ Model loaded on {device}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None
    
    # 데이터셋 이미지 수집
    wide_dir = dataset_config['wide_dir']
    narrow_dir = dataset_config['narrow_dir']
    label_dir = dataset_config['label_dir']
    
    wide_files = [f for f in os.listdir(wide_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not wide_files:
        print(f"❌ No images found in {wide_dir}")
        return None
    
    print(f"📸 Found {len(wide_files)} images")
    
    # 메트릭 초기화
    total_metrics = defaultdict(int)
    total_inference_time = 0
    processed_images = 0
    
    # Depth denormalizer
    depth_denormalizer = DepthDenormalizer()
    
    # 각 이미지에 대해 추론
    for filename in tqdm(wide_files, desc="Processing images"):
        wide_image_path = os.path.join(wide_dir, filename)
        narrow_image_path = os.path.join(narrow_dir, filename)
        label_path = os.path.join(label_dir, filename.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))
        
        # narrow 이미지 확인
        if not os.path.exists(narrow_image_path):
            continue
        
        # GT 라벨 로드
        gt_labels = load_ground_truth_labels(label_path)
        
        try:
            # 이미지 전처리
            wide_tensor, wide_original_size = preprocess_image(wide_image_path, IMAGE_SIZE)
            narrow_tensor, _ = preprocess_image(narrow_image_path, IMAGE_SIZE)
            
            # 듀얼 스트림 입력 생성
            dual_input = create_dual_stream_input(wide_tensor, narrow_tensor)
            dual_input = dual_input.to(device)
            
            # 추론
            start_time = time.time()
            with torch.no_grad():
                predictions = pytorch_model(dual_input)
                
                # NMS 적용
                from ultralytics.utils.ops import non_max_suppression
                nc = getattr(pytorch_model, 'nc', 80)
                
                predictions = non_max_suppression(
                    predictions,
                    conf_thres=CONFIDENCE_THRESHOLD,
                    iou_thres=IOU_THRESHOLD,
                    classes=None,
                    agnostic=False,
                    max_det=300,
                    nc=nc
                )
            
            inference_time = time.time() - start_time
            total_inference_time += inference_time
            
            # 후처리
            detections = postprocess_results_with_depth(predictions, wide_original_size, IMAGE_SIZE)
            
            # 평가
            img_height, img_width = wide_original_size
            metrics = evaluate_detections(detections, gt_labels, img_width, img_height, iou_threshold=0.5)
            
            # 누적
            for key, value in metrics.items():
                total_metrics[key] += value
            
            processed_images += 1
            
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            continue
    
    if processed_images == 0:
        print("❌ No images processed successfully")
        return None
    
    # 최종 메트릭 계산
    tp = total_metrics['true_positives']
    fp = total_metrics['false_positives']
    fn = total_metrics['false_negatives']
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    tl_precision = (total_metrics['traffic_light_matched'] / total_metrics['traffic_light_detections'] 
                   if total_metrics['traffic_light_detections'] > 0 else 0)
    tl_recall = (total_metrics['traffic_light_matched'] / total_metrics['traffic_light_gt'] 
                if total_metrics['traffic_light_gt'] > 0 else 0)
    
    avg_inference_time = total_inference_time / processed_images
    
    results = {
        'model_path': model_path,
        'model_name': Path(model_path).name,
        'dataset': dataset_config['name'],
        'processed_images': processed_images,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'traffic_light_gt': total_metrics['traffic_light_gt'],
        'traffic_light_detections': total_metrics['traffic_light_detections'],
        'traffic_light_matched': total_metrics['traffic_light_matched'],
        'traffic_light_precision': tl_precision,
        'traffic_light_recall': tl_recall,
        'avg_inference_time': avg_inference_time,
        'total_inference_time': total_inference_time,
    }
    
    print(f"\n📊 Results for {Path(model_path).name} on {dataset_config['name']}:")
    print(f"  Images Processed: {processed_images}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1_score:.4f}")
    print(f"  False Positives: {fp}")
    print(f"  Traffic Light Detection Rate: {tl_recall:.4f} ({total_metrics['traffic_light_matched']}/{total_metrics['traffic_light_gt']})")
    print(f"  Avg Inference Time: {avg_inference_time*1000:.2f}ms")
    
    return results


def main():
    """메인 함수"""
    
    print("🚀 Starting Multi-Model Evaluation...")
    print(f"📦 Total models to evaluate: {len(MODEL_PATHS)}")
    print(f"📊 Total datasets: {len(DATASETS)}")
    
    # GPU 사용 가능 여부 확인
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Using device: {device}")
    
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 출력 디렉토리 생성
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    
    # 모든 결과 수집
    all_results = []
    
    # 각 모델과 데이터셋 조합에 대해 평가
    for model_path in MODEL_PATHS:
        if not os.path.exists(model_path):
            print(f"⚠️  Model not found: {model_path}")
            continue
        
        for dataset_config in DATASETS:
            result = evaluate_model_on_dataset(model_path, dataset_config, device=device)
            if result:
                all_results.append(result)
    
    if not all_results:
        print("❌ No results to report")
        return
    
    # 결과를 DataFrame으로 변환
    df = pd.DataFrame(all_results)
    
    # CSV 저장
    csv_path = output_dir / "model_comparison_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Results saved to: {csv_path}")
    
    # 상세 리포트 생성
    report_path = output_dir / "evaluation_report.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("MULTI-MODEL EVALUATION REPORT\n")
        f.write("=" * 100 + "\n\n")
        
        f.write(f"Total Models Evaluated: {len(set(df['model_name']))}\n")
        f.write(f"Total Datasets: {len(set(df['dataset']))}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}\n")
        f.write(f"IoU Threshold: {IOU_THRESHOLD}\n\n")
        
        # 데이터셋별 최고 성능 모델
        for dataset_name in df['dataset'].unique():
            f.write(f"\n{'='*100}\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"{'='*100}\n\n")
            
            dataset_df = df[df['dataset'] == dataset_name].copy()
            dataset_df = dataset_df.sort_values('f1_score', ascending=False)
            
            f.write("Ranking by F1 Score:\n")
            f.write("-" * 100 + "\n")
            
            for idx, row in dataset_df.iterrows():
                f.write(f"\n{row['model_name']}:\n")
                f.write(f"  Precision: {row['precision']:.4f}\n")
                f.write(f"  Recall: {row['recall']:.4f}\n")
                f.write(f"  F1 Score: {row['f1_score']:.4f}\n")
                f.write(f"  False Positives: {row['false_positives']}\n")
                f.write(f"  Traffic Light Precision: {row['traffic_light_precision']:.4f}\n")
                f.write(f"  Traffic Light Recall: {row['traffic_light_recall']:.4f}\n")
                f.write(f"  Traffic Light Matched: {row['traffic_light_matched']}/{row['traffic_light_gt']}\n")
                f.write(f"  Avg Inference Time: {row['avg_inference_time']*1000:.2f}ms\n")
        
        # 전체 요약
        f.write(f"\n{'='*100}\n")
        f.write("OVERALL SUMMARY\n")
        f.write(f"{'='*100}\n\n")
        
        f.write("Best Models by Metric:\n\n")
        
        best_f1 = df.loc[df['f1_score'].idxmax()]
        f.write(f"Best F1 Score: {best_f1['model_name']} on {best_f1['dataset']} ({best_f1['f1_score']:.4f})\n")
        
        best_precision = df.loc[df['precision'].idxmax()]
        f.write(f"Best Precision: {best_precision['model_name']} on {best_precision['dataset']} ({best_precision['precision']:.4f})\n")
        
        best_recall = df.loc[df['recall'].idxmax()]
        f.write(f"Best Recall: {best_recall['model_name']} on {best_recall['dataset']} ({best_recall['recall']:.4f})\n")
        
        least_fp = df.loc[df['false_positives'].idxmin()]
        f.write(f"Least False Positives: {least_fp['model_name']} on {least_fp['dataset']} ({least_fp['false_positives']})\n")
        
        best_tl_recall = df.loc[df['traffic_light_recall'].idxmax()]
        f.write(f"Best Traffic Light Recall: {best_tl_recall['model_name']} on {best_tl_recall['dataset']} ({best_tl_recall['traffic_light_recall']:.4f})\n")
        
        fastest = df.loc[df['avg_inference_time'].idxmin()]
        f.write(f"Fastest Inference: {fastest['model_name']} on {fastest['dataset']} ({fastest['avg_inference_time']*1000:.2f}ms)\n")
    
    print(f"📄 Detailed report saved to: {report_path}")
    
    # 콘솔에 요약 출력
    print("\n" + "="*100)
    print("EVALUATION SUMMARY")
    print("="*100)
    
    for dataset_name in df['dataset'].unique():
        print(f"\n📊 {dataset_name}:")
        dataset_df = df[df['dataset'] == dataset_name].sort_values('f1_score', ascending=False)
        
        for idx, row in dataset_df.head(3).iterrows():
            print(f"  {row['model_name']}: F1={row['f1_score']:.4f}, "
                  f"P={row['precision']:.4f}, R={row['recall']:.4f}, "
                  f"FP={row['false_positives']}, TL_R={row['traffic_light_recall']:.4f}")
    
    print("\n🎉 Evaluation completed successfully!")


if __name__ == "__main__":
    main()
