#!/usr/bin/env python3
"""
Dual Stream ONNX Model Test Script
Tests the exported dual-stream YOLO model with real dual-camera images

실제 dual-camera 이미지로 추론:
python test_dual_onnx_inference.py

특정 이미지 지정:
python test_dual_onnx_inference.py --image scene0002_20250422_07540461.jpg

다른 ONNX 모델 사용:
python test_dual_onnx_inference.py --model /path/to/other/model.onnx
"""

import numpy as np
import cv2
import onnxruntime as ort
from pathlib import Path
import argparse
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import os

def find_image_pairs(wide_dir, narrow_dir):
    """
    두 폴더에서 동일한 이름의 이미지 쌍을 찾기
    
    Args:
        wide_dir (str): Wide 이미지 폴더 경로
        narrow_dir (str): Narrow 이미지 폴더 경로
    
    Returns:
        list: [(wide_path, narrow_path), ...] 형태의 이미지 쌍 리스트
    """
    wide_path = Path(wide_dir)
    narrow_path = Path(narrow_dir)
    
    if not wide_path.exists():
        raise FileNotFoundError(f"Wide images directory not found: {wide_dir}")
    if not narrow_path.exists():
        raise FileNotFoundError(f"Narrow images directory not found: {narrow_dir}")
    
    # 지원하는 이미지 확장자
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Wide 이미지 파일들 찾기
    wide_images = {}
    for img_path in wide_path.iterdir():
        if img_path.suffix.lower() in image_extensions:
            wide_images[img_path.name] = img_path
    
    # Narrow 이미지 파일들 찾기
    narrow_images = {}
    for img_path in narrow_path.iterdir():
        if img_path.suffix.lower() in image_extensions:
            narrow_images[img_path.name] = img_path
    
    # 매칭되는 이미지 쌍 찾기
    image_pairs = []
    for name in wide_images.keys():
        if name in narrow_images:
            image_pairs.append((wide_images[name], narrow_images[name]))
    
    print(f"📂 Found {len(wide_images)} wide images, {len(narrow_images)} narrow images")
    print(f"🔗 Found {len(image_pairs)} matching image pairs")
    
    if not image_pairs:
        raise ValueError("No matching image pairs found!")
    
    return image_pairs

def preprocess_image(image_path, target_size=(640, 640)):
    """
    이미지를 ONNX 모델 입력에 맞게 전처리
    
    Args:
        image_path (str): 이미지 파일 경로
        target_size (tuple): 타겟 크기 (height, width)
    
    Returns:
        np.ndarray: 전처리된 이미지 [3, H, W]
        tuple: 원본 이미지 크기 (height, width)
        tuple: 스케일링 정보 (ratio, (pad_w, pad_h))
    """
    # 이미지 로드
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # BGR -> RGB 변환
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_shape = image.shape[:2]  # (H, W)
    
    # 리사이즈 (letterbox padding) - Ultralytics 방식
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    # 🔧 스케일 비율 계산 (Ultralytics 방식)
    ratio = min(target_h / h, target_w / w)
    new_h, new_w = int(round(h * ratio)), int(round(w * ratio))
    
    # 리사이즈
    resized = cv2.resize(image, (new_w, new_h))
    
    # 🔧 패딩 계산 (Ultralytics 방식)
    dw, dh = target_w - new_w, target_h - new_h  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    
    # 정수 패딩 값 계산
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    # 패딩 추가
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    
    # 정규화 [0, 255] -> [0, 1]
    normalized = padded.astype(np.float32) / 255.0
    
    # HWC -> CHW
    transposed = normalized.transpose(2, 0, 1)
    
    # 🔧 스케일링 정보 반환 (scale_boxes용)
    ratio_pad = (ratio, (left, top))
    
    # 🔧 디버깅: preprocess_image 정보 출력
    print(f"🔍 preprocess_image debug:")
    print(f"  original_shape: {original_shape}")
    print(f"  ratio: {ratio}")
    print(f"  new_h, new_w: {new_h}, {new_w}")
    print(f"  dw, dh: {dw}, {dh}")
    print(f"  top, bottom, left, right: {top}, {bottom}, {left}, {right}")
    print(f"  final ratio_pad: {ratio_pad}")
    
    return transposed, original_shape, ratio_pad

def xyxy2xywh_numpy(boxes):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) to (x, y, width, height) format.
    
    Args:
        boxes (np.ndarray): Input boxes in xyxy format
    
    Returns:
        np.ndarray: Boxes in xywh format
    """
    boxes_xywh = boxes.copy()
    boxes_xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2  # x center
    boxes_xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2  # y center
    boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]        # width
    boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]        # height
    return boxes_xywh

def xywh2xyxy_numpy(boxes):
    """
    Convert bounding box coordinates from (x, y, width, height) to (x1, y1, x2, y2) format.
    
    Args:
        boxes (np.ndarray): Input boxes in xywh format
    
    Returns:
        np.ndarray: Boxes in xyxy format
    """
    boxes_xyxy = boxes.copy()
    half_w = boxes[:, 2] / 2
    half_h = boxes[:, 3] / 2
    boxes_xyxy[:, 0] = boxes[:, 0] - half_w  # x1
    boxes_xyxy[:, 1] = boxes[:, 1] - half_h  # y1
    boxes_xyxy[:, 2] = boxes[:, 0] + half_w  # x2
    boxes_xyxy[:, 3] = boxes[:, 1] + half_h  # y2
    return boxes_xyxy

def scale_boxes_numpy(img1_shape, boxes, img0_shape, ratio_pad=None):
    """
    Numpy 버전의 scale_boxes - dual_stream_inference.py와 동일한 방식
    
    Args:
        img1_shape (tuple): 모델 입력 이미지 크기 (height, width)
        boxes (np.ndarray): 박스 좌표 [N, 4] (x1, y1, x2, y2)
        img0_shape (tuple): 원본 이미지 크기 (height, width)
        ratio_pad (tuple): 무시됨 - dual_stream_inference.py 방식 사용
    
    Returns:
        np.ndarray: 스케일링된 박스 좌표
    """
    # 🔧 dual_stream_inference.py와 동일한 방식
    target_size = img1_shape[0]  # 640
    original_height, original_width = img0_shape
    scale = min(target_size / original_width, target_size / original_height)
    
    # 패딩 계산 (dual_stream_inference.py 방식)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    delta_w = target_size - new_width
    delta_h = target_size - new_height
    left = delta_w // 2
    top = delta_h // 2
    
    # 🔧 디버깅용 출력
    print(f"DEBUG scale_boxes_numpy (dual_stream_inference.py style):")
    print(f"  img1_shape: {img1_shape}")
    print(f"  img0_shape: {img0_shape}")
    print(f"  scale: {scale}")
    print(f"  new_width: {new_width}, new_height: {new_height}")
    print(f"  delta_w: {delta_w}, delta_h: {delta_h}")
    print(f"  left: {left}, top: {top}")
    if len(boxes) > 0:
        print(f"  input boxes[0]: {boxes[0]}")
    
    # 박스 복사
    boxes_scaled = boxes.copy()
    
    # 🔧 dual_stream_inference.py와 동일한 패딩 제거 로직
    boxes_scaled[:, 0] = np.maximum(0, boxes_scaled[:, 0] - left)   # x1
    boxes_scaled[:, 1] = np.maximum(0, boxes_scaled[:, 1] - top)    # y1
    boxes_scaled[:, 2] = np.maximum(0, boxes_scaled[:, 2] - left)   # x2
    boxes_scaled[:, 3] = np.maximum(0, boxes_scaled[:, 3] - top)    # y2
    
    if len(boxes_scaled) > 0:
        print(f"  after padding removal[0]: {boxes_scaled[0]}")
    
    # 스케일링 (dual_stream_inference.py 방식)
    boxes_scaled[:, 0] = boxes_scaled[:, 0] / scale  # x1
    boxes_scaled[:, 1] = boxes_scaled[:, 1] / scale  # y1
    boxes_scaled[:, 2] = boxes_scaled[:, 2] / scale  # x2
    boxes_scaled[:, 3] = boxes_scaled[:, 3] / scale  # y2
    
    if len(boxes_scaled) > 0:
        print(f"  after scaling[0]: {boxes_scaled[0]}")
    
    # 이미지 경계 내로 클립
    boxes_scaled[:, [0, 2]] = np.clip(boxes_scaled[:, [0, 2]], 0, img0_shape[1])  # x coordinates
    boxes_scaled[:, [1, 3]] = np.clip(boxes_scaled[:, [1, 3]], 0, img0_shape[0])  # y coordinates
    
    if len(boxes_scaled) > 0:
        print(f"  final result[0]: {boxes_scaled[0]}")
    
    return boxes_scaled

def postprocess_detections(outputs, conf_threshold=0.5, iou_threshold=0.45):
    """
    YOLO 출력을 후처리하여 detection 결과 추출
    """
    detections = outputs[0][0].T  # [8400, 32]
    
    # ===================== 💡 수정된 부분 시작 =====================
    
    # 원본 cx, cy, w, h 값을 별도 변수에 저장하여 원본 데이터가 바뀌는 것을 방지합니다.
    cx = detections[:, 0]
    cy = detections[:, 1]
    w = detections[:, 2]
    h = detections[:, 3]
    
    # 저장된 원본 값을 사용하여 x1, y1, x2, y2를 정확하게 계산합니다.
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    
    # 계산된 좌표들로 새로운 boxes 배열을 생성합니다.
    boxes = np.stack((x1, y1, x2, y2), axis=1)

    # ===================== 💡 수정된 부분 끝 =====================

    # 신뢰도 및 클래스
    confidences = detections[:, 4:]  # [8400, 28]
    class_ids = np.argmax(confidences, axis=1)
    max_confidences = np.max(confidences, axis=1)
    
    # 신뢰도 필터링
    valid_indices = max_confidences > conf_threshold
    
    if not np.any(valid_indices):
        return []
    
    filtered_boxes = boxes[valid_indices]
    filtered_confidences = max_confidences[valid_indices]
    filtered_class_ids = class_ids[valid_indices]
    
    # NMS (간단한 버전)
    results = []
    for i in range(len(filtered_boxes)):
        box = filtered_boxes[i]
        conf = filtered_confidences[i]
        class_id = filtered_class_ids[i]
        
        results.append((box[0], box[1], box[2], box[3], conf, int(class_id)))
    
    return results

def visualize_dual_results(wide_image_path, narrow_image_path, detections, wide_original_shape, wide_ratio_pad, save_path=None):
    """
    Dual-camera Detection 결과를 시각화 (preprocess_image에서 계산한 ratio_pad 사용)
    
    Args:
        wide_image_path (str): Wide 이미지 경로
        narrow_image_path (str): Narrow 이미지 경로
        detections (list): Detection 결과
        wide_original_shape (tuple): Wide 이미지 원본 크기 (H, W)
        wide_ratio_pad (tuple): preprocess_image에서 계산한 ratio_pad 정보
        save_path (str): 저장 경로 (optional)
    """
    # 이미지 로드
    wide_image = cv2.imread(str(wide_image_path))
    wide_image = cv2.cvtColor(wide_image, cv2.COLOR_BGR2RGB)
    
    narrow_image = cv2.imread(str(narrow_image_path))
    narrow_image = cv2.cvtColor(narrow_image, cv2.COLOR_BGR2RGB)
    
    # 🔧 실제 이미지 크기 확인
    print(f"🔍 Image size verification:")
    print(f"  Wide image actual shape: {wide_image.shape}")
    print(f"  Expected wide_original_shape: {wide_original_shape}")
    print(f"  Narrow image actual shape: {narrow_image.shape}")
    
    # 실제 이미지 크기와 전달받은 크기가 다른 경우 경고
    if wide_image.shape[:2] != wide_original_shape:
        print(f"⚠️  WARNING: Image shape mismatch!")
        print(f"  Using actual image shape: {wide_image.shape[:2]}")
        wide_original_shape = wide_image.shape[:2]
    
    # 서브플롯 생성
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Wide 이미지 표시
    ax1.imshow(wide_image)
    ax1.set_title(f'Wide FOV - {Path(wide_image_path).name}', fontsize=14)
    
    # Narrow 이미지 표시
    ax2.imshow(narrow_image)
    ax2.set_title(f'Narrow FOV - {Path(narrow_image_path).name}', fontsize=14)
    
    # 🔧 Detection 시각화
    if detections:
        print(f"🔍 Visualizing detection results:")
        
        # Detection 좌표를 numpy 배열로 변환
        detection_boxes = np.array([[x1, y1, x2, y2] for x1, y1, x2, y2, _, _ in detections])
        
        print(f"📍 Original detection boxes (640x640 coordinates):")
        for i, box in enumerate(detection_boxes[:3]):  # 처음 3개만
            w, h = box[2] - box[0], box[3] - box[1]
            print(f"  Box {i+1}: x1={box[0]:.1f}, y1={box[1]:.1f}, x2={box[2]:.1f}, y2={box[3]:.1f} (w={w:.1f}, h={h:.1f})")
        
        # scale_boxes_numpy 사용 (파란색) - val.py 스타일
        scaled_boxes = scale_boxes_numpy(
            img1_shape=(640, 640),           # 모델 입력 크기
            boxes=detection_boxes,           # Detection 박스들
            img0_shape=wide_original_shape,  # 원본 이미지 크기
            ratio_pad=None                   # val.py처럼 재계산
        )
        
        for i, (detection, scaled_box) in enumerate(zip(detections[:5], scaled_boxes[:5])):  # 처음 5개만
            x1, y1, x2, y2, conf, class_id = detection
            x1_scaled, y1_scaled, x2_scaled, y2_scaled = scaled_box
            
            # 파란색 박스 그리기 (letterbox 스케일링)
            rect_scaled = patches.Rectangle(
                (x1_scaled, y1_scaled), 
                x2_scaled - x1_scaled, 
                y2_scaled - y1_scaled,
                linewidth=2, edgecolor='blue', facecolor='none'
            )
            ax1.add_patch(rect_scaled)
            
            # 라벨 추가
            ax1.text(x1_scaled, y1_scaled + 15, f'LETTERBOX {class_id}: {conf:.3f}',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='blue', alpha=0.7),
                    fontsize=8, color='white')
        
        print(f"🎨 Blue boxes: dual_stream_inference.py-style letterbox scaling")
    
    ax1.axis('off')
    ax2.axis('off')
    
    # 전체 제목
    fig.suptitle(f'Dual Stream YOLO Detection Results - {len(detections)} detections', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📁 Result saved to: {save_path}")
    
    plt.show()

def test_dual_onnx_inference(wide_image_path, narrow_image_path, onnx_path, conf_threshold=0.5, result_dir=None):
    """
    듀얼 스트림 ONNX 모델로 추론 테스트
    
    Args:
        wide_image_path (str): Wide 이미지 경로
        narrow_image_path (str): Narrow 이미지 경로
        onnx_path (str): ONNX 모델 경로
        conf_threshold (float): 신뢰도 임계값
        result_dir (str): 결과 저장 폴더
    """
    print(f"🚀 Testing Dual Stream ONNX Inference")
    print(f"📷 Wide Image: {wide_image_path}")
    print(f"📷 Narrow Image: {narrow_image_path}")
    print(f"🧠 Model: {onnx_path}")
    print("=" * 80)
    
    # 이미지 존재 확인
    if not Path(wide_image_path).exists():
        print(f"❌ Wide image not found: {wide_image_path}")
        return
    if not Path(narrow_image_path).exists():
        print(f"❌ Narrow image not found: {narrow_image_path}")
        return
    
    # ONNX 모델 존재 확인
    if not Path(onnx_path).exists():
        print(f"❌ ONNX model not found: {onnx_path}")
        return
    
    try:
        # 1. 이미지 전처리
        print("📂 Loading and preprocessing images...")
        wide_image, wide_original_shape, wide_ratio_pad = preprocess_image(wide_image_path)
        narrow_image, narrow_original_shape, narrow_ratio_pad = preprocess_image(narrow_image_path)
        print(f"   - Wide original shape: {wide_original_shape}")
        print(f"   - Narrow original shape: {narrow_original_shape}")
        print(f"   - Wide ratio_pad: {wide_ratio_pad}")
        print(f"   - Processed shapes: {wide_image.shape}, {narrow_image.shape}")
        
        # 2. 듀얼 스트림 입력 구성
        print(f"🔄 Creating dual stream input...")
        dual_input = np.stack([wide_image, narrow_image], axis=0)[None, ...]  # [1, 2, 3, 640, 640]
        print(f"   - Dual stream shape: {dual_input.shape}")
        
        # 3. ONNX 모델 로드
        print("🧠 Loading ONNX model...")
        session = ort.InferenceSession(onnx_path)
        
        # 모델 정보 출력
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()
        print(f"   - Input: {input_info.name} {input_info.shape}")
        print(f"   - Outputs: {len(output_info)} outputs")
        for i, out in enumerate(output_info):
            print(f"     - Output {i}: {out.name} {out.shape}")
        
        # 4. 추론 실행
        print("⚡ Running inference...")
        start_time = time.time()
        
        outputs = session.run(None, {'images': dual_input})
        
        inference_time = time.time() - start_time
        print(f"   - Inference time: {inference_time*1000:.1f} ms")
        
        # 5. 결과 후처리
        print("📊 Post-processing results...")
        detections = postprocess_detections(outputs, conf_threshold=conf_threshold)
        print(f"   - Found {len(detections)} detections")
        
        if detections:
            print("\n🎯 Detection Results:")
            for i, (x1, y1, x2, y2, conf, class_id) in enumerate(detections):
                print(f"   {i+1}. Class {class_id}: {conf:.3f} | Box: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
        
        # 6. 결과 시각화 및 저장
        print("\n🖼️  Visualizing results...")
        if result_dir:
            os.makedirs(result_dir, exist_ok=True)
            result_filename = f"{Path(wide_image_path).stem}_dual_result.png"
            result_path = Path(result_dir) / result_filename
        else:
            result_path = Path(wide_image_path).parent / f"{Path(wide_image_path).stem}_dual_result.png"
        
        # 🔧 preprocess_image에서 계산한 ratio_pad 사용
        visualize_dual_results(
            wide_image_path, narrow_image_path, detections, 
            wide_original_shape, wide_ratio_pad, save_path=result_path
        )
        
        # 7. 성능 요약
        print(f"\n📈 Performance Summary:")
        print(f"   - Wide image: {Path(wide_image_path).name}")
        print(f"   - Narrow image: {Path(narrow_image_path).name}")
        print(f"   - Inference time: {inference_time*1000:.1f} ms")
        print(f"   - Detections: {len(detections)}")
        print(f"   - Confidence threshold: {conf_threshold}")
        
        return outputs, detections
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    parser = argparse.ArgumentParser(description='Test Dual Stream ONNX Model with Real Dual-Camera Images')
    parser.add_argument('--wide-dir', type=str, 
                       default='/home/byounggun/ultralytics/swm_total/images',
                       help='Wide images directory')
    parser.add_argument('--narrow-dir', type=str,
                       default='/home/byounggun/ultralytics/swm_total/narrow_images',
                       help='Narrow images directory')
    parser.add_argument('--image', type=str, default=None,
                       help='Specific image name to test (optional, random if not specified)')
    parser.add_argument('--model', type=str,
                       default='/home/byounggun/ultralytics/runs/train/exp108/weights/best_dual_stream.onnx',
                       help='Path to ONNX model')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--result-dir', type=str,
                       default='/home/byounggun/ultralytics/swm_total/inference_onnx',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # 이미지 쌍 찾기
    try:
        image_pairs = find_image_pairs(args.wide_dir, args.narrow_dir)
    except Exception as e:
        print(f"❌ Error finding image pairs: {e}")
        return
    
    # 테스트할 이미지 선택
    if args.image:
        # 특정 이미지 지정
        selected_pair = None
        for wide_path, narrow_path in image_pairs:
            if wide_path.name == args.image:
                selected_pair = (wide_path, narrow_path)
                break
        
        if selected_pair is None:
            print(f"❌ Specified image not found: {args.image}")
            print(f"Available images: {[p[0].name for p in image_pairs[:5]]}...")
            return
        
        wide_image_path, narrow_image_path = selected_pair
        print(f"🎯 Testing specific image: {args.image}")
        
    else:
        # 랜덤 선택
        wide_image_path, narrow_image_path = random.choice(image_pairs)
        print(f"🎲 Randomly selected: {wide_image_path.name}")
    
    # 테스트 실행
    outputs, detections = test_dual_onnx_inference(
        wide_image_path=wide_image_path,
        narrow_image_path=narrow_image_path,
        onnx_path=args.model,
        conf_threshold=args.conf,
        result_dir=args.result_dir
    )
    
    if outputs is not None:
        print("\n✅ Test completed successfully!")
        print(f"📁 Results saved to: {args.result_dir}")
    else:
        print("\n❌ Test failed!")

if __name__ == "__main__":
    main() 