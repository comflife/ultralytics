#!/usr/bin/env python3
"""
Dual Stream ONNX Model Test Script
Tests the exported dual-stream YOLO model with real images
"""

import numpy as np
import cv2
import onnxruntime as ort
from pathlib import Path
import argparse
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def preprocess_image(image_path, target_size=(640, 640)):
    """
    이미지를 ONNX 모델 입력에 맞게 전처리
    
    Args:
        image_path (str): 이미지 파일 경로
        target_size (tuple): 타겟 크기 (height, width)
    
    Returns:
        np.ndarray: 전처리된 이미지 [3, H, W]
        tuple: 원본 이미지 크기 (height, width)
    """
    # 이미지 로드
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # BGR -> RGB 변환
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_shape = image.shape[:2]  # (H, W)
    
    # 리사이즈 (letterbox padding)
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    # 비율 계산
    ratio = min(target_h / h, target_w / w)
    new_h, new_w = int(h * ratio), int(w * ratio)
    
    # 리사이즈
    resized = cv2.resize(image, (new_w, new_h))
    
    # 패딩 추가
    padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)  # gray padding
    
    # 중앙에 배치
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    # 정규화 [0, 255] -> [0, 1]
    normalized = padded.astype(np.float32) / 255.0
    
    # HWC -> CHW
    transposed = normalized.transpose(2, 0, 1)
    
    return transposed, original_shape

def create_narrow_fov(wide_image, narrow_bbox=None):
    """
    Wide FOV 이미지에서 Narrow FOV 시뮬레이션
    
    Args:
        wide_image (np.ndarray): Wide 이미지 [3, H, W]
        narrow_bbox (dict): Narrow FOV 영역 정보
    
    Returns:
        np.ndarray: Narrow FOV 이미지 [3, H, W]
    """
    if narrow_bbox is None:
        # 기본 narrow bbox (중앙 영역)
        narrow_bbox = {
            'center_x': 0.5,
            'center_y': 0.5, 
            'width': 0.4,
            'height': 0.4
        }
    
    _, H, W = wide_image.shape
    
    # YOLO bbox -> 픽셀 좌표
    center_x = int(narrow_bbox['center_x'] * W)
    center_y = int(narrow_bbox['center_y'] * H)
    bbox_w = int(narrow_bbox['width'] * W)
    bbox_h = int(narrow_bbox['height'] * H)
    
    # 크롭 영역 계산
    x1 = max(0, center_x - bbox_w // 2)
    y1 = max(0, center_y - bbox_h // 2)
    x2 = min(W, x1 + bbox_w)
    y2 = min(H, y1 + bbox_h)
    
    # 크롭
    cropped = wide_image[:, y1:y2, x1:x2]
    
    # 원본 크기로 리사이즈
    # CHW -> HWC
    cropped_hwc = cropped.transpose(1, 2, 0)
    resized_hwc = cv2.resize(cropped_hwc, (W, H))
    # HWC -> CHW
    narrow_image = resized_hwc.transpose(2, 0, 1)
    
    return narrow_image

def postprocess_detections(outputs, conf_threshold=0.5, iou_threshold=0.45):
    """
    YOLO 출력을 후처리하여 detection 결과 추출
    
    Args:
        outputs (list): ONNX 모델 출력
        conf_threshold (float): 신뢰도 임계값
        iou_threshold (float): NMS IoU 임계값
    
    Returns:
        list: Detection 결과 [(x1, y1, x2, y2, conf, class_id), ...]
    """
    detections = outputs[0]  # [1, 32, 8400]
    detections = detections[0]  # [32, 8400]
    
    # 형태 변환: [32, 8400] -> [8400, 32]
    detections = detections.T
    
    # 박스 좌표 [cx, cy, w, h] -> [x1, y1, x2, y2]
    boxes = detections[:, :4]
    boxes[:, 0] = detections[:, 0] - detections[:, 2] / 2  # x1 = cx - w/2
    boxes[:, 1] = detections[:, 1] - detections[:, 3] / 2  # y1 = cy - h/2
    boxes[:, 2] = detections[:, 0] + detections[:, 2] / 2  # x2 = cx + w/2
    boxes[:, 3] = detections[:, 1] + detections[:, 3] / 2  # y2 = cy + h/2
    
    # 신뢰도 및 클래스
    confidences = detections[:, 4:]  # [8400, 28] (28 classes)
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

def visualize_results(image_path, detections, save_path=None):
    """
    Detection 결과를 시각화
    
    Args:
        image_path (str): 원본 이미지 경로
        detections (list): Detection 결과
        save_path (str): 저장 경로 (optional)
    """
    # 원본 이미지 로드
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    
    # Detection 박스 그리기
    for detection in detections:
        x1, y1, x2, y2, conf, class_id = detection
        
        # 좌표를 원본 이미지 크기에 맞게 스케일링
        h, w = image.shape[:2]
        x1 = x1 * w / 640
        y1 = y1 * h / 640
        x2 = x2 * w / 640
        y2 = y2 * h / 640
        
        # 박스 그리기
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        
        # 라벨 추가
        ax.text(x1, y1 - 5, f'Class {class_id}: {conf:.3f}',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7),
                fontsize=10, color='white')
    
    ax.set_title(f'Dual Stream YOLO Detection Results\n{len(detections)} detections found')
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📁 Result saved to: {save_path}")
    
    plt.show()

def test_dual_onnx_inference(image_path, onnx_path, method='duplicate', conf_threshold=0.5):
    """
    듀얼 스트림 ONNX 모델로 추론 테스트
    
    Args:
        image_path (str): 테스트 이미지 경로
        onnx_path (str): ONNX 모델 경로
        method (str): 듀얼 스트림 생성 방법 ('duplicate' or 'crop')
        conf_threshold (float): 신뢰도 임계값
    """
    print(f"🚀 Testing Dual Stream ONNX Inference")
    print(f"📷 Image: {image_path}")
    print(f"🧠 Model: {onnx_path}")
    print(f"🔧 Method: {method}")
    print("=" * 60)
    
    # 이미지 존재 확인
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    # ONNX 모델 존재 확인
    if not Path(onnx_path).exists():
        print(f"❌ ONNX model not found: {onnx_path}")
        return
    
    try:
        # 1. 이미지 전처리
        print("📂 Loading and preprocessing image...")
        wide_image, original_shape = preprocess_image(image_path)
        print(f"   - Original shape: {original_shape}")
        print(f"   - Processed shape: {wide_image.shape}")
        
        # 2. 듀얼 스트림 생성
        print(f"🔄 Creating dual stream input using '{method}' method...")
        if method == 'duplicate':
            # 같은 이미지 두 번 사용
            narrow_image = wide_image.copy()
            print("   - Using same image for both streams")
        elif method == 'crop':
            # 중앙 크롭으로 narrow FOV 시뮬레이션
            narrow_image = create_narrow_fov(wide_image)
            print("   - Created narrow FOV by cropping center region")
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # 3. 듀얼 스트림 입력 구성
        dual_input = np.stack([wide_image, narrow_image], axis=0)[None, ...]  # [1, 2, 3, 640, 640]
        print(f"   - Dual stream shape: {dual_input.shape}")
        
        # 4. ONNX 모델 로드
        print("🧠 Loading ONNX model...")
        session = ort.InferenceSession(onnx_path)
        
        # 모델 정보 출력
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()
        print(f"   - Input: {input_info.name} {input_info.shape}")
        print(f"   - Outputs: {len(output_info)} outputs")
        for i, out in enumerate(output_info):
            print(f"     - Output {i}: {out.name} {out.shape}")
        
        # 5. 추론 실행
        print("⚡ Running inference...")
        start_time = time.time()
        
        outputs = session.run(None, {'images': dual_input})
        
        inference_time = time.time() - start_time
        print(f"   - Inference time: {inference_time*1000:.1f} ms")
        
        # 6. 결과 후처리
        print("📊 Post-processing results...")
        detections = postprocess_detections(outputs, conf_threshold=conf_threshold)
        print(f"   - Found {len(detections)} detections")
        
        if detections:
            print("\n🎯 Detection Results:")
            for i, (x1, y1, x2, y2, conf, class_id) in enumerate(detections):
                print(f"   {i+1}. Class {class_id}: {conf:.3f} | Box: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
        
        # 7. 결과 시각화
        print("\n🖼️  Visualizing results...")
        result_path = Path(image_path).parent / f"{Path(image_path).stem}_dual_result.png"
        visualize_results(image_path, detections, save_path=result_path)
        
        # 8. 성능 요약
        print(f"\n📈 Performance Summary:")
        print(f"   - Image: {Path(image_path).name}")
        print(f"   - Method: {method}")
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
    parser = argparse.ArgumentParser(description='Test Dual Stream ONNX Model')
    parser.add_argument('--image', type=str, 
                       default='/home/byounggun/ultralytics/swm/images/scene0002_20250422_07540461.jpg',
                       help='Path to test image')
    parser.add_argument('--model', type=str,
                       default='/home/byounggun/ultralytics/runs/train/exp225/weights/best_dual_stream.onnx',
                       help='Path to ONNX model')
    parser.add_argument('--method', choices=['duplicate', 'crop'], default='crop',
                       help='Method to create dual stream input')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold')
    
    args = parser.parse_args()
    
    # 테스트 실행
    outputs, detections = test_dual_onnx_inference(
        image_path=args.image,
        onnx_path=args.model,
        method=args.method,
        conf_threshold=args.conf
    )
    
    if outputs is not None:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!")

if __name__ == "__main__":
    main() 