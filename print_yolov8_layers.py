# print_yolov8_layers.py
"""
YOLOv8 (ultralytics) backbone/head 전체 레이어를 순서대로 출력하는 유틸리티
- yolov8_dual.py, yolov8.py, ultralytics 모델 모두 사용 가능
- 모델 구조만 출력 (가중치 불필요)
"""
from ultralytics import YOLO
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolov8s.pt", help="YOLOv8 가중치 또는 yaml 경로")
    args = parser.parse_args()

    # 모델 로드 (가중치 or 구조)
    model = YOLO(args.weights)
    nn_model = model.model  # ultralytics nn.Module

    print("\n[YOLOv8 전체 레이어 구조 출력]")
    print(f"모델: {args.weights}")
    print(f"총 레이어 수: {len(list(nn_model.modules()))}")
    print("----------------------------------------")
    for i, layer in enumerate(nn_model.model):
        print(f"[Layer {i:2d}] {layer.__class__.__name__} | {layer}")
    print("----------------------------------------")
    print("[head 모듈(Detection head 등)]")
    for i, layer in enumerate(nn_model.head):
        print(f"[Head {i:2d}] {layer.__class__.__name__} | {layer}")
    print("----------------------------------------")
    print("[backbone 모듈]")
    for i, layer in enumerate(nn_model.backbone):
        print(f"[Backbone {i:2d}] {layer.__class__.__name__} | {layer}")
    print("----------------------------------------")
    print("[전체 nn.Module 트리]")
    print(nn_model)
