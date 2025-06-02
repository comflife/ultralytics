# print_dualyolo_layers.py
"""
DualYOLOv8 모델 전체 레이어 구조를 출력하는 유틸리티
- yolov8_dual.py에서 직접 import
- backbone, neck/head 포함
"""
import argparse
from yolov8_dual import DualYOLOv8

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=None, help="가중치(pt) 또는 구조(yaml) 경로")
    parser.add_argument("--img_size", type=int, default=640)
    args = parser.parse_args()

    # 더미 카메라 파라미터
    import numpy as np
    K = np.eye(3, dtype=np.float32)
    P = np.zeros((3, 4), dtype=np.float32)

    # 모델 로드
    model = DualYOLOv8(
        yolo_weights_path=args.weights,
        wide_K=K, wide_P=P,
        narrow_K=K, narrow_P=P,
        img_w=args.img_size, img_h=args.img_size, img_size=args.img_size
    )
    nn_model = model.model  # nn.ModuleList (YOLOv8 backbone+head)

    print("\n[DualYOLOv8 전체 레이어 구조 출력]")
    print(f"모델: {args.weights}")
    print(f"총 레이어 수: {len(nn_model)}")
    print("----------------------------------------")
    for i, layer in enumerate(nn_model):
        print(f"[Layer {i:2d}] {layer.__class__.__name__} | {layer}")
    print("----------------------------------------")
    print("[전체 nn.Module 트리]")
    # print(model)
