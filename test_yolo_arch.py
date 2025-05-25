"""
Test for YOLOv8 backbone and detection head extraction.
"""
import torch
from ultralytics import YOLO

def test_yolo_backbone_head(yolo_weights_path='yolov8s.pt'):
    print(f"Loading YOLOv8 model from: {yolo_weights_path}")
    yolo_model_full = YOLO(yolo_weights_path)
    model = yolo_model_full.model
    print(f"Type of model: {type(model)}")
    print(f"model.model type: {type(model.model)}")
    print(f"model.model length: {len(model.model)}")
    # Detect 레이어 위치 찾기
    detect_idx = None
    for idx, m in enumerate(model.model):
        if m.__class__.__name__ == 'Detect':
            detect_idx = idx
            break
    assert detect_idx is not None, "Detect layer not found in model!"
    backbone = model.model[:detect_idx]
    detection_head = model.model[detect_idx:]
    print(f"Backbone (up to Detect, idx={detect_idx}): {backbone}")
    print(f"Detection head (from Detect): {detection_head}")
    # Assertions
    assert hasattr(model, 'model'), "model does not have attribute 'model'"
    assert isinstance(backbone, torch.nn.Sequential), "Backbone is not a nn.Sequential"
    assert isinstance(detection_head, torch.nn.Sequential), "Detection head is not a nn.Sequential"
    print("Backbone and detection head extraction (multi-layer) successful.")

if __name__ == "__main__":
    # You can specify a different weights path if needed
    test_yolo_backbone_head()
