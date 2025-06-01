import os
import random
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics.utils.ops import non_max_suppression

# === 1. Import DualYOLOv8 ===
from yolov8_dual import DualYOLOv8

# === 2. Set paths and select common image ===
wide_dir = "/home/byounggun/ultralytics/traffic_train/wide/images"
narrow_dir = "/home/byounggun/ultralytics/traffic_train/narrow/images"
img_w, img_h = 1920, 1080

wide_files = set([f for f in os.listdir(wide_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
narrow_files = set([f for f in os.listdir(narrow_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
common_files = list(wide_files & narrow_files)
assert len(common_files) > 0, "No common image files found in both folders."

filename = random.choice(common_files)
wide_path = os.path.join(wide_dir, filename)
narrow_path = os.path.join(narrow_dir, filename)
print(f"Selected file: {filename}")

# === 3. 모델 로드 (카메라 파라미터 없이) ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pt_path = '/home/byounggun/ultralytics/runs/dual/exp/dual_yolov8_epoch40.pt'  # 실제 pt 경로로 수정
from yolov8_dual import DualYOLOv8

# 커스텀 DualYOLOv8 구조는 코드로 정의되어 있으므로, pt 파일(state_dict)만으로 추론
assert pt_path.endswith('.pt') and os.path.exists(pt_path), f"{pt_path} not found"
model = DualYOLOv8(yolo_weights_path=None, img_size=640)  # 구조만 생성 (코드 기반)
state = torch.load(pt_path, map_location=device)
model.load_state_dict(state)
model.to(device)
model.eval()

# === 4. 이미지 전처리 ===
def preprocess(img_path, img_size=640, device='cpu'):
    img = Image.open(img_path).convert("RGB").resize((img_size, img_size))
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img.to(device)

wide_tensor = preprocess(wide_path, 640, device)
narrow_tensor = preprocess(narrow_path, 640, device)

# === 5. 추론 ===
with torch.no_grad():
    pred = model(wide_tensor, narrow_tensor)
    pred = pred if isinstance(pred, torch.Tensor) else pred[0]
    from ultralytics.utils.ops import non_max_suppression
    results = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, max_det=100)

# === 6. 시각화 및 결과 출력 ===
class_names = model.names if hasattr(model, 'names') and model.names is not None else [str(i) for i in range(model.nc)]
orig_img = cv2.imread(wide_path)
orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
H, W, _ = orig_img.shape

for det in results:
    if det is not None and len(det):
        det = det.cpu().numpy()
        boxes = det[:, :4]
        scores = det[:, 4]
        cls_ids = det[:, 5].astype(int)
        # 640 기준 bbox를 원본 크기로 변환
        scale_x = W / 640
        scale_y = H / 640
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        for box, score, cls_id in zip(boxes, scores, cls_ids):
            x1, y1, x2, y2 = box.astype(int)
            label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(orig_img, f'{label} {score:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            print(f"Detected: {label} | Confidence: {score:.2f} | Box: [{x1}, {y1}, {x2}, {y2}]")
    else:
        print("No detections.")

plt.figure(figsize=(12,8))
plt.imshow(orig_img)
plt.title(f'Inference result: {filename}')
plt.axis('off')
plt.show()

# Path to your trained dual model .pt file
pt_path = 'dual_yolov8_best.pt'  # <-- EDIT THIS to your actual checkpoint
model = DualYOLOv8(
    yolo_weights_path=pt_path,
    wide_K=wide_K, wide_P=wide_P, narrow_K=narrow_K, narrow_P=narrow_P,
    img_w=img_w, img_h=img_h, img_size=640
)
model.to(device)
model.eval()

# === 5. Image preprocessing ===
def load_and_preprocess_image(image_path, img_size=640, device='cpu'):
    img = Image.open(image_path).convert("RGB").resize((img_size, img_size))
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    return img.to(device)

wide_tensor = load_and_preprocess_image(wide_path, img_size=640, device=device)
narrow_tensor = load_and_preprocess_image(narrow_path, img_size=640, device=device)

# === 6. Inference ===
with torch.no_grad():
    pred = model(wide_tensor, narrow_tensor)
    # pred shape: (B, num_anchors, nc+5) or (B, num_boxes, nc+5)
    # For YOLOv8, output is usually (B, num_boxes, nc+5)

    # Apply NMS (assume batch size 1)
    pred = pred if isinstance(pred, torch.Tensor) else pred[0]
    results = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, max_det=100)

# === 7. Visualization and print results ===
# Get class names from model
class_names = model.names if hasattr(model, 'names') and model.names is not None else [str(i) for i in range(model.nc)]

orig_img = cv2.imread(wide_path)
orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
H, W, _ = orig_img.shape

for det in results:
    if det is not None and len(det):
        # Rescale boxes from 640x640 to original image size
        det = det.cpu().numpy()
        boxes = det[:, :4]
        scores = det[:, 4]
        cls_ids = det[:, 5].astype(int)
        # xyxy to original scale
        scale_x = W / 640
        scale_y = H / 640
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        for box, score, cls_id in zip(boxes, scores, cls_ids):
            x1, y1, x2, y2 = box.astype(int)
            label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(orig_img, f'{label} {score:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            print(f"Detected: {label} | Confidence: {score:.2f} | Box: [{x1}, {y1}, {x2}, {y2}]")
    else:
        print("No detections.")

plt.figure(figsize=(12,8))
plt.imshow(orig_img)
plt.title(f'Inference result: {filename}')
plt.axis('off')
plt.show()
