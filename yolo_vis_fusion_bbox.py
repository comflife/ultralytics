import os
import random
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from yolov8_dual import DualYOLOv8

# === 1. 경로 및 파일 선택 ===
wide_dir = "/home/byounggun/ultralytics/traffic_train/wide/images"
narrow_dir = "/home/byounggun/ultralytics/traffic_train/narrow/images"
img_w, img_h = 1920, 1080

wide_files = set([f for f in os.listdir(wide_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
narrow_files = set([f for f in os.listdir(narrow_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
common_files = list(wide_files & narrow_files)
assert len(common_files) > 0, "두 폴더에 공통된 이미지 파일이 없습니다."

filename = random.choice(common_files)
wide_path = os.path.join(wide_dir, filename)
narrow_path = os.path.join(narrow_dir, filename)
print(f"선택된 파일: {filename}")

# === 2. 카메라 파라미터 (실제 bbox에는 사용하지 않음) ===
wide_K = np.array([[559.258761, 0, 928.108242],
                   [0, 565.348774, 518.787048],
                   [0, 0, 1]])
wide_P = np.array([[535.711792, 0, 924.086569, 0],
                   [0, 558.997375, 510.222325, 0],
                   [0, 0, 1, 0]])
narrow_K = np.array([[2651.127798, 0, 819.397071],
                     [0, 2635.360938, 896.163803],
                     [0, 0, 1]])
narrow_P = np.array([[2407.709780, 0, 801.603047, 0],
                     [0, 2544.697607, 897.250521, 0],
                     [0, 0, 1, 0]])

# === 3. DualYOLOv8 모델 로드 ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DualYOLOv8(
    yolo_weights_path='yolov8s.pt',
    wide_K=wide_K, wide_P=wide_P,
    narrow_K=narrow_K, narrow_P=narrow_P,
    img_w=img_w, img_h=img_h, img_size=640
).to(device)
model.eval()

def load_and_preprocess_image(image_path, img_size=640, device='cpu'):
    img = Image.open(image_path).convert("RGB").resize((img_size, img_size))
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    return img.to(device)

# === 4. wide/narrow 이미지 전처리 ===
wide_tensor = load_and_preprocess_image(wide_path, img_size=640, device=device)
narrow_tensor = load_and_preprocess_image(narrow_path, img_size=640, device=device)

# === 5. 시각화 함수 ===
def show_feature_map(feat, orig_path, box, title="", img_size=640):
    fmap = feat.mean(0)
    fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
    fmap = (fmap * 255).cpu().numpy().astype(np.uint8)
    fmap = cv2.resize(fmap, (img_size, img_size))
    heatmap = cv2.applyColorMap(fmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    orig = Image.open(orig_path).convert("RGB").resize((img_size, img_size))
    orig_np = np.array(orig)
    overlay = (0.5 * heatmap + 0.5 * orig_np).astype(np.uint8)
    if box.max() > img_size:
        scale_x = img_size / img_w
        scale_y = img_size / img_h
        box_scaled = np.stack([[int(x*scale_x), int(y*scale_y)] for x, y in box], axis=0)
    else:
        box_scaled = box
    for img in [heatmap, overlay]:
        cv2.polylines(img, [box_scaled.reshape(-1, 1, 2)], isClosed=True, color=(255,0,0), thickness=2)
    print(f"{title} | min: {feat.min().item():.4f} max: {feat.max().item():.4f} mean: {feat.mean().item():.4f} std: {feat.std().item():.4f}")
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(heatmap)
    plt.title(title + " (Heatmap)")
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(overlay)
    plt.title(title + " (Overlay)")
    plt.axis('off')
    plt.show()

# === 6. backbone 저장 포인트별 feature map 및 fusion 시각화 (YOLO 비율 기반 bbox) ===
print("model.save =", model.save)
# 원하는 YOLO label 비율
x_center_norm = 0.5
y_center_norm = 0.5
width_norm = 1 / 4.494  # ≈ 0.2225
height_norm = 1 / 4.552 # ≈ 0.2197

for idx in model.save:
    with torch.no_grad():
        # Wide feature
        feat_wide = model.extract_single_backbone_feature(wide_tensor, save_idx=idx).squeeze(0).cpu()
        # Narrow feature
        feat_narrow = model.extract_single_backbone_feature(narrow_tensor, save_idx=idx).squeeze(0).cpu()
        # === YOLO 비율 기반 bbox 계산 ===
        C, H, W = feat_wide.shape
        cx = x_center_norm * W
        cy = y_center_norm * H
        w = width_norm * W
        h = height_norm * H
        fx_min = int(round(cx - w / 2))
        fx_max = int(round(cx + w / 2))
        fy_min = int(round(cy - h / 2))
        fy_max = int(round(cy + h / 2))
        fx_min = max(fx_min, 0)
        fy_min = max(fy_min, 0)
        fx_max = min(fx_max, W)
        fy_max = min(fy_max, H)
        region_w = fx_max - fx_min
        region_h = fy_max - fy_min
        print(f"[DEBUG] [YOLO BBOX] fusion bbox: fx_min={fx_min}, fx_max={fx_max}, fy_min={fy_min}, fy_max={fy_max}, width={region_w}, height={region_h}")
        # fusion feature 생성
        if region_w > 0 and region_h > 0:
            narrow_region = torch.nn.functional.interpolate(
                feat_narrow.unsqueeze(0), size=(region_h, region_w), mode='bilinear', align_corners=False
            ).squeeze(0)
            narrow_full = torch.zeros_like(feat_wide)
            narrow_full[:, fy_min:fy_max, fx_min:fx_max] = narrow_region
            fusion_feat = narrow_full + feat_wide
        else:
            print("[WARNING] Zero-size fusion region! Skipping fusion for this layer.")
            fusion_feat = None
        # 시각화
        # YOLO bbox 시각화용
        fusion_box = np.array([
            [fx_min, fy_min],
            [fx_max-1, fy_min],
            [fx_max-1, fy_max-1],
            [fx_min, fy_max-1]
        ], dtype=np.int32)
        show_feature_map(feat_wide, wide_path, fusion_box, title=f"Wide | Layer idx {idx}", img_size=640)
        show_feature_map(feat_narrow, narrow_path, fusion_box, title=f"Narrow | Layer idx {idx}", img_size=640)
        if fusion_feat is not None:
            show_feature_map(fusion_feat, wide_path, fusion_box, title=f"Wide+Narrow Fusion | Layer idx {idx}", img_size=640)
