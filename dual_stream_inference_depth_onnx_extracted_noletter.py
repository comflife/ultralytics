#!/usr/bin/env python3
"""
Dual Stream YOLO ONNX Inference Script with Depth Visualization (Extracted Model Version, No Reverse Letterbox)
Original file: dual_stream_inference_depth_onnx_extracted.py
Modification: Removed all reverse letterbox handling so boxes stay in model (640x640) coordinate space.
NOTE: Drawing now overlays model-space boxes on the original image without geometric correction, which may cause misalignment if the original image is not a square 640x640 letterboxed image. Adjust if needed.
"""
import sys
import os
from pathlib import Path
import numpy as np
import cv2
import time
import random
import argparse
import onnxruntime as ort

SCRIPT_DIR = Path(__file__).parent.absolute()
ULTRALYTICS_ROOT = SCRIPT_DIR
sys.path.insert(0, str(ULTRALYTICS_ROOT))
print(f"Using local ultralytics from: {ULTRALYTICS_ROOT}")

# ===================================================================
# Configuration
# ===================================================================
ONNX_PATH = "/home/byounggun/ultralytics/best_dual_input_depth_extracted2.onnx"
WIDE_DIR = "/home/byounggun/ultralytics/swm_dual_split/val/images/"
NARROW_DIR = "/home/byounggun/ultralytics/swm_dual_split/val/val_narrow_images/"
LABEL_DIR = "/home/byounggun/ultralytics/swm_dual_split/val/labels/"
OUTPUT_DIR = "inference_results_depth_onnx_extracted_noletter"
CONFIDENCE_THRESHOLD = 0.1
IOU_THRESHOLD = 0.35
IMAGE_SIZE = 640
DEP_MIN = 0.1
DEP_MAX = 419.1

try:
    from ultralytics import YOLO  # noqa: F401
    import ultralytics  # noqa: F401
except ImportError as e:
    print(f"Failed to import ultralytics: {e}")
    sys.exit(1)

class DepthDenormalizer:
    def __init__(self, min_depth=DEP_MIN, max_depth=DEP_MAX):
        self.min_depth = min_depth
        self.max_depth = max_depth
    def denormalize_depth(self, normalized_depth):
        return normalized_depth * (self.max_depth - self.min_depth) + self.min_depth

def load_ground_truth_labels(label_path):
    if not os.path.exists(label_path):
        return []
    gt_labels = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    gt_labels.append([
                        int(parts[0]),
                        float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]),
                        float(parts[5])
                    ])
    except Exception as e:
        print(f"Label read error: {e}")
    return gt_labels

def convert_normalized_to_pixel_coords(bbox, img_width, img_height):
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

def preprocess_image(image_path, target_size=640):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    original_height, original_width = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    scale = min(target_size / original_width, target_size / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    resized = cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    delta_w = target_size - new_width
    delta_h = target_size - new_height
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[114, 114, 114])
    normalized = padded.astype(np.float32) / 255.0
    tensor = normalized.transpose(2, 0, 1)[np.newaxis, :]
    return tensor, (original_height, original_width), image

def create_dual_stream_inputs(wide_tensor, narrow_tensor):
    return {'images_wide': wide_tensor, 'images_narrow': narrow_tensor}

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def compute_dfl_dir(reg_output, y, x, start, num_bins=16):
    vals = reg_output[0, start:start + num_bins, y, x]
    exp_vals = np.exp(vals)
    sum_exp = np.sum(exp_vals)
    if sum_exp <= 0.0:
        sum_exp = 1e-6
    softmax = exp_vals / sum_exp
    return np.sum(np.arange(num_bins) * softmax)

def box_iou(x1, y1, x2, y2, ax1, ay1, ax2, ay2):
    xx1 = max(x1, ax1)
    yy1 = max(y1, ay1)
    xx2 = min(x2, ax2)
    yy2 = min(y2, ay2)
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    area1 = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area2 = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    denom = area1 + area2 - inter + 1e-6
    return inter / denom if denom > 0.0 else 0.0

def custom_postprocess(outputs, orig_size, img_size=640.0, conf_thres=0.1, iou_thres=0.35):
    """Postprocess WITHOUT reverse letterbox. Coordinates remain in the model input space (0..img_size)."""
    num_levels = len(outputs) // 3
    detections = []
    for li in range(num_levels):
        reg = outputs[li * 3]
        cls = outputs[li * 3 + 1]
        dep = outputs[li * 3 + 2]
        _, reg_C, H, W = reg.shape
        num_bins = reg_C // 4
        stride = img_size / H
        for y in range(H):
            for x in range(W):
                d_l = compute_dfl_dir(reg, y, x, 0, num_bins)
                d_t = compute_dfl_dir(reg, y, x, num_bins, num_bins)
                d_r = compute_dfl_dir(reg, y, x, 2 * num_bins, num_bins)
                d_b = compute_dfl_dir(reg, y, x, 3 * num_bins, num_bins)
                anchor_x = x + 0.5
                anchor_y = y + 0.5
                x1 = (anchor_x - d_l) * stride
                y1 = (anchor_y - d_t) * stride
                x2 = (anchor_x + d_r) * stride
                y2 = (anchor_y + d_b) * stride
                max_conf = 0.0
                max_class = -1
                for k in range(28):
                    prob = sigmoid(cls[0, k, y, x])
                    if prob > max_conf:
                        max_conf = prob
                        max_class = k
                if max_conf > conf_thres:
                    dep_norm = sigmoid(dep[0, 0, y, x]) if dep is not None else 0.0
                    detections.append([x1, y1, x2, y2, max_conf, max_class, dep_norm])
    detections = sorted(detections, key=lambda d: d[4], reverse=True)
    keep = []
    for i in range(len(detections)):
        if detections[i][4] == 0.0:
            continue
        keep.append(detections[i])
        for j in range(i + 1, len(detections)):
            if detections[j][4] == 0.0:
                continue
            if detections[i][5] != detections[j][5]:
                continue
            if box_iou(*detections[i][:4], *detections[j][:4]) > iou_thres:
                detections[j][4] = 0.0
    return [d for d in detections if d[4] > 0.0]

def draw_detections_with_depth(image, detections, gt_labels, depth_denormalizer, class_names=None):
    if class_names is None:
        class_names = [f"class_{i}" for i in range(28)]
    result_image = image.copy()
    img_height, img_width = image.shape[:2]
    pred_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    gt_color = (128, 128, 128)
    for gt_label in gt_labels:
        class_id, x_center, y_center, width, height, normalized_depth = gt_label
        x1, y1, x2, y2 = convert_normalized_to_pixel_coords([x_center, y_center, width, height], img_width, img_height)
        gt_depth_original = depth_denormalizer.denormalize_depth(normalized_depth)
        cv2.rectangle(result_image, (x1, y1), (x2, y2), gt_color, 2, lineType=cv2.LINE_AA)
        label_text = f"GT-{class_names[int(class_id)]}: {gt_depth_original:.2f}m"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(result_image, (x1, y1 - th - 10), (x1 + tw, y1), gt_color, -1)
        cv2.putText(result_image, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    for det in detections:
        x1, y1, x2, y2, confidence, class_id, depth_norm = det
        depth_original = depth_denormalizer.denormalize_depth(depth_norm)
        color = pred_colors[int(class_id) % len(pred_colors)]
        cv2.rectangle(result_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
        class_name = class_names[int(class_id)] if int(class_id) < len(class_names) else f"class_{int(class_id)}"
        text = f"PRED-{class_name}: {confidence:.2f}, D:{depth_original:.2f}m"
        text_y = int(y2) + 25
        if text_y > img_height - 30:
            text_y = int(y1) - 10
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(result_image, (int(x1), text_y - th), (int(x1) + tw, text_y + 5), color, -1)
        cv2.putText(result_image, text, (int(x1), text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return result_image

def main():
    parser = argparse.ArgumentParser(description="Dual Stream YOLO ONNX Inference (No Reverse Letterbox)")
    parser.add_argument('--wide-img', type=str, help='Path to the wide-angle image.')
    parser.add_argument('--narrow-img', type=str, help='Path to the narrow-angle image.')
    args = parser.parse_args()
    depth_denormalizer = DepthDenormalizer()
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    try:
        ort_session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f"Failed to load ONNX model: {e}")
        return
    input_names = [i.name for i in ort_session.get_inputs()]
    print(f"Model inputs: {input_names}")
    use_cmd_args = args.wide_img and args.narrow_img
    if use_cmd_args:
        WIDE_IMAGE_PATH = args.wide_img
        NARROW_IMAGE_PATH = args.narrow_img
        filename = Path(WIDE_IMAGE_PATH).name
    else:
        wide_files = [f for f in os.listdir(WIDE_DIR) if f.lower().endswith('.jpg')]
        if not wide_files:
            print(f"No JPG files in {WIDE_DIR}")
            return
        filename = random.choice(wide_files)
        WIDE_IMAGE_PATH = os.path.join(WIDE_DIR, filename)
        NARROW_IMAGE_PATH = os.path.join(NARROW_DIR, filename)
    LABEL_PATH = os.path.join(LABEL_DIR, filename.replace('.jpg', '.txt'))
    if not os.path.exists(NARROW_IMAGE_PATH):
        print(f"Missing narrow image: {NARROW_IMAGE_PATH}")
        return
    gt_labels = load_ground_truth_labels(LABEL_PATH)
    try:
        wide_tensor, wide_original_size, wide_image = preprocess_image(WIDE_IMAGE_PATH, IMAGE_SIZE)
        narrow_tensor, _, _ = preprocess_image(NARROW_IMAGE_PATH, IMAGE_SIZE)
    except Exception as e:
        print(f"Preprocess error: {e}")
        return
    dual_inputs = create_dual_stream_inputs(wide_tensor, narrow_tensor)
    start_time = time.time()
    try:
        onnx_outputs = ort_session.run(None, dual_inputs)
    except Exception as e:
        print(f"Inference failed: {e}")
        return
    inf_time = time.time() - start_time
    detections = custom_postprocess(onnx_outputs, wide_original_size, IMAGE_SIZE, CONFIDENCE_THRESHOLD, IOU_THRESHOLD)
    print(f"Detections: {len(detections)}  (time {inf_time:.3f}s)")
    for i, d in enumerate(detections):
        x1, y1, x2, y2, conf, cls, depth_norm = d
        depth_orig = depth_denormalizer.denormalize_depth(depth_norm)
        print(f"  {i+1}: cls={int(cls)} conf={conf:.3f} depth={depth_orig:.2f}m box=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")
    result_image = draw_detections_with_depth(wide_image, detections, gt_labels, depth_denormalizer)
    output_path = Path(OUTPUT_DIR) / f"dual_stream_depth_result_onnx_extracted_noletter_{filename}"
    cv2.imwrite(str(output_path), result_image)
    print(f"Saved image: {output_path}")

if __name__ == "__main__":
    main()
