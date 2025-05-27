import os
import cv2
import matplotlib.pyplot as plt

# Classes from traffic.yaml
CLASSES = [
    "Green", "Red", "Green-up", "Empty-count-down", "Count-down",
    "Yellow", "Empty", "Green-right", "Green-left", "Red-yellow"
]

IMAGES_DIR = "/home/byounggun/ultralytics/traffic_train/narrow/images"
LABELS_DIR = "/home/byounggun/ultralytics/traffic_train/narrow/labels"

# Get all image files
image_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.lower().endswith('.jpg') or f.lower().endswith('.png')])

for img_name in image_files:
    img_path = os.path.join(IMAGES_DIR, img_name)
    label_name = os.path.splitext(img_name)[0] + ".txt"
    label_path = os.path.join(LABELS_DIR, label_name)

    # Read image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load {img_path}")
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    # Draw boxes
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.strip().split()
                cls_id = int(parts[0])
                x_center, y_center, bw, bh = map(float, parts[1:])
                # YOLO format to box coords
                x1 = int((x_center - bw / 2) * w)
                y1 = int((y_center - bh / 2) * h)
                x2 = int((x_center + bw / 2) * w)
                y2 = int((y_center + bh / 2) * h)
                color = (0, 255, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label = CLASSES[cls_id] if 0 <= cls_id < len(CLASSES) else str(cls_id)
                cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4)

    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(img_name)
    plt.axis('off')
    plt.show()
    # Optionally, break after N images
    # if idx > 10:
    #     break
