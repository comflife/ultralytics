from ultralytics import YOLO

# 간단하게 직접 호출
model = YOLO('models/yolov8-dual.yaml')
model.train(
    data='ultralytics/cfg/datasets/swm_dual.yaml',
    epochs=250,
    batch=16,
    imgsz=640
)