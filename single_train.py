from ultralytics import YOLO

# Load a model
model = YOLO("yolov8.yaml")  # build a new model from YAML


# Train the model
results = model.train(data="ultralytics/cfg/datasets/for_swm3.yaml", epochs=100, imgsz=640, device=3)