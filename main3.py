from ultralytics import YOLO

# def freeze_layer(trainer):
#     model = trainer.model
#     num_freeze = 21
#     print(f"Freezing {num_freeze} layers")
#     freeze = [f'model.{x}.' for x in range(num_freeze)]  # layers to freeze 
#     for k, v in model.named_parameters(): 
#         v.requires_grad = True  # train all layers 
#         if any(x in k for x in freeze): 
#             print(f'freezing {k}') 
#             v.requires_grad = False 
#     print(f"{num_freeze} layers are freezed.")
# model=YOLO("yolov8s.pt")
# model.add_callback("on_train_start", freeze_layer)
# results = model.train(data="traffic_wide.yaml", epochs=200, imgsz=640, batch=256)


# Load a model
model = YOLO("yolov8s.yaml")  # build a new model from YAML
model = YOLO("yolov8s.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolov8s.yaml").load("yolov8s.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="traffic_wide_narrow.yaml", epochs=100, imgsz=640, batch=128, auto_augment="ada", device=[0,1])
#train10