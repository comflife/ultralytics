#!/bin/bash

# Run dual-stream YOLO training with pretrained weights
echo "Starting dual-stream YOLO training with pretrained weights..."

# python train_yolov8_dual_v1.py \
#   --cfg models/yolov8-dual.yaml \
#   --data ultralytics/cfg/datasets/swm_dual.yaml \
#   --epochs 1 \
#   --batch-size 16 \
#   --imgsz 640 \


python train_yolov8_dual_v5_pre.py \
  --weights /home/byounggun/ultralytics/yolov8n.pt \
  --cfg models/yolov8n-dual2.yaml \
  --data ultralytics/cfg/datasets/for_swm2.yaml \
  --epochs 500 \
  --batch-size 128 \
  --imgsz 640 \
  --device 1
