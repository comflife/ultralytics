#!/bin/bash

# Run dual-stream YOLO training
echo "Starting dual-stream YOLO training..."

# python train_yolov8_dual_v1.py \
#   --cfg models/yolov8-dual.yaml \
#   --data ultralytics/cfg/datasets/swm_dual.yaml \
#   --epochs 1 \
#   --batch-size 16 \
#   --imgsz 640 \


python train_yolov8_dual_v1.py \
  --cfg models/yolov8-dual.yaml \
  --data ultralytics/cfg/datasets/for_swm3.yaml \
  --epochs 250 \
  --batch-size 64 \
  # --monitor-gradients \
  # --gradient-log-interval 10 \
  # --gradient-check-interval 50 \
  # --detect-anomaly \
  --imgsz 640 \
  --device 3
