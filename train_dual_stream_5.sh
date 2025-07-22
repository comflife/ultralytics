#!/bin/bash

# Run dual-stream YOLO training
echo "Starting dual-stream YOLO training..."

# python train_yolov8_dual_v1.py \
#   --cfg models/yolov8-dual.yaml \
#   --data ultralytics/cfg/datasets/swm_dual.yaml \
#   --epochs 1 \
#   --batch-size 16 \
#   --imgsz 640 \


python train_yolov8_dual_v3.py \
  --cfg models/yolov8s-dual.yaml \
  --data ultralytics/cfg/datasets/for_swm2.yaml \
  --epochs 150 \
  --batch-size 24 \
  --imgsz 640 \
  --device 2
