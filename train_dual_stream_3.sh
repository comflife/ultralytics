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
  --cfg models/yolov8-dual_2.yaml \
  --data ultralytics/cfg/datasets/for_swm2.yaml \
  --epochs 100 \
  --batch-size 32 \
  --imgsz 640 \
  --device 2
