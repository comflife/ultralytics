#!/bin/bash

# Run dual-stream YOLO training
echo "Starting dual-stream YOLO training..."

python train_yolov8_dual_v0.py \
  --cfg models/yolov8-dual.yaml \
  --data ultralytics/cfg/datasets/swm_dual.yaml \
  --epochs 100 \
  --batch-size 16 \
  --imgsz 640 \
  --dual-stream
