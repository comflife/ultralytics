#!/bin/bash

# Run dual-stream YOLO training with updated data loader
echo "Starting dual-stream YOLO training with dual stream data loading..."

python train_yolov8_dual_v0.py \
  --cfg models/yolov8-dual.yaml \
  --data ultralytics/cfg/datasets/swm_dual.yaml \
  --epochs 100 \
  --batch-size 16 \
  --imgsz 640 \
  --dual-stream
