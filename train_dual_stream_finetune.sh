#!/bin/bash

# Fine-tune dual-stream YOLO with frozen backbone
echo "Starting dual-stream YOLO fine-tuning with frozen backbone..."

python train_yolov8_dual_v7_finetune.py \
  --weights /home/byounggun/ultralytics/runs/train/exp361/weights/best.pt \
  --data ultralytics/cfg/datasets/katri.yaml \
  --epochs 50 \
  --batch-size 64 \
  --imgsz 640 \
  --device 1 \
  --project runs/finetune \
  --name katri_overfit
