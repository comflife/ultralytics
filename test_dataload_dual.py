# -*- coding: utf-8 -*-
import os
import torch
from dataload_dual import DualImageDataset

if __name__ == '__main__':
    # 경로는 yolov8_dual.py에서 사용하는 기본값을 참고
    wide_img_dir = '/home/byounggun/ultralytics/traffic_train/wide/images'
    narrow_img_dir = '/home/byounggun/ultralytics/traffic_train/narrow/images'
    label_dir = '/home/byounggun/ultralytics/traffic_train/wide/labels'
    img_size = 640
    sample_ratio = 1.0  # 전체 데이터 사용, 일부만 테스트하려면 0.05 등으로 변경

    dataset = DualImageDataset(
        wide_img_dir=wide_img_dir,
        narrow_img_dir=narrow_img_dir,
        label_dir=label_dir,
        img_size=img_size,
        transform=None,
        sample_ratio=sample_ratio
    )

    print(f"[INFO] 전체 샘플 수: {len(dataset)}")
    print(f"[INFO] wide 이미지 예시 경로: {dataset.wide_image_paths[:3]}")
    print(f"[INFO] narrow 이미지 예시 경로: {dataset.narrow_image_paths[:3]}")
    print(f"[INFO] 라벨 예시 경로: {dataset.label_paths[:3]}")

    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"\n[샘플 {i}] wide_img shape: {sample['wide_img'].shape}, dtype: {sample['wide_img'].dtype}")
        print(f"[샘플 {i}] narrow_img shape: {sample['narrow_img'].shape}, dtype: {sample['narrow_img'].dtype}")
        print(f"[샘플 {i}] labels shape: {sample['labels'].shape}, dtype: {sample['labels'].dtype}")
        print(f"[샘플 {i}] labels 내용: {sample['labels']}")
