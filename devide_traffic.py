# -*- coding: utf-8 -*-
import os
import shutil

# Load a model

traffic_train_data = "/home/byounggun/ultralytics/traffic_dataset/train/images"
traffic_train_label = "/home/byounggun/ultralytics/traffic_dataset/train/yolo"

traffic_train2_data = "/home/byounggun/ultralytics/traffic_dataset/test_1/images"
traffic_train2_label = "/home/byounggun/ultralytics/traffic_dataset/test_1/yolo"

traffic_test_data = "/home/byounggun/ultralytics/traffic_dataset/test_2/images"
traffic_test_label = "/home/byounggun/ultralytics/traffic_dataset/test_2/yolo"


# traffics/wide_images_only/train
# traffics/wide_images_only/val
# traffics/wide_images_only/test

import glob

def make_wide_only():
    # 디렉토리 생

    # wide가 포함된 이미지 파일들 찾기
    train1_imgs = glob.glob(f"{traffic_train_data}/*wide*.jpg")
    train2_imgs = glob.glob(f"{traffic_train2_data}/*wide*.jpg")
    test_imgs = glob.glob(f"{traffic_test_data}/*wide*.jpg")

    # wide가 포함된 라벨 파일들 찾기
    train1_labels = [img.replace('images', 'yolo').replace('.jpg', '.txt') for img in train1_imgs]
    train2_labels = [img.replace('images', 'yolo').replace('.jpg', '.txt') for img in train2_imgs]
    test_labels = [img.replace('images', 'yolo').replace('.jpg', '.txt') for img in test_imgs]

    # 파일 복사
    def copy_files(files, dest_dir):
        for f in files:
            if os.path.exists(f):
                shutil.copy(f, dest_dir)

    # 학습용 데이터 복사
    copy_files(train1_imgs + train2_imgs, f"traffics/wide_images_only/train/images")
    copy_files(train1_labels + train2_labels, f"traffics/wide_images_only/train/labels")
    
    # 검증용 데이터 복사
    copy_files(test_imgs, f"traffics/wide_images_only/val/images")
    copy_files(test_labels, f"traffics/wide_images_only/val/labels")

    print(f"Copied {len(train1_imgs) + len(train2_imgs)} training images")
    print(f"Copied {len(test_imgs)} validation images")


def make_narrow_only():

    # narrow가 포함된 이미지 파일들 찾기
    train1_imgs = glob.glob(f"{traffic_train_data}/*narrow*.jpg")
    train2_imgs = glob.glob(f"{traffic_train2_data}/*narrow*.jpg")
    test_imgs = glob.glob(f"{traffic_test_data}/*narrow*.jpg")

    # narrow가 포함된 라벨 파일들 찾기
    train1_labels = [img.replace('images', 'yolo').replace('.jpg', '.txt') for img in train1_imgs]
    train2_labels = [img.replace('images', 'yolo').replace('.jpg', '.txt') for img in train2_imgs]
    test_labels = [img.replace('images', 'yolo').replace('.jpg', '.txt') for img in test_imgs]

    # 파일 복사
    def copy_files(files, dest_dir):
        for f in files:
            if os.path.exists(f):
                shutil.copy(f, dest_dir)

    # 학습용 데이터 복사
    copy_files(train1_imgs + train2_imgs, f"traffics/narrow_images_only/train/images")
    copy_files(train1_labels + train2_labels, f"traffics/narrow_images_only/train/labels")
    
    # 검증용 데이터 복사
    copy_files(test_imgs, f"traffics/narrow_images_only/val/images")
    copy_files(test_labels, f"traffics/narrow_images_only/val/labels")

    print(f"Copied {len(train1_imgs) + len(train2_imgs)} training images")
    print(f"Copied {len(test_imgs)} validation images")

def make_wide_narrow():
    # 디렉토리 생성
    output_base = "traffics/wide_narrow"


    # wide와 narrow 이미지 파일들 찾기
    # Wide 이미지
    train1_wide = glob.glob(f"{traffic_train_data}/*wide*.jpg")
    train2_wide = glob.glob(f"{traffic_train2_data}/*wide*.jpg")
    test_wide = glob.glob(f"{traffic_test_data}/*wide*.jpg")
    
    # Narrow 이미지
    train1_narrow = glob.glob(f"{traffic_train_data}/*narrow*.jpg")
    train2_narrow = glob.glob(f"{traffic_train2_data}/*narrow*.jpg")
    test_narrow = glob.glob(f"{traffic_test_data}/*narrow*.jpg")

    # 라벨 파일 경로 생성
    def get_label_paths(img_paths):
        return [img.replace('images', 'yolo').replace('.jpg', '.txt') for img in img_paths]

    # Wide 라벨
    train1_wide_labels = get_label_paths(train1_wide)
    train2_wide_labels = get_label_paths(train2_wide)
    test_wide_labels = get_label_paths(test_wide)
    
    # Narrow 라벨
    train1_narrow_labels = get_label_paths(train1_narrow)
    train2_narrow_labels = get_label_paths(train2_narrow)
    test_narrow_labels = get_label_paths(test_narrow)

    # 파일 복사 함수
    def copy_files(files, dest_dir):
        for f in files:
            if os.path.exists(f):
                shutil.copy(f, dest_dir)

    # 학습용 데이터 복사 (wide + narrow)
    all_train_imgs = train1_wide + train2_wide + train1_narrow + train2_narrow
    all_train_labels = train1_wide_labels + train2_wide_labels + train1_narrow_labels + train2_narrow_labels
    
    # 검증용 데이터 복사 (wide + narrow)
    all_val_imgs = test_wide + test_narrow
    all_val_labels = test_wide_labels + test_narrow_labels

    # 파일 복사
    copy_files(all_train_imgs, "traffics/wide_narrow/train/images")
    copy_files(all_train_labels, "traffics/wide_narrow/train/labels")
    copy_files(all_val_imgs, "traffics/wide_narrow/val/images")
    copy_files(all_val_labels, "traffics/wide_narrow/val/labels")

    print(f"Copied {len(all_train_imgs)} training images (wide + narrow)")
    print(f"Copied {len(all_val_imgs)} validation images (wide + narrow)")

if __name__ == "__main__":
    make_wide_only()
    make_narrow_only()
    make_wide_narrow()

    
    