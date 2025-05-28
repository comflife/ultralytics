# dataload_dual.py
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ultralytics.utils import LOGGER

class DualImageDataset(Dataset):
    """
    Wide/Narrow 이미지 페어를 반환하는 데이터셋. 라벨은 wide에만 존재.
    """
    def __init__(self, wide_img_dir, narrow_img_dir, label_dir, img_size=640, transform=None, prefix=""):
        self.wide_img_dir = wide_img_dir
        self.narrow_img_dir = narrow_img_dir
        self.label_dir = label_dir  # For wide images only
        self.img_size = img_size
        self.transform = transform
        self.prefix = prefix

        self.wide_image_paths = []
        self.narrow_image_paths = []
        self.label_paths = []

        possible_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        LOGGER.info(f"Attempting to find image pairs: Wide dir: '{wide_img_dir}', Narrow dir: '{narrow_img_dir}', Label dir: '{label_dir}'")

        if not os.path.isdir(wide_img_dir):
            raise FileNotFoundError(f"Wide image directory not found: {wide_img_dir}")
        if not os.path.isdir(narrow_img_dir):
            raise FileNotFoundError(f"Narrow image directory not found: {narrow_img_dir}")
        if not os.path.isdir(label_dir):
            raise FileNotFoundError(f"Label directory not found: {label_dir}")

        for f_name_ext in os.listdir(self.wide_img_dir):
            f_name, f_ext = os.path.splitext(f_name_ext)
            if f_ext.lower() not in possible_extensions:
                continue

            wide_path = os.path.join(self.wide_img_dir, f_name_ext)
            narrow_path_found = None
            for ext_option in possible_extensions:
                potential_narrow_path = os.path.join(self.narrow_img_dir, f_name + ext_option)
                if os.path.exists(potential_narrow_path):
                    narrow_path_found = potential_narrow_path
                    break
            label_path = os.path.join(self.label_dir, f_name + '.txt')

            if narrow_path_found and os.path.exists(label_path):
                self.wide_image_paths.append(wide_path)
                self.narrow_image_paths.append(narrow_path_found)
                self.label_paths.append(label_path)

    def __len__(self):
        return len(self.wide_image_paths)

    def __getitem__(self, idx):
        wide_img = cv2.imread(self.wide_image_paths[idx])
        wide_img = cv2.cvtColor(wide_img, cv2.COLOR_BGR2RGB)
        narrow_img = cv2.imread(self.narrow_image_paths[idx])
        narrow_img = cv2.cvtColor(narrow_img, cv2.COLOR_BGR2RGB)
        label_path = self.label_paths[idx]
        # YOLO 라벨: class x_center y_center w h (normalized)
        labels = np.loadtxt(label_path).reshape(-1, 5) if os.path.getsize(label_path) > 0 else np.zeros((0, 5))

        if self.transform:
            augmented = self.transform(image=wide_img, image1=narrow_img)
            wide_img = augmented["image"]
            narrow_img = augmented["image1"]
        else:
            wide_img = cv2.resize(wide_img, (self.img_size, self.img_size))
            narrow_img = cv2.resize(narrow_img, (self.img_size, self.img_size))
            wide_img = torch.from_numpy(wide_img).permute(2, 0, 1).float() / 255.0
            narrow_img = torch.from_numpy(narrow_img).permute(2, 0, 1).float() / 255.0

        return {
            "wide_img": wide_img,
            "narrow_img": narrow_img,
            "labels": torch.from_numpy(labels).float()
        }

def get_dual_train_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.3),
        A.Normalize(),
        ToTensorV2(transpose_mask=True),
    ], additional_targets={"image1": "image"})

def get_dual_val_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(),
        ToTensorV2(transpose_mask=True),
    ], additional_targets={"image1": "image"})

def dual_collate_fn(batch):
    wide_imgs = torch.stack([b["wide_img"] for b in batch])
    narrow_imgs = torch.stack([b["narrow_img"] for b in batch])
    labels = [b["labels"] for b in batch]
    return {"wide_img": wide_imgs, "narrow_img": narrow_imgs, "labels": labels}
