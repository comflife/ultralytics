# dataload_siamese.py
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ultralytics.utils import LOGGER

class SiameseDataset(Dataset):
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
            else:
                if not os.path.exists(wide_path): # Should not happen as we are listing from wide_img_dir
                     LOGGER.warning(f"Wide image path issue (should not occur): {wide_path}")
                if not narrow_path_found:
                    pass # LOGGER.warning(f"Skipping {wide_path}: Corresponding narrow image not found for base name '{f_name}'.")
                if not os.path.exists(label_path) and narrow_path_found : # Only log if narrow was found but label missing
                    pass # LOGGER.warning(f"Skipping {wide_path}: Label file {label_path} not found.")
        
        if not self.wide_image_paths:
            LOGGER.error(f"No suitable image pairs found. Check paths and naming conventions.")
            LOGGER.error(f"Searched in: Wide='{wide_img_dir}', Narrow='{narrow_img_dir}', Labels='{label_dir}'")
            raise FileNotFoundError(f"No image pairs found. Please check dataset structure and paths.")
        
        LOGGER.info(f"Found {len(self.wide_image_paths)} image pairs for Siamese dataset.")

    def __len__(self):
        return len(self.wide_image_paths)

    def __getitem__(self, index):
        wide_img_path = self.wide_image_paths[index]
        narrow_img_path = self.narrow_image_paths[index]
        label_path = self.label_paths[index]

        wide_img = cv2.imread(wide_img_path)
        wide_img = cv2.cvtColor(wide_img, cv2.COLOR_BGR2RGB)
        
        narrow_img = cv2.imread(narrow_img_path)
        narrow_img = cv2.cvtColor(narrow_img, cv2.COLOR_BGR2RGB)

        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls = int(float(parts[0])) # Class ID
                        x_c, y_c, w, h = map(float, parts[1:]) # BBox coords
                        labels.append([x_c, y_c, w, h, cls]) # Format for Albumentations: x,y,w,h,cls
        
        labels_np = np.array(labels, dtype=np.float32)
        # Ensure labels_np is 2D even if empty, with 5 columns
        if labels_np.ndim == 1 and labels_np.size == 0: # Empty list resulted in empty array
            labels_np = np.empty((0, 5), dtype=np.float32)
        elif labels_np.ndim == 1 and labels_np.size > 0 : # Single label
            labels_np = labels_np.reshape(1,5)


        # Clamp bbox coordinates to valid range [0.0, 1.0] to avoid floating point precision issues
        if labels_np.shape[0] > 0:
            # Clamp coordinates to [0.0, 1.0] range to avoid precision errors
            labels_np[:, :4] = np.clip(labels_np[:, :4], 0.0, 1.0)
            
            # Ensure width and height don't cause out-of-bounds issues when converted to xmin, ymin
            # x_min = x_center - width/2 must be >= 0
            # y_min = y_center - height/2 must be >= 0
            # x_max = x_center + width/2 must be <= 1
            # y_max = y_center + height/2 must be <= 1
            x_center, y_center = labels_np[:, 0], labels_np[:, 1]
            width, height = labels_np[:, 2], labels_np[:, 3]
            
            # Adjust width to ensure x_min >= 0 and x_max <= 1
            max_width = np.minimum(2 * x_center, 2 * (1 - x_center))
            width = np.minimum(width, max_width)
            
            # Adjust height to ensure y_min >= 0 and y_max <= 1
            max_height = np.minimum(2 * y_center, 2 * (1 - y_center))
            height = np.minimum(height, max_height)
            
            # Update the array with adjusted width and height
            labels_np[:, 2] = width
            labels_np[:, 3] = height
            
            # Apply a small epsilon margin to avoid floating point precision issues
            epsilon = 1e-6
            labels_np[:, :4] = np.clip(labels_np[:, :4], epsilon, 1.0 - epsilon)
            
        bboxes_for_aug = labels_np[:, :4].tolist() if labels_np.shape[0] > 0 else []
        class_labels_for_aug = labels_np[:, 4].astype(int).tolist() if labels_np.shape[0] > 0 else []

        if self.transform:
            augmented_wide = self.transform(image=wide_img, bboxes=bboxes_for_aug, class_labels=class_labels_for_aug)
            wide_img_tensor = augmented_wide['image']
            
            # Process augmented bboxes
            augmented_bboxes_list = augmented_wide['bboxes']
            augmented_class_labels_list = augmented_wide['class_labels']
            
            if augmented_bboxes_list: # If there are any bboxes left after augmentation
                final_labels_list = []
                for i, bbox in enumerate(augmented_bboxes_list):
                    cls_label = augmented_class_labels_list[i]
                    # bbox is (x_center, y_center, width, height) from Albumentations YOLO format
                    final_labels_list.append([cls_label] + list(bbox)) 
                final_labels_np = np.array(final_labels_list, dtype=np.float32)
            else:
                final_labels_np = np.empty((0, 5), dtype=np.float32) # cls, x, y, w, h

            # Apply transform to narrow image (only image part, no labels/bboxes)
            # This will use its own random parameters for augmentations like ColorJitter.
            # Geometric transforms (like Flip, Rotate) will also be independently random.
            narrow_img_transformed = self.transform(image=narrow_img, bboxes=[], class_labels=[]) 
            narrow_img_tensor = narrow_img_transformed['image']
        else:
            # Basic resize and ToTensor if no advanced transform pipeline provided
            h_orig, w_orig, _ = wide_img.shape
            wide_img_resized = cv2.resize(wide_img, (self.img_size, self.img_size))
            narrow_img_resized = cv2.resize(narrow_img, (self.img_size, self.img_size))
            
            wide_img_tensor = torch.from_numpy(wide_img_resized.transpose(2, 0, 1)).float() / 255.0
            narrow_img_tensor = torch.from_numpy(narrow_img_resized.transpose(2, 0, 1)).float() / 255.0
            
            # Labels: cls, x,y,w,h format from original file
            if labels_np.shape[0] > 0:
                 final_labels_np = np.concatenate(
                    (labels_np[:, 4][:, np.newaxis], labels_np[:, :4]), axis=1 
                )
            else:
                final_labels_np = np.empty((0,5), dtype=np.float32)


        return wide_img_tensor, narrow_img_tensor, torch.from_numpy(final_labels_np)


def get_siamese_train_transforms(img_size):
    return A.Compose([
        A.Resize(height=img_size, width=img_size), # Resize first to make sure subsequent ops are on consistent size
        A.HorizontalFlip(p=0.5),
        # A.RandomResizedCrop might be too aggressive if wide/narrow views are very different
        # A.ShiftScaleRotate(p=0.3, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5), # Reduced hue jitter
        A.ToGray(p=0.01),
        # Add some blur or noise occasionally if relevant
        # A.GaussianBlur(blur_limit=(3, 7), p=0.1),
        # A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=0.1),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Standard ImageNet normalization
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.1))

def get_siamese_val_transforms(img_size):
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])) # Though bboxes not used for narrow val

def siamese_collate_fn(batch):
    wide_imgs, narrow_imgs, labels_list = zip(*batch)

    wide_imgs_stacked = torch.stack(wide_imgs, 0)
    narrow_imgs_stacked = torch.stack(narrow_imgs, 0)

    processed_labels = []
    for i, item_labels_tensor in enumerate(labels_list): # item_labels_tensor is (N_objs, 5) [cls, x,y,w,h]
        if item_labels_tensor.shape[0] > 0:
            batch_idx_tensor = torch.full((item_labels_tensor.shape[0], 1), float(i), dtype=item_labels_tensor.dtype, device=item_labels_tensor.device)
            # Concatenate to: batch_idx, cls, x, y, w, h
            processed_labels.append(torch.cat((batch_idx_tensor, item_labels_tensor), dim=1))
    
    if processed_labels:
        labels_stacked = torch.cat(processed_labels, 0)
    else:
        labels_stacked = torch.empty((0, 6), dtype=torch.float32) # Ensure dtype is float for consistency

    return wide_imgs_stacked, narrow_imgs_stacked, labels_stacked