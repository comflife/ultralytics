import os
import shutil
from pathlib import Path

# Base directories
base_dir = Path("/home/byounggun/ultralytics")
swm_images_dir = base_dir / "swm" / "images"
swm_narrow_images_dir = base_dir / "swm" / "narrow_images"
swm_labels_dir = base_dir / "swm" / "labels"

# Output directory and files (changed to swm-data)
output_dir = base_dir / "datasets" / "swm-data"
output_dir.mkdir(parents=True, exist_ok=True)

# Create train and val directories with wide/narrow subdirectories
for split in ["train", "val"]:
    for modality in ["wide", "narrow"]:
        (output_dir / split / "images" / modality).mkdir(parents=True, exist_ok=True)
    (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

# Get all image files from both directories
swm_images = set([f.name for f in swm_images_dir.glob("*.jpg")])
swm_narrow_images = set([f.name for f in swm_narrow_images_dir.glob("*.jpg")])

# Find common images (intersection)
common_images = swm_images.intersection(swm_narrow_images)
print(f"Found {len(common_images)} common images")

# Sort for consistent ordering
common_images = sorted(list(common_images))

# Split into train and val (80-20 split)
split_idx = int(len(common_images) * 0.8)
train_images = common_images[:split_idx]
val_images = common_images[split_idx:]

print(f"Train images: {len(train_images)}")
print(f"Val images: {len(val_images)}")

# Write train-all-04.txt with actual paths (not placeholders)
with open(output_dir / "train-all-04.txt", "w") as f:
    for img_name in train_images:
        # Write both wide and narrow paths
        f.write(f"datasets/swm-data/train/images/wide/{img_name}\n")

# Write test-all-20.txt with actual paths (not placeholders)  
with open(output_dir / "test-all-20.txt", "w") as f:
    for img_name in val_images:
        # Write both wide and narrow paths
        f.write(f"datasets/swm-data/val/images/wide/{img_name}\n")

# Copy actual image files to the new structure
print("Copying images...")
for i, img_name in enumerate(train_images):
    # Copy wide images
    src_wide = swm_images_dir / img_name
    dst_wide = output_dir / "train" / "images" / "wide" / img_name
    if src_wide.exists():
        shutil.copy2(src_wide, dst_wide)
    
    # Copy narrow images
    src_narrow = swm_narrow_images_dir / img_name
    dst_narrow = output_dir / "train" / "images" / "narrow" / img_name
    if src_narrow.exists():
        shutil.copy2(src_narrow, dst_narrow)
    
    # Copy labels (assuming they exist)
    label_name = img_name.replace('.jpg', '.txt')
    src_label = swm_labels_dir / label_name
    dst_label = output_dir / "train" / "labels" / label_name
    if src_label.exists():
        shutil.copy2(src_label, dst_label)
    
    if i % 50 == 0:
        print(f"Copied {i}/{len(train_images)} train images")

print("Copying validation images...")
for i, img_name in enumerate(val_images):
    # Copy wide images
    src_wide = swm_images_dir / img_name
    dst_wide = output_dir / "val" / "images" / "wide" / img_name
    if src_wide.exists():
        shutil.copy2(src_wide, dst_wide)
    
    # Copy narrow images
    src_narrow = swm_narrow_images_dir / img_name
    dst_narrow = output_dir / "val" / "images" / "narrow" / img_name
    if src_narrow.exists():
        shutil.copy2(src_narrow, dst_narrow)
    
    # Copy labels
    label_name = img_name.replace('.jpg', '.txt')
    src_label = swm_labels_dir / label_name
    dst_label = output_dir / "val" / "labels" / label_name
    if src_label.exists():
        shutil.copy2(src_label, dst_label)
    
    if i % 50 == 0:
        print(f"Copied {i}/{len(val_images)} val images")

print("Dataset structure created successfully!")
print(f"Output directory: {output_dir}")
print(f"Train file: {output_dir}/train-all-04.txt")
print(f"Val file: {output_dir}/test-all-20.txt")
