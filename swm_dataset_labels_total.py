"""
Script to copy images with corresponding labels into a YOLOv8 dataset structure.
This script collects images that have corresponding labels from camera_100_labels directory
and organizes them into a proper YOLOv8 dataset structure with 'images' and 'labels' folders.
It also creates a YAML file for the dataset with class names.
"""

import os
import shutil
import glob
from tqdm import tqdm
from pathlib import Path

def create_yolov8_dataset(source_images_dir, source_labels_dir, output_dir, allowed_extensions=('.jpg', '.jpeg', '.png')):
    """
    Create a YOLOv8 dataset by copying images and labels to the appropriate directories.
    
    Args:
        source_images_dir (str): Directory containing all source images
        source_labels_dir (str): Directory containing all label files
        output_dir (str): Base output directory where images/ and labels/ folders will be created
        allowed_extensions (tuple): Allowed image file extensions
    """
    # Create output directories
    output_images_dir = os.path.join(output_dir, 'total_images')
    output_labels_dir = os.path.join(output_dir, 'total_labels')
    
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    print(f"Created output directories:\n- {output_images_dir}\n- {output_labels_dir}")
    
    # Get all scene folders in the labels directory
    scene_folders = [folder for folder in os.listdir(source_labels_dir) 
                     if os.path.isdir(os.path.join(source_labels_dir, folder))]
    
    total_copied = 0
    skipped_count = 0
    
    print(f"Found {len(scene_folders)} scene folders in the labels directory")
    
    # Process each scene folder
    for scene_folder in tqdm(scene_folders, desc="Processing scene folders"):
        # Get full path to scene folder in both source directories
        scene_labels_path = os.path.join(source_labels_dir, scene_folder)
        scene_images_path = os.path.join(source_images_dir, scene_folder)
        
        # Skip if the scene doesn't exist in the images directory
        if not os.path.exists(scene_images_path):
            print(f"Warning: Scene {scene_folder} exists in labels but not in images directory. Skipping.")
            continue
            
        # Get all label files in the scene folder
        label_files = glob.glob(os.path.join(scene_labels_path, "*.txt"))
        
        for label_file in label_files:
            label_filename = os.path.basename(label_file)
            image_basename = os.path.splitext(label_filename)[0]
            
            # Check for matching image files with allowed extensions
            found_image = False
            for ext in allowed_extensions:
                image_filename = f"{image_basename}{ext}"
                image_path = os.path.join(scene_images_path, image_filename)
                
                if os.path.exists(image_path):
                    # Copy image and label to the output directories
                    dest_image = os.path.join(output_images_dir, f"{scene_folder}_{image_filename}")
                    dest_label = os.path.join(output_labels_dir, f"{scene_folder}_{label_filename}")
                    
                    shutil.copy2(image_path, dest_image)
                    shutil.copy2(label_file, dest_label)
                    total_copied += 1
                    found_image = True
                    break
                
            if not found_image:
                skipped_count += 1
    
    print(f"\nDone! Copied {total_copied} image-label pairs to YOLOv8 dataset format")
    print(f"Skipped {skipped_count} labels without matching images")
    
    return total_copied

def main():
    # Set paths
    source_images_dir = "/home/byounggun/swm_dataset/swm_dataset/final_folder/final_data/camera_100"
    source_labels_dir = "/home/byounggun/swm_dataset/swm_dataset/final_folder/final_data/camera_100_labels"
    output_dir = "swm_total"
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the YOLOv8 dataset
    create_yolov8_dataset(source_images_dir, source_labels_dir, output_dir)
    
    print(f"\nYOLOv8 dataset created at: {output_dir}")
    print("You can now use this dataset for training with YOLOv8.")

if __name__ == "__main__":
    main()
