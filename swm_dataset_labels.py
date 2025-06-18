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
import yaml

def create_yaml_file(output_dir, yaml_template_path):
    """
    Create a YAML file for the SWM dataset based on the COCO YAML template.
    
    Args:
        output_dir (str): Directory where the dataset YAML file will be saved
        yaml_template_path (str): Path to the COCO YAML template file
    """
    # Define class names
    class_names = {
        -1: "ignore",
        0: "car",
        1: "suv",
        2: "van",
        3: "regular_truck",
        4: "large_truck",
        5: "bus",
        6: "bicyclist",
        7: "motorcyclist",
        8: "scooter",
        9: "pedestrian",
        10: "stroller",
        11: "rubber_cone",
        12: "traffic_drum",
        13: "speed_30",
        14: "speed_40",
        15: "speed_50",
        16: "speed_60",
        17: "speed_70",
        18: "speed_80",
        19: "speed_90",
        20: "speed_100",
        21: "speed_110",
        22: "green_on",
        23: "yellow_on",
        24: "red_on",
        25: "green_left_on",
        26: "red_left_on",
        27: "unknown"
    }
    
    # Create yaml content as a dictionary with mixed types
    dataset_dict = {}
    
    # First create the names dictionary
    names_dict = {i: name for i, name in class_names.items() if i >= 0}  # Skip -1 index
    
    # Build the complete dictionary with proper typing
    dataset_dict = {
        "names": names_dict,
        "nc": len(names_dict),
        "path": str(output_dir),
        "train": str(os.path.join(output_dir, "images")),
        "val": str(os.path.join(output_dir, "images"))  # Using same directory for validation
    }
    
    # Determine output path (same directory as the COCO YAML)
    yaml_dir = os.path.dirname(yaml_template_path)
    yaml_output_path = os.path.join(yaml_dir, "swm.yaml")
    
    # Write YAML file
    with open(yaml_output_path, 'w') as f:
        yaml.safe_dump(dataset_dict, f, sort_keys=False)
    
    print(f"Created dataset YAML file at: {yaml_output_path}")
    return yaml_output_path

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
    output_images_dir = os.path.join(output_dir, 'images')
    output_labels_dir = os.path.join(output_dir, 'labels')
    
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
    output_dir = "swm"
    yaml_template_path = "/home/byounggun/ultralytics/ultralytics/cfg/datasets/coco.yaml"
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the YOLOv8 dataset
    create_yolov8_dataset(source_images_dir, source_labels_dir, output_dir)
    
    # Create the YAML file for the dataset
    yaml_output_path = create_yaml_file(os.path.abspath(output_dir), yaml_template_path)
    
    print(f"\nYOLOv8 dataset created at: {output_dir}")
    print(f"Dataset YAML file created at: {yaml_output_path}")
    print("You can now use this dataset for training with YOLOv8 using the command:")
    print(f"yolo train data={yaml_output_path}")

if __name__ == "__main__":
    main()
