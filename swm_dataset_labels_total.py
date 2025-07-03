"""
Script to copy all images and corresponding labels from a source directory to an output directory.
This script recursively finds all images and their corresponding label files and copies them
to the output directory in a flat structure.
"""

import os
import shutil
import glob
from tqdm import tqdm
from pathlib import Path

def copy_all_files(source_dir, output_dir, allowed_extensions=('.jpg', '.jpeg', '.png')):
    """
    Copy all images and their corresponding labels from source directory to output directory.
    
    Args:
        source_dir (str): Source directory containing images and labels
        output_dir (str): Output directory where all files will be copied
        allowed_extensions (tuple): Allowed image file extensions
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Copying files from {source_dir} to {output_dir}")
    
    total_copied = 0
    
    # Walk through all subdirectories in source
    for root, dirs, files in os.walk(source_dir):
        for file in tqdm(files, desc=f"Processing {os.path.basename(root)}"):
            file_path = os.path.join(root, file)
            
            # Check if it's an image file
            if any(file.lower().endswith(ext) for ext in allowed_extensions):
                # Copy image file
                dest_image = os.path.join(output_dir, file)
                
                # Handle duplicate filenames by adding directory prefix
                if os.path.exists(dest_image):
                    relative_path = os.path.relpath(root, source_dir)
                    prefix = relative_path.replace(os.sep, '_')
                    dest_image = os.path.join(output_dir, f"{prefix}_{file}")
                
                shutil.copy2(file_path, dest_image)
                total_copied += 1
                
                # Look for corresponding label file
                image_basename = os.path.splitext(file)[0]
                label_file = os.path.join(root, f"{image_basename}.txt")
                
                if os.path.exists(label_file):
                    dest_label = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(dest_image))[0]}.txt")
                    shutil.copy2(label_file, dest_label)
            
            # Also copy standalone .txt files (labels without images)
            elif file.lower().endswith('.txt'):
                dest_label = os.path.join(output_dir, file)
                
                # Handle duplicate filenames by adding directory prefix
                if os.path.exists(dest_label):
                    relative_path = os.path.relpath(root, source_dir)
                    prefix = relative_path.replace(os.sep, '_')
                    dest_label = os.path.join(output_dir, f"{prefix}_{file}")
                
                shutil.copy2(file_path, dest_label)
    
    print(f"\nDone! Copied {total_copied} files to {output_dir}")
    return total_copied

def main():
    # Set paths directly in code
    source_dir = "/home/byounggun/swm_dataset/swm_dataset/final_folder/final_data/camera_30"
    output_dir = "swm_total/narrow_images"
    
    if not os.path.exists(source_dir):
        print(f"Error: Source directory {source_dir} does not exist!")
        return
    
    # Copy all files
    copy_all_files(source_dir, output_dir)
    
    print(f"\nAll files copied from {source_dir} to {output_dir}")

if __name__ == "__main__":
    main()
