#!/usr/bin/env python3
"""
Visualize the position of the narrow camera image on the wide camera image.
Shows how the dual-stream model aligns and concatenates the two camera views.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import cv2

# Narrow camera bounding box in normalized coordinates (from SpatialAlignedMultiStreamConv)
narrow_bbox = {
    'center_x': 0.499289,
    'center_y': 0.499912,
    'width': 0.286041,
    'height': 0.291975
}

def create_visualization(img_width=1920, img_height=1080, output_path='narrow_position_on_wide.png'):
    """
    Create a visualization showing where the narrow image is positioned on the wide image.
    
    Args:
        img_width: Width of the wide image in pixels
        img_height: Height of the wide image in pixels
        output_path: Path to save the output image
    """
    
    # Calculate absolute pixel coordinates
    cx = narrow_bbox['center_x'] * img_width
    cy = narrow_bbox['center_y'] * img_height
    bw = narrow_bbox['width'] * img_width
    bh = narrow_bbox['height'] * img_height
    
    # Calculate bbox corners
    left = cx - bw / 2
    top = cy - bh / 2
    right = left + bw
    bottom = top + bh
    
    print("=" * 60)
    print("DUAL-STREAM IMAGE ALIGNMENT INFO")
    print("=" * 60)
    print(f"\n📐 Wide Image Dimensions: {img_width} × {img_height} pixels")
    print(f"\n📷 Narrow Camera ROI (Region of Interest):")
    print(f"   Center: ({cx:.1f}, {cy:.1f}) pixels")
    print(f"   Size: {bw:.1f} × {bh:.1f} pixels")
    print(f"   Bounding Box:")
    print(f"      Left:   {left:.1f} px")
    print(f"      Right:  {right:.1f} px")
    print(f"      Top:    {top:.1f} px")
    print(f"      Bottom: {bottom:.1f} px")
    print(f"\n📊 Coverage: {(bw*bh)/(img_width*img_height)*100:.1f}% of wide image")
    print("=" * 60)
    
    # Create figure with larger size
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # ==================== Left Plot: Overview ====================
    ax1 = axes[0]
    
    # Draw wide image background (light blue)
    wide_rect = patches.Rectangle((0, 0), img_width, img_height, 
                                   linewidth=3, edgecolor='blue', 
                                   facecolor='lightblue', alpha=0.3, 
                                   label='Wide Camera View')
    ax1.add_patch(wide_rect)
    
    # Draw narrow image ROI (red with hatching)
    narrow_rect = FancyBboxPatch((left, top), bw, bh,
                                  linewidth=4, edgecolor='red', 
                                  facecolor='lightcoral', alpha=0.5,
                                  boxstyle="round,pad=10",
                                  label='Narrow Camera ROI')
    ax1.add_patch(narrow_rect)
    
    # Add center crosshair
    ax1.plot([cx-50, cx+50], [cy, cy], 'r-', linewidth=3)
    ax1.plot([cx, cx], [cy-50, cy+50], 'r-', linewidth=3)
    ax1.plot(cx, cy, 'ro', markersize=15, label='ROI Center')
    
    # Add corner markers
    corner_size = 40
    for corner_x, corner_y in [(left, top), (right, top), (left, bottom), (right, bottom)]:
        ax1.plot(corner_x, corner_y, 'r^', markersize=12)
    
    # Add dimension annotations
    ax1.annotate('', xy=(left, top-100), xytext=(right, top-100),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax1.text(cx, top-150, f'Width: {bw:.0f}px ({narrow_bbox["width"]*100:.1f}%)', 
            ha='center', fontsize=14, color='red', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax1.annotate('', xy=(left-100, top), xytext=(left-100, bottom),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax1.text(left-200, cy, f'Height: {bh:.0f}px\n({narrow_bbox["height"]*100:.1f}%)', 
            ha='center', va='center', fontsize=14, color='red', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax1.set_xlim(-300, img_width + 300)
    ax1.set_ylim(img_height + 300, -300)
    ax1.set_aspect('equal')
    ax1.set_xlabel('Width (pixels)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Height (pixels)', fontsize=14, fontweight='bold')
    ax1.set_title('Dual-Stream Camera Alignment\n(Narrow ROI on Wide Image)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax1.legend(loc='upper right', fontsize=12, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # ==================== Right Plot: Detailed View ====================
    ax2 = axes[1]
    
    # Create a more detailed visualization with grid
    img_array = np.ones((img_height, img_width, 3), dtype=np.uint8) * 200
    
    # Draw wide image border
    cv2.rectangle(img_array, (0, 0), (img_width-1, img_height-1), (0, 0, 255), 5)
    
    # Draw narrow ROI with thick border
    cv2.rectangle(img_array, (int(left), int(top)), (int(right), int(bottom)), 
                 (255, 0, 0), 8)
    
    # Add semi-transparent overlay
    overlay = img_array.copy()
    cv2.rectangle(overlay, (int(left), int(top)), (int(right), int(bottom)), 
                 (255, 100, 100), -1)
    img_array = cv2.addWeighted(img_array, 0.7, overlay, 0.3, 0)
    
    # Draw grid lines
    grid_step = 100
    for x in range(0, img_width, grid_step):
        cv2.line(img_array, (x, 0), (x, img_height), (150, 150, 150), 1)
    for y in range(0, img_height, grid_step):
        cv2.line(img_array, (0, y), (img_width, y), (150, 150, 150), 1)
    
    # Draw center crosshair
    cv2.line(img_array, (int(cx)-40, int(cy)), (int(cx)+40, int(cy)), (255, 0, 0), 4)
    cv2.line(img_array, (int(cx), int(cy)-40), (int(cx), int(cy)+40), (255, 0, 0), 4)
    cv2.circle(img_array, (int(cx), int(cy)), 12, (255, 0, 0), -1)
    
    # Convert BGR to RGB for matplotlib
    img_array_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    
    ax2.imshow(img_array_rgb)
    ax2.set_xlabel('Width (pixels)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Height (pixels)', fontsize=14, fontweight='bold')
    ax2.set_title('Detailed View with Grid\n(100px grid spacing)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Add text annotations on the image
    text_props = dict(boxstyle='round', facecolor='white', alpha=0.9)
    ax2.text(img_width/2, 80, 'Wide Camera (Full View)', 
            ha='center', fontsize=16, color='blue', fontweight='bold', 
            bbox=text_props)
    ax2.text(cx, cy, f'Narrow\nCamera\nROI', 
            ha='center', va='center', fontsize=14, color='red', fontweight='bold',
            bbox=text_props)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {output_path}\n")
    
    return fig

def create_processing_diagram(output_path='dual_stream_processing.png'):
    """
    Create a diagram showing how the dual-stream model processes the images.
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Dual-Stream YOLO Processing Pipeline', 
           ha='center', fontsize=20, fontweight='bold')
    
    # Step 1: Input images
    wide_box = FancyBboxPatch((0.5, 7), 1.5, 1.2, 
                              boxstyle="round,pad=0.1", 
                              edgecolor='blue', facecolor='lightblue', linewidth=3)
    narrow_box = FancyBboxPatch((0.5, 5.3), 1.5, 1.2, 
                                boxstyle="round,pad=0.1", 
                                edgecolor='red', facecolor='lightcoral', linewidth=3)
    ax.add_patch(wide_box)
    ax.add_patch(narrow_box)
    ax.text(1.25, 7.6, 'Wide Image\n(3 channels)', ha='center', va='center', 
           fontsize=12, fontweight='bold')
    ax.text(1.25, 5.9, 'Narrow Image\n(3 channels)', ha='center', va='center', 
           fontsize=12, fontweight='bold')
    
    # Arrow 1
    ax.annotate('', xy=(2.5, 6.5), xytext=(2.1, 6.5),
               arrowprops=dict(arrowstyle='->', lw=3, color='black'))
    ax.text(2.3, 6.8, 'Concat', ha='center', fontsize=11, fontweight='bold')
    
    # Step 2: Concatenated input
    concat_box = FancyBboxPatch((2.5, 5.8), 1.5, 1.4, 
                                boxstyle="round,pad=0.1", 
                                edgecolor='purple', facecolor='lavender', linewidth=3)
    ax.add_patch(concat_box)
    ax.text(3.25, 6.5, 'Concatenated\nInput\n(6 channels)', ha='center', va='center', 
           fontsize=12, fontweight='bold')
    
    # Arrow 2
    ax.annotate('', xy=(4.5, 6.5), xytext=(4.1, 6.5),
               arrowprops=dict(arrowstyle='->', lw=3, color='black'))
    ax.text(4.3, 6.8, 'Split', ha='center', fontsize=11, fontweight='bold')
    
    # Step 3: MultiStreamConv
    stream1_box = FancyBboxPatch((4.5, 7), 1.5, 0.7, 
                                 boxstyle="round,pad=0.05", 
                                 edgecolor='blue', facecolor='lightblue', linewidth=2)
    stream2_box = FancyBboxPatch((4.5, 5.8), 1.5, 0.7, 
                                 boxstyle="round,pad=0.05", 
                                 edgecolor='red', facecolor='lightcoral', linewidth=2)
    ax.add_patch(stream1_box)
    ax.add_patch(stream2_box)
    ax.text(5.25, 7.35, 'Stream 1\n(Wide)', ha='center', va='center', 
           fontsize=11, fontweight='bold')
    ax.text(5.25, 6.15, 'Stream 2\n(Narrow)', ha='center', va='center', 
           fontsize=11, fontweight='bold')
    
    # Arrow 3
    ax.annotate('', xy=(6.5, 6.5), xytext=(6.1, 6.5),
               arrowprops=dict(arrowstyle='->', lw=3, color='black'))
    ax.text(6.3, 6.8, 'Align', ha='center', fontsize=11, fontweight='bold')
    
    # Step 4: Spatial Alignment
    align_box = FancyBboxPatch((6.5, 5.8), 2, 1.4, 
                               boxstyle="round,pad=0.1", 
                               edgecolor='green', facecolor='lightgreen', linewidth=3)
    ax.add_patch(align_box)
    ax.text(7.5, 6.8, 'SpatialAligned\nMultiStreamConv', ha='center', va='top', 
           fontsize=11, fontweight='bold')
    ax.text(7.5, 6.2, '• Upsample narrow\n• Align to ROI\n• Fuse features', 
           ha='center', va='top', fontsize=9, style='italic')
    
    # Arrow 4
    ax.annotate('', xy=(7.5, 5.3), xytext=(7.5, 5.7),
               arrowprops=dict(arrowstyle='->', lw=3, color='black'))
    
    # Step 5: YOLO Backbone
    backbone_box = FancyBboxPatch((6.5, 3.8), 2, 1.3, 
                                  boxstyle="round,pad=0.1", 
                                  edgecolor='orange', facecolor='lightyellow', linewidth=3)
    ax.add_patch(backbone_box)
    ax.text(7.5, 4.8, 'YOLO Backbone', ha='center', va='top', 
           fontsize=12, fontweight='bold')
    ax.text(7.5, 4.3, 'C2f, Conv, SPPF', ha='center', va='top', fontsize=10)
    
    # Arrow 5
    ax.annotate('', xy=(7.5, 3.3), xytext=(7.5, 3.7),
               arrowprops=dict(arrowstyle='->', lw=3, color='black'))
    
    # Step 6: Detection Head
    head_box = FancyBboxPatch((6.5, 1.8), 2, 1.3, 
                              boxstyle="round,pad=0.1", 
                              edgecolor='darkred', facecolor='mistyrose', linewidth=3)
    ax.add_patch(head_box)
    ax.text(7.5, 2.8, 'Detection Head', ha='center', va='top', 
           fontsize=12, fontweight='bold')
    ax.text(7.5, 2.3, 'Boxes + Classes + Depth', ha='center', va='top', fontsize=10)
    
    # Arrow 6
    ax.annotate('', xy=(7.5, 1.3), xytext=(7.5, 1.7),
               arrowprops=dict(arrowstyle='->', lw=3, color='black'))
    
    # Step 7: Output
    output_box = FancyBboxPatch((6.5, 0.5), 2, 0.7, 
                                boxstyle="round,pad=0.05", 
                                edgecolor='black', facecolor='lightgray', linewidth=3)
    ax.add_patch(output_box)
    ax.text(7.5, 0.85, 'Detections + Depth', ha='center', va='center', 
           fontsize=12, fontweight='bold')
    
    # Add legend on the right
    legend_y = 7.5
    ax.text(9.5, legend_y, 'Key Points:', ha='center', fontsize=12, fontweight='bold')
    ax.text(9.5, legend_y-0.5, '✓ No pretrained weights', ha='center', fontsize=10)
    ax.text(9.5, legend_y-0.9, '✓ 6-channel input', ha='center', fontsize=10)
    ax.text(9.5, legend_y-1.3, '✓ Spatial alignment', ha='center', fontsize=10)
    ax.text(9.5, legend_y-1.7, '✓ Depth estimation', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Processing diagram saved to: {output_path}\n")
    
    return fig

if __name__ == '__main__':
    # Create visualizations
    print("\n🎨 Creating visualizations...\n")
    
    # Create the main position visualization
    create_visualization(img_width=1920, img_height=1080, 
                        output_path='narrow_position_on_wide.png')
    
    # Create the processing pipeline diagram
    create_processing_diagram(output_path='dual_stream_processing.png')
    
    print("=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print("1. Your model trains from scratch (no pretrained weights)")
    print("2. Narrow image covers ~28.6% width × ~29.2% height of wide image")
    print("3. Narrow image is centered at approximately (50%, 50%) of wide image")
    print("4. Both images are concatenated along channel dimension (6 channels total)")
    print("=" * 60)
    print("\n✨ Done! Check the generated PNG files.\n")
