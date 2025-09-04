#!/usr/bin/env python3
"""
Test script to validate letterbox coordinate transformation
This script helps verify that your C code coordinate transformation matches Python implementation
"""

import numpy as np
from pathlib import Path

def letterbox_transform(img_shape, target_size=640):
    """
    Calculate letterbox transformation parameters
    Same logic as ultralytics LetterBox class
    """
    h, w = img_shape[:2]
    
    # Calculate gain (scale factor)
    gain = min(target_size / w, target_size / h)
    
    # Calculate new dimensions after scaling
    new_w = int(w * gain)
    new_h = int(h * gain)
    
    # Calculate padding
    pad_w = (target_size - new_w) / 2
    pad_h = (target_size - new_h) / 2
    
    return {
        'gain': gain,
        'pad_w': pad_w,
        'pad_h': pad_h,
        'new_w': new_w,
        'new_h': new_h,
        'orig_w': w,
        'orig_h': h
    }

def unscale_coords(coords, transform_params):
    """
    Transform coordinates from letterbox space back to original image space
    Same logic as ultralytics scale_boxes function
    
    Args:
        coords: [x1, y1, x2, y2] in letterbox coordinates
        transform_params: dictionary from letterbox_transform()
    
    Returns:
        [x1, y1, x2, y2] in original image coordinates
    """
    x1, y1, x2, y2 = coords
    
    # Remove padding
    x1 -= transform_params['pad_w']
    y1 -= transform_params['pad_h']
    x2 -= transform_params['pad_w']
    y2 -= transform_params['pad_h']
    
    # Scale back to original size
    gain = transform_params['gain']
    x1 /= gain
    y1 /= gain
    x2 /= gain
    y2 /= gain
    
    # Clip to original image bounds
    orig_w = transform_params['orig_w']
    orig_h = transform_params['orig_h']
    x1 = max(0, min(x1, orig_w))
    y1 = max(0, min(y1, orig_h))
    x2 = max(0, min(x2, orig_w))
    y2 = max(0, min(y2, orig_h))
    
    return [x1, y1, x2, y2]

def test_coordinate_transformation():
    """Test coordinate transformation with various image sizes"""
    
    print("🧪 Testing Letterbox Coordinate Transformation")
    print("=" * 60)
    
    # Test cases: (width, height)
    test_cases = [
        (1920, 1080),  # 16:9 HD
        (1280, 720),   # 16:9 HD smaller
        (800, 600),    # 4:3
        (640, 640),    # Square (no transform needed)
        (480, 640),    # Portrait
        (1024, 768),   # 4:3 larger
    ]
    
    for orig_w, orig_h in test_cases:
        print(f"\n📐 Original image: {orig_w}x{orig_h}")
        
        # Calculate transformation parameters
        transform_params = letterbox_transform((orig_h, orig_w))
        
        print(f"   Gain: {transform_params['gain']:.4f}")
        print(f"   Padding: ({transform_params['pad_w']:.1f}, {transform_params['pad_h']:.1f})")
        print(f"   New size: {transform_params['new_w']}x{transform_params['new_h']}")
        
        # Test with sample bounding box in letterbox space
        # Example: center box in 640x640 letterbox space
        letterbox_coords = [200, 150, 450, 400]  # x1, y1, x2, y2
        
        # Transform back to original image space
        original_coords = unscale_coords(letterbox_coords, transform_params)
        
        print(f"   Letterbox coords: {letterbox_coords}")
        print(f"   Original coords:  {[f'{x:.1f}' for x in original_coords]}")
        
        # Verify coordinates are within bounds
        x1, y1, x2, y2 = original_coords
        assert 0 <= x1 <= orig_w, f"x1 out of bounds: {x1}"
        assert 0 <= y1 <= orig_h, f"y1 out of bounds: {y1}"
        assert 0 <= x2 <= orig_w, f"x2 out of bounds: {x2}"
        assert 0 <= y2 <= orig_h, f"y2 out of bounds: {y2}"
        assert x1 < x2, f"Invalid box: x1={x1} >= x2={x2}"
        assert y1 < y2, f"Invalid box: y1={y1} >= y2={y2}"
        
        print("   ✅ Coordinates valid")

def generate_c_test_code():
    """Generate C code for testing the same transformations"""
    
    print("\n" + "=" * 60)
    print("🔧 C Code Test Cases")
    print("=" * 60)
    
    test_cases = [
        (1920, 1080),
        (1280, 720),
        (800, 600),
        (640, 640),
    ]
    
    print("// Add this to your C code for testing:")
    print("void test_letterbox_transformation() {")
    
    for orig_w, orig_h in test_cases:
        transform_params = letterbox_transform((orig_h, orig_w))
        letterbox_coords = [200, 150, 450, 400]
        original_coords = unscale_coords(letterbox_coords, transform_params)
        
        print(f"    // Test case: {orig_w}x{orig_h}")
        print(f"    letterbox_params_t params_{orig_w}x{orig_h} = {{")
        print(f"        .gain = {transform_params['gain']:.6f}f,")
        print(f"        .pad_w = {transform_params['pad_w']:.6f}f,")
        print(f"        .pad_h = {transform_params['pad_h']:.6f}f,")
        print(f"        .orig_w = {orig_w}.0f,")
        print(f"        .orig_h = {orig_h}.0f")
        print(f"    }};")
        
        print(f"    float test_coords[4] = {{{letterbox_coords[0]}.0f, {letterbox_coords[1]}.0f, {letterbox_coords[2]}.0f, {letterbox_coords[3]}.0f}};")
        print(f"    unscale_coords(&test_coords[0], &test_coords[1], &test_coords[2], &test_coords[3], &params_{orig_w}x{orig_h});")
        print(f"    // Expected result: ({original_coords[0]:.1f}, {original_coords[1]:.1f}, {original_coords[2]:.1f}, {original_coords[3]:.1f})")
        print(f"    printf(\"Result: (%.1f, %.1f, %.1f, %.1f)\\n\", test_coords[0], test_coords[1], test_coords[2], test_coords[3]);")
        print()
    
    print("}")

def compare_with_ultralytics():
    """Compare with actual ultralytics scale_boxes function"""
    try:
        import torch
        from ultralytics.utils.ops import scale_boxes
        
        print("\n" + "=" * 60)
        print("🔬 Comparison with Ultralytics scale_boxes")
        print("=" * 60)
        
        # Test with actual ultralytics function
        orig_shape = (1080, 1920)  # (height, width)
        img_shape = (640, 640)     # model input shape
        
        # Sample bounding box in model coordinate space
        boxes = torch.tensor([[200, 150, 450, 400, 0.9, 0]])  # x1,y1,x2,y2,conf,cls
        
        # Use ultralytics scale_boxes
        scaled_boxes = scale_boxes(img_shape, boxes[:, :4], orig_shape)
        ultralytics_result = scaled_boxes[0].tolist()
        
        # Use our implementation
        transform_params = letterbox_transform(orig_shape)
        our_result = unscale_coords([200, 150, 450, 400], transform_params)
        
        print(f"Original shape: {orig_shape}")
        print(f"Model input shape: {img_shape}")
        print(f"Input coordinates: [200, 150, 450, 400]")
        print(f"Ultralytics result: {[f'{x:.1f}' for x in ultralytics_result]}")
        print(f"Our result:         {[f'{x:.1f}' for x in our_result]}")
        
        # Check if results match (within tolerance)
        tolerance = 1.0  # 1 pixel tolerance
        matches = all(abs(a - b) < tolerance for a, b in zip(ultralytics_result, our_result))
        
        if matches:
            print("✅ Results match ultralytics implementation!")
        else:
            print("❌ Results don't match - check implementation")
            
    except ImportError:
        print("⚠️ Ultralytics not available for comparison")

if __name__ == "__main__":
    test_coordinate_transformation()
    generate_c_test_code()
    compare_with_ultralytics()
    
    print("\n" + "=" * 60)
    print("📝 Summary:")
    print("1. Your C code needs the letterbox unscaling transformation")
    print("2. Use the corrected custom_postproc_corrected.c")
    print("3. Update original image dimensions in the code")
    print("4. Test with the generated C test cases")
    print("=" * 60)