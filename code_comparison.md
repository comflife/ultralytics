# Key Differences Between Original and Corrected Code

## Problem: Missing Letterbox Coordinate Transformation

### ORIGINAL CODE (Incorrect):
```c
// Directly use stride-based coordinates (WRONG - these are in letterbox space!)
float x1 = (anchor_x - d_l) * stride;
float y1 = (anchor_y - d_t) * stride;
float x2 = (anchor_x + d_r) * stride;
float y2 = (anchor_y + d_b) * stride;

// Store directly without coordinate transformation
dets[det_count].x1 = x1;  // ❌ Still in 640x640 letterbox coordinates!
dets[det_count].y1 = y1;  // ❌ Not in original image coordinates!
```

### CORRECTED CODE (Fixed):
```c
// 1. Calculate letterbox coordinates first
float x1_letterbox = (anchor_x - d_l) * stride;
float y1_letterbox = (anchor_y - d_t) * stride;
float x2_letterbox = (anchor_x + d_r) * stride;
float y2_letterbox = (anchor_y + d_b) * stride;

// 2. Transform to original image coordinates
float x1_orig = x1_letterbox;
float y1_orig = y1_letterbox;
float x2_orig = x2_letterbox;
float y2_orig = y2_letterbox;

unscale_coords(&x1_orig, &y1_orig, &x2_orig, &y2_orig, &letterbox_params);

// 3. Store original image coordinates
dets[det_count].x1 = x1_orig;  // ✅ Correct original image coordinates
dets[det_count].y1 = y1_orig;  // ✅ Correct original image coordinates
```

## New Functions Added:

### 1. Letterbox Parameters Structure:
```c
typedef struct {
    float gain;      // scale factor: min(640/orig_w, 640/orig_h)
    float pad_w;     // width padding: (640 - orig_w * gain) / 2
    float pad_h;     // height padding: (640 - orig_h * gain) / 2
    float orig_w;    // original image width
    float orig_h;    // original image height
} letterbox_params_t;
```

### 2. Letterbox Parameter Calculation:
```c
static letterbox_params_t calculate_letterbox_params(float orig_w, float orig_h, float target_size) {
    letterbox_params_t params;
    params.orig_w = orig_w;
    params.orig_h = orig_h;
    
    // Same logic as ultralytics scale_boxes function
    params.gain = fminf(target_size / orig_w, target_size / orig_h);
    
    float new_w = orig_w * params.gain;
    float new_h = orig_h * params.gain;
    params.pad_w = (target_size - new_w) / 2.0f;
    params.pad_h = (target_size - new_h) / 2.0f;
    
    return params;
}
```

### 3. Coordinate Unscaling Function:
```c
static void unscale_coords(float* x1, float* y1, float* x2, float* y2, 
                          const letterbox_params_t* params) {
    // Step 1: Remove padding (same as ultralytics scale_boxes)
    *x1 -= params->pad_w;
    *y1 -= params->pad_h;
    *x2 -= params->pad_w;
    *y2 -= params->pad_h;
    
    // Step 2: Scale back to original size
    *x1 /= params->gain;
    *y1 /= params->gain;
    *x2 /= params->gain;
    *y2 /= params->gain;
    
    // Step 3: Clip to original image bounds
    *x1 = fmaxf(0.0f, fminf(*x1, params->orig_w));
    *y1 = fmaxf(0.0f, fminf(*y1, params->orig_h));
    *x2 = fmaxf(0.0f, fminf(*x2, params->orig_w));
    *y2 = fmaxf(0.0f, fminf(*y2, params->orig_h));
}
```

## What You Need To Do:

### ⚠️ CRITICAL: Update Original Image Dimensions
```c
// In custom_postproc_run function, replace these placeholder values:
float original_width = 1920.0f;   // ← UPDATE: Replace with actual width
float original_height = 1080.0f;  // ← UPDATE: Replace with actual height
```

**Options to get original image dimensions:**
1. **Modify function signature** to pass dimensions as parameters
2. **Add to global variables** set from your application
3. **Read from configuration** or header data

### Reference Implementation
The corrected code follows the exact same logic as ultralytics:
- `ultralytics/utils/ops.py` → `scale_boxes()` function
- `ultralytics/data/augment.py` → `LetterBox` class
- `dual_stream_inference_depth.py` → `postprocess_results_with_depth()` function

This ensures your NPU post-processing produces identical results to the Python inference pipeline.