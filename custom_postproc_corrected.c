#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h> // expf, fmaxf, fminf

#include "enlight_network.h"  // enlight_act_tensor_t, enlight_get_tensor_data_by_off, enlight_get_tensor_dimensions

#ifdef BARE_METAL_FW_DEV
extern int _printf(const char *format, ...);
# define ENLIGHT_CUSTOM_PRINT _printf
#else
# define ENLIGHT_CUSTOM_PRINT printf
#endif

#define enlight_custom_err(...) do { ENLIGHT_CUSTOM_PRINT(__VA_ARGS__); } while(0)
#define enlight_custom_log(...) do { ENLIGHT_CUSTOM_PRINT(__VA_ARGS__); } while(0)

#ifdef ENLIGHT_CUSTOM_DEBUG
# define enlight_custom_dbg(...) do { ENLIGHT_CUSTOM_PRINT(__VA_ARGS__); } while(0)
#else
# define enlight_custom_dbg(...) do {} while(0)
#endif

// Keep depth normalization (used only for logging here)
static const float DEPTH_MIN = 0.1f;
static const float DEPTH_MAX = 419.1f;
static float denormalize_depth(float normalized_depth) {
    return normalized_depth * (DEPTH_MAX - DEPTH_MIN) + DEPTH_MIN;
}

// 🔴 CRITICAL FIX: Add letterbox parameters
// These should match your training configuration
static const float MODEL_INPUT_SIZE = 640.0f;  // Model input size (640x640)

// 🔴 ADD: Letterbox unscaling parameters
typedef struct {
    float gain;      // scale factor
    float pad_w;     // width padding 
    float pad_h;     // height padding
    float orig_w;    // original image width
    float orig_h;    // original image height
} letterbox_params_t;

// 🔴 ADD: Function to calculate letterbox parameters
static letterbox_params_t calculate_letterbox_params(float orig_w, float orig_h, float target_size) {
    letterbox_params_t params;
    params.orig_w = orig_w;
    params.orig_h = orig_h;
    
    // Calculate gain (same as in ultralytics/utils/ops.py scale_boxes)
    params.gain = fminf(target_size / orig_w, target_size / orig_h);
    
    // Calculate padding (same as ultralytics letterbox)
    float new_w = orig_w * params.gain;
    float new_h = orig_h * params.gain;
    params.pad_w = (target_size - new_w) / 2.0f;
    params.pad_h = (target_size - new_h) / 2.0f;
    
    enlight_custom_log("Letterbox params: gain=%.4f, pad_w=%.2f, pad_h=%.2f\n", 
                       params.gain, params.pad_w, params.pad_h);
    return params;
}

// 🔴 ADD: Function to unscale coordinates from letterbox to original image
static void unscale_coords(float* x1, float* y1, float* x2, float* y2, const letterbox_params_t* params) {
    // Remove padding first (same as ultralytics scale_boxes)
    *x1 -= params->pad_w;
    *y1 -= params->pad_h;
    *x2 -= params->pad_w;
    *y2 -= params->pad_h;
    
    // Scale back to original size
    *x1 /= params->gain;
    *y1 /= params->gain;
    *x2 /= params->gain;
    *y2 /= params->gain;
    
    // Clip to original image bounds
    *x1 = fmaxf(0.0f, fminf(*x1, params->orig_w));
    *y1 = fmaxf(0.0f, fminf(*y1, params->orig_h));
    *x2 = fmaxf(0.0f, fminf(*x2, params->orig_w));
    *y2 = fmaxf(0.0f, fminf(*y2, params->orig_h));
}

void custom_postproc_init() {
    enlight_custom_log("Depth normalization info loaded: min=%.1f, max=%.1f\n", DEPTH_MIN, DEPTH_MAX);
    enlight_custom_log("Model input size: %.0f\n", MODEL_INPUT_SIZE);
}

// Group tensors by feature map HxW
typedef struct {
    int H, W;
    int reg_idx;  // DFL tensor index
    int cls_idx;  // class-score tensor index
    int dep_idx;  // depth tensor index (C == 1)
    int reg_C;
    int cls_C;
} level_group_t;

static int find_or_add_level(level_group_t* L, int* n, int H, int W, int cap) {
    for (int i = 0; i < *n; i++) if (L[i].H == H && L[i].W == W) return i;
    if (*n >= cap) return -1;
    L[*n].H = H; L[*n].W = W;
    L[*n].reg_idx = -1; L[*n].cls_idx = -1; L[*n].dep_idx = -1;
    L[*n].reg_C = 0; L[*n].cls_C = 0;
    return (*n)++;
}

// DFL decode for one direction: sum k * softmax(v_k)
static float compute_dfl_dir(enlight_act_tensor_t* t, int y, int x, int start) {
    float sum_exp = 0.0f;
    float softmax[16];
    for (int k = 0; k < 16; k++) {
        float val = enlight_get_tensor_data_by_off(t, y, x, start + k);
        sum_exp += expf(val);
    }
    if (sum_exp <= 0.0f) sum_exp = 1e-6f;
    for (int k = 0; k < 16; k++) {
        float val = enlight_get_tensor_data_by_off(t, y, x, start + k);
        softmax[k] = expf(val) / sum_exp;
    }
    float d = 0.0f;
    for (int k = 0; k < 16; k++) d += (float)k * softmax[k];
    return d;
}

static float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

typedef struct {
    float x1, y1, x2, y2;
    float conf;
    int   class_id;
    float dep_m; // depth (meters) for logging
} detection_t;

static int compare_dets(const void* a, const void* b) {
    float ca = ((const detection_t*)a)->conf;
    float cb = ((const detection_t*)b)->conf;
    return (ca < cb) ? 1 : ((ca > cb) ? -1 : 0);
}

static float box_iou(float x1, float y1, float x2, float y2,
                     float ax1, float ay1, float ax2, float ay2) {
    float xx1 = fmaxf(x1, ax1);
    float yy1 = fmaxf(y1, ay1);
    float xx2 = fminf(x2, ax2);
    float yy2 = fminf(y2, ay2);
    float inter = fmaxf(0.0f, xx2 - xx1) * fmaxf(0.0f, yy2 - yy1);
    float area1 = fmaxf(0.0f, x2 - x1) * fmaxf(0.0f, y2 - y1);
    float area2 = fmaxf(0.0f, ax2 - ax1) * fmaxf(0.0f, ay2 - ay1);
    float denom = area1 + area2 - inter + 1e-6f;
    return (denom > 0.0f) ? (inter / denom) : 0.0f;
}

static void apply_nms(detection_t* dets, int num, float iou_thres) {
    for (int i = 0; i < num; i++) {
        if (dets[i].conf == 0.0f) continue;
        for (int j = i + 1; j < num; j++) {
            if (dets[j].conf == 0.0f) continue;
            if (dets[i].class_id != dets[j].class_id) continue;
            float iou = box_iou(dets[i].x1, dets[i].y1, dets[i].x2, dets[i].y2,
                                dets[j].x1, dets[j].y1, dets[j].x2, dets[j].y2);
            if (iou > iou_thres) dets[j].conf = 0.0f;
        }
    }
}

// Write CUSTOM payload as float array to avoid struct layout mismatch
// Format: [0]=count, then per det: [x1,y1,x2,y2,score,class_id] (6 floats)
static void write_custom_payload(void* post_output, const detection_t* dets, int det_count) {
    if (!post_output) {
        enlight_custom_log("ERROR: post_output is NULL!\n");
        return;
    }
    
    float* buf = (float*)post_output;
    enlight_custom_log("post_output address: %p\n", post_output);
    
    // Clear the buffer first to ensure clean state
    for (int i = 0; i < 1600; i++) buf[i] = 0.0f;  // Clear enough space for 256 boxes + count
    
    int write_count = 0;
    // Count valid (non-suppressed) first to keep contiguous boxes
    for (int i = 0; i < det_count; i++) {
        if (dets[i].conf > 0.0f) write_count++;
    }
    if (write_count > 256) write_count = 256;

    // Write count at index 0
    buf[0] = (float)write_count;
    
    // Write each box: [x1,y1,x2,y2,score,class_id]
    int w = 0;
    for (int i = 0; i < det_count && w < write_count; i++) {
        if (dets[i].conf <= 0.0f) continue;
        int base = 1 + w * 6;
        buf[base + 0] = dets[i].x1;
        buf[base + 1] = dets[i].y1;
        buf[base + 2] = dets[i].x2;
        buf[base + 3] = dets[i].y2;
        buf[base + 4] = dets[i].conf;
        buf[base + 5] = (float)dets[i].class_id;
        w++;
    }
    
    // Debug: log the first 12 floats to verify they're written correctly
    enlight_custom_log("Wrote %d boxes to CUSTOM buffer. First 12 floats:\n", write_count);
    for (int i = 0; i < 12; i++) {
        enlight_custom_log("  buf[%d] = %.3f\n", i, buf[i]);
    }
}

int custom_postproc_run(
    int num_output,
    enlight_act_tensor_t** output_tensors,
    void* post_output // CUSTOM output buffer (treated as float[])
)
{
    // Critical address logging - must match app's resultLane!
    enlight_custom_log("====== CUSTOM_POSTPROC_RUN CALLED ======\n");
    enlight_custom_log("post_output address: %p\n", post_output);
    enlight_custom_log("CRITICAL: This MUST match app's resultLane address!\n");
    
    // 🔴 CRITICAL: You need to pass original image dimensions here
    // This should come from your application - you need to modify the interface
    // For now, using placeholder values - YOU MUST UPDATE THESE!
    float original_width = 1920.0f;   // ⚠️ UPDATE: Replace with actual original image width
    float original_height = 1080.0f;  // ⚠️ UPDATE: Replace with actual original image height
    
    // Calculate letterbox parameters
    letterbox_params_t letterbox_params = calculate_letterbox_params(
        original_width, original_height, MODEL_INPUT_SIZE);
    
    int dims[4];

    // 1) Group tensors per level (H,W) and identify reg/cls/dep
    level_group_t groups[16]; int gnum = 0;
    for (int i = 0; i < 16; i++) {
        groups[i].H = groups[i].W = 0;
        groups[i].reg_idx = groups[i].cls_idx = groups[i].dep_idx = -1;
        groups[i].reg_C = groups[i].cls_C = 0;
    }

    for (int t = 0; t < num_output; t++) {
        enlight_act_tensor_t* tensor = output_tensors[t];
        if (!tensor) continue;
        enlight_get_tensor_dimensions(tensor, dims); // N,C,H,W
        int C = dims[1], H = dims[2], W = dims[3];
        int gi = find_or_add_level(groups, &gnum, H, W, 16);
        if (gi < 0) continue;

        if (C == 1) {
            if (groups[gi].dep_idx < 0) groups[gi].dep_idx = t;
        } else if (C >= 8 && (C % 4) == 0) {
            if (C > groups[gi].reg_C) {
                if (groups[gi].reg_idx >= 0) {
                    groups[gi].cls_idx = groups[gi].reg_idx;
                    groups[gi].cls_C = groups[gi].reg_C;
                }
                groups[gi].reg_idx = t;
                groups[gi].reg_C = C;
            } else if (C > groups[gi].cls_C) {
                groups[gi].cls_idx = t;
                groups[gi].cls_C = C;
            }
        } else if (C > 1) {
            if (groups[gi].cls_idx < 0 || C > groups[gi].cls_C) {
                groups[gi].cls_idx = t;
                groups[gi].cls_C = C;
            }
        }
    }

    // 2) Decode all levels
    const float conf_thres = 0.25f;
    const float iou_thres  = 0.35f;

    detection_t dets[10000];
    int det_count = 0;

    for (int gi = 0; gi < gnum; gi++) {
        level_group_t* G = &groups[gi];
        if (G->reg_idx < 0 || G->cls_idx < 0) {
            enlight_custom_log("Skipping level %d: missing reg or cls\n", gi);
            continue;
        }

        float stride = MODEL_INPUT_SIZE / (float)G->H;
        enlight_custom_log("\n[Level %d] HxW = %dx%d, stride=%g\n", gi, G->H, G->W, stride);

        enlight_act_tensor_t* tR = output_tensors[G->reg_idx];
        enlight_act_tensor_t* tC = output_tensors[G->cls_idx];
        enlight_act_tensor_t* tD = (G->dep_idx >= 0) ? output_tensors[G->dep_idx] : NULL;

        int num_bins = G->reg_C / 4;

        for (int y = 0; y < G->H; y++) {
            for (int x = 0; x < G->W; x++) {
                float d_l = compute_dfl_dir(tR, y, x, 0);
                float d_t = compute_dfl_dir(tR, y, x, num_bins);
                float d_r = compute_dfl_dir(tR, y, x, 2 * num_bins);
                float d_b = compute_dfl_dir(tR, y, x, 3 * num_bins);

                float anchor_x = (x + 0.5f);
                float anchor_y = (y + 0.5f);

                // 🔴 IMPORTANT: These coordinates are in letterbox space (640x640)
                float x1_letterbox = (anchor_x - d_l) * stride;
                float y1_letterbox = (anchor_y - d_t) * stride;
                float x2_letterbox = (anchor_x + d_r) * stride;
                float y2_letterbox = (anchor_y + d_b) * stride;

                float max_conf = 0.0f;
                int max_class = -1;
                for (int k = 0; k < G->cls_C; k++) {
                    float logit = enlight_get_tensor_data_by_off(tC, y, x, k);
                    float prob = sigmoid(logit);
                    if (prob > max_conf) { max_conf = prob; max_class = k; }
                }

                if (max_conf > conf_thres && det_count < (int)(sizeof(dets)/sizeof(dets[0]))) {
                    float dep_norm = (tD) ? sigmoid(enlight_get_tensor_data_by_off(tD, y, x, 0)) : 0.0f;
                    float dep_m = denormalize_depth(dep_norm);

                    // 🔴 CRITICAL FIX: Transform coordinates from letterbox space to original image space
                    float x1_orig = x1_letterbox;
                    float y1_orig = y1_letterbox;
                    float x2_orig = x2_letterbox;
                    float y2_orig = y2_letterbox;
                    
                    unscale_coords(&x1_orig, &y1_orig, &x2_orig, &y2_orig, &letterbox_params);

                    dets[det_count].x1 = x1_orig;
                    dets[det_count].y1 = y1_orig;
                    dets[det_count].x2 = x2_orig;
                    dets[det_count].y2 = y2_orig;
                    dets[det_count].conf = max_conf;
                    dets[det_count].class_id = max_class;
                    dets[det_count].dep_m = dep_m;
                    det_count++;
                }
            }
        }
    }

    // 3) Sort and NMS
    qsort(dets, det_count, sizeof(detection_t), compare_dets);
    apply_nms(dets, det_count, iou_thres);

    // 4) Logging (with depth)
    int final_count = 0;
    enlight_custom_log("\nTotal detections after NMS: %d\n", det_count);
    enlight_custom_log("All detections (sorted by conf):\n");
    for (int i = 0; i < det_count; i++) {
        if (dets[i].conf > 0.0f) {
            enlight_custom_log("  Box: (%.2f, %.2f, %.2f, %.2f), conf=%.3f, class=%d, depth=%.2fm\n",
                dets[i].x1, dets[i].y1, dets[i].x2, dets[i].y2, dets[i].conf, dets[i].class_id, dets[i].dep_m);
            final_count++;
        }
    }

    // 5) Write CUSTOM payload for app (count + 6 floats/box)
    write_custom_payload(post_output, dets, det_count);
    
    // 6) Final verification - read back what we wrote
    if (post_output) {
        float* verify_buf = (float*)post_output;
        enlight_custom_log("FINAL: Wrote to %p, count=%.0f\n", post_output, verify_buf[0]);
    }

    // Return final (non-suppressed) count
    return final_count;
}