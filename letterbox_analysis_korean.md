# Letterbox Preprocessing and Bounding Box Decoding Analysis

## Problem Analysis

당신의 C 코드에서 bbox가 이상한 곳에 찍히는 문제는 **letterbox 전처리와 좌표 변환** 때문입니다.

### 핵심 문제

1. **학습 시 letterbox 적용**: `train_dual_stream_5.sh`로 학습할 때 이미지가 letterbox로 전처리됩니다
2. **모델 출력 좌표계**: 모델은 letterbox된 좌표계(640x640)에서 bbox를 출력합니다  
3. **좌표 변환 누락**: 당신의 C 코드는 이 좌표를 원본 이미지 좌표계로 변환하지 않습니다

## Letterbox 전처리 과정

### 1. Letterbox란?
```python
# ultralytics/data/augment.py의 LetterBox 클래스
def __call__(self, labels=None, image=None):
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2
```

### 2. 좌표 변환 공식

**Forward (Original → Letterbox):**
```c
gain = min(640.0f / original_width, 640.0f / original_height);
pad_w = (640.0f - original_width * gain) / 2.0f;
pad_h = (640.0f - original_height * gain) / 2.0f;

letterbox_x = original_x * gain + pad_w;
letterbox_y = original_y * gain + pad_h;
```

**Reverse (Letterbox → Original):**
```c
original_x = (letterbox_x - pad_w) / gain;
original_y = (letterbox_y - pad_h) / gain;
```

## 당신 코드의 문제점

### 원본 코드:
```c
float x1 = (anchor_x - d_l) * stride;  // ❌ letterbox 좌표계
float y1 = (anchor_y - d_t) * stride;  // ❌ letterbox 좌표계
```

### 수정된 코드:
```c
// 1. letterbox 좌표로 먼저 계산
float x1_letterbox = (anchor_x - d_l) * stride;
float y1_letterbox = (anchor_y - d_t) * stride;

// 2. 원본 이미지 좌표로 변환
unscale_coords(&x1_orig, &y1_orig, &x2_orig, &y2_orig, &letterbox_params);
```

## 해결책

### 1. 주요 수정사항

1. **letterbox 파라미터 계산**:
   ```c
   typedef struct {
       float gain;      // scale factor
       float pad_w;     // width padding 
       float pad_h;     // height padding
       float orig_w;    // original image width
       float orig_h;    // original image height
   } letterbox_params_t;
   ```

2. **좌표 변환 함수**:
   ```c
   static void unscale_coords(float* x1, float* y1, float* x2, float* y2, 
                             const letterbox_params_t* params) {
       // Remove padding first
       *x1 -= params->pad_w;
       *y1 -= params->pad_h;
       *x2 -= params->pad_w;
       *y2 -= params->pad_h;
       
       // Scale back to original size
       *x1 /= params->gain;
       *y1 /= params->gain;
       *x2 /= params->gain;
       *y2 /= params->gain;
   }
   ```

### 2. 필요한 추가 작업

**⚠️ CRITICAL**: 원본 이미지 크기를 C 함수에 전달해야 합니다:
```c
// 현재 하드코딩된 값들을 실제 값으로 교체하세요
float original_width = 1920.0f;   // ← 실제 원본 이미지 너비
float original_height = 1080.0f;  // ← 실제 원본 이미지 높이
```

## 검증 방법

### 1. Python 참조 구현과 비교
`dual_stream_inference_depth.py`의 결과와 비교:
```python
# 이 함수는 올바른 letterbox 변환을 수행합니다
def postprocess_results_with_depth(predictions, original_size, target_size=640):
    scale = min(target_size / original_width, target_size / original_height)
    # ... padding 계산 및 좌표 변환
```

### 2. 좌표 변환 검증
```c
// 디버그 로그 추가
enlight_custom_log("Before unscale: (%.2f,%.2f,%.2f,%.2f)\n", 
                   x1_letterbox, y1_letterbox, x2_letterbox, y2_letterbox);
enlight_custom_log("After unscale: (%.2f,%.2f,%.2f,%.2f)\n", 
                   x1_orig, y1_orig, x2_orig, y2_orig);
```

## 핵심 포인트

1. **DFL 디코딩은 정확합니다** - 문제가 아닙니다
2. **letterbox 좌표 변환이 누락**되어 있었습니다
3. **원본 이미지 크기 정보**가 필요합니다
4. **ultralytics의 scale_boxes 함수**와 동일한 로직을 구현했습니다

수정된 코드를 사용하시면 bbox가 올바른 위치에 표시될 것입니다.