#!/usr/bin/env python3
"""
실제 학습된 모델의 SpatialAlignedMultiStreamConv 레이어 구조를 분석합니다.
"""

import torch
import sys
from pathlib import Path

# Add ultralytics to path
sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO

# 모델 로드
model_path = "/home/byounggun/ultralytics/runs/train/exp354/weights/best.pt"

print("="*80)
print("ANALYZING TRAINED MODEL STRUCTURE")
print("="*80)
print(f"Model: {model_path}\n")

model = YOLO(model_path)

print("Model loaded successfully!")
print(f"Model type: {type(model.model)}")

# 모델 구조 출력
print("\n" + "="*80)
print("BACKBONE LAYERS")
print("="*80)

for idx, (name, module) in enumerate(model.model.model.named_children()):
    print(f"\nLayer {idx}: {name}")
    print(f"  Type: {type(module).__name__}")
    
    # 특별히 SpatialAlignedMultiStreamConv를 찾기
    if 'SpatialAligned' in type(module).__name__:
        print(f"\n  ⭐ FOUND SpatialAlignedMultiStreamConv!")
        print(f"  Components:")
        for subname, submodule in module.named_children():
            print(f"    - {subname}: {submodule}")
        
        # Check for downscaler
        if hasattr(module, 'downscaler'):
            print(f"\n  ✓ Has downscaler: {module.downscaler}")
        else:
            print(f"\n  ✗ NO downscaler attribute")
        
        # Check upsampler
        if hasattr(module, 'upsampler'):
            print(f"  ✓ Has upsampler:")
            if isinstance(module.upsampler, torch.nn.Sequential):
                for i, layer in enumerate(module.upsampler):
                    print(f"      [{i}] {layer}")
            else:
                print(f"      {module.upsampler}")
        
        # Test forward pass
        print(f"\n  Testing forward pass:")
        test_input = torch.randn(1, 64, 320, 320)  # After MultiStreamConv
        print(f"    Input shape: {test_input.shape}")
        
        with torch.no_grad():
            # Manual step-by-step
            wide = module.split_wide(test_input)
            narrow = module.split_narrow(test_input)
            print(f"    After split: wide={wide.shape}, narrow={narrow.shape}")
            
            wide_proc = module.wide_processor(wide)
            narrow_proc = module.narrow_processor(narrow)
            print(f"    After processor: wide_proc={wide_proc.shape}, narrow_proc={narrow_proc.shape}")
            
            # Check if downscaler exists and is used
            if hasattr(module, 'downscaler'):
                narrow_down = module.downscaler(narrow_proc)
                print(f"    After downscaler: narrow_down={narrow_down.shape}")
                input_to_upsample = narrow_down
            else:
                print(f"    No downscaler - using narrow_proc directly")
                input_to_upsample = narrow_proc
            
            narrow_upsampled = module.upsampler(input_to_upsample)
            print(f"    After upsampler: narrow_upsampled={narrow_upsampled.shape}")
            
            # Cropping
            if narrow_upsampled.shape[2:] != wide_proc.shape[2:]:
                target_h, target_w = wide_proc.shape[2:]
                narrow_final = narrow_upsampled[:, :, :target_h, :target_w]
                print(f"    After crop: narrow_final={narrow_final.shape}")
            else:
                narrow_final = narrow_upsampled
                print(f"    No crop needed")
            
            fused = wide_proc + narrow_final
            print(f"    After fusion (add): fused={fused.shape}")
            
            output = module.fusion_conv(fused)
            print(f"    Final output: {output.shape}")

print("\n" + "="*80)
print("LAYER SEQUENCE IN BACKBONE")
print("="*80)

# 전체 백본 시퀀스 확인
print("\nFirst few layers:")
for idx in range(min(5, len(model.model.model))):
    layer = model.model.model[idx]
    print(f"  [{idx}] {type(layer).__name__}")
    
    # 입력 테스트
    if idx == 0:
        test_input = torch.randn(1, 6, 640, 640)  # Initial input
        print(f"      Input: {test_input.shape}")
    
    with torch.no_grad():
        test_output = layer(test_input)
        print(f"      Output: {test_output.shape}")
        test_input = test_output

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("Layer flow:")
print("  [0] MultiStreamConv: 640×640 → 320×320")
print("  [1] SpatialAlignedMultiStreamConv:")
if hasattr(model.model.model[1], 'downscaler'):
    print("      • Has downscaler: narrow 320×320 → 160×160")
    print("      • Has upsampler: narrow 160×160 → 320×320")
else:
    print("      • NO downscaler")
    print("      • Has upsampler: narrow 320×320 → 640×640")
print("      • Fusion: wide + narrow (addition)")
print("  [2] Conv: next layer downsampling")
print("="*80 + "\n")
