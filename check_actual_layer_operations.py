#!/usr/bin/env python3
"""
실제 SpatialAlignedMultiStreamConv 레이어의 연산을 확인하는 스크립트
"""

import torch
import sys
from pathlib import Path

# Add ultralytics to path
sys.path.insert(0, str(Path(__file__).parent))

from ultralytics.nn.modules.conv import SpatialAlignedMultiStreamConv

# 테스트용 레이어 생성
print("="*70)
print("SpatialAlignedMultiStreamConv Layer Structure Analysis")
print("="*70)

# YAML에서 사용되는 파라미터: [128, [512, 512], 3]
# c1=64 (이전 레이어 출력), c2=128, max_hw=[512, 512], k=3
c1 = 64
c2 = 128
max_hw = [512, 512]
k = 3

layer = SpatialAlignedMultiStreamConv(c1=c1, c2=c2, max_hw=max_hw, k=k)

print(f"\nLayer Parameters:")
print(f"  Input channels (c1): {c1}")
print(f"  Output channels (c2): {c2}")
print(f"  Kernel size (k): {k}")
print(f"  Max HW: {max_hw}")

print(f"\nLayer Components:")
for name, module in layer.named_children():
    print(f"  {name}: {module}")

print(f"\n\nLet's check if there's a downscaler:")
if hasattr(layer, 'downscaler'):
    print(f"  ✓ downscaler exists: {layer.downscaler}")
else:
    print(f"  ✗ NO downscaler found!")

if hasattr(layer, 'upsampler'):
    print(f"  ✓ upsampler exists: {layer.upsampler}")
    for i, sublayer in enumerate(layer.upsampler):
        print(f"      [{i}] {sublayer}")

print(f"\n{'='*70}")
print(f"Forward Pass Simulation")
print(f"{'='*70}")

# 테스트 입력 생성 (MultiStreamConv 출력: 320x320)
batch_size = 1
H, W = 320, 320
test_input = torch.randn(batch_size, c1, H, W)

print(f"\nInput shape: {test_input.shape}")

# Forward pass
with torch.no_grad():
    # 중간 단계 확인
    wide = layer.split_wide(test_input)
    narrow = layer.split_narrow(test_input)
    print(f"After split:")
    print(f"  wide: {wide.shape}")
    print(f"  narrow: {narrow.shape}")
    
    wide_proc = layer.wide_processor(wide)
    narrow_proc = layer.narrow_processor(narrow)
    print(f"After processor:")
    print(f"  wide_proc: {wide_proc.shape}")
    print(f"  narrow_proc: {narrow_proc.shape}")
    
    # Check if downscaling happens
    if hasattr(layer, 'downscaler'):
        narrow_down = layer.downscaler(narrow_proc)
        print(f"After downscaler:")
        print(f"  narrow_down: {narrow_down.shape}")
        narrow_to_upsample = narrow_down
    else:
        print(f"No downscaler - using narrow_proc directly")
        narrow_to_upsample = narrow_proc
    
    narrow_upsampled = layer.upsampler(narrow_to_upsample)
    print(f"After upsampler:")
    print(f"  narrow_upsampled: {narrow_upsampled.shape}")
    
    # Cropping if needed
    if narrow_upsampled.shape[2:] != wide_proc.shape[2:]:
        target_h, target_w = wide_proc.shape[2:]
        narrow_upsampled_cropped = narrow_upsampled[:, :, :target_h, :target_w]
        print(f"After cropping to match wide:")
        print(f"  narrow_upsampled_cropped: {narrow_upsampled_cropped.shape}")
    else:
        narrow_upsampled_cropped = narrow_upsampled
        print(f"No cropping needed - shapes already match")
    
    fused = wide_proc + narrow_upsampled_cropped
    print(f"After fusion (wide + narrow):")
    print(f"  fused: {fused.shape}")
    
    output = layer.fusion_conv(fused)
    print(f"After fusion_conv:")
    print(f"  output: {output.shape}")

print(f"\n{'='*70}")
print(f"CONCLUSION")
print(f"{'='*70}")
print(f"Current implementation:")
print(f"  1. wide_processor: {H}×{W} → {wide_proc.shape[2]}×{wide_proc.shape[3]} (stride=1)")
print(f"  2. narrow_processor: {H}×{W} → {narrow_proc.shape[2]}×{narrow_proc.shape[3]} (stride=1)")
if hasattr(layer, 'downscaler'):
    print(f"  3. downscaler: {narrow_proc.shape[2]}×{narrow_proc.shape[3]} → {narrow_down.shape[2]}×{narrow_down.shape[3]} (pool 2×2)")
    print(f"  4. upsampler: {narrow_down.shape[2]}×{narrow_down.shape[3]} → {narrow_upsampled.shape[2]}×{narrow_upsampled.shape[3]} (×2)")
else:
    print(f"  3. NO downscaler - skipped")
    print(f"  4. upsampler: {narrow_proc.shape[2]}×{narrow_proc.shape[3]} → {narrow_upsampled.shape[2]}×{narrow_upsampled.shape[3]} (×2)")
print(f"  5. fusion: add wide + narrow")
print(f"  6. output: {output.shape[2]}×{output.shape[3]}")
print(f"{'='*70}\n")
