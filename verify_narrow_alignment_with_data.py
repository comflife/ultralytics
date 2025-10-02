#!/usr/bin/env python3
"""
실제 데이터를 사용하여 narrow 이미지가 wide 이미지 위에 어떻게 정렬되는지 확인하는 스크립트.
SpatialAlignedMultiStreamConv의 처리 과정을 시각화합니다.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image

# 데이터셋 경로
WIDE_DIR = "/home/byounggun/ultralytics/swm_dual_split/train/images"
NARROW_DIR = "/home/byounggun/ultralytics/swm_dual_split/train/train_narrow_images"

# Narrow 이미지 ROI 정보 (SpatialAlignedMultiStreamConv에서 사용)
NARROW_BBOX = {
    'center_x': 0.499289,
    'center_y': 0.499912,
    'width': 0.286041,
    'height': 0.291975
}


def get_image_pairs(wide_dir, narrow_dir, num_samples=5):
    """Wide와 Narrow 이미지 쌍을 찾습니다."""
    wide_dir = Path(wide_dir)
    narrow_dir = Path(narrow_dir)
    
    if not wide_dir.exists():
        print(f"❌ Wide directory not found: {wide_dir}")
        return []
    if not narrow_dir.exists():
        print(f"❌ Narrow directory not found: {narrow_dir}")
        return []
    
    # 이미지 파일 찾기
    wide_images = sorted(list(wide_dir.glob("*.jpg")) + list(wide_dir.glob("*.png")))
    narrow_images = sorted(list(narrow_dir.glob("*.jpg")) + list(narrow_dir.glob("*.png")))
    
    print(f"📂 Found {len(wide_images)} wide images")
    print(f"📂 Found {len(narrow_images)} narrow images")
    
    # 파일명 기반으로 매칭 (확장자 제외)
    pairs = []
    for wide_path in wide_images[:num_samples]:
        wide_stem = wide_path.stem
        # Narrow 이미지 찾기
        narrow_candidates = [n for n in narrow_images if n.stem == wide_stem]
        if narrow_candidates:
            pairs.append((wide_path, narrow_candidates[0]))
        else:
            print(f"⚠️  No matching narrow image for: {wide_path.name}")
    
    return pairs


def simulate_spatial_alignment(wide_img, narrow_img, target_size=(640, 640)):
    """
    SpatialAlignedMultiStreamConv의 처리 과정을 정확히 시뮬레이션합니다.
    
    ✅ 실제 모델 분석 결과 (best.pt):
    
    0. 입력: [B, 6, 640, 640] (wide 3ch + narrow 3ch concatenated)
    
    1. MultiStreamConv [64, 3, 2]: 640×640 → 320×320 (stride=2 다운샘플링)
    
    2. SpatialAlignedMultiStreamConv [128, [512, 512], 3]:
       - split: 16채널 → wide(8ch) + narrow(8ch)
       - wide_processor: Conv(3×3, s=1) → 8ch, 320×320 유지
       - narrow_processor: Conv(3×3, s=1) → 8ch, 320×320 유지
       - ⚠️ NO DOWNSCALER! (다운샘플링 없음)
       - upsampler: Upsample(×2, nearest) + Conv(3×3) → 8ch, 640×640 (2배 확대!)
       - narrow cropped: 640×640 → 320×320 (center crop to match wide)
       - fused: wide(320×320) + narrow_cropped(320×320) → Addition
       - fusion_conv: Conv → 32ch 출력
    
    3. Conv [128, 3, 2]: 320×320 → 160×160 (stride=2 다운샘플링)
    
    핵심: Narrow는 downsampling 없이 바로 2배 upsampling됨!
          그래서 wide(320×320)보다 큰 640×640이 된 후 crop됨
    """
    H, W = target_size
    
    # 1. 이미지 리사이즈 (numpy array로 변환)
    if isinstance(wide_img, Image.Image):
        wide_img = np.array(wide_img)
    if isinstance(narrow_img, Image.Image):
        narrow_img = np.array(narrow_img)
    
    # PIL로 리사이즈
    wide_pil = Image.fromarray(wide_img)
    narrow_pil = Image.fromarray(narrow_img)
    wide_resized = np.array(wide_pil.resize((W, H), Image.BILINEAR))
    narrow_resized = np.array(narrow_pil.resize((W, H), Image.BILINEAR))
    
    print(f"   Initial size: {W}×{H}")
    
    # 2. MultiStreamConv 시뮬레이션: stride=2 다운샘플링
    # 640×640 → 320×320
    after_multistream_size = (H // 2, W // 2)
    wide_after_ms = np.array(Image.fromarray(wide_resized).resize(
        (after_multistream_size[1], after_multistream_size[0]), Image.BILINEAR))
    narrow_after_ms = np.array(Image.fromarray(narrow_resized).resize(
        (after_multistream_size[1], after_multistream_size[0]), Image.BILINEAR))
    
    print(f"   After MultiStreamConv (s=2): {after_multistream_size[1]}×{after_multistream_size[0]}")
    
    # 3. SpatialAlignedMultiStreamConv 시뮬레이션
    # Wide processor: 320×320 유지 (stride=1)
    wide_tensor = torch.from_numpy(wide_after_ms).permute(2, 0, 1).unsqueeze(0).float()
    narrow_tensor = torch.from_numpy(narrow_after_ms).permute(2, 0, 1).unsqueeze(0).float()
    
    # Wide processor: Conv(3×3, s=1) - 해상도 유지
    wide_proc = wide_tensor  # 시각화용으로 단순화 (실제로는 Conv 통과)
    print(f"   Wide after processor (s=1): {wide_proc.shape}")
    
    # Narrow processor: Conv(3×3, s=1) - 해상도 유지
    narrow_proc = narrow_tensor  # 시각화용으로 단순화
    print(f"   Narrow after processor (s=1): {narrow_proc.shape}")
    
    # ⚠️ NO DOWNSCALING! narrow_proc를 바로 업샘플링
    # Narrow upsampler: Upsample(×2, nearest) + Conv(3×3)
    # 320×320 → 640×640 (2배 확대!)
    upsampler = torch.nn.Upsample(scale_factor=2, mode='nearest')
    narrow_upsampled = upsampler(narrow_proc)
    print(f"   Narrow after upsampling (×2, NO downscale before!): {narrow_upsampled.shape}")
    
    # 해상도 맞추기: narrow(640×640)를 wide(320×320) 크기로 crop
    # 실제 코드: if narrow_upsampled.shape[2:] != wide_proc.shape[2:]:
    #               narrow_upsampled = narrow_upsampled[:, :, :target_h, :target_w]
    target_h, target_w = wide_proc.shape[2:]
    if narrow_upsampled.shape[2:] != (target_h, target_w):
        # Top-left crop (실제 코드처럼)
        narrow_upsampled_cropped = narrow_upsampled[:, :, :target_h, :target_w]
        print(f"   Narrow after crop (top-left) to match wide: {narrow_upsampled_cropped.shape}")
    else:
        narrow_upsampled_cropped = narrow_upsampled
        print(f"   Narrow - no crop needed (shapes already match)")
    
    # 4. Fusion: wide + narrow (element-wise addition)
    # 실제 코드: fused = wide_proc + narrow_upsampled
    fused = wide_proc + narrow_upsampled_cropped
    print(f"   Fused (wide + narrow): {fused.shape}")
    
    # Tensor를 numpy로 변환 (시각화용)
    wide_proc_np = torch.clamp(wide_proc, 0, 255).squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
    narrow_proc_np = torch.clamp(narrow_proc, 0, 255).squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
    narrow_upsampled_np = torch.clamp(narrow_upsampled, 0, 255).squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
    narrow_cropped_np = torch.clamp(narrow_upsampled_cropped, 0, 255).squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
    fused_np = torch.clamp(fused, 0, 255).squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
    
    # 5. 시각화를 위한 오버레이 생성
    # Narrow 업샘플링 결과를 wide 위에 오버레이 (crop된 부분만)
    overlay = wide_after_ms.copy()
    # Narrow cropped를 wide에 블렌딩
    overlay = (wide_proc_np * 0.5 + narrow_cropped_np * 0.5).astype(np.uint8)
    
    return {
        'wide_resized': wide_resized,
        'narrow_resized': narrow_resized,
        'wide_after_ms': wide_after_ms,
        'narrow_after_ms': narrow_after_ms,
        'wide_proc': wide_proc_np,
        'narrow_proc': narrow_proc_np,
        'narrow_upsampled': narrow_upsampled_np,
        'narrow_cropped': narrow_cropped_np,
        'fused': fused_np,
        'overlay': overlay,
        'sizes': {
            'initial': (H, W),
            'after_multistream': after_multistream_size,
            'narrow_after_proc': narrow_proc.shape[2:],
            'narrow_upsampled': narrow_upsampled.shape[2:],
            'final_fused': fused.shape[2:]
        }
    }


def visualize_alignment(wide_path, narrow_path, output_path):
    """이미지 정렬 과정을 시각화합니다."""
    
    print(f"\n{'='*70}")
    print(f"Processing: {wide_path.name}")
    print(f"{'='*70}")
    
    # 이미지 로드 (PIL 사용)
    try:
        wide_img = Image.open(str(wide_path)).convert('RGB')
        narrow_img = Image.open(str(narrow_path)).convert('RGB')
    except Exception as e:
        print(f"❌ Failed to load images: {e}")
        return False
    
    # numpy array로 변환
    wide_img = np.array(wide_img)
    narrow_img = np.array(narrow_img)
    
    print(f"✓ Wide image shape: {wide_img.shape}")
    print(f"✓ Narrow image shape: {narrow_img.shape}")
    
    # 정렬 시뮬레이션
    results = simulate_spatial_alignment(wide_img, narrow_img, target_size=(640, 640))
    
    sizes = results['sizes']
    print(f"✓ Processing sizes:")
    print(f"   Initial: {sizes['initial']}")
    print(f"   After MultiStreamConv: {sizes['after_multistream']}")
    print(f"   Narrow after processor: {sizes['narrow_after_proc']}")
    print(f"   Narrow upsampled: {sizes['narrow_upsampled']}")
    print(f"   Final fused: {sizes['final_fused']}")
    
    # 시각화 생성 - 4x3 그리드로 변경
    fig = plt.figure(figsize=(20, 20))
    
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)
    
    # Row 1: 원본 이미지들
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(results['wide_resized'])
    ax1.set_title(f'1. Wide Input\n({sizes["initial"][1]}×{sizes["initial"][0]})', 
                 fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(results['narrow_resized'])
    ax2.set_title(f'2. Narrow Input\n({sizes["initial"][1]}×{sizes["initial"][0]})', 
                 fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.text(0.5, 0.5, 
            'Concatenated\nInput\n\n[Wide 3ch + Narrow 3ch]\n= 6 channels',
            ha='center', va='center', fontsize=11, fontweight='bold',
            transform=ax3.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax3.axis('off')
    
    # Row 2: MultiStreamConv 결과
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(results['wide_after_ms'])
    ax4.set_title(f'3. Wide after MultiStreamConv\n(stride=2: {sizes["after_multistream"][1]}×{sizes["after_multistream"][0]})', 
                 fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(results['narrow_after_ms'])
    ax5.set_title(f'4. Narrow after MultiStreamConv\n(stride=2: {sizes["after_multistream"][1]}×{sizes["after_multistream"][0]})', 
                 fontsize=12, fontweight='bold')
    ax5.axis('off')
    
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.text(0.5, 0.5, 
            'MultiStreamConv\n\nStride=2\nDownsampling\n\n640×640 → 320×320',
            ha='center', va='center', fontsize=11, fontweight='bold',
            transform=ax6.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax6.axis('off')
    
    # Row 3: SpatialAlignedMultiStreamConv - Processing
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.imshow(results['wide_proc'])
    ax7.set_title(f'5. Wide Processor\n(Conv s=1: {sizes["after_multistream"][1]}×{sizes["after_multistream"][0]} 유지)', 
                 fontsize=12, fontweight='bold')
    ax7.axis('off')
    
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.imshow(results['narrow_proc'])
    ax8.set_title(f'6. Narrow Processor\n(Conv s=1: {sizes["narrow_after_proc"][1]}×{sizes["narrow_after_proc"][0]} 유지)', 
                 fontsize=12, fontweight='bold')
    ax8.axis('off')
    
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.imshow(results['narrow_upsampled'])
    ax9.set_title(f'7. Narrow UPSAMPLED! ⚠️\n(×2: {sizes["narrow_upsampled"][1]}×{sizes["narrow_upsampled"][0]})', 
                 fontsize=12, fontweight='bold', color='red')
    ax9.axis('off')
    
    # Row 4: Fusion
    ax10 = fig.add_subplot(gs[3, 0])
    ax10.imshow(results['narrow_cropped'])
    ax10.set_title(f'8. Narrow Cropped (top-left)\n({sizes["final_fused"][1]}×{sizes["final_fused"][0]}) to match wide', 
                 fontsize=12, fontweight='bold')
    ax10.axis('off')
    
    ax11 = fig.add_subplot(gs[3, 1])
    ax11.imshow(results['overlay'])
    ax11.set_title(f'9. Overlay (0.5×wide + 0.5×narrow)\n({sizes["final_fused"][1]}×{sizes["final_fused"][0]})', 
                 fontsize=12, fontweight='bold')
    ax11.axis('off')
    
    ax12 = fig.add_subplot(gs[3, 2])
    ax12.imshow(results['fused'])
    ax12.set_title(f'10. FUSED (wide + narrow)\n({sizes["final_fused"][1]}×{sizes["final_fused"][0]})', 
                 fontsize=12, fontweight='bold', color='green')
    ax12.axis('off')
    
    # 전체 제목
    fig.suptitle(f'SpatialAlignedMultiStreamConv: NO Downscaling, Direct ×2 Upsampling!\n{wide_path.name}', 
                fontsize=16, fontweight='bold', y=0.995)
    
    # 정보 텍스트 추가
    info_text = (
        f"� Actual Implementation (from best.pt model):\n"
        f"1. MultiStreamConv: {sizes['initial'][1]}×{sizes['initial'][0]} → {sizes['after_multistream'][1]}×{sizes['after_multistream'][0]} (stride=2)\n"
        f"2. Wide Processor: Conv(3×3, s=1) - {sizes['after_multistream'][1]}×{sizes['after_multistream'][0]} (no change)\n"
        f"3. Narrow Processor: Conv(3×3, s=1) - {sizes['narrow_after_proc'][1]}×{sizes['narrow_after_proc'][0]} (no change)\n"
        f"4. ⚠️  NO DOWNSCALER - narrow goes directly to upsampler!\n"
        f"5. Narrow Upsampler: Upsample(×2, nearest) + Conv - {sizes['narrow_after_proc'][1]}×{sizes['narrow_after_proc'][0]} → {sizes['narrow_upsampled'][1]}×{sizes['narrow_upsampled'][0]}\n"
        f"6. Crop narrow (top-left): {sizes['narrow_upsampled'][1]}×{sizes['narrow_upsampled'][0]} → {sizes['final_fused'][1]}×{sizes['final_fused'][0]}\n"
        f"7. Fusion: wide + narrow (element-wise addition)"
    )
    fig.text(0.5, 0.005, info_text, ha='center', fontsize=9, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.95), family='monospace')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved visualization to: {output_path}")
    
    return True


def create_comparison_grid(image_pairs, output_path='alignment_comparison.png'):
    """여러 이미지 쌍의 정렬 결과를 그리드로 비교합니다."""
    
    if not image_pairs:
        print("❌ No image pairs to process")
        return
    
    num_pairs = len(image_pairs)
    fig, axes = plt.subplots(num_pairs, 3, figsize=(18, 6*num_pairs))
    
    if num_pairs == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (wide_path, narrow_path) in enumerate(image_pairs):
        print(f"\n[{idx+1}/{num_pairs}] Processing {wide_path.name}...")
        
        # 이미지 로드 (PIL 사용)
        try:
            wide_img = np.array(Image.open(str(wide_path)).convert('RGB'))
            narrow_img = np.array(Image.open(str(narrow_path)).convert('RGB'))
        except Exception as e:
            print(f"   ⚠️ Skipping due to load error: {e}")
            continue
        
        # 정렬 시뮬레이션
        results = simulate_spatial_alignment(wide_img, narrow_img)
        sizes = results['sizes']
        
        # 플롯
        axes[idx, 0].imshow(results['wide_after_ms'])
        axes[idx, 0].set_title(f'Wide ({sizes["after_multistream"][1]}×{sizes["after_multistream"][0]}): {wide_path.name}', fontsize=9)
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(results['narrow_upsampled'])
        axes[idx, 1].set_title(f'Narrow Upsampled (×2: {sizes["narrow_upsampled"][1]}×{sizes["narrow_upsampled"][0]})', fontsize=9)
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(results['fused'])
        axes[idx, 2].set_title(f'Fused ({sizes["final_fused"][1]}×{sizes["final_fused"][0]})', fontsize=9)
        axes[idx, 2].axis('off')
    
    plt.suptitle('Dual-Stream Alignment Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Saved comparison grid to: {output_path}")


def main():
    print("\n" + "="*70)
    print("🔍 DUAL-STREAM ALIGNMENT VERIFICATION WITH REAL DATA")
    print("="*70)
    
    # 이미지 쌍 찾기
    image_pairs = get_image_pairs(WIDE_DIR, NARROW_DIR, num_samples=5)
    
    if not image_pairs:
        print("\n❌ No image pairs found. Please check your dataset paths.")
        return
    
    print(f"\n✓ Found {len(image_pairs)} image pairs to process\n")
    
    # 출력 디렉토리 생성
    output_dir = Path("alignment_verification_results")
    output_dir.mkdir(exist_ok=True)
    
    # 각 이미지 쌍에 대해 상세 시각화
    success_count = 0
    for idx, (wide_path, narrow_path) in enumerate(image_pairs):
        output_path = output_dir / f"alignment_{idx+1}_{wide_path.stem}.png"
        if visualize_alignment(wide_path, narrow_path, output_path):
            success_count += 1
    
    # 비교 그리드 생성
    if success_count > 0:
        print(f"\n{'='*70}")
        print("Creating comparison grid...")
        create_comparison_grid(image_pairs, output_dir / "alignment_comparison.png")
    
    # 요약
    print(f"\n{'='*70}")
    print("📊 SUMMARY")
    print(f"{'='*70}")
    print(f"✓ Processed: {success_count}/{len(image_pairs)} image pairs")
    print(f"✓ Output directory: {output_dir.absolute()}")
    print(f"\n💡 Key findings (VERIFIED FROM ACTUAL MODEL):")
    print(f"   ")
    print(f"   Layer 0: MultiStreamConv [64, 3, 2]")
    print(f"            640×640 → 320×320 (stride=2 downsampling)")
    print(f"   ")
    print(f"   Layer 1: SpatialAlignedMultiStreamConv [128, [512, 512], 3]")
    print(f"            ✓ Wide processor: Conv(3×3, s=1) → 320×320 (no change)")
    print(f"            ✓ Narrow processor: Conv(3×3, s=1) → 320×320 (no change)")
    print(f"            ✗ NO DOWNSCALER! (verified from best.pt)")
    print(f"            ✓ Narrow upsampler: Upsample(×2, nearest) + Conv")
    print(f"                                 320×320 → 640×640 (2배 확대!)")
    print(f"            ✓ Narrow cropped: 640×640 → 320×320 (top-left crop to match wide)")
    print(f"            ✓ Fusion: wide(320×320) + narrow_cropped(320×320)")
    print(f"   ")
    print(f"   Layer 2: Conv [128, 3, 2]")
    print(f"            320×320 → 160×160 (stride=2 downsampling)")
    print(f"   ")
    print(f"   ⚠️  CRITICAL: Narrow는 downsampling 없이 바로 ×2 upsampling됨!")
    print(f"       → Narrow view가 더 높은 해상도로 확대되어 wide와 융합")
    print(f"       → Top-left 부분이 crop되어 wide와 정렬")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
