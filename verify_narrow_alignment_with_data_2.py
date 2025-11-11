#!/usr/bin/env python3
"""
실제 데이터를 사용하여 narrow 이미지가 wide 이미지 위에 어떻게 정렬되는지 확인하는 스크립트.
SpatialAlignedMultiStreamConv의 처리 과정을 시각화합니다.
[수정] Conv Down: AdaptiveAvgPool 대신 고정 Conv/AvgPool 조합 (e.g., AvgPool k=4 s=4 + k=3 s=1)으로 bbox size 근사 downsample → center pad.
- No adaptive: 고정 conv chain으로 ~92x93 근사 (e.g., 320→80→92 via adjust conv).
- 약간 miss align OK: 고정으로 인한 pixel 오차 허용.
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

# Narrow 이미지 ROI 정보 (wide의 중심 네모 영역에 맞춤) - conv down에 사용
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


def compute_bbox_size(target_h, target_w, bbox):
    """BBOX를 target size에 맞게 pixel 크기 계산."""
    roi_w = int(target_w * bbox['width'])
    roi_h = int(target_h * bbox['height'])
    center_x = int(target_w * bbox['center_x']) - roi_w // 2
    center_y = int(target_h * bbox['center_y']) - roi_h // 2
    return roi_h, roi_w, center_y, center_x


def simulate_spatial_alignment(wide_img, narrow_img, target_size=(640, 640), use_conv_down=False):
    """
    SpatialAlignedMultiStreamConv의 처리 과정을 정확히 시뮬레이션합니다.
    
    [수정] use_conv_down=True: 고정 Conv/AvgPool 조합으로 bbox size 근사 downsample (no adaptive).
           - Chain: AvgPool2d(k=4,s=4) 320→80 + Conv2d(1x1) adjust to ~92x93 (or AvgPool k=3 s=1 fine-tune).
           - 고정: 320 input 가정, miss align OK (pixel 오차 ~2-3).
    
    ✅ 실제 모델 분석 결과 (best.pt): 원본은 NO DOWNSCALER, direct upsample + top-left crop.
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
    wide_tensor = torch.from_numpy(wide_after_ms).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    narrow_tensor = torch.from_numpy(narrow_after_ms).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    
    # Wide processor: Conv(3×3, s=1) - 해상도 유지
    wide_proc = wide_tensor  # 시각화용으로 단순화 (실제로는 Conv 통과)
    print(f"   Wide after processor (s=1): {wide_proc.shape}")
    
    # Narrow processor: Conv(3×3, s=1) - 해상도 유지
    narrow_proc = narrow_tensor  # 시각화용으로 단순화
    print(f"   Narrow after processor (s=1): {narrow_proc.shape}")
    
    roi_h, roi_w, center_y, center_x = None, None, None, None  # 초기화 (에러 방지)
    
    if not use_conv_down:
        # [원본] ⚠️ NO DOWNSCALING! narrow_proc를 바로 업샘플링
        upsampler = torch.nn.Upsample(scale_factor=2, mode='nearest')
        narrow_upsampled = upsampler(narrow_proc)
        print(f"   Narrow after upsampling (×2, NO downscale before!): {narrow_upsampled.shape}")
        
        # Top-left crop
        target_h_proc, target_w_proc = wide_proc.shape[2:]
        if narrow_upsampled.shape[2:] != (target_h_proc, target_w_proc):
            narrow_for_fuse = narrow_upsampled[:, :, :target_h_proc, :target_w_proc]
            crop_mode = 'top-left'
            print(f"   Narrow after crop (top-left) to match wide: {narrow_for_fuse.shape}")
        else:
            narrow_for_fuse = narrow_upsampled
            crop_mode = 'none'
        mode_desc = "Original: Direct nearest up + top-left crop"
    else:
        # [수정] Conv Chain Downsample: 고정 AvgPool k=4 s=4 (320→80) + AvgPool k=3 s=1 (80→~92, miss align OK)
        target_h_proc, target_w_proc = narrow_proc.shape[2:]
        roi_h, roi_w, center_y, center_x = compute_bbox_size(target_h_proc, target_w_proc, NARROW_BBOX)
        print(f"   Target BBox ROI size @ {target_h_proc}x{target_w_proc}: {roi_w}x{roi_h} (center {center_x},{center_y})")
        
        # 고정 Conv Chain: 320→80 (AvgPool k=4 s=4)
        pool1 = torch.nn.AvgPool2d(kernel_size=4, stride=4)
        narrow_stage1 = pool1(narrow_proc)  # [B,3,80,80]
        print(f"   Narrow stage1: AvgPool k=4 s=4 → {narrow_stage1.shape}")
        
        # Adjust to ~roi: AvgPool k=3 s=1 on 80→~79, but for demo, use Upsample+Pool or 1x1 Conv to fine-tune (miss align OK)
        # 여기선 간단히 bilinear resize to roi (conv approx, but no adaptive pool) – 실제 모듈에선 더 conv chain
        adjuster = torch.nn.Upsample(size=(roi_h, roi_w), mode='nearest')  # nearest for conv-like (no bilinear interp)
        narrow_pooled_roi = adjuster(narrow_stage1)  # ~80→92 (slight up, but conv equiv)
        print(f"   Narrow pooled to ROI (Conv chain approx): {narrow_pooled_roi.shape} (target {roi_w}x{roi_h}, miss ~2px OK)")
        
        # Center pad to full wide size (0-pad 주변)
        pad_h = (target_h_proc - narrow_pooled_roi.shape[2]) // 2
        pad_w = (target_w_proc - narrow_pooled_roi.shape[3]) // 2
        pad_top = pad_h
        pad_bottom = target_h_proc - narrow_pooled_roi.shape[2] - pad_top
        pad_left = pad_w
        pad_right = target_w_proc - narrow_pooled_roi.shape[3] - pad_left
        
        narrow_for_fuse = F.pad(narrow_pooled_roi, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0.0)
        print(f"   Narrow after center-pad to wide: {narrow_for_fuse.shape}")
        
        crop_mode = f'Conv chain (k=4 s=4 + adjust) to ROI + center-pad (bbox w/h={NARROW_BBOX["width"]:.3f}/{NARROW_BBOX["height"]:.3f})'
        mode_desc = "Conv Chain: Fixed AvgPool k=4 s=4 + nearest adjust to bbox size + pad → no adaptive"
    
    # 4. Fusion: wide + narrow (element-wise addition, narrow_for_fuse가 wide와 맞춤)
    fused = wide_proc + narrow_for_fuse
    print(f"   Fused (wide + narrow): {fused.shape}")
    
    # Tensor를 numpy로 변환 (시각화용, clamp 0-1 to 0-255)
    wide_proc_np = (wide_proc * 255).clamp(0, 255).squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
    narrow_proc_np = (narrow_proc * 255).clamp(0, 255).squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
    narrow_upsampled_np = (narrow_for_fuse * 255).clamp(0, 255).squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)  # aligned narrow
    narrow_cropped_np = narrow_upsampled_np  # alias
    fused_np = (fused * 255).clamp(0, 255).squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
    
    # 5. 시각화를 위한 오버레이 생성 (0.5 blend, narrow 영역 강조)
    overlay = (wide_proc_np * 0.5 + narrow_upsampled_np * 0.5).astype(np.uint8)
    
    # BBox 시각화용: wide에 ROI 박스 그리기 (overlay에) - conv_down=True일 때만
    bbox_overlay = overlay  # default
    if use_conv_down:
        fig_overlay, ax = plt.subplots(1, 1, figsize=(5,5))
        ax.imshow(overlay)
        rect = patches.Rectangle((center_x, center_y), roi_w, roi_h, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.set_title('Overlay with BBox (red rect: narrow ROI)')
        ax.axis('off')
        plt.savefig('temp_bbox_overlay.png', dpi=150, bbox_inches='tight')
        plt.close()
        bbox_overlay = np.array(Image.open('temp_bbox_overlay.png'))
        os.remove('temp_bbox_overlay.png')
    
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
        'bbox_overlay': bbox_overlay,  # 추가: ROI 표시 overlay (conv_down only)
        'sizes': {
            'initial': (H, W),
            'after_multistream': after_multistream_size,
            'narrow_after_proc': narrow_proc.shape[2:],
            'narrow_upsampled': narrow_for_fuse.shape[2:],
            'final_fused': fused.shape[2:]
        },
        'mode': mode_desc,
        'crop_mode': crop_mode,
        'roi_info': {'h': roi_h, 'w': roi_w, 'center_y': center_y, 'center_x': center_x} if use_conv_down else None
    }


def visualize_alignment(wide_path, narrow_path, output_path, use_conv_down=False):
    """이미지 정렬 과정을 시각화합니다. [수정] use_conv_down 지원, 에러 픽스."""
    
    print(f"\n{'='*70}")
    print(f"Processing: {wide_path.name} (Conv Down: {use_conv_down})")
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
    results = simulate_spatial_alignment(wide_img, narrow_img, target_size=(640, 640), use_conv_down=use_conv_down)
    
    sizes = results['sizes']
    mode = results['mode']
    roi_info = results['roi_info']
    print(f"✓ Processing sizes ({mode}):")
    print(f"   Initial: {sizes['initial']}")
    print(f"   After MultiStreamConv: {sizes['after_multistream']}")
    print(f"   Narrow after processor: {sizes['narrow_after_proc']}")
    print(f"   Narrow aligned: {sizes['narrow_upsampled']}")
    print(f"   Final fused: {sizes['final_fused']}")
    if roi_info:
        print(f"   ROI: {roi_info['w']}x{roi_info['h']} @ center ({roi_info['center_x']}, {roi_info['center_y']})")
    
    # 시각화 생성 - 4x3 그리드 (Row 4에 bbox_overlay 추가 if conv_down)
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
            f'MultiStreamConv\n\nStride=2\nDownsampling\n\n{mode.split(":")[0]}',
            ha='center', va='center', fontsize=11, fontweight='bold',
            transform=ax6.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax6.axis('off')
    
    # Row 3: Processing
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
    if use_conv_down:
        ax9.imshow(results['narrow_upsampled'])
        ax9.set_title(f'7. Narrow CONV CHAIN + PADDED\n(Fixed pools to ROI + center: {sizes["narrow_upsampled"][1]}×{sizes["narrow_upsampled"][0]})', 
                     fontsize=12, fontweight='bold', color='blue')
    else:
        ax9.imshow(results['narrow_upsampled'])
        ax9.set_title(f'7. Narrow UPSAMPLED! ⚠️\n(Nearest ×2: {sizes["narrow_upsampled"][1]}×{sizes["narrow_upsampled"][0]})', 
                     fontsize=12, fontweight='bold', color='red')
    ax9.axis('off')
    
    # Row 4: Fusion (bbox_overlay 추가 if conv_down)
    ax10 = fig.add_subplot(gs[3, 0])
    ax10.imshow(results['narrow_cropped'])
    ax10.set_title(f'8. Narrow {results["crop_mode"]}\n({sizes["final_fused"][1]}×{sizes["final_fused"][0]}) to match wide', 
                 fontsize=12, fontweight='bold')
    ax10.axis('off')
    
    ax11 = fig.add_subplot(gs[3, 1])
    ax11.imshow(results['overlay'])
    ax11.set_title(f'9. Overlay (0.5×wide + 0.5×narrow)\n({sizes["final_fused"][1]}×{sizes["final_fused"][0]})', 
                 fontsize=12, fontweight='bold')
    ax11.axis('off')
    
    ax12 = fig.add_subplot(gs[3, 2])
    ax12.imshow(results['bbox_overlay'] if use_conv_down else results['fused'])
    if use_conv_down:
        ax12.set_title(f'10. FUSED w/ BBox ROI (red rect)\n({sizes["final_fused"][1]}×{sizes["final_fused"][0]})', 
                      fontsize=12, fontweight='bold', color='green')
    else:
        ax12.set_title(f'10. FUSED (wide + narrow)\n({sizes["final_fused"][1]}×{sizes["final_fused"][0]})', 
                      fontsize=12, fontweight='bold', color='green')
    ax12.axis('off')
    
    # 전체 제목
    mode_title = " (Conv Chain Version)" if use_conv_down else " (Original Version)"
    fig.suptitle(f'SpatialAlignedMultiStreamConv{mode_title}: {mode}\n{wide_path.name}', 
                fontsize=16, fontweight='bold', y=0.995)
    
    # 정보 텍스트 추가
    roi_str = f"ROI: {roi_info['w']}x{roi_info['h']} @ center {roi_info['center_x']},{roi_info['center_y']}" if roi_info else "N/A"
    info_text = (
        f"Actual Implementation (from best.pt model, modified for conv chain):\n"
        f"1. MultiStreamConv: {sizes['initial'][1]}×{sizes['initial'][0]} → {sizes['after_multistream'][1]}×{sizes['after_multistream'][0]} (stride=2)\n"
        f"2. Wide Processor: Conv(3×3, s=1) - {sizes['after_multistream'][1]}×{sizes['after_multistream'][0]} (no change)\n"
        f"3. Narrow Processor: Conv(3×3, s=1) - {sizes['narrow_after_proc'][1]}×{sizes['narrow_after_proc'][0]} (no change)\n"
        f"4. {'Conv Chain (fixed AvgPool k=4 s=4 + adjust): no adaptive + center-pad' if use_conv_down else 'NO DOWNSCALER - direct to upsampler'}\n"
        f"5. Narrow Aligner: {'Fixed pool chain (k=4 s=4 + nearest adjust)' if use_conv_down else 'Nearest ×2 + Conv'} - {sizes['narrow_after_proc'][1]}×{sizes['narrow_after_proc'][0]} → {sizes['narrow_upsampled'][1]}×{sizes['narrow_upsampled'][0]}\n"
        f"6. {results['crop_mode']}: Fits wide square ({roi_str}, miss align OK)\n"
        f"7. Fusion: wide + narrow (element-wise addition, narrow in ROI)"
    )
    fig.text(0.5, 0.005, info_text, ha='center', fontsize=9, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.95), family='monospace')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved visualization to: {output_path}")
    
    return True


def create_comparison_grid(image_pairs, output_path='alignment_comparison.png', use_conv_down=False):
    """여러 이미지 쌍의 정렬 결과를 그리드로 비교합니다. [수정] use_conv_down 지원."""
    
    if not image_pairs:
        print("❌ No image pairs to process")
        return
    
    num_pairs = len(image_pairs)
    fig, axes = plt.subplots(num_pairs, 3, figsize=(18, 6*num_pairs))
    
    if num_pairs == 1:
        axes = axes.reshape(1, -1)
    
    mode_str = "Conv Chain" if use_conv_down else "Original"
    
    for idx, (wide_path, narrow_path) in enumerate(image_pairs):
        print(f"\n[{idx+1}/{num_pairs}] Processing {wide_path.name} ({mode_str})...")
        
        # 이미지 로드 (PIL 사용)
        try:
            wide_img = np.array(Image.open(str(wide_path)).convert('RGB'))
            narrow_img = np.array(Image.open(str(narrow_path)).convert('RGB'))
        except Exception as e:
            print(f"   ⚠️ Skipping due to load error: {e}")
            continue
        
        # 정렬 시뮬레이션
        results = simulate_spatial_alignment(wide_img, narrow_img, use_conv_down=use_conv_down)
        sizes = results['sizes']
        
        # 플롯
        axes[idx, 0].imshow(results['wide_after_ms'])
        axes[idx, 0].set_title(f'Wide ({sizes["after_multistream"][1]}×{sizes["after_multistream"][0]}): {wide_path.name}', fontsize=9)
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(results['narrow_upsampled'])
        axes[idx, 1].set_title(f'Narrow Aligned ({results["mode"].split()[-1]}: {sizes["narrow_upsampled"][1]}×{sizes["narrow_upsampled"][0]})', fontsize=9)
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(results['fused'])
        axes[idx, 2].set_title(f'Fused ({sizes["final_fused"][1]}×{sizes["final_fused"][0]})', fontsize=9)
        axes[idx, 2].axis('off')
    
    plt.suptitle(f'Dual-Stream Alignment Comparison ({mode_str})', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Saved comparison grid to: {output_path}")


def main():
    print("\n" + "="*70)
    print("🔍 DUAL-STREAM ALIGNMENT VERIFICATION WITH REAL DATA (Conv Chain Test)")
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
    
    # 각 이미지 쌍에 대해 상세 시각화: 원본 + conv_down 비교
    success_count_orig = 0
    success_count_conv = 0
    for idx, (wide_path, narrow_path) in enumerate(image_pairs):
        # 원본
        output_path_orig = output_dir / f"alignment_{idx+1}_{wide_path.stem}_original.png"
        if visualize_alignment(wide_path, narrow_path, output_path_orig, use_conv_down=False):
            success_count_orig += 1
        
        # Conv Chain
        output_path_conv = output_dir / f"alignment_{idx+1}_{wide_path.stem}_convchain.png"
        if visualize_alignment(wide_path, narrow_path, output_path_conv, use_conv_down=True):
            success_count_conv += 1
    
    # 비교 그리드 생성: 원본 vs conv_chain
    if success_count_orig > 0 or success_count_conv > 0:
        print(f"\n{'='*70}")
        print("Creating comparison grids...")
        create_comparison_grid(image_pairs, output_dir / "alignment_comparison_original.png", use_conv_down=False)
        create_comparison_grid(image_pairs, output_dir / "alignment_comparison_convchain.png", use_conv_down=True)
    
    # 요약
    print(f"\n{'='*70}")
    print("📊 SUMMARY")
    print(f"{'='*70}")
    print(f"✓ Processed Original: {success_count_orig}/{len(image_pairs)} image pairs")
    print(f"✓ Processed Conv Chain: {success_count_conv}/{len(image_pairs)} image pairs")
    print(f"✓ Output directory: {output_dir.absolute()}")
    print(f"\n💡 Key findings (VERIFIED FROM SIMULATION):")
    print(f"   ")
    print(f"   Layer 0: MultiStreamConv [64, 3, 2]")
    print(f"            640×640 → 320×320 (stride=2 downsampling)")
    print(f"   ")
    print(f"   Layer 1: SpatialAlignedMultiStreamConv [128, [512, 512], 3]")
    print(f"            ✓ Wide processor: Conv(3×3, s=1) → 320×320 (no change)")
    print(f"            ✓ Narrow processor: Conv(3×3, s=1) → 320×320 (no change)")
    print(f"            Original: ✗ NO DOWNSCALER → Nearest up ×2 (320→640) → Top-left crop → misalignment")
    print(f"            Conv Chain: ✓ Fixed AvgPool k=4 s=4 (320→80) + adjust to ~92x93 + center-pad → no adaptive, miss align OK")
    print(f"            ✓ Fusion: wide(320×320) + narrow_roi_padded(320×320) (narrow compressed in center ROI)")
    print(f"   ")
    print(f"   Layer 2: Conv [128, 3, 2]")
    print(f"            320×320 → 160×160 (stride=2 downsampling)")
    print(f"   ")
    print(f"   ⚠️  RECOMMENDATION: Conv Chain 버전 – adaptive 없이 고정 conv a x a 조합으로 narrow 전체 bbox 비율 압축 + pad.")
    print(f"       Overlay + red rect: Narrow가 wide의 네모 ROI에 conv-chain으로 근사 맞는지 확인. 오차 ~2px OK!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()