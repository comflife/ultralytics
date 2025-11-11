#!/usr/bin/env python3
"""
실제 데이터를 사용하여 narrow 이미지가 wide 이미지 위에 어떻게 정렬되는지 확인하는 스크립트.
SpatialAlignedMultiStreamConv의 '수정 전 원본' 로직을 정확히 시뮬레이션합니다.

[수정된 시뮬레이션 로직]
- Conv Down: AvgPool k=4 s=4로 320x320 -> 80x80 다운샘플링.
- No Resize: 모듈의 1x1 Conv는 공간 크기를 바꾸지 않으므로, 80x80 크기를 그대로 유지.
- Center Pad: 80x80 피쳐맵을 320x320 크기에 맞춰 중앙 정렬.
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

# Narrow 이미지 ROI 정보 (wide의 중심 네모 영역에 맞춤)
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
    
    wide_images = sorted(list(wide_dir.glob("*.jpg")) + list(wide_dir.glob("*.png")))
    narrow_images = sorted(list(narrow_dir.glob("*.jpg")) + list(narrow_dir.glob("*.png")))
    
    print(f"📂 Found {len(wide_images)} wide images")
    print(f"📂 Found {len(narrow_images)} narrow images")
    
    pairs = []
    for wide_path in wide_images[:num_samples]:
        wide_stem = wide_path.stem
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
    SpatialAlignedMultiStreamConv의 처리 과정을 시뮬레이션합니다.
    
    [수정] use_conv_down=True: 실제 모듈 로직을 반영.
             - Chain: AvgPool2d(k=4,s=4)로 320→80 다운샘플링.
             - No resize: 1x1 Conv는 크기를 바꾸지 않으므로 80x80을 그대로 사용.
             - Center Pad: 80x80을 320x320 중앙에 배치.
    """
    H, W = target_size
    
    if isinstance(wide_img, Image.Image):
        wide_img = np.array(wide_img)
    if isinstance(narrow_img, Image.Image):
        narrow_img = np.array(narrow_img)
    
    wide_pil = Image.fromarray(wide_img)
    narrow_pil = Image.fromarray(narrow_img)
    wide_resized = np.array(wide_pil.resize((W, H), Image.BILINEAR))
    narrow_resized = np.array(narrow_pil.resize((W, H), Image.BILINEAR))
    
    print(f"    Initial size: {W}×{H}")
    
    after_multistream_size = (H // 2, W // 2)
    wide_after_ms = np.array(Image.fromarray(wide_resized).resize(
        (after_multistream_size[1], after_multistream_size[0]), Image.BILINEAR))
    narrow_after_ms = np.array(Image.fromarray(narrow_resized).resize(
        (after_multistream_size[1], after_multistream_size[0]), Image.BILINEAR))
    
    print(f"    After MultiStreamConv (s=2): {after_multistream_size[1]}×{after_multistream_size[0]}")
    
    wide_tensor = torch.from_numpy(wide_after_ms).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    narrow_tensor = torch.from_numpy(narrow_after_ms).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    
    wide_proc = wide_tensor
    print(f"    Wide after processor (s=1): {wide_proc.shape}")
    
    narrow_proc = narrow_tensor
    print(f"    Narrow after processor (s=1): {narrow_proc.shape}")
    
    roi_info_for_viz = None

    if not use_conv_down:
        # [원본 비교용] 단순 업샘플링 + Top-left crop
        upsampler = torch.nn.Upsample(scale_factor=2, mode='nearest')
        narrow_upsampled = upsampler(narrow_proc)
        print(f"    Narrow after upsampling (×2, NO downscale before!): {narrow_upsampled.shape}")
        
        target_h_proc, target_w_proc = wide_proc.shape[2:]
        narrow_for_fuse = narrow_upsampled[:, :, :target_h_proc, :target_w_proc]
        crop_mode = 'top-left'
        print(f"    Narrow after crop (top-left) to match wide: {narrow_for_fuse.shape}")
        
        mode_desc = "Original Sim: Direct nearest up + top-left crop"
    else:
        # [수정된 시뮬레이션] 실제 모듈 로직 (AvgPool -> 1x1 Conv -> Pad)
        target_h_proc, target_w_proc = narrow_proc.shape[2:]
        
        # 고정 Conv Chain: 320→80 (AvgPool k=4 s=4)
        pool1 = torch.nn.AvgPool2d(kernel_size=4, stride=4)
        narrow_stage1 = pool1(narrow_proc)  # [B,3,80,80]
        print(f"    Narrow stage1: AvgPool k=4 s=4 → {narrow_stage1.shape}")

        # 실제 모듈의 1x1 Conv는 공간 크기를 바꾸지 않음 (No-Op for size)
        narrow_small = narrow_stage1 # 사이즈 변경 없음
        print(f"    Narrow after 1x1 Conv sim (no spatial change): {narrow_small.shape}")
        
        # Center pad to full wide size (0-pad 주변)
        pad_h = (target_h_proc - narrow_small.shape[2]) // 2
        pad_w = (target_w_proc - narrow_small.shape[3]) // 2
        pad_top = pad_h
        pad_bottom = target_h_proc - narrow_small.shape[2] - pad_top
        pad_left = pad_w
        pad_right = target_w_proc - narrow_small.shape[3] - pad_left
        
        narrow_for_fuse = F.pad(narrow_small, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0.0)
        print(f"    Narrow after center-pad to wide: {narrow_for_fuse.shape}")
        
        crop_mode = 'Pool(k=4,s=4) -> No-Resize -> Center-Pad'
        mode_desc = "Module Sim: Fixed AvgPool k=4 s=4 + center-pad (NO RESIZE)"

        # 시각화용 BBox 정보 계산
        roi_h, roi_w, center_y, center_x = compute_bbox_size(target_h_proc, target_w_proc, NARROW_BBOX)
        roi_info_for_viz = {'h': roi_h, 'w': roi_w, 'center_y': center_y, 'center_x': center_x}
        
    fused = wide_proc + narrow_for_fuse
    print(f"    Fused (wide + narrow): {fused.shape}")
    
    wide_proc_np = (wide_proc * 255).clamp(0, 255).squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
    narrow_proc_np = (narrow_proc * 255).clamp(0, 255).squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
    narrow_aligned_np = (narrow_for_fuse * 255).clamp(0, 255).squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
    fused_np = (fused * 255).clamp(0, 255).squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
    
    overlay = (wide_proc_np * 0.5 + narrow_aligned_np * 0.5).astype(np.uint8)
    
    bbox_overlay = overlay
    if use_conv_down and roi_info_for_viz:
        fig_overlay, ax = plt.subplots(1, 1, figsize=(5,5))
        ax.imshow(overlay)
        # 참고용으로만 BBox를 그림 (실제 narrow 크기와는 다름)
        rect = patches.Rectangle((roi_info_for_viz['center_x'], roi_info_for_viz['center_y']), roi_info_for_viz['w'], roi_info_for_viz['h'], linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.set_title('Overlay with Target BBox (red rect)')
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
        'narrow_aligned': narrow_aligned_np,
        'fused': fused_np,
        'overlay': overlay,
        'bbox_overlay': bbox_overlay,
        'sizes': {
            'initial': (H, W),
            'after_multistream': after_multistream_size,
            'narrow_after_proc': narrow_proc.shape[2:],
            'narrow_small': narrow_small.shape[2:] if use_conv_down else 'N/A',
            'final_fused': fused.shape[2:]
        },
        'mode': mode_desc,
        'crop_mode': crop_mode,
        'roi_info': roi_info_for_viz
    }


def visualize_alignment(wide_path, narrow_path, output_path, use_conv_down=False):
    """이미지 정렬 과정을 시각화합니다."""
    
    print(f"\n{'='*70}")
    print(f"Processing: {wide_path.name} (Mode: {'Module Sim' if use_conv_down else 'Original Sim'})")
    print(f"{'='*70}")
    
    try:
        wide_img = Image.open(str(wide_path)).convert('RGB')
        narrow_img = Image.open(str(narrow_path)).convert('RGB')
    except Exception as e:
        print(f"❌ Failed to load images: {e}")
        return False
    
    results = simulate_spatial_alignment(np.array(wide_img), np.array(narrow_img), target_size=(640, 640), use_conv_down=use_conv_down)
    
    sizes = results['sizes']
    mode = results['mode']
    roi_info = results['roi_info']
    print(f"✓ Processing sizes ({mode}):")
    print(f"    Initial: {sizes['initial']}")
    print(f"    After MultiStreamConv: {sizes['after_multistream']}")
    if use_conv_down:
        print(f"    Narrow after pool: {sizes['narrow_small']}")
    print(f"    Final fused: {sizes['final_fused']}")

    fig, axes = plt.subplots(4, 3, figsize=(18, 22))

    # Row 1: Inputs
    axes[0, 0].imshow(results['wide_resized'])
    axes[0, 0].set_title(f'1. Wide Input\n({sizes["initial"][1]}×{sizes["initial"][0]})', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(results['narrow_resized'])
    axes[0, 1].set_title(f'2. Narrow Input\n({sizes["initial"][1]}×{sizes["initial"][0]})', fontsize=12)
    axes[0, 1].axis('off')

    axes[0, 2].text(0.5, 0.5, 'Input Feature', ha='center', va='center', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')

    # Row 2: After MultiStreamConv
    axes[1, 0].imshow(results['wide_after_ms'])
    axes[1, 0].set_title(f'3. Wide after MultiStreamConv (s=2)\n({sizes["after_multistream"][1]}×{sizes["after_multistream"][0]})', fontsize=12)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(results['narrow_after_ms'])
    axes[1, 1].set_title(f'4. Narrow after MultiStreamConv (s=2)\n({sizes["after_multistream"][1]}×{sizes["after_multistream"][0]})', fontsize=12)
    axes[1, 1].axis('off')
    
    axes[1, 2].text(0.5, 0.5, 'After s=2 Conv', ha='center', va='center', fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')

    # Row 3: Processing
    axes[2, 0].imshow(results['wide_proc'])
    axes[2, 0].set_title(f'5. Wide after Processor (s=1)', fontsize=12)
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(results['narrow_proc'])
    axes[2, 1].set_title(f'6. Narrow after Processor (s=1)', fontsize=12)
    axes[2, 1].axis('off')

    if use_conv_down:
        axes[2, 2].imshow(results['narrow_aligned'])
        axes[2, 2].set_title(f'7. Narrow after Pool + Pad\n(Final size: {sizes["final_fused"][1]}x{sizes["final_fused"][0]})', fontsize=12, color='blue', fontweight='bold')
    else:
        axes[2, 2].imshow(results['narrow_aligned'])
        axes[2, 2].set_title(f'7. Narrow Upsampled & Cropped ⚠️\n(Final size: {sizes["final_fused"][1]}x{sizes["final_fused"][0]})', fontsize=12, color='red', fontweight='bold')
    axes[2, 2].axis('off')
        
    # Row 4: Fusion
    axes[3, 0].imshow(results['wide_proc'])
    axes[3, 0].set_title(f'8. Wide for Fusion', fontsize=12)
    axes[3, 0].axis('off')
    
    axes[3, 1].imshow(results['overlay'])
    axes[3, 1].set_title(f'9. Overlay (Wide + Aligned Narrow)', fontsize=12)
    axes[3, 1].axis('off')
    
    axes[3, 2].imshow(results['bbox_overlay' if use_conv_down else 'fused'])
    title_text = f'10. FUSED (Wide + Narrow)\nRed box is target ROI for reference' if use_conv_down else '10. FUSED (Wide + Narrow)'
    axes[3, 2].set_title(title_text, fontsize=12, color='green', fontweight='bold')
    axes[3, 2].axis('off')

    mode_title = " (Module Simulation)" if use_conv_down else " (Original Baseline Sim)"
    fig.suptitle(f'SpatialAlignedMultiStreamConv Verification{mode_title}\n{wide_path.name}', fontsize=16, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"✅ Saved visualization to: {output_path}")
    
    return True


def main():
    print("\n" + "="*70)
    print("🔍 DUAL-STREAM ALIGNMENT VERIFICATION (Simulating Original Module)")
    print("="*70)
    
    image_pairs = get_image_pairs(WIDE_DIR, NARROW_DIR, num_samples=3)
    
    if not image_pairs:
        print("\n❌ No image pairs found. Please check your dataset paths.")
        return
    
    print(f"\n✓ Found {len(image_pairs)} image pairs to process\n")
    
    output_dir = Path("alignment_verification_results")
    output_dir.mkdir(exist_ok=True)
    
    for idx, (wide_path, narrow_path) in enumerate(image_pairs):
        # Original Baseline Simulation
        output_path_orig = output_dir / f"alignment_{idx+1}_{wide_path.stem}_baseline.png"
        visualize_alignment(wide_path, narrow_path, output_path_orig, use_conv_down=False)
        
        # Actual Module Simulation
        output_path_conv = output_dir / f"alignment_{idx+1}_{wide_path.stem}_module_sim.png"
        visualize_alignment(wide_path, narrow_path, output_path_conv, use_conv_down=True)
    
    print(f"\n{'='*70}")
    print("📊 SUMMARY")
    print(f"{'='*70}")
    print(f"✓ Processing complete. Results are saved in '{output_dir.absolute()}'")
    print("💡 Two images generated per sample:")
    print("   - `..._baseline.png`: Shows simple upsampling + top-left crop for comparison.")
    print("   - `..._module_sim.png`: Accurately shows your module's logic (Pool -> No-Resize -> Center-Pad).")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()