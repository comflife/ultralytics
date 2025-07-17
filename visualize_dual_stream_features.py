#!/usr/bin/env python3
"""
Dual Stream YOLO Feature Visualization Script
Extracts and visualizes feature maps from specific layers in dual-stream YOLO models
"""

import sys
import os
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict
import time
import random
import glob
from PIL import Image

# 🔧 로컬 ultralytics 모듈을 우선적으로 사용하도록 설정
SCRIPT_DIR = Path(__file__).parent.absolute()
ULTRALYTICS_ROOT = SCRIPT_DIR
sys.path.insert(0, str(ULTRALYTICS_ROOT))

print(f"🔧 Using local ultralytics from: {ULTRALYTICS_ROOT}")

# ============================================================
# 🛠️ 설정 부분 - 여기를 수정하세요!
# ============================================================

# 모델 파일 경로
MODEL_PATH = "/home/byounggun/ultralytics/runs/train/exp33/weights/best.pt"

# 입력 이미지 디렉토리 경로
WIDE_IMAGE_DIR = "/home/byounggun/ultralytics/swm_total/images"
NARROW_IMAGE_DIR = "/home/byounggun/ultralytics/swm_total/narrow_images"

# 출력 설정
OUTPUT_DIR = "/home/byounggun/ultralytics/models/debug_features"
IMAGE_SIZE = 640
NUM_RANDOM_IMAGES = 3  # 랜덤으로 시각화할 이미지 세트 개수

# 시각화할 레이어 타입들
TARGET_LAYERS = [
    'MultiStreamConv',
    'SpatialAlignedMultiStreamConv', 
    'MultiStreamC3'
]

# ============================================================

try:
    from ultralytics import YOLO
    import ultralytics
    print(f"✅ Using ultralytics from: {ultralytics.__file__}")
except ImportError as e:
    print(f"❌ Failed to import ultralytics: {e}")
    sys.exit(1)


class FeatureExtractor:
    """Feature map extraction을 위한 클래스"""
    
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.feature_maps = defaultdict(list)
        self.hooks = []
        self.layer_names = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        """모델의 타겟 레이어들에 hook 등록"""
        print("🔍 Searching for target layers...")
        
        def make_hook(layer_name):
            def hook_fn(module, input, output):
                print(f"📊 Capturing features from {layer_name}: {output.shape}")
                self.feature_maps[layer_name].append(output.detach().cpu())
            return hook_fn
        
        # 모델의 모든 모듈을 순회하면서 타겟 레이어 찾기
        for name, module in self.model.named_modules():
            module_type = type(module).__name__
            
            if module_type in self.target_layers:
                print(f"✅ Found target layer: {name} ({module_type})")
                hook = module.register_forward_hook(make_hook(f"{name}_{module_type}"))
                self.hooks.append(hook)
                self.layer_names.append(f"{name}_{module_type}")
    
    def clear_features(self):
        """저장된 feature map 초기화"""
        self.feature_maps.clear()
    
    def remove_hooks(self):
        """등록된 hook들 제거"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def preprocess_image(image_path, target_size=640):
    """이미지를 모델 입력에 맞게 전처리"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    original_height, original_width = image.shape[:2]
    print(f"📸 Loaded image {Path(image_path).name}: {original_width}x{original_height}")
    
    # RGB로 변환
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 리사이즈 (letterbox 적용)
    scale = min(target_size / original_width, target_size / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    resized = cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # 패딩 추가
    delta_w = target_size - new_width
    delta_h = target_size - new_height
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, 
                               cv2.BORDER_CONSTANT, value=[114, 114, 114])
    
    # 정규화 및 텐서 변환
    normalized = padded.astype(np.float32) / 255.0
    tensor = torch.from_numpy(normalized).permute(2, 0, 1)  # HWC -> CHW
    
    return tensor, image_rgb


def get_random_image_pairs(wide_dir, narrow_dir, num_pairs=3):
    """디렉토리에서 랜덤한 이미지 쌍을 선택"""
    wide_dir = Path(wide_dir)
    narrow_dir = Path(narrow_dir)
    
    # 지원하는 이미지 확장자
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    
    # Wide 디렉토리의 모든 이미지 파일 찾기
    wide_files = []
    for ext in image_extensions:
        wide_files.extend(wide_dir.glob(ext))
        wide_files.extend(wide_dir.glob(ext.upper()))
    
    if not wide_files:
        raise ValueError(f"No image files found in {wide_dir}")
    
    # 매칭되는 narrow 이미지가 있는 쌍만 선택
    valid_pairs = []
    for wide_file in wide_files:
        narrow_file = narrow_dir / wide_file.name
        if narrow_file.exists():
            valid_pairs.append((wide_file, narrow_file))
    
    if not valid_pairs:
        raise ValueError(f"No matching image pairs found between {wide_dir} and {narrow_dir}")
    
    # 랜덤 선택
    num_pairs = min(num_pairs, len(valid_pairs))
    selected_pairs = random.sample(valid_pairs, num_pairs)
    
    print(f"🎲 Selected {num_pairs} random image pairs:")
    for i, (wide, narrow) in enumerate(selected_pairs):
        print(f"  {i+1}. {wide.name}")
    
    return selected_pairs


def create_dual_stream_input(wide_tensor, narrow_tensor):
    """두 이미지를 듀얼 스트림 형태로 결합"""
    dual_stream = torch.stack([wide_tensor, narrow_tensor], dim=0)  # [2, 3, H, W]
    dual_stream = dual_stream.unsqueeze(0)  # [1, 2, 3, H, W]
    
    print(f"🔗 Created dual stream input: {dual_stream.shape}")
    return dual_stream


def visualize_feature_map(feature_tensor, layer_name, output_dir, max_channels=16):
    """Feature map을 heatmap으로 시각화하고 저장"""
    
    if feature_tensor.dim() == 5:  # Dual stream: [B, 2, C, H, W]
        print(f"📊 Processing dual stream features: {feature_tensor.shape}")
        
        # Wide stream과 Narrow stream 분리
        wide_features = feature_tensor[0, 0]  # [C, H, W]
        narrow_features = feature_tensor[0, 1]  # [C, H, W]
        
        # Wide stream 시각화
        _save_feature_channels(wide_features, f"{layer_name}_wide", output_dir, max_channels)
        
        # Narrow stream 시각화  
        _save_feature_channels(narrow_features, f"{layer_name}_narrow", output_dir, max_channels)
        
        # 평균 feature map도 저장
        avg_wide = torch.mean(wide_features, dim=0).numpy()
        avg_narrow = torch.mean(narrow_features, dim=0).numpy()
        
        _save_heatmap(avg_wide, f"{layer_name}_wide_avg", output_dir)
        _save_heatmap(avg_narrow, f"{layer_name}_narrow_avg", output_dir)
        
    elif feature_tensor.dim() == 4:  # Single stream: [B, C, H, W]
        print(f"📊 Processing single stream features: {feature_tensor.shape}")
        
        features = feature_tensor[0]  # [C, H, W]
        _save_feature_channels(features, layer_name, output_dir, max_channels)
        
        # 평균 feature map 저장
        avg_features = torch.mean(features, dim=0).numpy()
        _save_heatmap(avg_features, f"{layer_name}_avg", output_dir)
    
    else:
        print(f"⚠️ Unexpected feature tensor shape: {feature_tensor.shape}")


def _save_feature_channels(features, name_prefix, output_dir, max_channels):
    """개별 채널들을 heatmap으로 저장"""
    C, H, W = features.shape
    num_channels = min(C, max_channels)
    
    # 그리드 레이아웃 계산
    cols = min(4, num_channels)
    rows = (num_channels + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i in range(num_channels):
        channel_data = features[i].numpy()
        
        ax = axes[i] if num_channels > 1 else axes[0]
        im = ax.imshow(channel_data, cmap='viridis', aspect='auto')
        ax.set_title(f'Channel {i}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 빈 subplot 숨기기
    for i in range(num_channels, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # 저장
    output_path = output_dir / f"{name_prefix}_channels.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Saved channel visualization: {output_path}")


def _save_heatmap(feature_array, name, output_dir):
    """단일 feature map을 heatmap으로 저장"""
    plt.figure(figsize=(8, 8))
    plt.imshow(feature_array, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title(f'{name}')
    plt.axis('off')
    
    output_path = output_dir / f"{name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Saved heatmap: {output_path}")


def create_feature_summary(feature_maps, output_dir):
    """Feature map들의 요약 정보 생성"""
    summary_path = output_dir / "feature_summary.txt"
    
    with open(summary_path, 'w') as f:
        f.write("Dual Stream YOLO Feature Visualization Summary\n")
        f.write("=" * 50 + "\n\n")
        
        for layer_name, features_list in feature_maps.items():
            f.write(f"Layer: {layer_name}\n")
            f.write(f"Number of feature maps captured: {len(features_list)}\n")
            
            if features_list:
                feature = features_list[0]
                f.write(f"Feature shape: {feature.shape}\n")
                f.write(f"Value range: [{feature.min():.4f}, {feature.max():.4f}]\n")
                f.write(f"Mean: {feature.mean():.4f}, Std: {feature.std():.4f}\n")
            
            f.write("\n")
    
    print(f"📄 Saved feature summary: {summary_path}")


def main():
    """메인 visualization 함수"""
    
    print("🚀 Starting Dual Stream YOLO Feature Visualization...")
    
    # 출력 디렉토리 생성
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 모델 로드
    print(f"📦 Loading model from: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
        if hasattr(model, 'model') and model.model is not None:
            pytorch_model = model.model
        elif hasattr(model, 'predictor') and hasattr(model.predictor, 'model') and model.predictor.model is not None:
            pytorch_model = model.predictor.model
        else:
            raise ValueError("Could not access the PyTorch model from YOLO object")
        
        pytorch_model.eval()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # 2. Feature extractor 설정
    print("🔧 Setting up feature extractor...")
    extractor = FeatureExtractor(pytorch_model, TARGET_LAYERS)
    
    if not extractor.layer_names:
        print("❌ No target layers found! Check your model architecture.")
        return
    
    # 3. 랜덤 이미지 쌍 선택
    print("🎲 Selecting random image pairs...")
    try:
        image_pairs = get_random_image_pairs(WIDE_IMAGE_DIR, NARROW_IMAGE_DIR, NUM_RANDOM_IMAGES)
    except Exception as e:
        print(f"❌ Failed to select image pairs: {e}")
        return
    
    # 4. 각 이미지 쌍에 대해 feature visualization 수행
    for pair_idx, (wide_path, narrow_path) in enumerate(image_pairs):
        print(f"\n🖼️ Processing image pair {pair_idx + 1}/{len(image_pairs)}: {wide_path.name}")
        
        # 이미지 전처리
        try:
            wide_tensor, wide_image = preprocess_image(wide_path, IMAGE_SIZE)
            narrow_tensor, narrow_image = preprocess_image(narrow_path, IMAGE_SIZE)
        except Exception as e:
            print(f"❌ Failed to preprocess images for pair {pair_idx + 1}: {e}")
            continue
        
        # 듀얼 스트림 입력 생성
        dual_input = create_dual_stream_input(wide_tensor, narrow_tensor)
        
        # Feature 추출을 위한 forward pass
        print("🔮 Running forward pass to extract features...")
        start_time = time.time()
        
        try:
            # 이전 feature map 초기화
            extractor.clear_features()
            
            with torch.no_grad():
                # Feature 추출을 위한 forward pass
                _ = pytorch_model(dual_input)
                
        except Exception as e:
            print(f"❌ Feature extraction failed for pair {pair_idx + 1}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        extraction_time = time.time() - start_time
        print(f"⚡ Feature extraction completed in {extraction_time:.3f}s")
        
        # Feature map 시각화
        print("🎨 Visualizing feature maps...")
        
        for layer_name in extractor.layer_names:
            if layer_name in extractor.feature_maps:
                features_list = extractor.feature_maps[layer_name]
                
                print(f"📊 Processing {layer_name}...")
                for i, feature_tensor in enumerate(features_list):
                    # 이미지별로 구분된 파일명 생성
                    image_name = wide_path.stem
                    layer_save_name = f"{image_name}_{layer_name}_pass{i}"
                    
                    visualize_feature_map(
                        feature_tensor, 
                        layer_save_name, 
                        output_dir
                    )
    
    # 전체 요약 정보 생성
    print("\n📄 Creating overall summary...")
    create_feature_summary(extractor.feature_maps, output_dir)
    
    # 정리
    extractor.remove_hooks()
    
    print(f"\n✅ Feature visualization completed!")
    print(f"📁 Results saved to: {output_dir}")
    print("🎉 All visualizations saved successfully!")


if __name__ == "__main__":
    main() 