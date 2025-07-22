#!/usr/bin/env python3
"""
Inference 결과에서 depth 값을 원본 스케일로 복구하는 유틸리티
"""

import json
import numpy as np
from pathlib import Path

class DepthDenormalizer:
    """Depth 값을 원본 스케일로 복구하는 클래스"""
    
    def __init__(self, norm_info_path="/home/byounggun/ultralytics/depth_normalization_info.json"):
        """
        Args:
            norm_info_path: 정규화 정보가 저장된 JSON 파일 경로
        """
        self.norm_info_path = norm_info_path
        self.norm_info = None
        self.load_normalization_info()
    
    def load_normalization_info(self):
        """정규화 정보 로드"""
        try:
            with open(self.norm_info_path, 'r') as f:
                self.norm_info = json.load(f)
            print(f"✅ 정규화 정보 로드 완료: {self.norm_info_path}")
            print(f"   원본 범위: [{self.norm_info['min_depth']:.6f}, {self.norm_info['max_depth']:.6f}]")
        except FileNotFoundError:
            print(f"❌ 정규화 정보 파일을 찾을 수 없습니다: {self.norm_info_path}")
            print("   normalize_depths.py를 먼저 실행해주세요.")
            self.norm_info = None
        except Exception as e:
            print(f"❌ 정규화 정보 로드 오류: {e}")
            self.norm_info = None
    
    def denormalize_depth(self, normalized_depth):
        """
        정규화된 depth 값을 원본 스케일로 복구
        
        Args:
            normalized_depth: 정규화된 depth 값 (0~1 범위)
            
        Returns:
            원본 스케일의 depth 값
        """
        if self.norm_info is None:
            print("⚠️ 정규화 정보가 없습니다. 원본 값을 그대로 반환합니다.")
            return normalized_depth
        
        min_depth = self.norm_info['min_depth']
        max_depth = self.norm_info['max_depth']
        
        # Min-Max 역정규화
        original_depth = normalized_depth * (max_depth - min_depth) + min_depth
        return original_depth
    
    def denormalize_batch_depths(self, normalized_depths):
        """
        배치의 정규화된 depth 값들을 원본 스케일로 복구
        
        Args:
            normalized_depths: 정규화된 depth 값들의 배열/리스트
            
        Returns:
            원본 스케일의 depth 값들
        """
        if isinstance(normalized_depths, (list, tuple)):
            return [self.denormalize_depth(d) for d in normalized_depths]
        elif isinstance(normalized_depths, np.ndarray):
            return np.array([self.denormalize_depth(d) for d in normalized_depths])
        else:
            return self.denormalize_depth(normalized_depths)
    
    def get_depth_range(self):
        """원본 depth 값의 범위 반환"""
        if self.norm_info is None:
            return None, None
        return self.norm_info['min_depth'], self.norm_info['max_depth']
    
    def get_depth_stats(self):
        """원본 depth 값의 통계 정보 반환"""
        if self.norm_info is None:
            return None
        return {
            'min': self.norm_info['min_depth'],
            'max': self.norm_info['max_depth'],
            'mean': self.norm_info['mean_depth'],
            'std': self.norm_info['std_depth'],
            'total_labels': self.norm_info['total_labels']
        }

def test_denormalization():
    """역정규화 테스트"""
    print("🧪 역정규화 테스트...")
    
    denormalizer = DepthDenormalizer()
    
    if denormalizer.norm_info is None:
        print("❌ 테스트를 위해 정규화 정보가 필요합니다.")
        return
    
    # 테스트 케이스
    test_cases = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    print("정규화된 값 -> 원본 값:")
    for norm_val in test_cases:
        orig_val = denormalizer.denormalize_depth(norm_val)
        print(f"   {norm_val:.2f} -> {orig_val:.6f}")
    
    # 통계 정보 출력
    stats = denormalizer.get_depth_stats()
    if stats:
        print(f"\n📊 원본 데이터 통계:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.6f}")
            else:
                print(f"   {key}: {value}")

if __name__ == "__main__":
    test_denormalization()
