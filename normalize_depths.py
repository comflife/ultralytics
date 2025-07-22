#!/usr/bin/env python3
"""
라벨 파일의 depth 값을 normalize하는 스크립트
"""

import os
import numpy as np
from pathlib import Path
import json

def analyze_and_normalize_depths():
    """라벨 파일들의 depth 값을 분석하고 normalize"""
    
    # 타겟 디렉토리들
    label_dirs = [
        "/home/byounggun/ultralytics/swm_dual_split/train/labels",
        "/home/byounggun/ultralytics/swm_dual_split/val/labels"
    ]
    
    all_depths = []
    total_labels = 0
    
    # 1단계: 모든 depth 값 수집
    print("🔍 1단계: 모든 라벨 파일에서 depth 값 수집...")
    
    for label_dir in label_dirs:
        label_path = Path(label_dir)
        if not label_path.exists():
            print(f"⚠️ 디렉토리가 존재하지 않습니다: {label_dir}")
            continue
            
        label_files = list(label_path.glob("*.txt"))
        print(f"📁 {label_dir}: {len(label_files)}개 파일")
        
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 6:  # class x y w h depth
                        depth = float(parts[5])
                        all_depths.append(depth)
                        total_labels += 1
                    elif len(parts) == 5:  # depth 값이 없는 경우
                        print(f"⚠️ depth 값이 없는 라벨: {label_file.name}")
                        
            except Exception as e:
                print(f"❌ 파일 읽기 오류: {label_file.name} - {e}")
    
    if not all_depths:
        print("❌ depth 값이 발견되지 않았습니다!")
        return
        
    # 2단계: 통계 분석
    all_depths = np.array(all_depths)
    min_depth = np.min(all_depths)
    max_depth = np.max(all_depths)
    mean_depth = np.mean(all_depths)
    std_depth = np.std(all_depths)
    
    print(f"\n📊 Depth 값 통계:")
    print(f"   총 라벨 수: {total_labels}")
    print(f"   최소값: {min_depth:.6f}")
    print(f"   최대값: {max_depth:.6f}")
    print(f"   평균값: {mean_depth:.6f}")
    print(f"   표준편차: {std_depth:.6f}")
    print(f"   범위: {max_depth - min_depth:.6f}")
    
    # 정규화 방법 선택
    print(f"\n🔧 정규화 방법:")
    print(f"   Min-Max 정규화: [0, 1] 범위로 변환")
    print(f"   공식: (value - {min_depth:.6f}) / {max_depth - min_depth:.6f}")
    
    # 정규화 정보 저장
    norm_info = {
        "min_depth": float(min_depth),
        "max_depth": float(max_depth),
        "mean_depth": float(mean_depth),
        "std_depth": float(std_depth),
        "total_labels": int(total_labels),
        "normalization_method": "min_max"
    }
    
    norm_info_path = "/home/byounggun/ultralytics/depth_normalization_info.json"
    with open(norm_info_path, 'w') as f:
        json.dump(norm_info, f, indent=2)
    print(f"💾 정규화 정보 저장: {norm_info_path}")
    
    # 3단계: 라벨 파일들 정규화
    print(f"\n🔄 3단계: 라벨 파일들 정규화 중...")
    
    processed_files = 0
    processed_labels = 0
    
    for label_dir in label_dirs:
        label_path = Path(label_dir)
        if not label_path.exists():
            continue
            
        label_files = list(label_path.glob("*.txt"))
        
        for label_file in label_files:
            try:
                # 원본 읽기
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                # 정규화된 라인들
                normalized_lines = []
                file_changed = False
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 6:  # class x y w h depth
                        original_depth = float(parts[5])
                        # Min-Max 정규화
                        normalized_depth = (original_depth - min_depth) / (max_depth - min_depth)
                        
                        # 새로운 라인 생성
                        parts[5] = f"{normalized_depth:.6f}"
                        normalized_line = " ".join(parts) + "\n"
                        normalized_lines.append(normalized_line)
                        
                        file_changed = True
                        processed_labels += 1
                    else:
                        # depth 값이 없는 라인은 그대로 유지
                        normalized_lines.append(line)
                
                # 파일 다시 쓰기 (변경된 경우만)
                if file_changed:
                    with open(label_file, 'w') as f:
                        f.writelines(normalized_lines)
                    processed_files += 1
                    
            except Exception as e:
                print(f"❌ 파일 처리 오류: {label_file.name} - {e}")
    
    print(f"\n✅ 정규화 완료!")
    print(f"   처리된 파일: {processed_files}개")
    print(f"   처리된 라벨: {processed_labels}개")
    
    # 샘플 확인
    print(f"\n📄 정규화 후 샘플 확인:")
    sample_file = None
    for label_dir in label_dirs:
        label_path = Path(label_dir)
        if label_path.exists():
            files = list(label_path.glob("*.txt"))
            if files:
                sample_file = files[0]
                break
    
    if sample_file:
        print(f"   파일: {sample_file.name}")
        with open(sample_file, 'r') as f:
            lines = f.readlines()[:3]
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 6:
                    print(f"   {line.strip()} (depth: {parts[5]})")

if __name__ == "__main__":
    print("🚀 Depth 값 정규화 시작...")
    analyze_and_normalize_depths()
    print("✅ 완료!")
