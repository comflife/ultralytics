#!/usr/bin/env python3
"""
Dual Stream Dataset Splitter
데이터셋을 8:2 비율로 train/validation으로 분할하는 스크립트

특징:
- validation set을 연속되지 않게 일정 간격으로 선택
- 원본 파일은 그대로 두고 복사본으로 새로운 구조 생성
- wide images, narrow images, labels 모두 대응되는 파일만 사용
"""

import os
import shutil
import glob
from pathlib import Path
import math
import random
from collections import defaultdict

def setup_directories(base_output_dir):
    """
    출력 디렉토리 구조 생성
    
    Args:
        base_output_dir (str): 출력 기본 디렉토리
        
    Returns:
        dict: 생성된 디렉토리 경로들
    """
    dirs = {
        'train_images': os.path.join(base_output_dir, 'train_images'),
        'val_images': os.path.join(base_output_dir, 'val_images'),
        'train_labels': os.path.join(base_output_dir, 'train_labels'),
        'val_labels': os.path.join(base_output_dir, 'val_labels'),
        'train_narrow_images': os.path.join(base_output_dir, 'train_narrow_images'),
        'val_narrow_images': os.path.join(base_output_dir, 'val_narrow_images')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        print(f"📁 디렉토리 생성: {dir_path}")
    
    return dirs

def get_valid_files(images_dir, narrow_images_dir, labels_dir):
    """
    세 종류의 파일이 모두 존재하는 유효한 파일들만 찾기
    
    Args:
        images_dir (str): wide images 디렉토리
        narrow_images_dir (str): narrow images 디렉토리
        labels_dir (str): labels 디렉토리
        
    Returns:
        list: 유효한 파일명 리스트 (확장자 없음)
    """
    # labels 파일을 기준으로 시작
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    label_names = [os.path.splitext(os.path.basename(f))[0] for f in label_files]
    
    valid_files = []
    missing_files = defaultdict(list)
    
    for name in label_names:
        wide_path = os.path.join(images_dir, f"{name}.jpg")
        narrow_path = os.path.join(narrow_images_dir, f"{name}.jpg")
        label_path = os.path.join(labels_dir, f"{name}.txt")
        
        # 세 파일이 모두 존재하는지 확인
        files_exist = {
            'wide': os.path.exists(wide_path),
            'narrow': os.path.exists(narrow_path),
            'label': os.path.exists(label_path)
        }
        
        if all(files_exist.values()):
            valid_files.append(name)
        else:
            # 누락된 파일 기록
            for file_type, exists in files_exist.items():
                if not exists:
                    missing_files[file_type].append(name)
    
    # 누락된 파일 통계 출력
    if missing_files:
        print(f"\n⚠️  누락된 파일들:")
        for file_type, names in missing_files.items():
            print(f"  {file_type}: {len(names)}개")
            if len(names) <= 5:  # 5개 이하면 모두 출력
                for name in names:
                    print(f"    - {name}")
            else:  # 5개 초과면 일부만 출력
                for name in names[:3]:
                    print(f"    - {name}")
                print(f"    ... 외 {len(names)-3}개")
    
    return sorted(valid_files)

def split_dataset_interval(file_list, val_ratio=0.2):
    """
    일정 간격으로 validation set을 선택하여 데이터셋 분할
    
    Args:
        file_list (list): 전체 파일 리스트
        val_ratio (float): validation 비율 (기본값: 0.2)
        
    Returns:
        tuple: (train_files, val_files)
    """
    total_files = len(file_list)
    val_count = int(total_files * val_ratio)
    
    # 일정 간격 계산 (validation을 고르게 분포시키기 위해)
    if val_count == 0:
        return file_list, []
    
    interval = total_files / val_count
    
    val_indices = []
    train_indices = []
    
    # 일정 간격으로 validation 인덱스 선택
    for i in range(val_count):
        idx = int(i * interval + interval/2)  # 간격의 중앙값 선택
        if idx < total_files:
            val_indices.append(idx)
    
    # 나머지는 train 인덱스
    for i in range(total_files):
        if i not in val_indices:
            train_indices.append(i)
    
    train_files = [file_list[i] for i in train_indices]
    val_files = [file_list[i] for i in val_indices]
    
    print(f"📊 데이터 분할 결과:")
    print(f"  전체: {total_files}개")
    print(f"  Train: {len(train_files)}개 ({len(train_files)/total_files:.1%})")
    print(f"  Validation: {len(val_files)}개 ({len(val_files)/total_files:.1%})")
    print(f"  Validation 간격: 약 {interval:.1f}개마다 1개씩 선택")
    
    return train_files, val_files

def copy_files(file_list, source_dirs, target_dirs, split_type):
    """
    파일들을 해당 디렉토리로 복사
    
    Args:
        file_list (list): 복사할 파일명 리스트
        source_dirs (dict): 소스 디렉토리 경로들
        target_dirs (dict): 타겟 디렉토리 경로들
        split_type (str): 'train' 또는 'val'
    """
    print(f"\n📋 {split_type.upper()} 파일 복사 중...")
    
    file_types = ['images', 'narrow_images', 'labels']
    extensions = {
        'images': '.jpg',
        'narrow_images': '.jpg', 
        'labels': '.txt'
    }
    
    success_counts = defaultdict(int)
    error_counts = defaultdict(int)
    
    for i, filename in enumerate(file_list):
        if (i + 1) % 100 == 0:  # 100개마다 진행상황 출력
            print(f"  진행: {i+1}/{len(file_list)}")
        
        for file_type in file_types:
            ext = extensions[file_type]
            source_file = os.path.join(source_dirs[file_type], f"{filename}{ext}")
            target_file = os.path.join(target_dirs[f"{split_type}_{file_type}"], f"{filename}{ext}")
            
            try:
                shutil.copy2(source_file, target_file)
                success_counts[file_type] += 1
            except Exception as e:
                error_counts[file_type] += 1
                print(f"❌ 복사 실패: {source_file} -> {target_file}")
                print(f"   에러: {e}")
    
    print(f"✅ {split_type.upper()} 복사 완료:")
    for file_type in file_types:
        success = success_counts[file_type]
        errors = error_counts[file_type]
        print(f"  {file_type}: {success}개 성공, {errors}개 실패")

def analyze_class_distribution(file_list, labels_dir):
    """
    클래스 분포 분석
    
    Args:
        file_list (list): 분석할 파일 리스트
        labels_dir (str): labels 디렉토리
        
    Returns:
        dict: 클래스별 개수
    """
    class_counts = defaultdict(int)
    total_objects = 0
    
    for filename in file_list:
        label_path = os.path.join(labels_dir, f"{filename}.txt")
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 1:
                            class_id = int(parts[0])
                            class_counts[class_id] += 1
                            total_objects += 1
    
    return dict(class_counts), total_objects

def main():
    """메인 함수"""
    
    # 🛠️ 설정 부분
    BASE_DIR = "/home/byounggun/ultralytics/swm_total"
    OUTPUT_DIR = "/home/byounggun/ultralytics/swm_dual_split"
    
    # 소스 디렉토리
    SOURCE_DIRS = {
        'images': os.path.join(BASE_DIR, 'images'),
        'narrow_images': os.path.join(BASE_DIR, 'narrow_images'),
        'labels': os.path.join(BASE_DIR, 'labels')
    }
    
    VAL_RATIO = 0.2  # 20% validation
    
    print("🚀 Dual Stream Dataset Splitter 시작")
    print(f"📂 소스 디렉토리:")
    for name, path in SOURCE_DIRS.items():
        print(f"  {name}: {path}")
    print(f"📂 출력 디렉토리: {OUTPUT_DIR}")
    print(f"📊 분할 비율: Train {(1-VAL_RATIO)*100:.0f}%, Validation {VAL_RATIO*100:.0f}%")
    
    # 1. 소스 디렉토리 존재 확인
    for name, path in SOURCE_DIRS.items():
        if not os.path.exists(path):
            print(f"❌ 디렉토리가 존재하지 않습니다: {path}")
            return
    
    # 2. 출력 디렉토리 구조 생성
    print(f"\n📁 출력 디렉토리 생성 중...")
    target_dirs = setup_directories(OUTPUT_DIR)
    
    # 3. 유효한 파일들 찾기
    print(f"\n🔍 유효한 파일들 검색 중...")
    valid_files = get_valid_files(
        SOURCE_DIRS['images'], 
        SOURCE_DIRS['narrow_images'], 
        SOURCE_DIRS['labels']
    )
    
    if not valid_files:
        print("❌ 유효한 파일을 찾을 수 없습니다.")
        return
    
    print(f"✅ 유효한 파일 {len(valid_files)}개 발견")
    
    # 4. 데이터셋 분할 (일정 간격으로)
    print(f"\n📊 데이터셋 분할 중...")
    train_files, val_files = split_dataset_interval(valid_files, VAL_RATIO)
    
    # 5. 클래스 분포 분석
    print(f"\n📈 클래스 분포 분석 중...")
    train_classes, train_objects = analyze_class_distribution(train_files, SOURCE_DIRS['labels'])
    val_classes, val_objects = analyze_class_distribution(val_files, SOURCE_DIRS['labels'])
    
    print(f"📊 객체 분포:")
    print(f"  Train: {train_objects}개 객체")
    print(f"  Validation: {val_objects}개 객체")
    
    # 상위 5개 클래스 분포 비교
    all_classes = set(train_classes.keys()) | set(val_classes.keys())
    if all_classes:
        print(f"\n📊 주요 클래스 분포 (상위 5개):")
        sorted_classes = sorted(all_classes, key=lambda x: train_classes.get(x, 0) + val_classes.get(x, 0), reverse=True)[:5]
        
        for class_id in sorted_classes:
            train_count = train_classes.get(class_id, 0)
            val_count = val_classes.get(class_id, 0)
            total_count = train_count + val_count
            train_ratio = train_count / total_count if total_count > 0 else 0
            val_ratio = val_count / total_count if total_count > 0 else 0
            print(f"  클래스 {class_id}: Train {train_count}개 ({train_ratio:.1%}), Val {val_count}개 ({val_ratio:.1%})")
    
    # 6. 파일 복사
    print(f"\n📁 파일 복사 시작...")
    
    # Train 파일 복사
    copy_files(train_files, SOURCE_DIRS, target_dirs, 'train')
    
    # Validation 파일 복사  
    copy_files(val_files, SOURCE_DIRS, target_dirs, 'val')
    
    # 7. 최종 결과 출력
    print(f"\n🎉 데이터셋 분할 완료!")
    print(f"📊 최종 결과:")
    print(f"  📂 Train: {len(train_files)}개 샘플")
    print(f"    - Wide images: {target_dirs['train_images']}")
    print(f"    - Narrow images: {target_dirs['train_narrow_images']}")
    print(f"    - Labels: {target_dirs['train_labels']}")
    print(f"  📂 Validation: {len(val_files)}개 샘플")
    print(f"    - Wide images: {target_dirs['val_images']}")
    print(f"    - Narrow images: {target_dirs['val_narrow_images']}")
    print(f"    - Labels: {target_dirs['val_labels']}")
    
    # 8. 검증을 위한 샘플 파일명 출력
    print(f"\n🔍 분할 검증 (처음 5개와 마지막 5개 샘플):")
    print(f"Train 샘플:")
    for i, name in enumerate(train_files[:5]):
        print(f"  {i+1}. {name}")
    if len(train_files) > 5:
        print(f"  ...")
        for i, name in enumerate(train_files[-3:], len(train_files)-2):
            print(f"  {i}. {name}")
    
    print(f"Validation 샘플:")
    for i, name in enumerate(val_files[:5]):
        print(f"  {i+1}. {name}")
    if len(val_files) > 5:
        print(f"  ...")
        for i, name in enumerate(val_files[-3:], len(val_files)-2):
            print(f"  {i}. {name}")
    
    # 9. 간격 분석 (validation이 얼마나 고르게 분포되었는지)
    if len(val_files) > 1:
        val_indices = [valid_files.index(name) for name in val_files]
        val_intervals = [val_indices[i+1] - val_indices[i] for i in range(len(val_indices)-1)]
        avg_interval = sum(val_intervals) / len(val_intervals)
        print(f"\n📏 Validation 샘플 간격 분석:")
        print(f"  평균 간격: {avg_interval:.1f}")
        print(f"  최소 간격: {min(val_intervals)}")
        print(f"  최대 간격: {max(val_intervals)}")

if __name__ == "__main__":
    main() 