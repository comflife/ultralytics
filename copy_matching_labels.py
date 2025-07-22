#!/usr/bin/env python3
"""
이미지 파일과 매칭되는 라벨 파일들을 복사하는 스크립트
"""

import os
import shutil
from pathlib import Path

def copy_matching_labels():
    # 소스 라벨 디렉토리
    source_labels_dir = Path("/home/byounggun/ultralytics/swm_total/labels")
    
    # 이미지 디렉토리와 타겟 라벨 디렉토리 쌍들
    image_label_pairs = [
        {
            "images_dir": "/home/byounggun/ultralytics/swm_dual_split/train/images",
            "target_labels_dir": "/home/byounggun/ultralytics/swm_dual_split/train/labels"
        },
        {
            "images_dir": "/home/byounggun/ultralytics/swm_dual_split/val/images", 
            "target_labels_dir": "/home/byounggun/ultralytics/swm_dual_split/val/labels"
        }
    ]
    
    for pair in image_label_pairs:
        images_dir = Path(pair["images_dir"])
        target_labels_dir = Path(pair["target_labels_dir"])
        
        print(f"\n처리 중: {images_dir} -> {target_labels_dir}")
        
        # 타겟 라벨 디렉토리 생성
        target_labels_dir.mkdir(parents=True, exist_ok=True)
        
        if not images_dir.exists():
            print(f"⚠️ 이미지 디렉토리가 존재하지 않습니다: {images_dir}")
            continue
            
        # 이미지 파일들 찾기
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(images_dir.glob(f"*{ext}"))
            image_files.extend(images_dir.glob(f"*{ext.upper()}"))
        
        print(f"📁 이미지 파일 개수: {len(image_files)}")
        
        copied_count = 0
        missing_count = 0
        
        for image_file in image_files:
            # 이미지 파일명에서 확장자 제거
            basename = image_file.stem
            
            # 대응하는 라벨 파일 경로
            label_file = source_labels_dir / f"{basename}.txt"
            target_label_file = target_labels_dir / f"{basename}.txt"
            
            if label_file.exists():
                # 라벨 파일 복사
                shutil.copy2(label_file, target_label_file)
                copied_count += 1
                if copied_count <= 5:  # 처음 5개만 출력
                    print(f"✅ 복사됨: {basename}.txt")
            else:
                missing_count += 1
                if missing_count <= 5:  # 처음 5개만 출력
                    print(f"❌ 라벨 없음: {basename}.txt")
        
        print(f"📊 결과: 복사됨 {copied_count}개, 누락됨 {missing_count}개")
        
        # 복사된 라벨 파일 몇 개 확인
        if copied_count > 0:
            sample_label = list(target_labels_dir.glob("*.txt"))[0]
            print(f"📄 샘플 라벨 파일 내용 ({sample_label.name}):")
            with open(sample_label, 'r') as f:
                lines = f.readlines()[:3]  # 처음 3줄만
                for line in lines:
                    parts = line.strip().split()
                    print(f"   {line.strip()} (컬럼 수: {len(parts)})")

if __name__ == "__main__":
    print("🚀 이미지-라벨 매칭 복사 시작...")
    copy_matching_labels()
    print("✅ 완료!")
