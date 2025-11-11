#!/usr/bin/env python3
"""
Rename Matched Pairs Script
Renames matched image pairs to have identical names and copies to new folder
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm

# ============================================================
# 🛠️ 설정 부분
# ============================================================

# 소스 디렉토리 (기존 자동 라벨링 결과)
SOURCE_ROOT = "/home/byounggun/ultralytics/finetune_katri"
SOURCE_WIDE_DIR = os.path.join(SOURCE_ROOT, "images")
SOURCE_NARROW_DIR = os.path.join(SOURCE_ROOT, "narrow_images")
SOURCE_LABEL_DIR = os.path.join(SOURCE_ROOT, "labels")

# 타겟 디렉토리 (이름 통일된 새 폴더)
TARGET_ROOT = "/home/byounggun/ultralytics/finetune_katri_name"
TARGET_WIDE_DIR = os.path.join(TARGET_ROOT, "images")
TARGET_NARROW_DIR = os.path.join(TARGET_ROOT, "narrow_images")
TARGET_LABEL_DIR = os.path.join(TARGET_ROOT, "labels")

# ============================================================


def main():
    """메인 함수"""
    
    print("🚀 Starting Matched Pairs Renaming and Copying...")
    print(f"📂 Source: {SOURCE_ROOT}")
    print(f"📂 Target: {TARGET_ROOT}")
    
    # 타겟 디렉토리 생성
    os.makedirs(TARGET_WIDE_DIR, exist_ok=True)
    os.makedirs(TARGET_NARROW_DIR, exist_ok=True)
    os.makedirs(TARGET_LABEL_DIR, exist_ok=True)
    
    print(f"✅ Created target directories")
    
    # 소스 파일 목록 가져오기
    wide_files = sorted([f for f in os.listdir(SOURCE_WIDE_DIR) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if not wide_files:
        print(f"❌ No images found in {SOURCE_WIDE_DIR}")
        return
    
    print(f"📊 Found {len(wide_files)} image pairs to process")
    
    # 통계
    success_count = 0
    error_count = 0
    
    # 각 파일 처리
    for idx, old_filename in enumerate(tqdm(wide_files, desc="Processing pairs")):
        try:
            # 새 파일명 생성 (단순 인덱스 기반)
            # 확장자 추출
            ext = os.path.splitext(old_filename)[1]
            new_filename = f"image_{idx:06d}{ext}"  # image_000000.jpg, image_000001.jpg, ...
            
            # 라벨 파일명
            old_label_name = os.path.splitext(old_filename)[0] + '.txt'
            new_label_name = os.path.splitext(new_filename)[0] + '.txt'
            
            # 소스 파일 경로
            src_wide_path = os.path.join(SOURCE_WIDE_DIR, old_filename)
            src_narrow_path = os.path.join(SOURCE_NARROW_DIR, old_filename)
            src_label_path = os.path.join(SOURCE_LABEL_DIR, old_label_name)
            
            # 타겟 파일 경로
            dst_wide_path = os.path.join(TARGET_WIDE_DIR, new_filename)
            dst_narrow_path = os.path.join(TARGET_NARROW_DIR, new_filename)
            dst_label_path = os.path.join(TARGET_LABEL_DIR, new_label_name)
            
            # 파일 존재 확인
            if not os.path.exists(src_wide_path):
                print(f"⚠️  Wide image not found: {old_filename}")
                error_count += 1
                continue
            
            if not os.path.exists(src_narrow_path):
                print(f"⚠️  Narrow image not found: {old_filename}")
                error_count += 1
                continue
            
            if not os.path.exists(src_label_path):
                print(f"⚠️  Label not found: {old_label_name}")
                error_count += 1
                continue
            
            # 파일 복사 (이름 변경하여)
            shutil.copy2(src_wide_path, dst_wide_path)
            shutil.copy2(src_narrow_path, dst_narrow_path)
            shutil.copy2(src_label_path, dst_label_path)
            
            success_count += 1
            
        except Exception as e:
            print(f"\n❌ Error processing {old_filename}: {e}")
            error_count += 1
            continue
    
    # 결과 출력
    print(f"\n{'='*60}")
    print(f"✅ Processing completed!")
    print(f"{'='*60}")
    print(f"📊 Statistics:")
    print(f"   Successfully processed: {success_count} pairs")
    print(f"   Errors: {error_count}")
    print(f"   Total: {len(wide_files)}")
    print(f"\n📂 Output location:")
    print(f"   Wide images: {TARGET_WIDE_DIR}")
    print(f"   Narrow images: {TARGET_NARROW_DIR}")
    print(f"   Labels: {TARGET_LABEL_DIR}")
    print(f"{'='*60}")
    
    # 최종 파일 개수 확인
    final_wide_count = len([f for f in os.listdir(TARGET_WIDE_DIR) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    final_narrow_count = len([f for f in os.listdir(TARGET_NARROW_DIR) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    final_label_count = len([f for f in os.listdir(TARGET_LABEL_DIR) 
                            if f.lower().endswith('.txt')])
    
    print(f"\n🎉 Final counts:")
    print(f"   Wide images: {final_wide_count}")
    print(f"   Narrow images: {final_narrow_count}")
    print(f"   Labels: {final_label_count}")
    
    if final_wide_count == final_narrow_count == final_label_count:
        print(f"✅ All counts match! Dataset is ready.")
    else:
        print(f"⚠️  Warning: Counts don't match!")


if __name__ == "__main__":
    main()
