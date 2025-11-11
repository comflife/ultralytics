#!/usr/bin/env python3
"""
Test script to verify Katri image matching logic between cam0 and cam1
"""

import os
from pathlib import Path
import re

# Katri 이미지 루트 디렉토리
KATRI_ROOT = "/home/byounggun/ultralytics/katri_images"


def extract_timestamp_from_filename(filename):
    """
    파일명에서 타임스탬프 추출
    예: projection_144149_268938.jpeg -> (144149, 268938)
    
    Args:
        filename (str): 파일명
    
    Returns:
        tuple: (시분초, 마이크로초) 또는 None
    """
    # projection_HHMMSS_MICROSEC.jpeg 패턴
    match = re.match(r'projection_(\d+)_(\d+)\.jpe?g', filename, re.IGNORECASE)
    if match:
        time_part = int(match.group(1))  # HHMMSS
        microsec = int(match.group(2))   # microseconds
        return (time_part, microsec)
    return None


def find_closest_match(target_timestamp, candidate_timestamps, threshold_ms=100):
    """
    가장 가까운 타임스탬프 매칭 찾기
    
    Args:
        target_timestamp (tuple): (시분초, 마이크로초)
        candidate_timestamps (dict): {filename: (시분초, 마이크로초)}
        threshold_ms (int): 허용 가능한 최대 시간차 (밀리초)
    
    Returns:
        str: 매칭된 파일명 또는 None
    """
    target_time, target_micro = target_timestamp
    
    min_diff = float('inf')
    best_match = None
    
    for filename, (cand_time, cand_micro) in candidate_timestamps.items():
        # 시간 차이 계산 (마이크로초 단위)
        time_diff_sec = abs(target_time - cand_time)
        
        # 시분초가 다르면 건너뛰기 (같은 초 내에서만 매칭)
        if time_diff_sec > 1:
            continue
        
        # 마이크로초 차이 계산
        total_diff_micro = abs(target_micro - cand_micro)
        
        if total_diff_micro < min_diff:
            min_diff = total_diff_micro
            best_match = filename
    
    # threshold 체크 (마이크로초를 밀리초로 변환)
    threshold_micro = threshold_ms * 1000
    if min_diff <= threshold_micro:
        return best_match, min_diff
    
    return None, None


def test_folder_pair_matching(base_name, cam0_folder, cam1_folder):
    """
    한 폴더 쌍의 매칭 결과 테스트
    
    Args:
        base_name (str): 폴더 기본 이름
        cam0_folder (str): cam0 폴더 경로
        cam1_folder (str): cam1 폴더 경로
    """
    print(f"\n{'='*80}")
    print(f"🔍 Testing folder pair: {base_name}")
    print(f"{'='*80}")
    
    # cam0 이미지 목록
    cam0_images = sorted([f for f in os.listdir(cam0_folder) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    # cam1 이미지 목록
    cam1_images = sorted([f for f in os.listdir(cam1_folder) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"📊 cam0 images: {len(cam0_images)}")
    print(f"📊 cam1 images: {len(cam1_images)}")
    
    # cam1 타임스탬프 딕셔너리 생성
    cam1_timestamps = {}
    for img in cam1_images:
        timestamp = extract_timestamp_from_filename(img)
        if timestamp:
            cam1_timestamps[img] = timestamp
    
    print(f"📊 cam1 parsed timestamps: {len(cam1_timestamps)}")
    
    # 매칭 결과
    matched_pairs = []
    unmatched_cam0 = []
    
    # cam0 이미지별로 cam1에서 가장 가까운 매칭 찾기
    for cam0_img in cam0_images:
        cam0_timestamp = extract_timestamp_from_filename(cam0_img)
        
        if not cam0_timestamp:
            print(f"⚠️  Failed to parse timestamp from: {cam0_img}")
            unmatched_cam0.append(cam0_img)
            continue
        
        # 가장 가까운 cam1 이미지 찾기
        match, time_diff = find_closest_match(cam0_timestamp, cam1_timestamps, threshold_ms=150)
        
        if match:
            matched_pairs.append((cam0_img, match, time_diff))
        else:
            unmatched_cam0.append(cam0_img)
    
    # 결과 출력
    print(f"\n✅ Matched pairs: {len(matched_pairs)}")
    print(f"❌ Unmatched cam0 images: {len(unmatched_cam0)}")
    print(f"📈 Match rate: {len(matched_pairs) / len(cam0_images) * 100:.1f}%")
    
    # 샘플 매칭 결과 출력 (처음 10개)
    print(f"\n📋 Sample matched pairs (first 10):")
    print(f"{'cam0 (wide)':<50} {'cam1 (narrow)':<50} {'Time diff (μs)':>15}")
    print("-" * 115)
    for i, (cam0_img, cam1_img, diff) in enumerate(matched_pairs[:10]):
        print(f"{cam0_img:<50} {cam1_img:<50} {diff:>15,}")
    
    if len(matched_pairs) > 10:
        print(f"... and {len(matched_pairs) - 10} more pairs")
    
    # 매칭 안된 샘플 출력
    if unmatched_cam0:
        print(f"\n❌ Sample unmatched cam0 images (first 5):")
        for img in unmatched_cam0[:5]:
            print(f"   {img}")
    
    return matched_pairs, unmatched_cam0


def main():
    """메인 함수"""
    
    print("🚀 Testing Katri Image Matching Logic")
    print(f"📂 Katri root: {KATRI_ROOT}")
    
    # Katri 폴더 목록 가져오기
    katri_folders = sorted([f for f in os.listdir(KATRI_ROOT) 
                           if os.path.isdir(os.path.join(KATRI_ROOT, f))])
    
    # cam0와 cam1 쌍으로 그룹화
    folder_pairs = {}
    for folder in katri_folders:
        # 예: 20250930_144149_cam0 -> (20250930_144149, cam0)
        if '_cam' in folder:
            base_name = folder.rsplit('_cam', 1)[0]
            cam_type = 'cam' + folder.rsplit('_cam', 1)[1]
            
            if base_name not in folder_pairs:
                folder_pairs[base_name] = {}
            folder_pairs[base_name][cam_type] = folder
    
    print(f"🎯 Found {len(folder_pairs)} folder pairs")
    print(f"\n📁 Folder pairs:")
    for base_name, cams in sorted(folder_pairs.items()):
        cam0_status = "✓" if 'cam0' in cams else "✗"
        cam1_status = "✓" if 'cam1' in cams else "✗"
        print(f"   {base_name}: cam0 {cam0_status}  cam1 {cam1_status}")
    
    # 첫 번째 폴더 쌍만 테스트
    print("\n" + "="*80)
    print("Testing FIRST folder pair in detail...")
    print("="*80)
    
    first_pair = list(folder_pairs.items())[0]
    base_name, cams = first_pair
    
    if 'cam0' not in cams or 'cam1' not in cams:
        print(f"❌ First pair {base_name} is missing cam0 or cam1")
        return
    
    cam0_folder = os.path.join(KATRI_ROOT, cams['cam0'])
    cam1_folder = os.path.join(KATRI_ROOT, cams['cam1'])
    
    matched_pairs, unmatched = test_folder_pair_matching(base_name, cam0_folder, cam1_folder)
    
    # 전체 통계
    print("\n" + "="*80)
    print("📊 SUMMARY - Testing all folder pairs")
    print("="*80)
    
    total_matched = 0
    total_cam0_images = 0
    
    for base_name, cams in folder_pairs.items():
        if 'cam0' not in cams or 'cam1' not in cams:
            continue
        
        cam0_folder = os.path.join(KATRI_ROOT, cams['cam0'])
        cam1_folder = os.path.join(KATRI_ROOT, cams['cam1'])
        
        cam0_images = [f for f in os.listdir(cam0_folder) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        cam1_images = [f for f in os.listdir(cam1_folder) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        cam1_timestamps = {}
        for img in cam1_images:
            timestamp = extract_timestamp_from_filename(img)
            if timestamp:
                cam1_timestamps[img] = timestamp
        
        matched_count = 0
        for cam0_img in cam0_images:
            cam0_timestamp = extract_timestamp_from_filename(cam0_img)
            if cam0_timestamp:
                match, _ = find_closest_match(cam0_timestamp, cam1_timestamps, threshold_ms=150)
                if match:
                    matched_count += 1
        
        total_matched += matched_count
        total_cam0_images += len(cam0_images)
        
        print(f"{base_name}: {matched_count}/{len(cam0_images)} matched "
              f"({matched_count/len(cam0_images)*100:.1f}%)")
    
    print(f"\n{'='*80}")
    print(f"🎉 TOTAL RESULTS:")
    print(f"   Total cam0 images: {total_cam0_images}")
    print(f"   Total matched pairs: {total_matched}")
    print(f"   Overall match rate: {total_matched/total_cam0_images*100:.1f}%")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
