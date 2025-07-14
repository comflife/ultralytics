#!/usr/bin/env python3
"""
스크립트: 레이블 파일의 클래스 ID를 1씩 줄이기
"""

import os
from pathlib import Path
import shutil
from datetime import datetime

def reduce_class_ids(label_dir):
    """레이블 파일들의 클래스 ID를 1씩 줄입니다."""
    
    label_path = Path(label_dir)
    if not label_path.exists():
        print(f"❌ 디렉토리가 존재하지 않습니다: {label_dir}")
        return
    
    # 백업 디렉토리 생성
    backup_dir = label_path.parent / f"{label_path.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(exist_ok=True)
    print(f"📁 백업 디렉토리 생성: {backup_dir}")
    
    # 통계 변수
    processed_files = 0
    empty_files = 0
    error_files = 0
    total_objects = 0
    class_stats = {}
    
    # 모든 .txt 파일 처리
    txt_files = list(label_path.glob("*.txt"))
    print(f"📋 총 {len(txt_files)}개의 레이블 파일 발견")
    
    for txt_file in txt_files:
        try:
            # 원본 파일 백업
            shutil.copy2(txt_file, backup_dir / txt_file.name)
            
            # 파일 읽기
            with open(txt_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 빈 파일 건너뛰기
            if not lines or all(not line.strip() for line in lines):
                empty_files += 1
                continue
            
            # 새로운 내용 준비
            new_lines = []
            file_objects = 0
            
            for line in lines:
                line = line.strip()
                if not line:
                    new_lines.append('\n')
                    continue
                
                parts = line.split()
                if len(parts) >= 5:  # class_id x y w h
                    try:
                        old_class_id = int(parts[0])
                        new_class_id = old_class_id - 1
                        
                        # 클래스 통계 업데이트
                        if old_class_id not in class_stats:
                            class_stats[old_class_id] = 0
                        class_stats[old_class_id] += 1
                        
                        # 새로운 라인 생성
                        new_parts = [str(new_class_id)] + parts[1:]
                        new_lines.append(' '.join(new_parts) + '\n')
                        file_objects += 1
                        
                    except ValueError:
                        print(f"⚠️  잘못된 클래스 ID 형식: {txt_file.name} - {line}")
                        new_lines.append(line + '\n')
                else:
                    print(f"⚠️  잘못된 라인 형식: {txt_file.name} - {line}")
                    new_lines.append(line + '\n')
            
            # 파일 쓰기
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            
            processed_files += 1
            total_objects += file_objects
            
            if processed_files % 100 == 0:
                print(f"✅ 처리됨: {processed_files}/{len(txt_files)} 파일")
                
        except Exception as e:
            print(f"❌ 오류 발생: {txt_file.name} - {e}")
            error_files += 1
    
    # 결과 출력
    print(f"\n🎉 처리 완료!")
    print(f"📊 통계:")
    print(f"   - 처리된 파일: {processed_files}")
    print(f"   - 빈 파일: {empty_files}")  
    print(f"   - 오류 파일: {error_files}")
    print(f"   - 총 객체 수: {total_objects}")
    
    print(f"\n📈 클래스별 통계 (변경 전 → 변경 후):")
    for old_class, count in sorted(class_stats.items()):
        new_class = old_class - 1
        print(f"   클래스 {old_class:2d} → {new_class:2d}: {count:4d}개 객체")
    
    print(f"\n💾 백업 위치: {backup_dir}")

if __name__ == "__main__":
    # Train labels 처리 (이미 완료됨)
    print("🎯 Train Labels 처리는 이미 완료되었습니다.")
    
    # Validation labels 처리
    print("\n🎯 Validation Labels 처리 시작...")
    val_label_directory = "/home/byounggun/ultralytics/swm_dual_split/val_labels"
    reduce_class_ids(val_label_directory) 