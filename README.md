
# Dual-Stream YOLO Training (최종 학습 스크립트 기반)

이 프로젝트는 Dual-Stream 구조의 YOLO 모델을 학습합니다. 학습은 `train_dual_stream_5.sh` 스크립트로 실행하며, 주요 설정은 다음과 같습니다.

## 1. 학습 스크립트 예시

```bash
python train_yolov8_dual_v4.py \
	--cfg models/yolov8s-dual.yaml \
	--data ultralytics/cfg/datasets/for_swm2.yaml \
	--epochs 150 \
	--batch-size 32 \
	--imgsz 640 \
	--device 3
```

- `train_yolov8_dual_v4.py`: Dual-Stream YOLO 학습을 위한 메인 코드입니다.
- `--cfg models/yolov8s-dual.yaml`: Dual-Stream 구조의 YOLOv8 모델 설정 파일을 사용합니다.
- `--data ultralytics/cfg/datasets/for_swm2.yaml`: 학습 및 검증에 사용할 데이터셋 구성을 지정합니다.
- `--epochs 150`: 총 150 에폭 동안 학습합니다.
- `--batch-size 32`: 배치 크기는 32입니다.
- `--imgsz 640`: 입력 이미지 크기는 640x640입니다.
- `--device 3`: GPU 3번을 사용합니다.

## 2. Dual-Stream 모델 구조 (`models/yolov8s-dual.yaml`)

- 입력: RGB + Depth 등 2종류의 이미지를 채널 방향으로 합쳐 입력합니다.
- 주요 backbone 구조:
	- `MultiStreamConv`, `SpatialAlignedMultiStreamConv` 등 dual-stream 처리를 위한 커스텀 레이어 사용
	- 중간 feature fusion 및 standard YOLOv8 구조와 유사한 head
- `with_depth: True`로 depth estimation도 활성화되어 있습니다.
- 클래스 수(`nc`): 28

## 3. 데이터셋 구성 (`ultralytics/cfg/datasets/for_swm2.yaml`)

- 28개 클래스가 정의되어 있습니다 (예: car, suv, van, pedestrian 등)
- 주요 경로:
	- `train_wide`, `train_narrow`, `val_wide`, `val_narrow`로 wide/narrow(혹은 RGB/Depth) 이미지가 쌍(pair)으로 제공
	- 라벨은 wide 이미지 기준으로 제공
- `with_depth: True`로 depth 정보도 활용

예시:
```yaml
names:
	0: car
	1: suv
	...
	27: unknown
nc: 28
with_depth: True
train_wide: .../train/images
train_narrow: .../train/train_narrow_images
val_wide: .../val/images
val_narrow: .../val/val_narrow_images
```

## 4. 요약

- 본 스크립트는 Dual-Stream YOLO 모델을 학습하기 위한 표준 실행 예시입니다.
- 모델 구조와 데이터셋 포맷은 각각 yaml 파일로 명확히 정의되어 있으니, 실험 목적에 맞게 수정하여 사용할 수 있습니다.