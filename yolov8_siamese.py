import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel # For potential type hints or structure reference
from ultralytics.utils import LOGGER
import os

class SiameseYOLOv8s(nn.Module):
    def __init__(self, yolo_weights_path='yolov8s.pt', siamese_lambda=0.1, feature_dim=128, args=None):
        """
        Siamese YOLOv8s 모델 초기화.

        Args:
            yolo_weights_path (str): YOLOv8s 사전 학습된 가중치 경로 (e.g., 'yolov8s.pt').
                                     None으로 설정 시, 랜덤 초기화된 YOLOv8s 구조 사용 (구조 정의 필요).
            siamese_lambda (float): 전체 손실에서 Siamese 손실의 가중치.
            feature_dim (int): Siamese 비교를 위한 최종 특징 벡터의 차원.
            args (dict, optional): YOLOv8 스타일 하이퍼파라미터 딕셔너리. (nc, reg_max 등)
        """
        super().__init__()
        self.siamese_lambda = siamese_lambda
        self.feature_dim = feature_dim

        LOGGER.info(f"Initializing SiameseYOLOv8s with weights: {yolo_weights_path}")

        if not os.path.exists(yolo_weights_path):
            LOGGER.warning(f"Pretrained weights not found at {yolo_weights_path}. YOLO components might be randomly initialized if not handled by fallback.")
            # Consider downloading if not exists, or raising error, based on strictness
            # For now, we assume YOLO() handles 'yolov8s.pt' by potentially downloading it.

        try:
            yolo_model_full = YOLO(yolo_weights_path)
            self.yolo_model = yolo_model_full.model  # Expose for v8DetectionLoss compatibility
            self.shared_backbone = yolo_model_full.model.model[0]
            self.detection_head = yolo_model_full.model.model[1] # This is an nn.Sequential

            self.stride = yolo_model_full.model.stride
            self.names = yolo_model_full.model.names
            self.nc = yolo_model_full.model.nc

            # Correctly access reg_max from the Detect layer within the head's Sequential modules
            if isinstance(self.detection_head, torch.nn.Sequential):
                # The Detect layer is typically the last module in the head sequence
                detect_module = self.detection_head[-1]
            else:
                # Fallback if self.detection_head is not Sequential (should be for YOLOv8)
                detect_module = self.detection_head 
            
            self.reg_max = detect_module.reg_max if hasattr(detect_module, 'reg_max') else 16
            
            from types import SimpleNamespace
            if args is not None:
                # dict로 강제 변환 후 SimpleNamespace로 변환
                if not isinstance(args, dict):
                    args = vars(args)
                self.args = SimpleNamespace(**args)
            else:
                self.args = SimpleNamespace(nc=self.nc, reg_max=self.reg_max)
            self.hyp = self.args

            # --- Ensure all required detection loss hyperparameters are present ---
            # 내부 YOLOv8 모델의 args/hyp도 SimpleNamespace로 강제 변환
            from types import SimpleNamespace
            if hasattr(self.yolo_model, 'args') and isinstance(self.yolo_model.args, dict):
                self.yolo_model.args = SimpleNamespace(**self.yolo_model.args)
            if hasattr(self.yolo_model, 'hyp') and isinstance(self.yolo_model.hyp, dict):
                self.yolo_model.hyp = SimpleNamespace(**self.yolo_model.hyp)

            # --- Ensure box, cls, dfl exist in both self.args and self.yolo_model.args ---
            loss_keys = {
                'box': getattr(self.args, 'box', getattr(self.args, 'box_gain', 7.5)),
                'cls': getattr(self.args, 'cls', getattr(self.args, 'cls_gain', 0.5)),
                'dfl': getattr(self.args, 'dfl', getattr(self.args, 'dfl_gain', 1.5)),
            }
            for k, v in loss_keys.items():
                setattr(self.args, k, v)
                if hasattr(self.yolo_model, 'args'):
                    setattr(self.yolo_model.args, k, v)

            # These are the typical keys needed by v8DetectionLoss: box, cls, dfl, etc.
            default_loss_keys = {
                'box': 7.5,        # Default from Ultralytics
                'cls': 0.5,
                'dfl': 1.5,
                'fl_gamma': 2.0,
                'iou_type': 'ciou',
                'anchor_t': 4.0,
                'label_smoothing': 0.0,
                'obj': 1.0,
                'lr0': 0.01,
                'weight_decay': 0.0005
            }
            # If self.args is a dict, update with defaults if missing
            if isinstance(self.args, dict):
                for k, v in default_loss_keys.items():
                    self.args.setdefault(k, v)
                # --- Map *_gain to expected keys if present ---
                mapping = {'box_gain': 'box', 'cls_gain': 'cls', 'dfl_gain': 'dfl'}
                for src, dst in mapping.items():
                    if src in self.args and dst not in self.args:
                        self.args[dst] = self.args[src]
            elif hasattr(self.args, '__dict__'):
                for k, v in default_loss_keys.items():
                    if not hasattr(self.args, k):
                        setattr(self.args, k, v)
                # --- Map *_gain to expected keys if present ---
                mapping = {'box_gain': 'box', 'cls_gain': 'cls', 'dfl_gain': 'dfl'}
                for src, dst in mapping.items():
                    if hasattr(self.args, src) and not hasattr(self.args, dst):
                        setattr(self.args, dst, getattr(self.args, src))
            # --- End ensure hyperparameters ---

            LOGGER.info(f"Successfully extracted backbone and head from {yolo_weights_path}. NC={self.nc}, reg_max={self.reg_max}")

        except Exception as e:
            LOGGER.error(f"Error loading YOLO model from {yolo_weights_path}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize SiameseYOLOv8s components from {yolo_weights_path}: {e}")

        # Siamese 특징 비교를 위한 프로젝션 헤드 설정을 위해 백본 출력 채널 수를 동적으로 확인
        # YOLOv8s에서는 일반적으로 backbone의 마지막 특징맵(P5)의 채널 수가 1024이지만
        # 모델 설정에 따라 달라질 수 있으므로 동적으로 결정합니다.
        
        # 더미 입력으로 백본을 실행하여 출력 채널 수 확인
        dummy_input = torch.zeros(1, 3, 64, 64)  # 작은 크기의 더미 이미지
        with torch.no_grad():
            backbone_outputs = self.shared_backbone(dummy_input)
            if isinstance(backbone_outputs, (list, tuple)):
                deepest_features = backbone_outputs[-1]  # P5_sppf_out
            else:
                deepest_features = backbone_outputs
                
            # 채널 수 확인 (C x H x W 에서 C 값)
            backbone_out_channels = deepest_features.shape[1]
            LOGGER.info(f"Detected backbone output channels: {backbone_out_channels}")
        
        self.siamese_projector = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global Average Pooling
            nn.Flatten(),
            nn.Linear(backbone_out_channels, backbone_out_channels // 2), # Reduce dim
            nn.ReLU(),
            nn.Linear(backbone_out_channels // 2, self.feature_dim) # Final projection
        )

        # Siamese 손실 함수 (코사인 임베딩 손실)
        self.cosine_embedding_loss = nn.CosineEmbeddingLoss()

    def _extract_siamese_features(self, backbone_outputs):
        """
        백본 출력에서 Siamese 비교를 위한 특징을 추출하고 프로젝션합니다.
        YOLOv8 백본(self.shared_backbone)은 [P3, P4, P5_sppf_out] 형태의 특징 맵 리스트를 출력합니다.
        가장 깊은 특징 맵인 P5_sppf_out (backbone_outputs[-1])을 사용합니다.
        """
        if isinstance(backbone_outputs, (list, tuple)):
            deepest_features = backbone_outputs[-1]  # P5_sppf_out, e.g., (batch, 1024, H/32, W/32) for YOLOv8s
        else: # In case the backbone structure outputs a single tensor
            deepest_features = backbone_outputs
        
        projected_features = self.siamese_projector(deepest_features)  # (batch_size, self.feature_dim)
        return projected_features

    def forward(self, x_wide, x_narrow, targets_wide=None):
        """
        모델의 순전파 연산.

        Args:
            x_wide (torch.Tensor): 광각 이미지 배치 (batch_size, 3, H, W).
            x_narrow (torch.Tensor): 협각 이미지 배치 (batch_size, 3, H, W).
            targets_wide (torch.Tensor, optional): 광각 이미지에 대한 타겟 값 (학습 시).
                                                  Ultralytics 형식: (num_targets, 6) [batch_idx, cls_idx, x, y, w, h].

        Returns:
            tuple: 학습 시에는 (detection_predictions_wide, loss_siamese) 반환.
                   추론 시에는 detection_predictions_wide 반환.
                   detection_predictions_wide는 YOLO 헤드의 원시 출력입니다.
                   실제 손실 계산 및 후처리는 학습/추론 스크립트에서 수행됩니다.
        """
        # 1. 공유 백본을 통한 특징 추출
        # feat_wide_tuple/feat_narrow_tuple: [P3_feat, P4_feat, P5_sppf_feat]
        feat_wide_tuple = self.shared_backbone(x_wide)
        feat_narrow_tuple = self.shared_backbone(x_narrow)

        # 2. Siamese 비교를 위한 특징 벡터 추출 및 프로젝션
        siamese_emb_wide = self._extract_siamese_features(feat_wide_tuple)
        siamese_emb_narrow = self._extract_siamese_features(feat_narrow_tuple)

        # 3. Siamese 손실 계산 (학습 시에만 필요하나, 여기서 계산하여 반환 가능)
        # 가정: x_wide와 x_narrow는 항상 동일한 장면의 "긍정적 쌍(positive pair)"
        # 따라서 코사인 유사도를 최대화 (target = 1)
        target_similarity = torch.ones(x_wide.size(0), device=siamese_emb_wide.device)
        loss_siamese = self.cosine_embedding_loss(siamese_emb_wide, siamese_emb_narrow, target_similarity)

        # 4. 광각 이미지에 대한 객체 탐지 경로
        # detection_head 대신 YOLOv8 공식 모델의 forward를 사용 (임시 해결)
        detection_preds_wide = self.yolo_model(x_wide)

        # print('DEBUG: detection_preds_wide type:', type(detection_preds_wide))
        # print('DEBUG: detection_preds_wide:', detection_preds_wide)
        if self.training:
            # 학습 시에는 (탐지 예측 결과, Siamese 손실) 반환
            # 실제 탐지 손실 계산은 외부 학습 스크립트에서 이뤄짐
            return detection_preds_wide, loss_siamese
        else:
            # 추론 시에는 광각 이미지에 대한 탐지 예측 결과만 반환
            return detection_preds_wide