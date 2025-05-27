import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel # For potential type hints or structure reference
from ultralytics.utils import LOGGER
import os
from pytorch_msssim import ssim, ms_ssim

class SiameseYOLOv8s(nn.Module):
    def extract_backbone_features(self, x):
        """
        DetectionModel의 backbone feature만 추출 (YOLOv8 공식 구조 기반)
        여러 scale의 feature map 리스트 반환 (보통 [P3, P4, P5_sppf_out])
        """
        y, outputs = [], []
        for m in self.yolo_model.model:
            if hasattr(m, 'f') and m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x)
        # self.yolo_model.save에 저장된 인덱스의 feature map만 반환
        return [y[i] for i in self.yolo_model.save]

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
            # DetectionModel 전체를 사용하며, backbone feature 추출은 별도 메서드로 처리
            model = yolo_model_full.model
            self.stride = model.stride
            self.names = model.names
            self.nc = model.nc

            # Detect 모듈(헤드)에서 reg_max 등 파라미터를 안전하게 추출
            detect_module = None
            for m in reversed(self.yolo_model.model):
                if m.__class__.__name__ == 'Detect':
                    detect_module = m
                    break
            self.reg_max = getattr(detect_module, 'reg_max', 16)

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

        # Siamese projector 입력 채널은 backbone feature 추출 메서드의 output(P5 등)에 맞춰 동적으로 결정됨
        dummy_input = torch.zeros(1, 3, 64, 64)  # 작은 크기의 더미 이미지
        with torch.no_grad():
            backbone_outputs = self.extract_backbone_features(dummy_input)
            if isinstance(backbone_outputs, (list, tuple)):
                deepest_features = backbone_outputs[-1]  # 항상 가장 깊은 feature map 사용 (P5_sppf_out)
            else:
                deepest_features = backbone_outputs
            backbone_out_channels = deepest_features.shape[1]
            LOGGER.info(f"[Siamese] Detected backbone output channels: {backbone_out_channels}")
        
        self.siamese_projector = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global Average Pooling
            nn.Flatten(),
            nn.Linear(backbone_out_channels, backbone_out_channels // 2), # Reduce dim
            nn.ReLU(),
            nn.Linear(backbone_out_channels // 2, self.feature_dim) # Final projection
        )

        # Siamese 손실 함수 (코사인 임베딩 손실)
        self.cosine_embedding_loss = nn.CosineEmbeddingLoss()

        # SSIM 손실 함수
        # self.ssim_loss = 

    def _extract_siamese_features(self, x):
        """
        입력 이미지를 받아 YOLOv8 backbone feature를 추출하고, Siamese projector에 통과시킴.
        """
        backbone_outputs = self.extract_backbone_features(x)
        if isinstance(backbone_outputs, (list, tuple)):
            deepest_features = backbone_outputs[-1]  # P5_sppf_out 등
        else:
            deepest_features = backbone_outputs
        projected_features = self.siamese_projector(deepest_features)  # (batch_size, self.feature_dim)
        return projected_features

    def get_narrow_roi_in_wide(self, batch_wide, narrow_K, wide_K, device):
        # Assume batch_wide: (B, 3, H, W), narrow_K/wide_K: (3,3) np.ndarray
        # Returns: mask (B, 1, H, W) torch.Tensor
        import cv2
        import numpy as np
        B, C, H, W = batch_wide.shape
        # Narrow 이미지 해상도는 wide와 동일하다고 가정 (입력 전 resize로 맞춘 경우)
        corners = np.array([
            [0, 0],
            [W-1, 0],
            [W-1, H-1],
            [0, H-1]
        ], dtype=np.float32).reshape(-1, 1, 2)
        # undistort 없이 단순 투영(동일 위치/방향 가정)
        # narrow_K, wide_K: np.ndarray
        narrow_K_inv = np.linalg.inv(narrow_K)
        wide_K = wide_K
        points_cam = []
        for pt in corners:
            norm = np.dot(narrow_K_inv, np.array([pt[0][0], pt[0][1], 1.0]))
            norm = norm / norm[2]
            points_cam.append(norm)
        points_cam = np.stack(points_cam, axis=0)  # (4, 3)
        # wide 카메라로 투영
        points_2d = []
        for pt in points_cam:
            proj = np.dot(wide_K, pt)
            proj = proj / proj[2]
            points_2d.append(proj[:2])
        points_2d = np.stack(points_2d, axis=0).astype(np.int32)  # (4,2)
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [points_2d], 1)
        mask = torch.from_numpy(mask).float().unsqueeze(0).to(device)  # (1, H, W)
        mask = mask.unsqueeze(0).repeat(B, 1, 1, 1)  # (B,1,H,W)
        return mask

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
        # --- camera intrinsic (예시, 실제론 __init__에서 받아야 함) ---
        # narrow_K = np.array([[2651.127798, 0, 819.397071], [0, 2635.360938, 896.163803], [0, 0, 1]])
        # wide_K = np.array([[559.258761, 0, 928.108242], [0, 565.348774, 518.787048], [0, 0, 1]])
        # 실제론 self.narrow_K, self.wide_K로 관리 권장
        narrow_K = getattr(self, 'narrow_K', None)
        wide_K = getattr(self, 'wide_K', None)
        if narrow_K is None or wide_K is None:
            import numpy as np
            narrow_K = np.array([[2651.127798, 0, 819.397071], [0, 2635.360938, 896.163803], [0, 0, 1]])
            wide_K = np.array([[559.258761, 0, 928.108242], [0, 565.348774, 518.787048], [0, 0, 1]])
            self.narrow_K = narrow_K
            self.wide_K = wide_K
        # --- wide 이미지에서 narrow view ROI만 마스킹 ---
        # mask = self.get_narrow_roi_in_wide(x_wide, self.narrow_K, self.wide_K, x_wide.device)  # (B,1,H,W)
        # x_wide_masked = x_wide * mask
        # Siamese 임베딩
        # siamese_emb_wide = self._extract_siamese_features(x_wide)
        # siamese_emb_narrow = self._extract_siamese_features(x_narrow)
        # # Siamese 손실
        # target_similarity = torch.ones(x_wide.size(0), device=siamese_emb_wide.device)
        # loss_siamese = self.cosine_embedding_loss(siamese_emb_wide, siamese_emb_narrow, target_similarity)

        # YOLO backbone feature 추출
        feat_wide = self.extract_backbone_features(x_wide)
        feat_narrow = self.extract_backbone_features(x_narrow)
        # 가장 깊은 feature map 사용 (예: P5)
        if isinstance(feat_wide, (list, tuple)):
            feat_wide = feat_wide[-1]
            feat_narrow = feat_narrow[-1]
        # Debug: requires_grad 체크
        if not feat_wide.requires_grad or not feat_narrow.requires_grad:
            from ultralytics.utils import LOGGER
            LOGGER.warning(f"[SiameseYOLOv8s] Backbone features do not require grad! Check backbone train/freeze state.")
        # SSIM 계산 전 feature map 정규화 ([0, 1] 범위)
        def minmax_norm(x):
            return (x - x.min()) / (x.max() - x.min() + 1e-8)
        feat_wide_norm = minmax_norm(feat_wide)
        feat_narrow_norm = minmax_norm(feat_narrow)
        ssim_score_tensor = ssim(feat_wide_norm, feat_narrow_norm, data_range=1.0, size_average=False)
        ssim_loss = 1 - ssim_score_tensor.mean()
        total_ssim_loss = ssim_loss
        
        # Detection
        detection_preds_wide = self.yolo_model(x_wide)
        if self.training:
            return detection_preds_wide, total_ssim_loss
        else:
            return detection_preds_wide