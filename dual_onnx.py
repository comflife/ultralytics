import torch
import torch.nn as nn
import numpy as np
import os

# --- DualYOLOv8 클래스 정의 (제공해주신 코드 그대로 포함) ---
class DualYOLOv8(nn.Module):
    def __init__(self, yolo_weights_path='yolov8s.pt', args=None,
                 wide_K=None, wide_P=None, narrow_K=None, narrow_P=None, img_w=1920, img_h=1080, img_size=640):
        super().__init__()
        from ultralytics import YOLO

        # Check if dummy camera parameters are provided for ONNX export
        if wide_K is None:
            print("[WARN] Using dummy wide_K. Please provide actual camera parameters for real use.")
            wide_K = np.array([[1000, 0, img_w / 2], [0, 1000, img_h / 2], [0, 0, 1]], dtype=np.float32)
        if wide_P is None:
            print("[WARN] Using dummy wide_P. Please provide actual camera parameters for real use.")
            wide_P = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32) @ np.eye(4, dtype=np.float32) # Identity pose
        if narrow_K is None:
            print("[WARN] Using dummy narrow_K. Please provide actual camera parameters for real use.")
            narrow_K = np.array([[800, 0, img_w / 2], [0, 800, img_h / 2], [0, 0, 1]], dtype=np.float32)
        if narrow_P is None:
            print("[WARN] Using dummy narrow_P. Please provide actual camera parameters for real use.")
            narrow_P = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32) @ np.eye(4, dtype=np.float32) # Identity pose

        # 수정: pt 파일이면 가중치까지 로드, 아니면 구조만 생성
        if yolo_weights_path is not None and yolo_weights_path.endswith('.pt') and os.path.exists(yolo_weights_path):
            yolo_model_full = YOLO(yolo_weights_path)
        elif yolo_weights_path is not None and yolo_weights_path.endswith('.yaml') and os.path.exists(yolo_weights_path):
            yolo_model_full = YOLO(yolo_weights_path)
        elif yolo_weights_path is None:
            # 구조만 생성 (커스텀 구조 코드 기반)
            yolo_model_full = YOLO('yolov8s.yaml')  # 또는 커스텀 구조 코드에 맞게 수정
        else:
            # os.path.exists() 체크가 중요합니다. 파일이 없으면 ultralytics가 다운로드 시도합니다.
            try:
                yolo_model_full = YOLO(yolo_weights_path)
            except Exception as e:
                raise ValueError(f"올바른 yolo_weights_path가 필요하거나 파일을 찾을 수 없습니다: {e}")

        self.yolo_model = yolo_model_full.model  # DetectionModel
        self.model = self.yolo_model.model  # for v8DetectionLoss compatibility (nn.ModuleList)
        self.stride = self.yolo_model.stride
        import yaml
        dataset_yaml = 'ultralytics/cfg/datasets/traffic.yaml' # 이 경로가 실제 프로젝트에 맞는지 확인
        self.nc = None
        self.names = None
        if os.path.exists(dataset_yaml):
            with open(dataset_yaml, 'r') as f:
                data_yaml = yaml.safe_load(f)
            if isinstance(data_yaml, dict):
                self.nc = data_yaml.get('nc', None)
                self.names = data_yaml.get('names', None)
        detect_head = self.model[-1]
        if self.nc is None:
            self.nc = getattr(detect_head, 'nc', getattr(self.yolo_model, 'nc', None))
        if self.names is None:
            self.names = getattr(detect_head, 'names', getattr(self.yolo_model, 'names', None))
        self.save = self.yolo_model.save  # backbone output indices for multi-scale features
        self.args = args
        # --- 카메라 파라미터 저장 ---
        self.wide_K = wide_K
        self.wide_P = wide_P
        self.narrow_K = narrow_K
        self.narrow_P = narrow_P
        self.img_w = img_w
        self.img_h = img_h
        self.img_size = img_size if img_size is not None else (args.img_size if args and hasattr(args, 'img_size') else 640)

        # --- Feature Fusion Normalization & Learnable Weights (Multi-Scale) ---
        # Use separate BN for wide/narrow at each fusion scale (YOLOv8 official style)
        fusion_channels = []
        for i in self.save:
            layer = self.model[i]
            if hasattr(layer, 'out_channels'):
                fusion_channels.append(layer.out_channels)
            elif hasattr(layer, 'cv2') and hasattr(layer.cv2, 'out_channels'):
                fusion_channels.append(layer.cv2.out_channels)

        # learnable parameter is not used in forward pass currently
        # self.fusion_weight = nn.Parameter(torch.ones(len(self.save), 2) * 0.5, requires_grad=True)

    def project_narrow_to_wide(self):
        """
        narrow 이미지의 4개 코너를 wide 이미지 평면에 투영하여 wide_corners(img_size 기준, float32, (4,2)) 반환
        (이미지와 feature는 dataloader에서 img_size로 resize되어 들어오므로, 추가 스케일링 불필요)
        """
        narrow_corners = np.array([
            [0, 0],
            [self.img_w-1, 0],
            [self.img_w-1, self.img_h-1],
            [0, self.img_h-1]
        ], dtype=np.float32)
        
        # Check if narrow_K is singular for inversion
        try:
            narrow_K_inv = np.linalg.inv(self.narrow_K)
        except np.linalg.LinAlgError:
            print("[ERROR] narrow_K is singular. Cannot compute inverse. Using identity for projection.")
            narrow_K_inv = np.eye(3, dtype=np.float32)

        rays = []
        for u, v in narrow_corners:
            pixel = np.array([u, v, 1.0])
            ray = narrow_K_inv @ pixel
            ray = ray / ray[2] if ray[2] != 0 else ray # Avoid division by zero
            rays.append(ray)
        rays = np.stack(rays, axis=0)  # (4, 3)
        
        wide_corners = []
        for ray in rays:
            X, Y, Z = ray[0], ray[1], 1.0
            # Ensure pt3d_wide has 4 components for 4x4 wide_P
            # If wide_P is 3x4, pt3d_wide should be 3 components (X,Y,Z)
            # Given that wide_P is usually 3x4 for projection matrix, and pt3d_wide is 4x1 (homogeneous)
            # The previous code implicitly assumes pt3d_wide is 4x1 and wide_P is 3x4.
            # Let's align with common projection: P @ [X, Y, Z, 1].T
            pt3d_wide = np.array([X, Y + 0.2, Z, 1.0]) # Homogeneous coordinate
            
            # Ensure P is 3x4 for standard projection
            if self.wide_P.shape == (4,4): # If P is a 4x4 transformation matrix (e.g. from RT)
                 proj = (self.wide_K @ self.wide_P[:3,:]) @ pt3d_wide # K @ [R|t] @ X
            elif self.wide_P.shape == (3,4): # If P is already the full projection matrix
                 proj = self.wide_P @ pt3d_wide
            else:
                raise ValueError(f"Unexpected wide_P shape: {self.wide_P.shape}")
            
            # Handle division by zero for projection
            if proj[2] != 0:
                proj = proj / proj[2]
            else:
                # If proj[2] is zero, it means the point is at infinity on the image plane.
                # This could cause issues. For dummy data, we can try to handle it.
                # For real data, it suggests an issue with camera parameters or point.
                print("[WARN] Projection Z-component is zero. Point is at infinity on image plane.")
                # Fallback: just use current x, y or some large value
                proj[0] = proj[0] if proj[2] != 0 else np.sign(proj[0]) * 1e6 if proj[0] !=0 else 0
                proj[1] = proj[1] if proj[2] != 0 else np.sign(proj[1]) * 1e6 if proj[1] !=0 else 0
                proj[2] = 1.0 # Set to 1 to avoid NaN later

            wide_corners.append([proj[0], proj[1]])
        wide_corners = np.array(wide_corners, dtype=np.float32)  # (4,2)
        
        # Scale corners to img_size if they are based on original img_w, img_h
        # Assuming project_narrow_to_wide returns corners scaled to img_size based on original comment
        # "이미지와 feature는 dataloader에서 img_size로 resize되어 들어오므로, 추가 스케일링 불필요"
        # If the K matrices are defined relative to original img_w, img_h:
        # Scale factors
        scale_x = self.img_size / self.img_w
        scale_y = self.img_size / self.img_h
        wide_corners[:, 0] *= scale_x
        wide_corners[:, 1] *= scale_y

        return wide_corners

    def extract_single_backbone_feature(self, x, save_idx):
        # x를 backbone에 통과시켜 save_idx에 해당하는 feature만 추출
        y = []
        for i, m in enumerate(self.model):
            if hasattr(m, 'f') and m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [y[j] for j in m.f]
            x = m(x)
            y.append(x)
            if i == save_idx:
                return x
        return None # Should not happen if save_idx is valid

    def forward(self, wide_img, narrow_img):
        """
        wide_img, narrow_img: (B, 3, 640, 640)
        카메라 파라미터 기반으로 narrow FOV가 wide feature map에서 어디에 위치하는지 계산 후 sum
        """
        # 1. narrow의 4개 코너를 wide로 투영 (이미 img_size 기준으로 반환)
        # ONNX export를 위해서는 numpy 연산을 torch 연산으로 바꾸거나,
        # forward 밖에서 project_narrow_to_wide를 미리 계산하여 상수로 넣어줘야 합니다.
        # 여기서는 ONNX 호환을 위해 forward 밖에서 계산된 값을 가정하거나,
        # torch tensor 연산으로 변환해야 합니다.
        # 현재 project_narrow_to_wide는 numpy를 사용하므로, ONNX export 시 이 부분을 주의해야 합니다.
        # 간단한 ONNX export를 위해 이 부분은 건너뛰거나, 해당 로직을 forward 밖에서 계산한 후
        # 결과를 모델의 버퍼로 등록하여 사용하는 방식으로 수정하는 것이 좋습니다.
        # 여기서는 ONNX export를 위해 project_narrow_to_wide()의 결과가 필요하지 않다고 가정하거나
        # 해당 결과는 추론 시 동적으로 사용되지 않는다고 가정합니다.
        # 실제 모델 동작을 위해선 이 부분의 ONNX 호환성을 심도있게 고려해야 합니다.

        # For ONNX export, project_narrow_to_wide should not contain numpy operations.
        # If the output of project_narrow_to_wide is dynamic per input, it needs to be
        # converted to torch operations or passed as an input tensor.
        # If it's static, it can be calculated once and registered as a buffer.
        # For simplicity in this ONNX export example, we will ignore the direct use of
        # wide_corners from this numpy function within the torch.jit.traceable forward pass.
        # However, the core logic relies on it. For a truly robust ONNX export, this part
        # needs to be refactored to use torch operations or pre-calculated static values.
        # Since the fusion logic uses fixed YOLO-like bbox, wide_corners is not directly used for fusion.
        # So we can safely bypass this for ONNX export if it's not used in the torch graph.

        y_wide, y_narrow = [], []
        y = [] # This will eventually hold the fused features and subsequent neck/head outputs
        save_outputs = dict()  # model idx -> y idx

        x_wide, x_narrow = wide_img, narrow_img
        
        # Iterate through backbone layers up to the last save point
        for i, m in enumerate(self.model[:self.save[-1]+1]):
            # Prepare inputs for current module 'm'
            # If m.f is -1, it means the input is the previous layer's output (y_wide[-1], y_narrow[-1])
            # If m.f is an int, it's a skip connection to y_wide[m.f] / y_narrow[m.f]
            # If m.f is a list, it's a concatenation of multiple previous layer outputs
            if not hasattr(m, 'f') or m.f == -1: # Sequential input
                xw_in = y_wide[-1] if len(y_wide) > 0 else x_wide
                xn_in = y_narrow[-1] if len(y_narrow) > 0 else x_narrow
            else: # Skip connection(s)
                if isinstance(m.f, int):
                    xw_in = y_wide[m.f]
                    xn_in = y_narrow[m.f]
                else: # list of indices
                    xw_in = [y_wide[j] for j in m.f]
                    xn_in = [y_narrow[j] for j in m.f]
            
            # Pass through the module
            xw_out = m(xw_in)
            xn_out = m(xn_in)
            
            # Store outputs for future use (skip connections)
            y_wide.append(xw_out)
            y_narrow.append(xn_out)
            
            # --- Perform spatial fusion at specific backbone output indices (self.save) ---
            if i in self.save:
                B, C, H, W = xw_out.shape
                
                # === YOLO label 비율 기반 bbox 계산 ===
                # 이 계산은 onnx graph에 포함됩니다.
                x_center_norm = 0.5
                y_center_norm = 0.5
                width_norm = 1 / 4.494
                height_norm = 1 / 4.552

                # Calculate pixel coordinates for the fusion region
                cx = x_center_norm * W
                cy = y_center_norm * H
                w = width_norm * W
                h = height_norm * H

                fx_min = torch.round(cx - w / 2).int()
                fx_max = torch.round(cx + w / 2).int()
                fy_min = torch.round(cy - h / 2).int()
                fy_max = torch.round(cy + h / 2).int()

                # Clamp to feature map dimensions
                fx_min = torch.clamp(fx_min, 0, W)
                fy_min = torch.clamp(fy_min, 0, H)
                fx_max = torch.clamp(fx_max, 0, W)
                fy_max = torch.clamp(fy_max, 0, H)

                region_w = fx_max - fx_min
                region_h = fy_max - fy_min

                # Fusion
                # Ensure region_w and region_h are at least 1 to avoid errors with interpolate
                region_w = torch.max(region_w, torch.tensor(1, device=region_w.device))
                region_h = torch.max(region_h, torch.tensor(1, device=region_h.device))

                narrow_region = torch.nn.functional.interpolate(
                    xn_out, size=(region_h, region_w), mode='bilinear', align_corners=False
                )
                
                # Create a zero tensor of the same shape as wide feature map
                narrow_full = torch.zeros_like(xw_out)
                
                # Place the interpolated narrow region into the full tensor
                # Ensure indices are within bounds for slicing, though clamp() should handle this.
                narrow_full[..., fy_min:fy_max, fx_min:fx_max] = narrow_region
                
                # Summing wide and narrow features
                xw_out_spatial = xw_out + narrow_full
                
                y.append(xw_out_spatial)
                save_outputs[i] = len(y) - 1 # Store index in 'y' list
            else:
                # If not a save point, just append the wide output
                y.append(xw_out)
                save_outputs[i] = len(y) - 1

        # 2. Neck and Head processing
        # Continue from the last backbone output (self.save[-1]+1) to the end of the model
        for i in range(self.save[-1]+1, len(self.model)):
            m = self.model[i]
            
            # Prepare input for current module 'm'
            if not hasattr(m, 'f') or m.f == -1: # Sequential input
                x_in = y[-1]
            else: # Skip connection(s)
                if isinstance(m.f, int):
                    x_in = y[save_outputs[m.f]]
                else: # list of indices
                    # Collect inputs based on indices stored in save_outputs
                    x_in = [y[save_outputs[j]] for j in m.f]
            
            out = m(x_in)
            y.append(out)
            
        return y[-1] # Return the final detection head output

# --- ONNX Export 스크립트 ---
if __name__ == "__main__":
    # 1. 모델 인스턴스화
    # 실제 환경에 맞는 경로와 파라미터를 사용하세요.
    # YOLOv8s 가중치 파일이 없다면 'yolov8s.yaml'로 변경하거나, 실행 시 다운로드될 것입니다.
    yolo_weights_path = 'yolov8s.pt'
    
    # 더미 카메라 파라미터 (ONNX export를 위해 필수)
    # 실제 사용 시에는 정확한 카메라 캘리브레이션 값으로 교체해야 합니다.
    img_w, img_h = 1920, 1080
    img_size = 640 # YOLO input size

    # K (Intrinsic Camera Matrix): focal_x, focal_y, principal_x, principal_y
    # Example: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    dummy_wide_K = np.array([
        [1000, 0, img_w / 2],
        [0, 1000, img_h / 2],
        [0, 0, 1]
    ], dtype=np.float32)

    dummy_narrow_K = np.array([
        [800, 0, img_w / 2],
        [0, 800, img_h / 2],
        [0, 0, 1]
    ], dtype=np.float32)

    # P (Projection Matrix): [R|t] or K[R|t]
    # For a simple case, assuming identity rotation and zero translation relative to world origin.
    # Or, if it's K[R|t], it's a 3x4 matrix.
    # Let's use a 3x4 identity-like matrix for simplicity, or 4x4 if representing RT.
    # The project_narrow_to_wide function handles 3x4 or 4x4 for P.
    dummy_wide_P = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ], dtype=np.float32) # Standard 3x4 projection matrix [I|0]

    dummy_narrow_P = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ], dtype=np.float32) # Standard 3x4 projection matrix [I|0]

    try:
        model = DualYOLOv8(
            yolo_weights_path=yolo_weights_path,
            wide_K=dummy_wide_K,
            wide_P=dummy_wide_P,
            narrow_K=dummy_narrow_K,
            narrow_P=dummy_narrow_P,
            img_w=img_w,
            img_h=img_h,
            img_size=img_size
        )
        print("DualYOLOv8 model initialized successfully.")
    except Exception as e:
        print(f"Error initializing DualYOLOv8 model: {e}")
        print("Please ensure 'yolov8s.pt' or 'yolov8s.yaml' is accessible or specified correctly.")
        exit()

    model.eval() # 모델을 추론 모드로 설정 (dropout, BatchNorm 등이 올바르게 동작)

    # 2. 더미 입력 텐서 생성
    # batch_size=1, 3 channels (RGB), img_size x img_size
    dummy_wide_img = torch.randn(1, 3, img_size, img_size)
    dummy_narrow_img = torch.randn(1, 3, img_size, img_size)

    # 3. ONNX 파일 경로 설정
    onnx_output_path = 'dualyolov8_model.onnx'

    # 4. ONNX 내보내기
    print(f"Exporting model to ONNX: {onnx_output_path}...")
    try:
        torch.onnx.export(
            model,                              # 내보낼 모델
            (dummy_wide_img, dummy_narrow_img), # 모델의 입력 (튜플 형식)
            onnx_output_path,                   # ONNX 파일 저장 경로
            verbose=False,                      # 자세한 출력 (디버깅용)
            input_names=['wide_img', 'narrow_img'], # 입력 텐서 이름
            output_names=['output'],            # 출력 텐서 이름
            opset_version=11,                   # ONNX opset 버전 (최신 버전 사용 권장)
            dynamic_axes={                      # 동적 축 설정 (배치 크기 등)
                'wide_img': {0: 'batch_size'},
                'narrow_img': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
            # do_constant_folding=True, # 상수 폴딩 활성화 (선택 사항)
            # export_params=True, # 학습된 파라미터 내보내기 여부
        )
        print(f"Model successfully exported to {onnx_output_path}")

        # 5. ONNX 모델 검증 (선택 사항, onnx 라이브러리 필요)
        import onnx
        onnx_model = onnx.load(onnx_output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model check successful!")

    except Exception as e:
        print(f"Error during ONNX export: {e}")

    print("\n--- Next Steps ---")
    print(f"1. You can now use Netron (https://netron.app/) to visualize '{onnx_output_path}'.")
    print(f"2. Or use `onnxruntime` to run inference with the exported model.")
    print("   Example: import onnxruntime; sess = onnxruntime.InferenceSession('dualyolov8_model.onnx')")