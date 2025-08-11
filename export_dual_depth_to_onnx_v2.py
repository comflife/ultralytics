#!/usr/bin/env python3
"""
YOLO Concatenated Dual-Stream Model with Depth ONNX Export Script
Exports a model trained on 4D concatenated input ([B, 6, H, W]) to ONNX format.
"""

"""
# 기본 변환
python export_dual_depth_to_onnx_v2.py /home/byounggun/ultralytics/runs/train/exp244/weights/best.pt

# 추론 예제도 함께 생성
python export_dual_depth_to_onnx_v2.py /home/byounggun/ultralytics/runs/train/exp198/weights/best.pt --create-example

# GPU 사용 및 dynamic shape 지원
python export_dual_depth_to_onnx.py /home/byounggun/ultralytics/runs/train/exp149/weights/best.pt --device cuda:0 --dynamic

# FP16 precision으로 변환
python export_dual_depth_to_onnx.py /home/byounggun/ultralytics/runs/train/exp149/weights/best.pt --half --device cuda:0
"""

import sys
import os
from pathlib import Path

# 🔧 로컬 ultralytics 모듈을 우선적으로 사용하도록 설정
SCRIPT_DIR = Path(__file__).parent.absolute()
ULTRALYTICS_ROOT = SCRIPT_DIR

# Python path에 현재 디렉토리를 최우선으로 추가
sys.path.insert(0, str(ULTRALYTICS_ROOT))

print(f"🔧 Using local ultralytics from: {ULTRALYTICS_ROOT}")

# 이제 로컬 ultralytics import
try:
    from ultralytics import YOLO
    import ultralytics

    print(f"✅ Using ultralytics from: {ultralytics.__file__}")
except ImportError as e:
    print(f"❌ Failed to import ultralytics: {e}")
    sys.exit(1)

import torch
import numpy as np
import json
import argparse

def verify_custom_modules_with_depth():
    """커스텀 모듈들이 제대로 로드되는지 확인 - Depth estimation 포함"""
    print("🔍 Verifying custom modules with depth support...")
    
    try:
        from ultralytics.nn.modules.conv import MultiStreamConv, SpatialAlignedMultiStreamConv
        print("✅ MultiStreamConv and SpatialAlignedMultiStreamConv imported successfully")
        from ultralytics.nn.modules.block import C2f
        print("✅ C2f imported successfully")
        from ultralytics.nn.modules.head import Detect
        print("✅ Detect head imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import custom modules: {e}")
        return False
    
    # 🔴 CHANGED: 4D Concatenated 입력으로 테스트
    try:
        # 4D 입력 테스트
        test_input = torch.randn(1, 6, 32, 32)  # [B, 6, H, W]
        
        # MultiStreamConv 테스트 (c1=6)
        conv_test = MultiStreamConv(6, 64)
        conv_output = conv_test(test_input)
        print(f"✅ MultiStreamConv test with 4D input: {test_input.shape} -> {conv_output.shape}")

        # SpatialAlignedMultiStreamConv 테스트 (c1=64)
        # MultiStreamConv의 출력을 입력으로 사용
        spatial_conv_test = SpatialAlignedMultiStreamConv(64, 128)
        spatial_conv_output = spatial_conv_test(conv_output)
        print(f"✅ SpatialAlignedMultiStreamConv test with 4D input: {conv_output.shape} -> {spatial_conv_output.shape}")
        
        # Depth 정보 확인
        print("🔍 Checking depth estimation support...")
        print("   -> Model should output [bbox + conf + cls + depth] format")
        print("   -> Expected output channels: nc + 5 + 1 (depth)")
        
        return True
    except Exception as e:
        print(f"❌ Failed to test custom modules: {e}")
        import traceback
        traceback.print_exc()
        return False

def load_depth_normalization_info(norm_info_path="/home/byounggun/ultralytics/depth_normalization_info.json"):
    """Depth 정규화 정보 로드"""
    try:
        with open(norm_info_path, 'r') as f:
            norm_info = json.load(f)
        print(f"✅ Depth normalization info loaded:")
        print(f"   - Min depth: {norm_info['min_depth']:.3f}")
        print(f"   - Max depth: {norm_info['max_depth']:.3f}")
        print(f"   - Mean depth: {norm_info['mean_depth']:.3f}")
        return norm_info
    except FileNotFoundError:
        print(f"⚠️  Depth normalization info not found: {norm_info_path}")
        print("   -> Using default values (may affect depth accuracy)")
        return {
            "min_depth": 0.1,
            "max_depth": 419.1,
            "mean_depth": 42.96,
            "normalization_method": "min_max"
        }
    except Exception as e:
        print(f"❌ Failed to load depth normalization info: {e}")
        return None

def format_output_info(output):
    """출력 정보를 포맷팅하는 헬퍼 함수"""
    if isinstance(output, torch.Tensor):
        return f"Tensor{tuple(output.shape)}"
    elif isinstance(output, (tuple, list)):
        return f"({', '.join([format_output_info(item) for item in output])})"
    else:
        return f"{type(output).__name__}"

def analyze_model_output_structure(model, dummy_input):
    """
    모델의 출력을 분석하여 구조와 주요 출력 텐서를 반환
    """
    print("🔍 Analyzing model output structure...")
    try:
        # 🔴 FIX: Unpack the tuple for multiple inputs
        outputs = model(*dummy_input)
        
        main_output = None
        # 🔴 FIX: The model returns a list [P3_output, P4_output, P5_output]
        # The final output for detection is a concatenation of these.
        if isinstance(outputs, (tuple, list)):
            # In export mode, the output is a list of tensors from different detection heads
            print(f"   - Model returns a tuple/list of {len(outputs)} detection heads.")
            for i, out in enumerate(outputs):
                print(f"     - Head {i}: shape={out.shape}, dtype={out.dtype}")
            
            # For ONNX export, we treat them as separate outputs.
            # The 'main_output' concept is less relevant here, but we can use the first.
            main_output = outputs[0]
        else:
            print(f"   - Model returns a single tensor.")
            main_output = outputs
            outputs = [outputs] # Ensure outputs is a list
            print(f"     - Output: shape={main_output.shape}, dtype={main_output.dtype}")
            
        return outputs, main_output
    except Exception as e:
        print(f"❌ Failed during model analysis: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def export_dual_depth_model_to_onnx(
    model_path: str,
    output_path: str = None,
    imgsz: int = 640,
    half: bool = False,
    dynamic: bool = False,
    opset: int = 11,
    device: str = "cpu"
):
    """
    Depth estimation이 포함된 4D Concatenated 듀얼 스트림 YOLO 모델을 ONNX로 export
    """
    
    print(f"\n🚀 Loading 4D Concatenated dual-stream YOLO model with depth from: {model_path}")
    
    if not verify_custom_modules_with_depth():
        print("❌ Custom modules verification failed!")
        return None
    
    depth_info = load_depth_normalization_info()
    if depth_info is None:
        print("❌ Failed to load depth normalization info!")
        return None
    
    try:
        print("📂 Loading model...")
        yolo_model = YOLO(model_path)
        from ultralytics.nn.modules.head import Detect

        # 래퍼 모델 정의: 두 개의 3채널 입력을 받아 하나로 합침
        class DualInputWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, images_wide, images_narrow):
                x = torch.cat((images_wide, images_narrow), dim=1)
                return self.model(x)

        # Set the original model to export mode before wrapping
        original_model = yolo_model.model
        original_model.eval()
        for m in original_model.modules():
            if hasattr(m, 'export'):
                m.export = True
            if isinstance(m, Detect):
                m.dynamic = dynamic
                m.export = True
        print("✅ Model set to export mode.")

        # 원본 모델을 래퍼로 감싸기
        model = DualInputWrapper(original_model)
        model.eval()
        print(f"✅ Model wrapped for dual 3-channel input.")
        
        print(f"🏗️  Model architecture:")
        print(f"   - Model type: {type(model)}")
        print(f"   - Number of classes: {getattr(yolo_model.model, 'nc', 'Unknown')}")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    device_obj = torch.device(device)
    if device != "cpu" and torch.cuda.is_available():
        model = model.to(device_obj)
        print(f"🔧 Using device: {device}")
    else:
        device = "cpu"
        model = model.to('cpu')
        print(f"🔧 Using device: {device}")
    
    if output_path is None:
        model_path_obj = Path(model_path)
        output_path = model_path_obj.parent / f"{model_path_obj.stem}_4d_dual_depth.onnx"
    
    # 🔴 CHANGED: 4D 입력 대신 두 개의 3채널 입력으로 변경
    print(f"\n📤 Starting DUAL 3-CHANNEL INPUT + DEPTH ONNX export...")
    print(f"⚙️  Export settings:")
    print(f"   - Input size: {imgsz}")
    print(f"   - Input format: Dual 3-Channel Input [B, 3, H, W] x 2")
    print(f"   - Output format: Detection + Depth [B, nc+6, anchors]")
    print(f"   - Half precision: {half}")
    print(f"   - Dynamic shapes: {dynamic}")
    print(f"   - ONNX opset: {opset}")
    print(f"   - Output path: {output_path}")
    
    try:
        # 🔴 CHANGED: 두 개의 3채널 더미 입력 생성
        if half and device != "cpu":
            model = model.half()
            dummy_input_wide = torch.randn(1, 3, imgsz, imgsz, dtype=torch.float16).to(device_obj)
            dummy_input_narrow = torch.randn(1, 3, imgsz, imgsz, dtype=torch.float16).to(device_obj)
        else:
            dummy_input_wide = torch.randn(1, 3, imgsz, imgsz, dtype=torch.float32).to(device_obj)
            dummy_input_narrow = torch.randn(1, 3, imgsz, imgsz, dtype=torch.float32).to(device_obj)
        
        dummy_input = (dummy_input_wide, dummy_input_narrow)
        print(f"🔍 Test forward pass with dual 3-channel input: {dummy_input_wide.shape}, {dummy_input_narrow.shape}")
        
        outputs, main_output = analyze_model_output_structure(model, dummy_input)
        
        if main_output is None:
            print("❌ Failed to get model output!")
            return None
        
        # 🔴 CHANGED: Dynamic axes를 두 개의 입력에 맞게 수정
        dynamic_axes = None
        if dynamic:
            dynamic_axes = {
                'images_wide': {0: 'batch_size', 2: 'height', 3: 'width'},
                'images_narrow': {0: 'batch_size', 2: 'height', 3: 'width'},
                'output0': {0: 'batch_size', 2: 'anchors'}
            }
            if isinstance(outputs, (tuple, list)) and len(outputs) > 1:
                for i in range(1, len(outputs)):
                    dynamic_axes[f'output{i}'] = {0: 'batch_size', 2: 'anchors'}
        
        # 🔴 CHANGED: 입력 이름을 두 개로 설정
        input_names = ['images_wide', 'images_narrow']
        output_names = [f'output{i}' for i in range(len(outputs))] if isinstance(outputs, (tuple, list)) else ['output0']
        
        print(f"🔗 Input names: {input_names}")
        print(f"🔗 Output names: {output_names}")
        
        print("🔄 Exporting with torch.onnx.export...")
        
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        
        print(f"✅ ONNX export successful!")
        print(f"📁 ONNX model saved to: {output_path}")
        
        # Verify the exported model
        try:
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("✅ ONNX model validation passed")
            
            print(f"\n📋 ONNX Model Info:")
            for i, input_tensor in enumerate(onnx_model.graph.input):
                shape_str = str([dim.dim_param if dim.dim_param else dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim])
                print(f"   - Input {i}: {input_tensor.name} {shape_str}")
                
            for i, output_tensor in enumerate(onnx_model.graph.output):
                shape_str = str([dim.dim_param if dim.dim_param else dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim])
                print(f"   - Output {i}: {output_tensor.name} {shape_str}")
                
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"   - File size: {file_size:.1f} MB")
            
            depth_info_path = Path(output_path).with_suffix('.json')
            with open(depth_info_path, 'w') as f:
                json.dump(depth_info, f, indent=2)
            print(f"   - Depth info saved: {depth_info_path}")
            
            # Test with ONNX Runtime
            print(f"\n🧪 Testing with ONNX Runtime...")
            import onnxruntime as ort
            session = ort.InferenceSession(str(output_path), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] if device != "cpu" else ['CPUExecutionProvider'])
            test_input_np_wide = dummy_input[0].cpu().numpy()
            test_input_np_narrow = dummy_input[1].cpu().numpy()
            ort_outputs = session.run(None, {'images_wide': test_input_np_wide, 'images_narrow': test_input_np_narrow})
            print(f"✅ ONNX Runtime test successful!")
            print(f"   - Input shapes: {test_input_np_wide.shape}, {test_input_np_narrow.shape}")
            for i, ort_output in enumerate(ort_outputs):
                print(f"   - Output {i} shape: {ort_output.shape}")

        except Exception as e:
            print(f"⚠️  ONNX model verification or test failed: {e}")
        
        return output_path
        
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_onnx_inference_example(onnx_path, depth_info_path, imgsz=640):
    """ONNX 모델 사용 예제 생성"""
    example_code = f'''#!/usr/-bin/env python3
"""
ONNX 4D Concatenated Dual-Stream YOLO with Depth Inference Example
Generated automatically for model: {os.path.basename(str(onnx_path))}
"""

import numpy as np
import onnxruntime as ort
import json
import cv2

def load_depth_info(path="{os.path.basename(str(depth_info_path))}"):
    """Load depth normalization info"""
    with open(path, 'r') as f:
        return json.load(f)

def denormalize_depth(normalized_depth, depth_info):
    """Convert normalized depth back to original scale"""
    min_depth = depth_info['min_depth']
    max_depth = depth_info['max_depth']
    return normalized_depth * (max_depth - min_depth) + min_depth

def preprocess_dual_images(wide_img_path, narrow_img_path, target_size={imgsz}):
    """Preprocess dual stream images into a 4D concatenated input"""
    # Load images
    wide_img = cv2.imread(wide_img_path)
    narrow_img = cv2.imread(narrow_img_path)
    
    # Convert BGR to RGB
    wide_img = cv2.cvtColor(wide_img, cv2.COLOR_BGR2RGB)
    narrow_img = cv2.cvtColor(narrow_img, cv2.COLOR_BGR2RGB)
    
    # Resize with letterbox
    def letterbox_resize(img, new_shape=({imgsz}, {imgsz}), color=(114, 114, 114)):
        shape = img.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2; dh /= 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img

    wide_processed = letterbox_resize(wide_img, (target_size, target_size))
    narrow_processed = letterbox_resize(narrow_img, (target_size, target_size))
    
    # Normalize to [0, 1] and transpose to CHW
    wide_chw = (wide_processed.astype(np.float32) / 255.0).transpose(2, 0, 1)
    narrow_chw = (narrow_processed.astype(np.float32) / 255.0).transpose(2, 0, 1)

    # 🔴 CHANGED: Concatenate along channel axis (axis=0) to create a [6, H, W] tensor
    concat_input = np.concatenate((wide_chw, narrow_chw), axis=0)
    
    # Add batch dimension -> [1, 6, H, W]
    final_input = np.expand_dims(concat_input, axis=0)
    
    return final_input

def postprocess_detections(outputs, conf_threshold=0.5, iou_threshold=0.45, depth_info=None):
    """Post-process ONNX outputs to get detections with depth"""
    # This is a simplified NMS. For production, consider a more robust implementation.
    output = outputs[0][0].T  # Transpose to [num_anchors, channels]
    
    # Filter by confidence
    mask = output[:, 4] > conf_threshold
    filtered_output = output[mask]
    
    boxes = []
    for row in filtered_output:
        x, y, w, h = row[0:4]
        conf = row[4]
        depth_norm = row[5]
        class_scores = row[6:]
        class_id = np.argmax(class_scores)
        
        # Convert xywh to xyxy
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        
        boxes.append([x1, y1, x2, y2, conf, depth_norm, class_id])
    
    boxes = np.array(boxes)
    if len(boxes) == 0:
        return []

    # Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes[:, :4].tolist(), boxes[:, 4].tolist(), conf_threshold, iou_threshold)
    
    detections = []
    for i in indices:
        x1, y1, x2, y2, conf, depth_norm, class_id = boxes[i]
        
        depth_original = denormalize_depth(depth_norm, depth_info) if depth_info else depth_norm
        
        detections.append({{
            'bbox': [float(x1), float(y1), float(x2), float(y2)],
            'confidence': float(conf),
            'class_id': int(class_id),
            'depth': float(depth_original)
        }})
        
    return detections

def main():
    # Load ONNX model
    session = ort.InferenceSession("{os.path.basename(str(onnx_path))}")
    
    # Load depth normalization info
    depth_info = load_depth_info()
    
    # Example usage (replace with your image paths)
    wide_img_path = "path/to/wide_image.jpg"
    narrow_img_path = "path/to/narrow_image.jpg"
    
    # Preprocess images
    final_input = preprocess_dual_images(wide_img_path, narrow_img_path)
    
    # Run inference
    outputs = session.run(None, {{'images': final_input}})
    
    # Post-process results
    detections = postprocess_detections(outputs, depth_info=depth_info)
    
    # Print results
    print(f"Found {{len(detections)}} detections:")
    for i, det in enumerate(detections):
        print(f"  {{i+1}}. Class: {{det['class_id']}}, Conf: {{det['confidence']:.2f}}, "
              f"Depth: {{det['depth']:.2f}}m, BBox: {{[int(c) for c in det['bbox']]}}")

if __name__ == "__main__":
    main()
'''
    
    return example_code

def main():
    parser = argparse.ArgumentParser(description='Export YOLO 4D Concatenated Dual-Stream model with Depth to ONNX')
    parser.add_argument('model', type=str, help='Path to trained .pt model file')
    parser.add_argument('--output', type=str, help='Output ONNX file path')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size')
    parser.add_argument('--half', action='store_true', help='Export in FP16 precision')
    parser.add_argument('--dynamic', action='store_true', help='Enable dynamic input shapes')
    parser.add_argument('--opset', type=int, default=11, help='ONNX opset version')
    parser.add_argument('--device', type=str, default='cpu', help='Export device')
    parser.add_argument('--create-example', action='store_true', help='Create inference example script')
    
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        print(f"❌ Model file not found: {args.model}")
        return
    
    print("🎯 YOLO 4D Concatenated Dual-Stream with Depth ONNX Export")
    print("=" * 60)
    
    result = export_dual_depth_model_to_onnx(
        model_path=args.model,
        output_path=args.output,
        imgsz=args.imgsz,
        half=args.half,
        dynamic=args.dynamic,
        opset=args.opset,
        device=args.device
    )
    
    if result:
        print(f"\n🎉 Export completed successfully!")
        print(f"📁 ONNX model: {result}")
        
        if args.create_example:
            depth_info_path = Path(result).with_suffix('.json')
            example_code = create_onnx_inference_example(result, depth_info_path, args.imgsz)
            
            example_path = Path(result).with_suffix('.py')
            with open(example_path, 'w') as f:
                f.write(example_code)
            print(f"📝 Inference example: {example_path}")
        
    else:
        print(f"\n💥 Export failed!")

if __name__ == "__main__":
    main()