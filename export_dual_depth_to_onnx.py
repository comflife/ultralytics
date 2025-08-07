#!/usr/bin/env python3
"""
YOLO Dual-Stream Model with Depth ONNX Export Script
Exports yolov8-dual.yaml trained model with depth estimation to ONNX format
"""

"""
# 기본 변환
python export_dual_depth_to_onnx.py /home/byounggun/ultralytics/runs/train/exp149/weights/best.pt

# 추론 예제도 함께 생성
python export_dual_depth_to_onnx.py /path/to/your/best.pt --create-example

# GPU 사용 및 dynamic shape 지원
python export_dual_depth_to_onnx.py /path/to/your/best.pt --device cuda:0 --dynamic

# FP16 precision으로 변환
python export_dual_depth_to_onnx.py /path/to/your/best.pt --half --device cuda:0
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
    except ImportError as e:
        print(f"❌ Failed to import MultiStreamConv: {e}")
        return False
    
    try:
        from ultralytics.nn.modules.block import MultiStreamC3
        print("✅ MultiStreamC3 imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import MultiStreamC3: {e}")
        return False
    
    try:
        from ultralytics.nn.modules.head import Detect
        print("✅ Detect head imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Detect head: {e}")
        return False
    
    # 테스트 인스턴스 생성 - DUAL STREAM 입력으로 테스트
    try:
        conv_test = MultiStreamConv(3, 64)
        spatial_conv_test = SpatialAlignedMultiStreamConv(3, 64)
        c3_test = MultiStreamC3(64, 64)
        
        # 듀얼 스트림 입력 테스트
        test_input = torch.randn(1, 2, 3, 32, 32)  # [B, 2, C, H, W]
        
        # MultiStreamConv 테스트
        conv_output = conv_test(test_input)
        print(f"✅ MultiStreamConv test: {test_input.shape} -> {conv_output.shape}")
        
        # 🚀 SpatialAlignedMultiStreamConv 테스트 (ONNX export를 위해 eval 모드로)
        spatial_conv_test.eval()  # inference 모드로 설정
        spatial_output = spatial_conv_test(test_input)
        print(f"✅ SpatialAlignedMultiStreamConv test: {test_input.shape} -> {spatial_output.shape}")
        
        # C3 블록 테스트
        c3_input = torch.randn(1, 2, 64, 16, 16)
        c3_output = c3_test(c3_input)
        print(f"✅ MultiStreamC3 test: {c3_input.shape} -> {c3_output.shape}")
        
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
    """모델 출력 구조 분석"""
    print("🔍 Analyzing model output structure...")
    
    with torch.no_grad():
        outputs = model(dummy_input)
    
    if isinstance(outputs, (tuple, list)):
        print(f"📊 Model outputs {len(outputs)} tensors:")
        for i, output in enumerate(outputs):
            if isinstance(output, torch.Tensor):
                print(f"   Output {i}: {output.shape}")
                # Detect head 출력 분석
                if len(output.shape) == 3:  # [batch, channels, anchors]
                    batch_size, channels, anchors = output.shape
                    print(f"     -> Detection format: {channels} channels")
                    print(f"     -> Expected: 4(bbox) + 1(conf) + nc(classes) + 1(depth)")
        main_output = outputs[0] if outputs else None
    else:
        print(f"📊 Model output: {outputs.shape}")
        main_output = outputs
    
    return outputs, main_output

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
    Depth estimation이 포함된 듀얼 스트림 YOLO 모델을 ONNX로 export
    
    Args:
        model_path (str): Path to trained .pt model file
        output_path (str): Output ONNX file path (optional)
        imgsz (int): Input image size
        half (bool): Export in FP16 precision
        dynamic (bool): Enable dynamic input shapes
        opset (int): ONNX opset version
        device (str): Device for export ('cpu', 'cuda:0', etc.)
    """
    
    print(f"\n🚀 Loading dual-stream YOLO model with depth from: {model_path}")
    
    # 환경 및 커스텀 모듈 확인
    if not verify_custom_modules_with_depth():
        print("❌ Custom modules verification failed!")
        return None
    
    # Depth 정규화 정보 로드
    depth_info = load_depth_normalization_info()
    if depth_info is None:
        print("❌ Failed to load depth normalization info!")
        return None
    
    # Load the trained model
    try:
        print("📂 Loading model...")
        yolo_model = YOLO(model_path)
        model = yolo_model.model  # PyTorch 모델 직접 접근
        model.eval()
        print(f"✅ Model loaded successfully")
        
        print(f"🏗️  Model architecture:")
        print(f"   - Model type: {type(model)}")
        print(f"   - Number of classes: {getattr(model, 'nc', 'Unknown')}")
        
        # Depth 지원 확인
        if hasattr(model, 'model') and hasattr(model.model[-1], 'with_depth'):
            print(f"   - Depth estimation: {model.model[-1].with_depth}")
        else:
            print(f"   - Depth estimation: Checking output structure...")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Set device
    device_obj = torch.device(device)
    if device != "cpu" and torch.cuda.is_available():
        model = model.to(device_obj)
        print(f"🔧 Using device: {device}")
    else:
        device = "cpu"
        model = model.to('cpu')
        print(f"🔧 Using device: {device}")
    
    # Generate output path if not provided
    if output_path is None:
        model_path_obj = Path(model_path)
        output_path = model_path_obj.parent / f"{model_path_obj.stem}_dual_depth.onnx"
    
    print(f"\n📤 Starting DUAL STREAM + DEPTH ONNX export...")
    print(f"⚙️  Export settings:")
    print(f"   - Input size: {imgsz}")
    print(f"   - Input format: DUAL STREAM [B, 2, 3, H, W]")
    print(f"   - Output format: Detection + Depth [B, nc+6, anchors]")
    print(f"   - Half precision: {half}")
    print(f"   - Dynamic shapes: {dynamic}")
    print(f"   - ONNX opset: {opset}")
    print(f"   - Output path: {output_path}")
    
    try:
        # 🎯 핵심: 듀얼 스트림 입력 생성
        if half and device != "cpu":
            model = model.half()
            dummy_input = torch.randn(1, 2, 3, imgsz, imgsz, dtype=torch.float16).to(device_obj)
        else:
            dummy_input = torch.randn(1, 2, 3, imgsz, imgsz, dtype=torch.float32).to(device_obj)
        
        print(f"🔍 Test forward pass with dual stream input: {dummy_input.shape}")
        
        # 테스트 forward pass 및 출력 구조 분석
        outputs, main_output = analyze_model_output_structure(model, dummy_input)
        
        if main_output is None:
            print("❌ Failed to get model output!")
            return None
        
        # Dynamic axes 설정
        dynamic_axes = None
        if dynamic:
            dynamic_axes = {
                'images': {0: 'batch_size', 3: 'height', 4: 'width'},  # [B, 2, 3, H, W]
                'output0': {0: 'batch_size', 2: 'anchors'}  # [B, channels, anchors]
            }
            
            # 다중 출력인 경우
            if isinstance(outputs, (tuple, list)) and len(outputs) > 1:
                for i in range(1, len(outputs)):
                    dynamic_axes[f'output{i}'] = {0: 'batch_size', 2: 'anchors'}
        
        input_names = ['images']  # dual stream input
        output_names = ['output0']
        
        # 다중 출력인 경우 output names 추가
        if isinstance(outputs, (tuple, list)) and len(outputs) > 1:
            output_names.extend([f'output{i}' for i in range(1, len(outputs))])
        
        print(f"🔗 Input names: {input_names}")
        print(f"🔗 Output names: {output_names}")
        
        print("🔄 Exporting with torch.onnx.export...")
        
        # 직접 torch.onnx.export 사용
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
            
            # Print model info
            print(f"\n📋 ONNX Model Info:")
            print(f"   - IR version: {onnx_model.ir_version}")
            print(f"   - Producer: {onnx_model.producer_name} {onnx_model.producer_version}")
            print(f"   - Graph inputs: {len(onnx_model.graph.input)}")
            print(f"   - Graph outputs: {len(onnx_model.graph.output)}")
            
            for i, input_tensor in enumerate(onnx_model.graph.input):
                shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
                print(f"   - Input {i}: {input_tensor.name} {shape}")
                
            for i, output_tensor in enumerate(onnx_model.graph.output):
                shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
                print(f"   - Output {i}: {output_tensor.name} {shape}")
                
            # 파일 크기 정보
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"   - File size: {file_size:.1f} MB")
            
            # Depth 정규화 정보를 ONNX 파일 옆에 저장
            depth_info_path = Path(output_path).with_suffix('.json')
            with open(depth_info_path, 'w') as f:
                json.dump(depth_info, f, indent=2)
            print(f"   - Depth info saved: {depth_info_path}")
            
            # 🧪 ONNX Runtime으로 테스트
            print(f"\n🧪 Testing with ONNX Runtime...")
            try:
                import onnxruntime as ort
                
                # ONNX Runtime 세션 생성
                providers = ['CPUExecutionProvider']
                if device != "cpu":
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                
                session = ort.InferenceSession(str(output_path), providers=providers)
                
                # 테스트 입력 생성 (듀얼 스트림)
                test_input = dummy_input.cpu().numpy()
                
                # 추론 실행
                ort_outputs = session.run(None, {'images': test_input})
                print(f"✅ ONNX Runtime test successful!")
                print(f"   - Input shape: {test_input.shape}")
                
                for i, ort_output in enumerate(ort_outputs):
                    print(f"   - Output {i} shape: {ort_output.shape}")
                    
                    # Detection head 출력 분석
                    if len(ort_output.shape) == 3:  # [batch, channels, anchors]
                        batch_size, channels, anchors = ort_output.shape
                        print(f"     -> Detection format: {channels} channels, {anchors} anchors")
                        if channels >= 6:  # bbox(4) + conf(1) + depth(1) + classes
                            print(f"     -> Includes depth estimation channel")
                
                # PyTorch vs ONNX 결과 비교
                if isinstance(outputs, (tuple, list)):
                    pytorch_output = outputs[0].cpu().numpy()
                else:
                    pytorch_output = outputs.cpu().numpy()
                onnx_output = ort_outputs[0]
                
                # 차이 계산
                diff = abs(pytorch_output - onnx_output).max()
                print(f"   - Max difference: {diff:.6f}")
                if diff < 1e-3:
                    print(f"   ✅ Results match well!")
                else:
                    print(f"   ⚠️  Large difference detected")
                
            except ImportError:
                print("⚠️  onnxruntime not found. Install with: pip install onnxruntime")
            except Exception as e:
                print(f"⚠️  ONNX Runtime test failed: {e}")
                
        except ImportError:
            print("⚠️  onnx package not found. Install with: pip install onnx")
        except Exception as e:
            print(f"⚠️  ONNX model validation failed: {e}")
        
        return output_path
        
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        print(f"🔍 Error details:")
        import traceback
        traceback.print_exc()
        return None

def create_onnx_inference_example(onnx_path, depth_info_path, imgsz=640):
    """ONNX 모델 사용 예제 생성"""
    example_code = f'''#!/usr/bin/env python3
"""
ONNX Dual-Stream YOLO with Depth Inference Example
Generated automatically for model: {onnx_path}
"""

import numpy as np
import onnxruntime as ort
import json
import cv2

def load_depth_info(path="{depth_info_path}"):
    """Load depth normalization info"""
    with open(path, 'r') as f:
        return json.load(f)

def denormalize_depth(normalized_depth, depth_info):
    """Convert normalized depth back to original scale"""
    min_depth = depth_info['min_depth']
    max_depth = depth_info['max_depth']
    return normalized_depth * (max_depth - min_depth) + min_depth

def preprocess_dual_images(wide_img_path, narrow_img_path, target_size={imgsz}):
    """Preprocess dual stream images"""
    # Load images
    wide_img = cv2.imread(wide_img_path)
    narrow_img = cv2.imread(narrow_img_path)
    
    # Convert BGR to RGB
    wide_img = cv2.cvtColor(wide_img, cv2.COLOR_BGR2RGB)
    narrow_img = cv2.cvtColor(narrow_img, cv2.COLOR_BGR2RGB)
    
    # Resize with letterbox
    def letterbox_resize(img, target_size):
        h, w = img.shape[:2]
        scale = min(target_size / w, target_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize
        resized = cv2.resize(img, (new_w, new_h))
        
        # Pad
        delta_w = target_size - new_w
        delta_h = target_size - new_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, 
                                  cv2.BORDER_CONSTANT, value=[114, 114, 114])
        return padded
    
    wide_processed = letterbox_resize(wide_img, target_size)
    narrow_processed = letterbox_resize(narrow_img, target_size)
    
    # Normalize to [0, 1]
    wide_processed = wide_processed.astype(np.float32) / 255.0
    narrow_processed = narrow_processed.astype(np.float32) / 255.0
    
    # Create dual stream input [1, 2, 3, H, W]
    dual_input = np.stack([
        wide_processed.transpose(2, 0, 1),    # CHW
        narrow_processed.transpose(2, 0, 1)   # CHW
    ], axis=0)[None, ...]  # Add batch dimension
    
    return dual_input

def postprocess_detections(outputs, conf_threshold=0.5, depth_info=None):
    """Post-process ONNX outputs to get detections with depth"""
    detections = []
    
    # Assuming output format: [batch, channels, anchors]
    # channels = [x, y, w, h, conf, depth, class_scores...]
    
    output = outputs[0]  # Main detection output
    batch_size, channels, num_anchors = output.shape
    
    for b in range(batch_size):
        for a in range(num_anchors):
            # Extract detection data
            x, y, w, h = output[b, 0:4, a]
            conf = output[b, 4, a]
            depth_norm = output[b, 5, a]  # Normalized depth
            class_scores = output[b, 6:, a]  # Class scores
            
            if conf > conf_threshold:
                # Find best class
                class_id = np.argmax(class_scores)
                class_conf = class_scores[class_id]
                
                # Denormalize depth
                if depth_info:
                    depth_original = denormalize_depth(depth_norm, depth_info)
                else:
                    depth_original = depth_norm
                
                detection = {{
                    'bbox': [x, y, w, h],
                    'confidence': float(conf),
                    'class_id': int(class_id),
                    'class_confidence': float(class_conf),
                    'depth': float(depth_original),
                    'depth_normalized': float(depth_norm)
                }}
                detections.append(detection)
    
    return detections

def main():
    # Load ONNX model
    session = ort.InferenceSession("{onnx_path}")
    
    # Load depth normalization info
    depth_info = load_depth_info()
    
    # Example usage
    wide_img_path = "path/to/wide_image.jpg"
    narrow_img_path = "path/to/narrow_image.jpg"
    
    # Preprocess images
    dual_input = preprocess_dual_images(wide_img_path, narrow_img_path)
    
    # Run inference
    outputs = session.run(None, {{'images': dual_input}})
    
    # Post-process results
    detections = postprocess_detections(outputs, conf_threshold=0.5, depth_info=depth_info)
    
    # Print results
    print(f"Found {{len(detections)}} detections:")
    for i, det in enumerate(detections):
        print(f"  {{i+1}}. Class: {{det['class_id']}}, Conf: {{det['confidence']:.3f}}, "
              f"Depth: {{det['depth']:.2f}}m, BBox: {{det['bbox']}}")

if __name__ == "__main__":
    main()
'''
    
    return example_code

def main():
    parser = argparse.ArgumentParser(description='Export YOLO Dual-Stream model with Depth to ONNX')
    parser.add_argument('model', type=str, help='Path to trained .pt model file')
    parser.add_argument('--output', type=str, help='Output ONNX file path')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size')
    parser.add_argument('--half', action='store_true', help='Export in FP16 precision')
    parser.add_argument('--dynamic', action='store_true', help='Enable dynamic input shapes')
    parser.add_argument('--opset', type=int, default=11, help='ONNX opset version')
    parser.add_argument('--device', type=str, default='cpu', help='Export device')
    parser.add_argument('--create-example', action='store_true', help='Create inference example script')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not Path(args.model).exists():
        print(f"❌ Model file not found: {args.model}")
        return
    
    print("🎯 YOLO Dual-Stream with Depth ONNX Export")
    print("=" * 60)
    
    # Export the model
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
        
        # Create inference example
        if args.create_example:
            depth_info_path = Path(result).with_suffix('.json')
            example_code = create_onnx_inference_example(result, depth_info_path, args.imgsz)
            
            example_path = Path(result).with_suffix('.py')
            with open(example_path, 'w') as f:
                f.write(example_code)
            print(f"📝 Inference example: {example_path}")
        
        print(f"\n💡 Key Features:")
        print(f"   ✅ Dual-stream input support")
        print(f"   ✅ Depth estimation output")
        print(f"   ✅ Automatic depth denormalization")
        print(f"   ✅ ONNX Runtime compatible")
        
    else:
        print(f"\n💥 Export failed!")

if __name__ == "__main__":
    main()
