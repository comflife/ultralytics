#!/usr/bin/env python3
"""
YOLO Dual-Stream Model ONNX Export Script
Exports yolov8-dual.yaml trained model to ONNX format with DUAL STREAM INPUT
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
import argparse

def verify_custom_modules():
    """커스텀 모듈들이 제대로 로드되는지 확인 - SpatialAlignedMultiStreamConv 포함"""
    print("🔍 Verifying custom modules...")
    
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
        
        # 🔧 ONNX 호환성 확인: SpatialAlignedMultiStreamConv의 torch.randn 사용 체크
        print("🔍 Checking ONNX compatibility for SpatialAlignedMultiStreamConv...")
        print("⚠️  Note: SpatialAlignedMultiStreamConv uses torch.randn() which may need attention in ONNX")
        print("   -> Ensuring model is in eval() mode for consistent behavior")
        
        return True
    except Exception as e:
        print(f"❌ Failed to test custom modules: {e}")
        import traceback
        traceback.print_exc()
        return False

def format_output_info(output):
    """출력 정보를 포맷팅하는 헬퍼 함수"""
    if isinstance(output, torch.Tensor):
        return f"Tensor{tuple(output.shape)}"
    elif isinstance(output, (tuple, list)):
        return f"({', '.join([format_output_info(item) for item in output])})"
    else:
        return f"{type(output).__name__}"

def export_dual_model_to_onnx_custom(
    model_path: str,
    output_path: str = None,
    imgsz: int = 640,
    half: bool = False,
    dynamic: bool = False,
    opset: int = 11,
    device: str = "cpu"
):
    """
    커스텀 듀얼 스트림 ONNX export - 올바른 입력 형태로
    
    Args:
        model_path (str): Path to trained .pt model file
        output_path (str): Output ONNX file path (optional)
        imgsz (int): Input image size
        half (bool): Export in FP16 precision
        dynamic (bool): Enable dynamic input shapes
        opset (int): ONNX opset version
        device (str): Device for export ('cpu', 'cuda:0', etc.)
    """
    
    print(f"\n🚀 Loading dual-stream YOLO model from: {model_path}")
    
    # 환경 및 커스텀 모듈 확인
    if not verify_custom_modules():
        print("❌ Custom modules verification failed!")
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
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
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
        output_path = model_path_obj.parent / f"{model_path_obj.stem}_dual_stream.onnx"
    
    print(f"\n📤 Starting DUAL STREAM ONNX export...")
    print(f"⚙️  Export settings:")
    print(f"   - Input size: {imgsz}")
    print(f"   - Input format: DUAL STREAM [B, 2, 3, H, W]")
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
        
        # 테스트 forward pass
        with torch.no_grad():
            test_output = model(dummy_input)
            output_info = format_output_info(test_output)
            print(f"✅ Forward pass successful: {dummy_input.shape} -> {output_info}")
            
            # 출력이 tuple인 경우 첫 번째 요소 사용 (detection output)
            if isinstance(test_output, (tuple, list)):
                main_output = test_output[0]
                print(f"🔍 Using main output: {main_output.shape}")
            else:
                main_output = test_output
        
        # Dynamic axes 설정
        dynamic_axes = None
        if dynamic:
            dynamic_axes = {
                'images': {0: 'batch_size', 3: 'height', 4: 'width'},  # [B, 2, 3, H, W]
                'output0': {0: 'batch_size', 2: 'anchors'}  # [B, classes+5, anchors]
            }
        
        input_names = ['images']  # dual stream input
        output_names = ['output0']
        
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
            verbose=False  # verbose 줄여서 깔끔하게
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
                print(f"   - Output shape: {ort_outputs[0].shape}")
                
                # PyTorch vs ONNX 결과 비교
                pytorch_output = main_output.cpu().numpy()
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

def main():
    parser = argparse.ArgumentParser(description='Export YOLO Dual-Stream model to ONNX with DUAL STREAM INPUT')
    parser.add_argument('model', type=str, help='Path to trained .pt model file', default='')
    parser.add_argument('--output', type=str, help='Output ONNX file path')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size')
    parser.add_argument('--half', action='store_true', help='Export in FP16 precision')
    parser.add_argument('--dynamic', action='store_true', help='Enable dynamic input shapes')
    parser.add_argument('--opset', type=int, default=11, help='ONNX opset version (default: 11 for stability)')
    parser.add_argument('--device', type=str, default='cpu', help='Export device')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not Path(args.model).exists():
        print(f"❌ Model file not found: {args.model}")
        return
    
    print("🎯 YOLO Dual-Stream ONNX Export (DUAL STREAM INPUT)")
    print("=" * 60)
    
    # Export the model
    result = export_dual_model_to_onnx_custom(
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
        print(f"\n💡 Usage example:")
        print(f"```python")
        print(f"import onnxruntime as ort")
        print(f"import numpy as np")
        print(f"")
        print(f"# Load ONNX model")
        print(f"session = ort.InferenceSession('{result}')")
        print(f"")
        print(f"# Prepare dual stream input [batch, 2, channels, height, width]")
        print(f"wide_stream = np.random.randn(1, 3, {args.imgsz}, {args.imgsz}).astype(np.float32)")
        print(f"narrow_stream = np.random.randn(1, 3, {args.imgsz}, {args.imgsz}).astype(np.float32)")
        print(f"dual_input = np.stack([wide_stream[0], narrow_stream[0]], axis=0)[None, ...]  # [1, 2, 3, {args.imgsz}, {args.imgsz}]")
        print(f"")
        print(f"# Run inference")
        print(f"outputs = session.run(None, {{'images': dual_input}})")
        print(f"print('Detection output shape:', outputs[0].shape)")
        print(f"```")
    else:
        print(f"\n💥 Export failed!")

if __name__ == "__main__":
    main() 