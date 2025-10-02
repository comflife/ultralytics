#!/usr/bin/env python3
"""
YOLO Dual-Stream (Dual 3-Channel Input) Model with Depth ONNX Export Script (Final Version)
Exports a model to accept two separate 3-channel inputs and merges them internally.
This version EXCLUDES the final box decoding and confidence score layers, outputting raw feature maps.
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
import json
import argparse
import traceback

# 로컬 ultralytics 모듈 경로 설정
# 이 스크립트가 있는 폴더에 ultralytics 소스 코드가 있다고 가정합니다.
# sys.path.insert(0, str(Path(__file__).parent.absolute()))

try:
    from ultralytics import YOLO
    import ultralytics
    print(f"✅ Using ultralytics from: {ultralytics.__file__}")
except ImportError as e:
    print(f"❌ Failed to import ultralytics: {e}")
    sys.exit(1)

def load_depth_normalization_info(norm_info_path="depth_normalization_info.json"):
    """Loads depth normalization info from a JSON file."""
    try:
        with open(norm_info_path, 'r') as f:
            return json.load(f)
    except Exception:
        print(f"⚠️ Depth normalization info not found. Using defaults.")
        return {"min_depth": 0.1, "max_depth": 419.1, "mean_depth": 42.96}

def export_raw_output_model_to_onnx(
    model_path: str, output_path: str = None, imgsz: int = 640,
    half: bool = False, dynamic: bool = False, opset: int = 11, device: str = "cpu"
):
    """
    Exports a YOLO model to ONNX, modifying it to return raw prediction outputs
    before the final decoding and post-processing steps.
    """
    print(f"\n🚀 Loading model from: {model_path}")
    depth_info = load_depth_normalization_info()

    try:
        yolo_model = YOLO(model_path)
        original_model = yolo_model.model

        # 🔴 [수정] 원시 예측 값(raw logits)만 출력하는 새로운 래퍼 클래스
        class RawOutputWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, images_wide, images_narrow):
                # 1. 두 개의 입력을 하나로 합칩니다.
                x = torch.cat((images_wide, images_narrow), dim=1)
                # 2. 모델의 forward를 호출하면 (최종 결과, 원시 결과) 튜플이 반환됩니다.
                _processed_output, raw_outputs = self.model(x)
                # 3. 우리는 원시 결과(raw_outputs)만 필요합니다.
                # raw_outputs는 3개의 텐서로 구성된 리스트이므로, 각각을 ONNX의 개별 출력으로 반환합니다.
                return raw_outputs[0], raw_outputs[1], raw_outputs[2]

        # 원본 모델을 eval 모드로 설정
        original_model.eval()
        for m in original_model.modules():
            if hasattr(m, 'export'):
                # 🔴 [수정] export 모드를 False로 유지하여 (최종결과, 원시결과) 튜플을 반환하게 함
                m.export = False
            if isinstance(m, ultralytics.nn.modules.head.Detect):
                m.dynamic = dynamic
        print("✅ Original model set to eval mode for raw output.")

        # 🔴 [수정] 새로운 RawOutputWrapper로 모델을 감쌈
        model = RawOutputWrapper(original_model)
        model.eval()
        print(f"✅ Model wrapped for DUAL INPUT and RAW OUTPUT.")
        
    except Exception as e:
        print(f"❌ Failed to load or wrap model: {e}")
        traceback.print_exc()
        return None

    device_obj = torch.device(device)
    model.to(device_obj)
    print(f"🔧 Using device: {device}")

    if output_path is None:
        p = Path(model_path)
        # 🔴 [수정] 출력 파일 이름 변경
        output_path = p.parent / f"{p.stem}_raw_output.onnx"

    print(f"\n📤 Starting RAW OUTPUT ONNX export...")
    print(f"   - Input size: {imgsz}x{imgsz}, Precision: {'FP16' if half else 'FP32'}")
    print(f"   - Dynamic shapes: {dynamic}, ONNX opset: {opset}")
    print(f"   - Output path: {output_path}")

    try:
        dtype = torch.float16 if half and device != 'cpu' else torch.float32
        dummy_input_wide = torch.randn(1, 3, imgsz, imgsz, dtype=dtype).to(device_obj)
        dummy_input_narrow = torch.randn(1, 3, imgsz, imgsz, dtype=dtype).to(device_obj)
        dummy_input = (dummy_input_wide, dummy_input_narrow)

        input_names = ['images_wide', 'images_narrow']
        
        # 🔴 [수정] 출력은 3개의 텐서이므로 이름을 3개 지정 (P3, P4, P5에 해당)
        output_names = ['output_small', 'output_medium', 'output_large']
        
        dynamic_axes = None
        if dynamic:
            dynamic_axes = {
                'images_wide': {0: 'batch', 2: 'height', 3: 'width'},
                'images_narrow': {0: 'batch', 2: 'height', 3: 'width'},
            }
            # 🔴 [수정] 각 출력은 [batch, channels, height, width] 4개 차원을 가지므로, 그에 맞게 dynamic_axes 설정
            for name in output_names:
                 dynamic_axes[name] = {0: 'batch', 2: 'height', 3: 'width'}
        
        print(f"🔗 Input names: {input_names}, Output names: {output_names}")

        torch.onnx.export(
            model, dummy_input, str(output_path),
            export_params=True, opset_version=opset, do_constant_folding=True,
            input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes,
            verbose=False
        )
        print(f"✅ ONNX export successful!")

        # 검증 및 테스트
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model validation passed.")
        
        import onnxruntime as ort
        print(f"🧪 Testing with ONNX Runtime...")
        session = ort.InferenceSession(str(output_path), providers=['CUDAExecutionProvider' if device != 'cpu' and ort.get_device() == 'GPU' else 'CPUExecutionProvider'])
        ort_inputs = {
            'images_wide': dummy_input_wide.cpu().numpy(),
            'images_narrow': dummy_input_narrow.cpu().numpy()
        }
        ort_outputs = session.run(None, ort_inputs)
        print("✅ ONNX Runtime test successful!")
        print(f"   - Number of outputs: {len(ort_outputs)}")
        for i, out in enumerate(ort_outputs):
            print(f"   - Output '{output_names[i]}' shape: {out.shape}")

        depth_info_path = Path(output_path).with_suffix('.json')
        with open(depth_info_path, 'w') as f:
            json.dump(depth_info, f, indent=2)
        print(f"   - Depth info saved: {depth_info_path}")

        return output_path

    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        traceback.print_exc()
        return None

def main():
    """Parses command-line arguments and runs the export function."""
    parser = argparse.ArgumentParser(description='Export YOLO Dual 3-Channel Input model (RAW OUTPUT) to ONNX')
    parser.add_argument('model', type=str, help='Path to trained .pt model file')
    parser.add_argument('--output', type=str, help='Output ONNX file path')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size')
    parser.add_argument('--half', action='store_true', help='Export in FP16 precision')
    parser.add_argument('--dynamic', action='store_true', help='Enable dynamic input shapes')
    parser.add_argument('--opset', type=int, default=11, help='ONNX opset version')
    parser.add_argument('--device', type=str, default='cpu', help='Export device')
    
    args = parser.parse_args()
    
    export_raw_output_model_to_onnx(
        model_path=args.model, output_path=args.output, imgsz=args.imgsz,
        half=args.half, dynamic=args.dynamic, opset=args.opset, device=args.device
    )

if __name__ == "__main__":
    main()