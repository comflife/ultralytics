#!/usr/bin/env python3
"""
Dual Stream 모델 구조 분석 도구
모델의 PT 파일을 로드하고 입력부터 각 레이어의 텐서 구조를 출력합니다.
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
import traceback
from typing import Dict, List, Any
import yaml

# 현재 디렉토리를 추가하여 ultralytics 모듈 로드
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
    import ultralytics
    print(f"✅ Using ultralytics from: {ultralytics.__file__}")
except ImportError as e:
    print(f"❌ Failed to import ultralytics: {e}")
    print("Please ensure you're in the ultralytics directory with custom modules")
    sys.exit(1)

# === 설정 ===
DEFAULT_MODEL_PATH = "runs/train/exp61/weights/best.pt"  # 가장 최근 dual stream 모델로 변경
DEFAULT_INPUT_SIZE = 640

class ModelStructureAnalyzer:
    """Dual Stream 모델 구조를 분석하는 클래스"""
    
    def __init__(self, model_path: str):
        """
        모델 구조 분석기 초기화
        
        Args:
            model_path (str): 분석할 .pt 모델 파일 경로
        """
        self.model_path = model_path
        self.model = None
        self.pytorch_model = None
        self.layer_outputs = {}
        self.hooks = []
        
    def load_model(self):
        """모델 로드 및 초기화"""
        print(f"🔄 Loading model from: {self.model_path}")
        
        try:
            # YOLO 모델 로드
            self.model = YOLO(self.model_path)
            
            # PyTorch 모델 추출
            if hasattr(self.model, 'model') and self.model.model is not None:
                self.pytorch_model = self.model.model
            elif hasattr(self.model, 'predictor') and hasattr(self.model.predictor, 'model'):
                self.pytorch_model = self.model.predictor.model
            else:
                raise ValueError("Could not access the PyTorch model from YOLO object")
            
            self.pytorch_model.eval()
            print(f"✅ Model loaded successfully!")
            
            # 모델 기본 정보 출력
            self._print_model_info()
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            traceback.print_exc()
            return False
            
        return True
    
    def _print_model_info(self):
        """모델 기본 정보 출력"""
        print("\n" + "="*80)
        print("📊 MODEL BASIC INFORMATION")
        print("="*80)
        
        # 파라미터 수 계산
        total_params = sum(p.numel() for p in self.pytorch_model.parameters())
        trainable_params = sum(p.numel() for p in self.pytorch_model.parameters() if p.requires_grad)
        
        print(f"📦 Model file: {self.model_path}")
        print(f"🔧 Total parameters: {total_params:,}")
        print(f"🎯 Trainable parameters: {trainable_params:,}")
        print(f"🔒 Frozen parameters: {total_params - trainable_params:,}")
        
        # 클래스 수 확인
        if hasattr(self.pytorch_model, 'nc'):
            print(f"📋 Number of classes: {self.pytorch_model.nc}")
        
        # 클래스 이름 확인
        if hasattr(self.pytorch_model, 'names') and self.pytorch_model.names:
            print(f"🏷️  Class names: {list(self.pytorch_model.names.values())}")
        
        # 스트라이드 정보
        if hasattr(self.pytorch_model, 'stride'):
            print(f"📏 Model stride: {self.pytorch_model.stride}")
    
    def _check_dual_stream_architecture(self):
        """Dual stream 아키텍처 여부 확인"""
        print("\n" + "="*80)
        print("🔍 DUAL STREAM ARCHITECTURE DETECTION")
        print("="*80)
        
        dual_stream_modules = []
        fusion_modules = []
        
        for name, module in self.pytorch_model.named_modules():
            module_type = type(module).__name__
            
            if 'MultiStream' in module_type:
                dual_stream_modules.append((name, module_type))
            elif 'Fusion' in module_type:
                fusion_modules.append((name, module_type))
        
        print(f"🔗 MultiStream modules found: {len(dual_stream_modules)}")
        for name, module_type in dual_stream_modules:
            print(f"   - {name}: {module_type}")
        
        print(f"🔀 Fusion modules found: {len(fusion_modules)}")
        for name, module_type in fusion_modules:
            print(f"   - {name}: {module_type}")
        
        is_dual_stream = len(dual_stream_modules) > 0 or len(fusion_modules) > 0
        print(f"\n{'✅' if is_dual_stream else '❌'} This is a {'dual-stream' if is_dual_stream else 'single-stream'} model")
        
        return is_dual_stream
    
    def create_hooks(self):
        """각 레이어의 출력을 캡처하기 위한 hook 생성"""
        def create_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.layer_outputs[name] = {
                        'shape': list(output.shape),
                        'dtype': str(output.dtype),
                        'device': str(output.device),
                        'requires_grad': output.requires_grad,
                        'min': float(output.min().item()),
                        'max': float(output.max().item()),
                        'mean': float(output.mean().item()),
                        'std': float(output.std().item())
                    }
                elif isinstance(output, (list, tuple)):
                    # 다중 출력의 경우
                    self.layer_outputs[name] = []
                    for i, out in enumerate(output):
                        if isinstance(out, torch.Tensor):
                            self.layer_outputs[name].append({
                                'shape': list(out.shape),
                                'dtype': str(out.dtype),
                                'device': str(out.device),
                                'requires_grad': out.requires_grad,
                                'min': float(out.min().item()),
                                'max': float(out.max().item()),
                                'mean': float(out.mean().item()),
                                'std': float(out.std().item())
                            })
            return hook
        
        # 모든 레이어에 hook 등록
        for name, module in self.pytorch_model.named_modules():
            if len(list(module.children())) == 0:  # leaf 모듈만
                hook = module.register_forward_hook(create_hook(name))
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """등록된 hook들을 제거"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def create_dummy_input(self, batch_size=1, image_size=640, is_dual_stream=True):
        """더미 입력 데이터 생성"""
        print("\n" + "="*80)
        print("🎯 CREATING DUMMY INPUT")
        print("="*80)
        
        if is_dual_stream:
            # Dual stream 입력: [B, 2, 3, H, W]
            dummy_input = torch.randn(batch_size, 2, 3, image_size, image_size)
            print(f"🔗 Created dual stream input: {dummy_input.shape}")
            print(f"   - Format: [batch_size, streams, channels, height, width]")
            print(f"   - Wide stream: {dummy_input[:, 0].shape}")
            print(f"   - Narrow stream: {dummy_input[:, 1].shape}")
        else:
            # Single stream 입력: [B, 3, H, W]
            dummy_input = torch.randn(batch_size, 3, image_size, image_size)
            print(f"📷 Created single stream input: {dummy_input.shape}")
            print(f"   - Format: [batch_size, channels, height, width]")
        
        return dummy_input
    
    def analyze_forward_pass(self, dummy_input):
        """Forward pass를 실행하고 각 레이어의 출력 분석"""
        print("\n" + "="*80)
        print("🚀 FORWARD PASS ANALYSIS")
        print("="*80)
        
        # Hook 생성
        self.create_hooks()
        
        try:
            with torch.no_grad():
                print("🔄 Running forward pass...")
                start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                
                if start_time:
                    start_time.record()
                
                output = self.pytorch_model(dummy_input)
                
                if end_time:
                    end_time.record()
                    torch.cuda.synchronize()
                    inference_time = start_time.elapsed_time(end_time)
                    print(f"⏱️  Inference time: {inference_time:.2f} ms")
                
                print("✅ Forward pass completed successfully!")
                
                # 최종 출력 정보
                self._print_output_info(output)
                
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            traceback.print_exc()
            return None
        finally:
            self.remove_hooks()
        
        return output
    
    def _print_output_info(self, output):
        """모델 최종 출력 정보 출력"""
        print(f"\n📤 FINAL MODEL OUTPUT:")
        
        if isinstance(output, torch.Tensor):
            print(f"   - Shape: {output.shape}")
            print(f"   - Type: {output.dtype}")
            print(f"   - Range: [{output.min().item():.6f}, {output.max().item():.6f}]")
        elif isinstance(output, (list, tuple)):
            print(f"   - Number of outputs: {len(output)}")
            for i, out in enumerate(output):
                if isinstance(out, torch.Tensor):
                    print(f"   - Output {i}: {out.shape} ({out.dtype})")
                    print(f"     Range: [{out.min().item():.6f}, {out.max().item():.6f}]")
    
    def print_layer_analysis(self):
        """각 레이어별 상세 분석 결과 출력"""
        print("\n" + "="*80)
        print("📋 DETAILED LAYER-BY-LAYER ANALYSIS")
        print("="*80)
        
        if not self.layer_outputs:
            print("❌ No layer outputs captured. Run analyze_forward_pass first.")
            return
        
        print(f"📊 Total layers analyzed: {len(self.layer_outputs)}")
        print("\n" + "-"*80)
        
        for i, (layer_name, output_info) in enumerate(self.layer_outputs.items()):
            print(f"\n🔍 Layer {i+1}: {layer_name}")
            print(f"   📐 Module type: {layer_name.split('.')[-1] if '.' in layer_name else 'root'}")
            
            if isinstance(output_info, dict):
                # 단일 출력
                self._print_tensor_info(output_info, indent="   ")
            elif isinstance(output_info, list):
                # 다중 출력
                for j, info in enumerate(output_info):
                    print(f"   📤 Output {j+1}:")
                    self._print_tensor_info(info, indent="      ")
            
            print("-" * 40)
    
    def _print_tensor_info(self, tensor_info: Dict, indent: str = ""):
        """텐서 정보를 포맷팅하여 출력"""
        shape = tensor_info['shape']
        print(f"{indent}📏 Shape: {shape}")
        print(f"{indent}🔧 Type: {tensor_info['dtype']}")
        print(f"{indent}💾 Device: {tensor_info['device']}")
        print(f"{indent}🎯 Requires grad: {tensor_info['requires_grad']}")
        print(f"{indent}📊 Stats: min={tensor_info['min']:.6f}, max={tensor_info['max']:.6f}")
        print(f"{indent}        mean={tensor_info['mean']:.6f}, std={tensor_info['std']:.6f}")
        
        # 형태별 해석
        if len(shape) == 4:  # [B, C, H, W]
            print(f"{indent}🖼️  Format: [batch={shape[0]}, channels={shape[1]}, height={shape[2]}, width={shape[3]}]")
        elif len(shape) == 5:  # [B, S, C, H, W] (dual stream)
            print(f"{indent}🔗 Format: [batch={shape[0]}, streams={shape[1]}, channels={shape[2]}, height={shape[3]}, width={shape[4]}]")
        elif len(shape) == 3:  # Detection output like [B, anchors, 5+classes]
            if shape[2] > 10:  # Likely detection output
                classes = shape[2] - 5
                print(f"{indent}🎯 Format: [batch={shape[0]}, anchors={shape[1]}, bbox+conf+classes={shape[2]} (classes={classes})]")
            else:
                print(f"{indent}📦 Format: [dim0={shape[0]}, dim1={shape[1]}, dim2={shape[2]}]")
        elif len(shape) == 2:  # [B, features]
            print(f"{indent}📊 Format: [batch={shape[0]}, features={shape[1]}]")
    
    def save_analysis_report(self, output_path: str = None):
        """분석 결과를 파일로 저장"""
        if output_path is None:
            model_name = Path(self.model_path).stem
            output_path = f"model_analysis_{model_name}.txt"
        
        print(f"\n💾 Saving analysis report to: {output_path}")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("DUAL STREAM MODEL STRUCTURE ANALYSIS REPORT\n")
                f.write("="*80 + "\n\n")
                f.write(f"Model: {self.model_path}\n")
                f.write(f"Generated: {torch.utils.data.get_worker_info()}\n\n")
                
                # 기본 정보
                total_params = sum(p.numel() for p in self.pytorch_model.parameters())
                trainable_params = sum(p.numel() for p in self.pytorch_model.parameters() if p.requires_grad)
                
                f.write("MODEL INFORMATION:\n")
                f.write(f"- Total parameters: {total_params:,}\n")
                f.write(f"- Trainable parameters: {trainable_params:,}\n")
                f.write(f"- Frozen parameters: {total_params - trainable_params:,}\n\n")
                
                # 레이어 분석
                f.write("LAYER ANALYSIS:\n")
                f.write("-" * 80 + "\n")
                
                for i, (layer_name, output_info) in enumerate(self.layer_outputs.items()):
                    f.write(f"\nLayer {i+1}: {layer_name}\n")
                    
                    if isinstance(output_info, dict):
                        f.write(f"  Shape: {output_info['shape']}\n")
                        f.write(f"  Type: {output_info['dtype']}\n")
                        f.write(f"  Stats: min={output_info['min']:.6f}, max={output_info['max']:.6f}\n")
                    elif isinstance(output_info, list):
                        for j, info in enumerate(output_info):
                            f.write(f"  Output {j+1}: {info['shape']} ({info['dtype']})\n")
            
            print(f"✅ Report saved successfully!")
            
        except Exception as e:
            print(f"❌ Failed to save report: {e}")
    
    def run_complete_analysis(self, batch_size=1, image_size=640):
        """전체 분석 프로세스 실행"""
        print("🚀 STARTING COMPLETE DUAL STREAM MODEL ANALYSIS")
        print("="*80)
        
        # 1. 모델 로드
        if not self.load_model():
            return False
        
        # 2. Dual stream 아키텍처 확인
        is_dual_stream = self._check_dual_stream_architecture()
        
        # 3. 더미 입력 생성
        dummy_input = self.create_dummy_input(batch_size, image_size, is_dual_stream)
        
        # 4. Forward pass 분석
        output = self.analyze_forward_pass(dummy_input)
        if output is None:
            return False
        
        # 5. 레이어별 상세 분석
        self.print_layer_analysis()
        
        # 6. 분석 리포트 저장
        self.save_analysis_report()
        
        print("\n" + "="*80)
        print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        return True

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze Dual Stream YOLO model structure')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH,
                        help=f'Path to model .pt file (default: {DEFAULT_MODEL_PATH})')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for analysis (default: 1)')
    parser.add_argument('--image-size', type=int, default=DEFAULT_INPUT_SIZE,
                        help=f'Input image size (default: {DEFAULT_INPUT_SIZE})')
    parser.add_argument('--output', type=str, help='Output report file path')
    
    args = parser.parse_args()
    
    # 모델 파일 존재 확인
    if not Path(args.model).exists():
        print(f"❌ Model file not found: {args.model}")
        print("\n🔍 Available model files:")
        
        # 가능한 모델 파일들 검색
        possible_paths = [
            "yolo11n.pt",
            "yolov8n.pt", 
            "runs/train/*/weights/best.pt",
            "runs/train/*/weights/last.pt"
        ]
        
        for pattern in possible_paths:
            for path in Path(".").glob(pattern):
                print(f"   - {path}")
        
        return False
    
    # 분석 실행
    analyzer = ModelStructureAnalyzer(args.model)
    success = analyzer.run_complete_analysis(
        batch_size=args.batch_size,
        image_size=args.image_size
    )
    
    if success:
        print(f"\n💡 To analyze different models, use:")
        print(f"   python {__file__} --model path/to/your/model.pt")
        print(f"\n📊 For detailed analysis with custom settings:")
        print(f"   python {__file__} --model {args.model} --batch-size 4 --image-size 1024")
    
    return success

if __name__ == "__main__":
    main()
