#!/usr/bin/env python3
"""
Dual Stream YOLO 백본과 헤드 레이어 통과 시뮬레이션
YAML 파일을 기반으로 각 레이어에서의 텐서 형태 변화를 출력합니다.
"""

import yaml
from pathlib import Path

def load_model_config(yaml_path):
    """YAML 설정 파일 로드"""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def calculate_output_size(input_size, kernel_size, stride, padding=None):
    """컨볼루션 레이어 출력 크기 계산"""
    if padding is None:
        # YOLO에서는 일반적으로 'same' padding을 사용하여 stride로 나눈 몫이 정확히 나오도록 함
        if stride == 1:
            padding = kernel_size // 2
        else:
            # stride > 1일 때는 일반적으로 크기가 정확히 stride로 나누어지도록 설계
            padding = (kernel_size - 1) // 2
    
    output_size = (input_size + 2 * padding - kernel_size) // stride + 1
    
    # YOLO의 일반적인 동작: stride=2일 때 정확히 절반이 되도록 조정
    if stride == 2 and output_size % 2 == 1:
        output_size = input_size // stride
    
    return output_size

def simulate_dual_stream_layers(config, input_size=640):
    """Dual stream 레이어 통과 시뮬레이션"""
    
    print("🚀 DUAL STREAM YOLO LAYER SIMULATION")
    print("=" * 80)
    print(f"📥 Input: Dual Stream [1, 2, 3, {input_size}, {input_size}]")
    print("   - Wide stream:  [1, 3, {}, {}]".format(input_size, input_size))
    print("   - Narrow stream: [1, 3, {}, {}]".format(input_size, input_size))
    print()
    
    # 현재 텐서 상태 추적
    current_batch = 1
    current_streams = 2  # dual stream
    current_channels = 3
    current_height = input_size
    current_width = input_size
    
    backbone_outputs = {}  # 백본 출력들 저장 (헤드에서 사용)
    
    print("🔧 BACKBONE LAYERS:")
    print("-" * 60)
    
    backbone = config.get('backbone', [])
    
    for i, layer in enumerate(backbone):
        if len(layer) < 3:
            continue
            
        from_layer, repeats, module, args = layer[0], layer[1], layer[2], layer[3] if len(layer) > 3 else []
        
        print(f"\n🔍 Layer {i}: {module}")
        print(f"   From: {from_layer}, Repeats: {repeats}, Args: {args}")
        
        # 이전 크기 출력
        if current_streams == 2:
            print(f"   📥 Input:  [batch={current_batch}, streams={current_streams}, channels={current_channels}, H={current_height}, W={current_width}]")
        else:
            print(f"   📥 Input:  [batch={current_batch}, channels={current_channels}, H={current_height}, W={current_width}]")
        
        # 모듈별 출력 계산
        if module == "MultiStreamConv":
            # args: [out_channels, kernel_size, stride, padding]
            out_channels = args[0] if args else 64
            kernel_size = args[1] if len(args) > 1 else 3
            stride = args[2] if len(args) > 2 else 1
            
            current_channels = out_channels
            current_height = calculate_output_size(current_height, kernel_size, stride)
            current_width = calculate_output_size(current_width, kernel_size, stride)
            # Dual stream 유지
            
        elif module == "SpatialAlignedMultiStreamConv":
            # args: [out_channels, kernel_size, stride]
            out_channels = args[0] if args else 64
            kernel_size = args[1] if len(args) > 1 else 3
            stride = args[2] if len(args) > 2 else 1
            
            current_channels = out_channels
            current_height = calculate_output_size(current_height, kernel_size, stride)
            current_width = calculate_output_size(current_width, kernel_size, stride)
            # Dual stream 유지
            
        elif module == "MultiStreamC3":
            # args: [out_channels]
            out_channels = args[0] if args else current_channels
            current_channels = out_channels
            # 크기는 변하지 않음
            
            # ⭐ MultiStreamC3에서 dual stream이 single stream으로 fusion됨!
            if current_streams == 2:
                current_streams = 1
                print(f"   🔀 Dual → Single stream FUSION in MultiStreamC3")
            
        elif module == "Conv":
            # args: [out_channels, kernel_size, stride]
            out_channels = args[0] if args else 128
            kernel_size = args[1] if len(args) > 1 else 3
            stride = args[2] if len(args) > 2 else 1
            
            current_channels = out_channels
            current_height = calculate_output_size(current_height, kernel_size, stride)
            current_width = calculate_output_size(current_width, kernel_size, stride)
            
            # Conv는 이미 fusion된 이후에 사용됨 (MultiStreamC3에서 fusion 완료)
                
        elif module == "C2f":
            # args: [out_channels, shortcut]
            out_channels = args[0] if args else current_channels
            current_channels = out_channels
            # 크기는 변하지 않음
            
        elif module == "SPPF":
            # args: [out_channels, kernel_size]
            out_channels = args[0] if args else current_channels
            current_channels = out_channels
            # 크기는 변하지 않음
        
        # 현재 크기 출력
        if current_streams == 2:
            print(f"   📤 Output: [batch={current_batch}, streams={current_streams}, channels={current_channels}, H={current_height}, W={current_width}]")
        else:
            print(f"   📤 Output: [batch={current_batch}, channels={current_channels}, H={current_height}, W={current_width}]")
        
        # 백본 출력 저장 (P3, P4, P5 레벨)
        if i in [4, 6, 8]:  # P3, P4, P5 해당 레이어들
            level_name = f"P{3 + (i-4)//2}"
            backbone_outputs[i] = {
                'name': level_name,
                'shape': [current_batch, current_channels, current_height, current_width]
            }
            print(f"   💾 Saved as {level_name} feature map")
    
    print("\n" + "=" * 80)
    print("🎯 HEAD LAYERS:")
    print("-" * 60)
    
    head = config.get('head', [])
    head_outputs = {}
    
    for i, layer in enumerate(head):
        if len(layer) < 3:
            continue
            
        from_layer, repeats, module, args = layer[0], layer[1], layer[2], layer[3] if len(layer) > 3 else []
        
        print(f"\n🔍 Head Layer {i}: {module}")
        print(f"   From: {from_layer}, Repeats: {repeats}, Args: {args}")
        
        # from_layer에 따른 입력 결정
        if isinstance(from_layer, list):
            if len(from_layer) == 2:  # Concat 연산
                # 이전 레이어들의 출력 크기 추정
                print(f"   🔗 Concatenating from layers: {from_layer}")
        
        # 이전 크기 출력
        print(f"   📥 Input:  [batch={current_batch}, channels={current_channels}, H={current_height}, W={current_width}]")
        
        # 모듈별 출력 계산
        if module == "nn.Upsample":
            # args: [size, scale_factor, mode]
            scale_factor = args[1] if len(args) > 1 else 2
            current_height *= scale_factor
            current_width *= scale_factor
            
        elif module == "Concat":
            # 채널 수가 증가 (정확한 계산은 복잡하므로 추정)
            if i == 1:  # P4와 concat
                current_channels += 256  # P4 채널 수 추정
            elif i == 4:  # P3와 concat  
                current_channels += 128  # P3 채널 수 추정
            
        elif module == "C2f":
            # args: [out_channels]
            out_channels = args[0] if args else current_channels
            current_channels = out_channels
            
        elif module == "Conv":
            # args: [out_channels, kernel_size, stride]
            out_channels = args[0] if args else 128
            kernel_size = args[1] if len(args) > 1 else 3
            stride = args[2] if len(args) > 2 else 1
            
            current_channels = out_channels
            current_height = calculate_output_size(current_height, kernel_size, stride)
            current_width = calculate_output_size(current_width, kernel_size, stride)
            
        elif module == "Detect":
            # Detection head - 최종 출력
            nc = config.get('nc', 28)  # 클래스 수
            anchors_per_level = current_height * current_width
            output_channels = nc + 5  # classes + box + confidence
            
            print(f"   🎯 Detection head for {nc} classes")
            print(f"   📤 Final Output: [batch={current_batch}, anchors={anchors_per_level}, predictions={output_channels}]")
            continue
        
        # 현재 크기 출력
        print(f"   📤 Output: [batch={current_batch}, channels={current_channels}, H={current_height}, W={current_width}]")
        
        # 중요한 출력 레벨 저장
        if i in [5, 8, 11]:  # P3, P4, P5 detection levels
            level = f"Detection_P{3 + (i-5)//3}"
            head_outputs[i] = {
                'name': level,
                'shape': [current_batch, current_channels, current_height, current_width]
            }
            print(f"   💾 Saved as {level}")
    
    print("\n" + "=" * 80)
    print("📊 SUMMARY:")
    print("-" * 60)
    print("🔧 Backbone Feature Maps:")
    for layer_idx, info in backbone_outputs.items():
        print(f"   {info['name']}: {info['shape']}")
    
    print("\n🎯 Head Outputs:")
    for layer_idx, info in head_outputs.items():
        print(f"   {info['name']}: {info['shape']}")
    
    print(f"\n✅ Model processes dual stream input [1, 2, 3, {input_size}, {input_size}]")
    print(f"   → Extracts features at multiple scales (P3, P4, P5)")
    print(f"   → Outputs detections for {config.get('nc', 28)} classes")

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simulate Dual Stream YOLO layer processing')
    parser.add_argument('--config', type=str, default='models/yolov8-dual.yaml',
                        help='Path to model YAML config file')
    parser.add_argument('--input-size', type=int, default=640,
                        help='Input image size (default: 640)')
    
    args = parser.parse_args()
    
    # 설정 파일 확인
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {args.config}")
        print("Looking for available config files...")
        
        # 가능한 설정 파일들 검색
        for pattern in ["models/*.yaml", "*.yaml"]:
            for path in Path(".").glob(pattern):
                if 'dual' in path.name.lower() or 'yolo' in path.name.lower():
                    print(f"   - {path}")
        return False
    
    # 설정 로드 및 시뮬레이션 실행
    try:
        config = load_model_config(config_path)
        simulate_dual_stream_layers(config, args.input_size)
        
        print(f"\n💡 To analyze with different input size:")
        print(f"   python {Path(__file__).name} --input-size 1024")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()
