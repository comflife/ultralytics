#!/usr/bin/env python3
"""
Example Usage Script for ONNX Model to Image Visualization

This script demonstrates how to use the onnx_model_to_image.py tool
with different ONNX models in the workspace.
"""

import os
import sys
from pathlib import Path
import subprocess
import glob

def find_onnx_models(search_dir="."):
    """Find all ONNX models in the workspace"""
    onnx_files = []
    
    # Search for ONNX files
    search_patterns = [
        "**/*.onnx",
        "runs/**/*.onnx",
        "models/**/*.onnx",
        "weights/**/*.onnx"
    ]
    
    for pattern in search_patterns:
        files = glob.glob(pattern, recursive=True)
        onnx_files.extend(files)
    
    # Remove duplicates and sort
    onnx_files = sorted(list(set(onnx_files)))
    return onnx_files

def run_visualization(model_path, visualization_type="overview"):
    """Run visualization for a specific model"""
    script_path = "onnx_model_to_image.py"
    
    if not Path(script_path).exists():
        print(f"❌ Visualization script not found: {script_path}")
        return False
    
    try:
        cmd = ["python", script_path, model_path]
        
        if visualization_type == "all":
            cmd.append("--visualize-all")
        elif visualization_type == "overview":
            cmd.append("--overview")
        elif visualization_type == "graph":
            cmd.append("--graph")
        elif visualization_type == "layers":
            cmd.append("--layers")
        elif visualization_type == "performance":
            cmd.append("--performance")
        elif visualization_type == "dual-stream":
            cmd.append("--dual-stream")
        
        print(f"🚀 Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Visualization completed successfully!")
            print(result.stdout)
            return True
        else:
            print("❌ Visualization failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
    
    except Exception as e:
        print(f"❌ Error running visualization: {e}")
        return False

def main():
    print("🎨 ONNX Model Visualization Examples")
    print("=" * 50)
    
    # Find available ONNX models
    onnx_models = find_onnx_models()
    
    if not onnx_models:
        print("❌ No ONNX models found in the workspace!")
        print("Please ensure you have exported some models to ONNX format.")
        return
    
    print(f"📂 Found {len(onnx_models)} ONNX model(s):")
    for i, model in enumerate(onnx_models[:10]):  # Show first 10
        size_mb = Path(model).stat().st_size / (1024 * 1024)
        print(f"   {i+1}. {model} ({size_mb:.1f} MB)")
    
    if len(onnx_models) > 10:
        print(f"   ... and {len(onnx_models) - 10} more")
    
    print("\n" + "="*50)
    
    # Example 1: Basic overview visualization
    print("\n🔍 Example 1: Basic Model Overview")
    print("-" * 30)
    
    # Use the first available model
    test_model = onnx_models[0]
    print(f"Using model: {test_model}")
    
    success = run_visualization(test_model, "overview")
    if success:
        print("✅ Basic overview completed!")
    
    # Example 2: Comprehensive analysis
    print("\n🔍 Example 2: Comprehensive Analysis")
    print("-" * 30)
    
    # Look for a dual-stream model
    dual_model = None
    for model in onnx_models:
        if any(keyword in model.lower() for keyword in ['dual', 'stream', 'depth']):
            dual_model = model
            break
    
    if dual_model:
        print(f"Using dual-stream model: {dual_model}")
        success = run_visualization(dual_model, "all")
        if success:
            print("✅ Comprehensive analysis completed!")
    else:
        print("No dual-stream model found, using first model for comprehensive analysis")
        success = run_visualization(test_model, "all")
        if success:
            print("✅ Comprehensive analysis completed!")
    
    # Example 3: Specific analysis types
    print("\n🔍 Example 3: Specific Analysis Types")
    print("-" * 30)
    
    analysis_types = ["graph", "layers", "performance"]
    
    for analysis_type in analysis_types:
        print(f"\nRunning {analysis_type} analysis...")
        success = run_visualization(test_model, analysis_type)
        if success:
            print(f"✅ {analysis_type.title()} analysis completed!")
        else:
            print(f"❌ {analysis_type.title()} analysis failed!")
    
    # Example 4: Batch processing
    print("\n🔍 Example 4: Batch Processing (Multiple Models)")
    print("-" * 30)
    
    # Process first 3 models for overview
    models_to_process = onnx_models[:min(3, len(onnx_models))]
    
    for i, model in enumerate(models_to_process):
        print(f"\nProcessing model {i+1}/{len(models_to_process)}: {Path(model).name}")
        success = run_visualization(model, "overview")
        if success:
            print(f"✅ Model {i+1} processed successfully!")
        else:
            print(f"❌ Model {i+1} processing failed!")
    
    print("\n" + "="*50)
    print("🎉 Example demonstrations completed!")
    print("\nGenerated visualizations can be found in:")
    print("  - ./visualizations/ (default)")
    print("  - Same directory as each model file")
    print("\nTo view images:")
    print("  eog visualizations/*.png")
    print("  # or")
    print("  firefox visualizations/")
    
    # Show manual usage examples
    print("\n💡 Manual Usage Examples:")
    print("=" * 30)
    print("# Basic overview:")
    print(f"python onnx_model_to_image.py {test_model} --overview")
    print()
    print("# All visualizations:")
    print(f"python onnx_model_to_image.py {test_model} --visualize-all")
    print()
    print("# Specific analysis:")
    print(f"python onnx_model_to_image.py {test_model} --graph --layers")
    print()
    print("# Custom output directory:")
    print(f"python onnx_model_to_image.py {test_model} --output-dir ./my_visualizations --visualize-all")

if __name__ == "__main__":
    main()
