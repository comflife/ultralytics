#!/usr/bin/env python3
"""
Compare ONNX model sizes and parameters.

Usage:
  python3 compare_onnx_model_size.py /path/to/model1.onnx /path/to/model2.onnx
  # You can pass two or more models; prints a table and diffs for two.
"""
from __future__ import annotations
import argparse
import os
from typing import Dict, Tuple, List

import onnx
from onnx import numpy_helper
import numpy as np


def human_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024 or unit == "TB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{n} {unit}"
        n /= 1024
    return f"{n:.2f} TB"


def tensor_numel(t: onnx.TensorProto) -> int:
    # Handle empty dims (scalars) and external data
    if t.dims:
        numel = int(np.prod(list(t.dims)))
    else:
        numel = 1
    return numel


DTYPE_BYTES = {
    onnx.TensorProto.FLOAT: 4,
    onnx.TensorProto.UINT8: 1,
    onnx.TensorProto.INT8: 1,
    onnx.TensorProto.UINT16: 2,
    onnx.TensorProto.INT16: 2,
    onnx.TensorProto.INT32: 4,
    onnx.TensorProto.INT64: 8,
    onnx.TensorProto.BOOL: 1,
    onnx.TensorProto.FLOAT16: 2,
    onnx.TensorProto.DOUBLE: 8,
    onnx.TensorProto.UINT32: 4,
    onnx.TensorProto.UINT64: 8,
    onnx.TensorProto.COMPLEX64: 8,
    onnx.TensorProto.COMPLEX128: 16,
    onnx.TensorProto.BFLOAT16: 2,
}


def tensor_nbytes(t: onnx.TensorProto) -> int:
    # Prefer raw_data length if present; otherwise infer from data_type and numel
    if t.raw_data:
        return len(t.raw_data)
    # If not raw_data, values could be stored in typed fields
    numel = tensor_numel(t)
    itemsize = DTYPE_BYTES.get(t.data_type, 0)
    if itemsize == 0:
        # Fallback: try converting via numpy_helper (may be expensive)
        try:
            arr = numpy_helper.to_array(t)
            return int(arr.nbytes)
        except Exception:
            return 0
    return numel * itemsize


def analyze_onnx(path: str) -> Dict[str, object]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    model = onnx.load(path)
    graph = model.graph

    # Initializers hold the trainable weights (and some constants)
    param_tensors: List[onnx.TensorProto] = list(graph.initializer)

    total_params = 0
    total_param_bytes = 0
    per_dtype_counts: Dict[str, int] = {}
    for t in param_tensors:
        numel = tensor_numel(t)
        nbytes = tensor_nbytes(t)
        total_params += numel
        total_param_bytes += nbytes
        dt_name = onnx.TensorProto.DataType.Name(t.data_type) if hasattr(onnx.TensorProto.DataType, 'Name') else str(t.data_type)
        per_dtype_counts[dt_name] = per_dtype_counts.get(dt_name, 0) + numel

    # File size on disk
    file_size = os.path.getsize(path)

    # Node count and graph stats
    node_count = len(graph.node)
    input_count = len(graph.input)
    output_count = len(graph.output)

    return {
        "path": path,
        "file_size": file_size,
        "file_size_h": human_bytes(file_size),
        "total_params": total_params,
        "total_param_bytes": total_param_bytes,
        "total_param_bytes_h": human_bytes(total_param_bytes),
        "node_count": node_count,
        "input_count": input_count,
        "output_count": output_count,
        "dtype_param_counts": per_dtype_counts,
    }


def print_summary(reports: List[Dict[str, object]]):
    print("ONNX Model Size and Parameter Summary")
    print("-" * 72)
    header = f"{'Model':60}  {'File Size':>10}  {'Params':>12}  {'Param Bytes':>12}  {'Nodes':>6}"
    print(header)
    print("-" * 72)
    for r in reports:
        name = os.path.basename(r["path"])[:60]
        print(
            f"{name:60}  {r['file_size_h']:>10}  {r['total_params']:>12,}  {r['total_param_bytes_h']:>12}  {r['node_count']:>6}"
        )
    print("-" * 72)


def print_diff(a: Dict[str, object], b: Dict[str, object]):
    def pct(delta, base):
        return 0.0 if base == 0 else (delta / base) * 100.0

    print("Detailed Diff (Model A -> Model B)")
    print(f"A: {a['path']}")
    print(f"B: {b['path']}")
    print("-" * 72)
    fs_delta = int(b["file_size"]) - int(a["file_size"])
    pb_delta = int(b["total_param_bytes"]) - int(a["total_param_bytes"])
    p_delta = int(b["total_params"]) - int(a["total_params"])
    n_delta = int(b["node_count"]) - int(a["node_count"])
    print(f"File size: {a['file_size_h']} -> {b['file_size_h']}  (Δ {human_bytes(abs(fs_delta))} {('+' if fs_delta>=0 else '-')}{pct(fs_delta, a['file_size']):.2f}%)")
    print(f"Param bytes: {a['total_param_bytes_h']} -> {b['total_param_bytes_h']}  (Δ {human_bytes(abs(pb_delta))} {('+' if pb_delta>=0 else '-')}{pct(pb_delta, a['total_param_bytes']):.2f}%)")
    print(f"Params: {a['total_params']:,} -> {b['total_params']:,}  (Δ {p_delta:+,} {pct(p_delta, a['total_params']):.2f}%)")
    print(f"Nodes: {a['node_count']:,} -> {b['node_count']:,}  (Δ {n_delta:+,})")


def main():
    parser = argparse.ArgumentParser(description="Compare ONNX model sizes and parameter counts.")
    parser.add_argument("models", nargs="+", help="Paths to .onnx files (2+ recommended)")
    args = parser.parse_args()

    reports = [analyze_onnx(p) for p in args.models]
    print_summary(reports)
    if len(reports) == 2:
        print_diff(reports[0], reports[1])


if __name__ == "__main__":
    main()
