#!/usr/bin/env python3
"""
Dual Stream YOLO ONNX Inference (Extracted) - Video Generator (Prediction Only) [v3]

v3 ADDITIONS:
1. Multi-session root processing: Provide --sessions-root containing multiple *_(cam0|cam1) folders.
   Example structure:
       sessions_root/
           20250918_152853_cam0/
           20250918_152853_cam1/
           20250918_160012_cam0/
           20250918_160012_cam1/
2. Automatic session grouping by directory name pattern '(session_id)_cam[01]'.
3. Pre-compute ALL file pairings before any inference and save mapping JSON (paired_files_v3.json).
4. Strict pairing mode (--strict-pair): assume perfect 1:1 chronological match; pair by sorted timestamp index up to min(len(cam0), len(cam1)).
5. Optional threshold pairing (--pair-max-diff) like v2 when not strict.
6. Option to output a single combined video (default) or per-session videos (--per-session-video).
7. Deterministic ordering: sessions sorted lexicographically, inside session by time.

You can still use single-session mode with --wide-dir / --narrow-dir like previous versions.

Usage (multi-session):
    python dual_stream_inference_depth_onnx_extracted_video_v3.py \
        --onnx best_dual_input_depth_extracted2.onnx \
        --sessions-root test_imgs \
        --strict-pair --out-dir inference_results_v3

Usage (single session same as v2):
    python dual_stream_inference_depth_onnx_extracted_video_v3.py \
        --wide-dir test_imgs/20250918_152853_cam0 \
        --narrow-dir test_imgs/20250918_152853_cam1

Dependencies: onnxruntime, numpy, opencv-python
"""

from __future__ import annotations

import sys
from pathlib import Path
import argparse
import time
import random
import json
import re
from typing import List, Tuple, Dict, Optional

import numpy as np  # noqa: F401 (used indirectly if custom_postprocess returns numpy arrays)
import cv2
import onnxruntime as ort

SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

DEFAULT_ONNX = '/home/byounggun/ultralytics/best_dual_input_depth_extracted2.onnx'
DEFAULT_OUTPUT_DIR = 'inference_results_depth_onnx_extracted_video'
DEFAULT_VIDEO_NAME = 'dual_stream_depth_inference_v3.mp4'
DEFAULT_WIDE_DIR = '/home/byounggun/ultralytics/test_imgs/20250918_152853_cam0'
DEFAULT_NARROW_DIR = '/home/byounggun/ultralytics/test_imgs/20250918_152853_cam1'

DEP_MIN = 0.1
DEP_MAX = 419.1

try:
    from dual_stream_inference_depth_onnx_extracted import (  # type: ignore
        DepthDenormalizer,
        preprocess_image,
        create_dual_stream_inputs,
        custom_postprocess,
        draw_detections_with_depth,
    )
except Exception as e:  # pragma: no cover - fallback minimal
    print(f"[WARN] Could not import helpers from single-image script: {e}")
    class DepthDenormalizer:  # minimal
        def __init__(self, min_depth=DEP_MIN, max_depth=DEP_MAX):
            self.min_depth = min_depth
            self.max_depth = max_depth
        def denormalize_depth(self, normalized_depth):
            return normalized_depth * (self.max_depth - self.min_depth) + self.min_depth
    def preprocess_image(image_path, target_size=640):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        oh, ow = image.shape[:2]
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        scale = min(target_size / ow, target_size / oh)
        nw, nh = int(ow * scale), int(oh * scale)
        resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        dw, dh = target_size - nw, target_size - nh
        top, bottom = dh // 2, dh - dh // 2
        left, right = dw // 2, dw - dw // 2
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[114,114,114])
        norm = padded.astype(np.float32) / 255.0
        tensor = norm.transpose(2,0,1)[np.newaxis,:]
        return tensor, (oh, ow), image
    def create_dual_stream_inputs(wide_tensor, narrow_tensor):
        return {'images_wide': wide_tensor, 'images_narrow': narrow_tensor}
    def draw_detections_with_depth(img, dets, gts, depth_denormalizer):
        for d in dets:
            x1,y1,x2,y2,conf,cls,depth = d
            cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
            cv2.putText(img,f"{int(cls)} {conf:.2f} {depth:.1f}",(int(x1),int(y1)-5),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,0),1,cv2.LINE_AA)
        return img
    def custom_postprocess(outputs, orig_size, img_size_f, conf_thr, iou_thr, reverse_lb):
        raise SystemExit("custom_postprocess unavailable in fallback; keep original script in place.")


def parse_timestamp_from_filename(path: Path) -> Optional[float]:
    stem = path.stem
    parts = stem.split('_')
    if len(parts) < 3:
        return None
    hhmmss = parts[-2]
    micro = parts[-1]
    if not (hhmmss.isdigit() and micro.isdigit() and len(hhmmss) == 6):
        return None
    h = int(hhmmss[0:2])
    m = int(hhmmss[2:4])
    s = int(hhmmss[4:6])
    us = int(micro[:6].ljust(6, '0'))
    return h*3600 + m*60 + s + us/1_000_000.0


def collect_images(dir_path: Path, exts=(".jpg", ".jpeg", ".png")) -> List[Path]:
    return sorted([p for p in dir_path.iterdir() if p.suffix.lower() in exts])


def pair_strict(wide_files: List[Path], narrow_files: List[Path]) -> Tuple[List[Tuple[Path, Path, float]], Dict[str, int]]:
    """Strict 1:1 by sorted timestamp index. Delta recorded (abs time difference)."""
    wt = [(p, parse_timestamp_from_filename(p)) for p in wide_files]
    nt = [(p, parse_timestamp_from_filename(p)) for p in narrow_files]
    wt = [x for x in wt if x[1] is not None]
    nt = [x for x in nt if x[1] is not None]
    wt.sort(key=lambda x: x[1])
    nt.sort(key=lambda x: x[1])
    n = min(len(wt), len(nt))
    pairs = []
    for i in range(n):
        wp, wtime = wt[i]
        np_, ntime = nt[i]
        delta = abs(wtime - ntime)
        pairs.append((wp, np_, delta))
    stats = {
        'wide_total': len(wide_files),
        'narrow_total': len(narrow_files),
        'paired': len(pairs),
        'unpaired_wide': len(wide_files) - len(pairs),
        'unpaired_narrow': len(narrow_files) - len(pairs),
    }
    return pairs, stats


def pair_by_nearest_time(
    wide_files: List[Path],
    narrow_files: List[Path],
    max_diff: float,
) -> Tuple[List[Tuple[Path, Path, float]], Dict[str, int]]:
    wide_times = [(p, parse_timestamp_from_filename(p)) for p in wide_files]
    narrow_times = [(p, parse_timestamp_from_filename(p)) for p in narrow_files]
    wide_times = [wt for wt in wide_times if wt[1] is not None]
    narrow_times = [nt for nt in narrow_times if nt[1] is not None]
    wide_times.sort(key=lambda x: x[1])
    narrow_times.sort(key=lambda x: x[1])
    i = j = 0
    result: List[Tuple[Path, Path, float]] = []
    while i < len(wide_times) and j < len(narrow_times):
        wp, wt = wide_times[i]
        np_, nt = narrow_times[j]
        delta = wt - nt
        abs_delta = abs(delta)
        if abs_delta <= max_diff:
            result.append((wp, np_, abs_delta))
            i += 1
            j += 1
        else:
            if wt < nt:
                i += 1
            else:
                j += 1
    stats = {
        'wide_total': len(wide_files),
        'narrow_total': len(narrow_files),
        'paired': len(result),
        'unpaired_wide': len(wide_files) - len(result),
        'unpaired_narrow': len(narrow_files) - len(result),
    }
    return result, stats


SESSION_DIR_REGEX = re.compile(r'^(?P<sid>.+)_cam(?P<cam>[01])$')


def discover_sessions(root: Path) -> Dict[str, Dict[str, Path]]:
    sessions: Dict[str, Dict[str, Path]] = {}
    for d in root.iterdir():
        if not d.is_dir():
            continue
        m = SESSION_DIR_REGEX.match(d.name)
        if not m:
            continue
        sid = m.group('sid')
        cam = m.group('cam')  # '0' or '1'
        sess = sessions.setdefault(sid, {})
        sess[cam] = d
    # Filter only complete sessions with both cams
    complete = {k: v for k, v in sessions.items() if '0' in v and '1' in v}
    return complete


def build_global_pairs(
    sessions: Dict[str, Dict[str, Path]],
    strict: bool,
    max_diff: float,
    limit_per_session: int = -1,
    shuffle: bool = False,
    seed: int = 0,
) -> Tuple[List[Tuple[Path, Path, str, float]], Dict[str, Dict[str, int]]]:
    """Return list of (wide_path, narrow_path, session_id, delta) across all sessions.
    Also returns stats per session.
    """
    rng = random.Random(seed)
    session_ids = sorted(sessions.keys())
    global_pairs: List[Tuple[Path, Path, str, float]] = []
    per_stats: Dict[str, Dict[str, int]] = {}
    for sid in session_ids:
        wide_dir = sessions[sid]['0']
        narrow_dir = sessions[sid]['1']
        wide_files = collect_images(wide_dir)
        narrow_files = collect_images(narrow_dir)
        if strict:
            pairs, stats = pair_strict(wide_files, narrow_files)
        else:
            pairs, stats = pair_by_nearest_time(wide_files, narrow_files, max_diff)
        per_stats[sid] = stats
        if limit_per_session > 0:
            pairs = pairs[:limit_per_session]
        for w, n, delta in pairs:
            global_pairs.append((w, n, sid, delta))
    if shuffle:
        rng.shuffle(global_pairs)
    return global_pairs, per_stats


def annotate_frame(wide_image_bgr, detections, depth_denormalizer):
    return draw_detections_with_depth(wide_image_bgr, detections, [], depth_denormalizer)


def main():
    ap = argparse.ArgumentParser(description='Dual Stream ONNX Video Inference v3 (multi-session + prepair)')
    ap.add_argument('--onnx', type=str, default=DEFAULT_ONNX, help='Path to ONNX model')
    # Single session args
    ap.add_argument('--wide-dir', type=str, help='Wide image directory (cam0)')
    ap.add_argument('--narrow-dir', type=str, help='Narrow image directory (cam1)')
    # Multi-session root
    ap.add_argument('--sessions-root', type=str, help='Root containing *_cam0 & *_cam1 subdirs')
    ap.add_argument('--strict-pair', action='store_true', help='Strict 1:1 chronological pairing (ignore threshold)')
    ap.add_argument('--pair-max-diff', type=float, default=0.15, help='Max timestamp delta for non-strict pairing (seconds)')
    ap.add_argument('--out-dir', type=str, default=DEFAULT_OUTPUT_DIR, help='Output directory root')
    ap.add_argument('--out-video', type=str, default=DEFAULT_VIDEO_NAME, help='Combined output video name (when not per-session)')
    ap.add_argument('--per-session-video', action='store_true', help='Produce one video per session instead of combined')
    ap.add_argument('--size', type=int, default=640, help='Inference image size')
    ap.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    ap.add_argument('--iou', type=float, default=0.35, help='IoU threshold for NMS')
    ap.add_argument('--fps', type=int, default=10, help='Video FPS')
    ap.add_argument('--max-frames', type=int, default=-1, help='Global limit of frames (after pairing aggregation)')
    ap.add_argument('--limit-per-session', type=int, default=-1, help='Limit number of pairs per session before aggregation')
    ap.add_argument('--save-frames', action='store_true', help='Save each annotated frame as image')
    ap.add_argument('--no-reverse-lb', action='store_true', help='Disable reverse letterbox mapping')
    ap.add_argument('--shuffle', action='store_true', help='Shuffle order (after building all pairs)')
    ap.add_argument('--seed', type=int, default=0, help='Random seed for shuffling')
    ap.add_argument('--verbose', action='store_true', help='Verbose per-frame logging')
    args = ap.parse_args()

    # Reverse letterbox flag (simple and correct)
    reverse_lb_flag = not args.no_reverse_lb

    # Build list of (wide_path, narrow_path, session_id, delta)
    if args.sessions_root:
        root = Path(args.sessions_root)
        if not root.is_dir():
            print('[ERROR] sessions_root invalid')
            return
        sessions = discover_sessions(root)
        if not sessions:
            print('[ERROR] No complete sessions (cam0+cam1) found under root.')
            return
        print(f"[INFO] Discovered {len(sessions)} sessions: {', '.join(sorted(sessions.keys()))}")
        global_pairs, per_stats = build_global_pairs(
            sessions,
            strict=args.strict_pair,
            max_diff=args.pair_max_diff,
            limit_per_session=args.limit_per_session,
            shuffle=args.shuffle,
            seed=args.seed,
        )
    else:
        # Allow running with zero CLI args by falling back to defaults
        if not args.wide_dir and not args.narrow_dir:
            wide_dir = Path(DEFAULT_WIDE_DIR)
            narrow_dir = Path(DEFAULT_NARROW_DIR)
            print(f"[INFO] Using default directories: wide={wide_dir} narrow={narrow_dir}")
        else:
            # If only one provided, still fall back for the other
            wide_dir = Path(args.wide_dir) if args.wide_dir else Path(DEFAULT_WIDE_DIR)
            narrow_dir = Path(args.narrow_dir) if args.narrow_dir else Path(DEFAULT_NARROW_DIR)
            if not args.wide_dir or not args.narrow_dir:
                print(f"[INFO] Mixed provided/default directories: wide={wide_dir} narrow={narrow_dir}")
        if not wide_dir.exists() or not narrow_dir.exists():
            print('[ERROR] Wide or narrow directory does not exist (after applying defaults).')
            return
        if not wide_dir.is_dir() or not narrow_dir.is_dir():
            print('[ERROR] Wide or narrow directory invalid (not a directory).')
            return
        wide_files = collect_images(wide_dir)
        narrow_files = collect_images(narrow_dir)
        if args.strict_pair:
            pairs, stats_single = pair_strict(wide_files, narrow_files)
        else:
            pairs, stats_single = pair_by_nearest_time(wide_files, narrow_files, args.pair_max_diff)
        per_stats = {'single_session': stats_single}
        global_pairs = [(w, n, 'single_session', d) for w, n, d in pairs]

    if not global_pairs:
        print('[ERROR] No pairs produced.')
        return

    if args.max_frames > 0:
        global_pairs = global_pairs[:args.max_frames]
        print(f"[INFO] Global frame limit applied: {len(global_pairs)}")

    print(f"[INFO] Total paired frames: {len(global_pairs)}")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    frames_dir = out_root / ('frames_v3_per_session' if args.per_session_video else 'frames_v3')
    if args.save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    # Save pairing map BEFORE inference
    pairing_json = out_root / 'paired_files_v3.json'
    with open(pairing_json, 'w') as f:
        json.dump({
            'mode': 'multi' if args.sessions_root else 'single',
            'strict': args.strict_pair,
            'pair_max_diff': args.pair_max_diff,
            'sessions_stats': per_stats,
            'total_pairs': len(global_pairs),
            'pairs': [
                {
                    'session': sid,
                    'wide': str(w),
                    'narrow': str(n),
                    'delta_ms': round(delta*1000, 3)
                } for w, n, sid, delta in global_pairs
            ]
        }, f, indent=2)
    print(f"[INFO] Pairing map saved: {pairing_json}")

    # ONNX session
    try:
        providers = ['CPUExecutionProvider']
        ort_session = ort.InferenceSession(args.onnx, providers=providers)
        print('[OK] ONNX model loaded.')
    except Exception as e:
        print(f'[ERROR] ONNX load failed: {e}')
        return

    depth_denormalizer = DepthDenormalizer()

    # Video writers
    combined_writer = None
    per_session_writers: Dict[str, cv2.VideoWriter] = {}
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    t0 = time.time()
    frame_count = 0
    summary = []

    def get_writer(session_id: str, frame_shape) -> cv2.VideoWriter:
        nonlocal combined_writer
        h, w = frame_shape[:2]
        if args.per_session_video:
            if session_id not in per_session_writers:
                vid_path = out_root / f"{session_id}.mp4"
                vw = cv2.VideoWriter(str(vid_path), fourcc, args.fps, (w, h))
                if not vw.isOpened():
                    raise RuntimeError(f"Failed to open per-session writer for {session_id}")
                per_session_writers[session_id] = vw
                print(f"[OK] Opened session video: {vid_path}")
            return per_session_writers[session_id]
        else:
            if combined_writer is None:
                vid_path = out_root / args.out_video
                combined_writer = cv2.VideoWriter(str(vid_path), fourcc, args.fps, (w, h))
                if not combined_writer.isOpened():
                    raise RuntimeError('Failed to open combined VideoWriter.')
                print(f"[OK] Opened combined video: {vid_path}")
            return combined_writer

    # reverse_lb_flag already computed earlier

    for wide_path, narrow_path, sid, delta in global_pairs:
        frame_t0 = time.time()
        try:
            wide_tensor, wide_size, wide_bgr = preprocess_image(str(wide_path), args.size)
            narrow_tensor, narrow_size, _ = preprocess_image(str(narrow_path), args.size)
        except Exception as ex:
            print(f'[WARN] Preprocess failed {wide_path.name} ({sid}): {ex}')
            continue

        dual_inputs = create_dual_stream_inputs(wide_tensor, narrow_tensor)
        try:
            outputs = ort_session.run(None, dual_inputs)
        except Exception as ex:
            print(f'[WARN] Inference failed {wide_path.name} ({sid}): {ex}')
            continue

        detections = custom_postprocess(outputs, wide_size, float(args.size), args.conf, args.iou, reverse_lb_flag)
        annotated = annotate_frame(wide_bgr, detections, depth_denormalizer)

        try:
            writer = get_writer(sid, annotated.shape)
        except Exception as ex:
            print(f'[ERROR] Video writer acquisition failed: {ex}')
            return
        writer.write(annotated)

        if args.save_frames:
            session_frame_dir = frames_dir / sid
            session_frame_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(session_frame_dir / f'{frame_count:06d}_{wide_path.name}'), annotated)

        elapsed_ms = round((time.time() - frame_t0) * 1000, 2)
        summary.append({
            'session': sid,
            'file_wide': wide_path.name,
            'file_narrow': narrow_path.name,
            'delta_ms': round(delta*1000, 3),
            'detections': len(detections),
            'time_ms': elapsed_ms
        })
        if args.verbose:
            print(f"Frame {frame_count}: [{sid}] {wide_path.name} + {narrow_path.name} det={len(detections)} delta={delta*1000:.1f}ms")
        frame_count += 1

    # Release writers
    if combined_writer is not None:
        combined_writer.release()
    for sid, vw in per_session_writers.items():
        vw.release()

    total_time = time.time() - t0
    eff_fps = frame_count/total_time if total_time > 0 else 0.0
    print(f'[DONE] {frame_count} frames processed in {total_time:.2f}s ({eff_fps:.2f} FPS)')

    # Summary JSON
    summary_path = out_root / 'video_inference_summary_v3.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'model': args.onnx,
            'frames': frame_count,
            'fps_config': args.fps,
            'effective_fps': round(eff_fps, 2),
            'conf': args.conf,
            'iou': args.iou,
            'reverse_letterbox': reverse_lb_flag,
            'strict_pair': args.strict_pair,
            'pair_max_diff': args.pair_max_diff,
            'sessions_stats': per_stats,
            'global_pairs': len(global_pairs),
            'per_session_video': args.per_session_video,
            'frames_detail': summary,
        }, f, indent=2)
    print(f'[INFO] Summary JSON: {summary_path}')


if __name__ == '__main__':
    main()
