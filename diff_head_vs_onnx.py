#!/usr/bin/env python3
"""
Diff tool: Compare ONNX head outputs vs head text dump files (head0/1/2.txt).
Purpose: Diagnose mismatch (channel/spatial ordering, group order) causing biased boxes.

Features:
- Runs ONNX inference on a dual-stream image pair.
- Loads head text files (DFL/CLS/DEP) into dense tensors.
- Computes statistics of differences (mean/median/max abs diff) over non-zero dump entries.
- Samples several cells where dump provided data and prints side-by-side first 64 DFL logits (grouped 16*4).
- Supports alternative DFL group order test (--dfl-order) for downstream decoding check (not decoding here, just annotation).
- Optional output summary file.

Usage examples:
python diff_head_vs_onnx.py wide.jpg narrow.jpg --head-base /path/to/head
python diff_head_vs_onnx.py wide.jpg narrow.jpg --head-files head0.txt head1.txt head2.txt
"""
import os, sys, argparse, random, time
from pathlib import Path
import numpy as np
import onnxruntime as ort
import cv2
from math import sqrt

# Default config (adjust if needed)
ONNX_PATH = "/home/byounggun/ultralytics/runs/train/exp350/weights/best_dual_input_depth_extracted.onnx"
# Default directories overridden per user request to point to external download folders.
# You can still override them via CLI: --wide-dir /path --narrow-dir /path --image-name FILENAME.jpg
WIDE_DIR = "/home/byounggun/ultralytics/camera_wideview_100image"
NARROW_DIR = "/home/byounggun/ultralytics/camera_narrowview_100image"
DEFAULT_IMAGE_NAME = "20250422_07540580.jpg"
IMAGE_SIZE = 640
OUTPUT_DIR = "diff_head_compare_results"
DEP_MIN = 0.1
DEP_MAX = 419.1
CONF_THRES_DEFAULT = 0.25
IOU_THRES_DEFAULT = 0.35

# Shapes per level
LEVEL_SHAPES = [
    {'reg': (1,64,80,80), 'cls': (1,28,80,80), 'dep': (1,1,80,80)},
    {'reg': (1,64,40,40), 'cls': (1,28,40,40), 'dep': (1,1,40,40)},
    {'reg': (1,64,20,20), 'cls': (1,28,20,20), 'dep': (1,1,20,20)},
]

# ------------ Image preprocessing (same letterbox) ------------
def preprocess_image(image_path, target_size=640):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(image_path)
    h, w = img.shape[:2]
    scale = min(target_size / w, target_size / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = target_size - nw, target_size - nh
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,114,114))
    arr = padded.astype(np.float32)/255.0
    arr = arr.transpose(2,0,1)[np.newaxis,:]
    return arr, (h,w)

# ------------ Head file loader ------------
def load_head_text(path: str, reg_shape, cls_shape, dep_shape):
    reg = np.zeros(reg_shape, dtype=np.float32)
    cls = np.zeros(cls_shape, dtype=np.float32)
    dep = np.zeros(dep_shape, dtype=np.float32)
    filled = {'reg':0,'cls':0,'dep':0}
    try:
        with open(path,'r') as f:
            for line in f:
                if not line.strip() or line.startswith('#'): continue
                parts = line.strip().split()
                if len(parts)!=5: continue
                role, c, y, x, val = parts
                c = int(c); y = int(y); x = int(x); v = float(val)
                if role=='DFL':
                    if 0<=c<reg_shape[1] and 0<=y<reg_shape[2] and 0<=x<reg_shape[3]:
                        reg[0,c,y,x]=v; filled['reg']+=1
                elif role=='CLS':
                    if 0<=c<cls_shape[1] and 0<=y<cls_shape[2] and 0<=x<cls_shape[3]:
                        cls[0,c,y,x]=v; filled['cls']+=1
                elif role=='DEP':
                    if 0<=y<dep_shape[2] and 0<=x<dep_shape[3]:
                        dep[0,0,y,x]=v; filled['dep']+=1
    except FileNotFoundError:
        return None, None, None, {'error':'file_not_found'}
    return reg, cls, dep, filled

# ------------ Separated per-type files loader (p3_dfl.txt, p3_cls.txt, p3_dep.txt etc.) ------------
def load_level_separated(head_dir: str, level_tag: str, reg_shape, cls_shape, dep_shape):
    """Load separate DFL/CLS/DEP files for a level.
    Expected filenames: {level_tag}_dfl.txt, {level_tag}_cls.txt, {level_tag}_dep.txt
    Line formats supported:
      DFL file: either 'c y x val' or 'DFL c y x val'
      CLS file: either 'c y x val' or 'CLS c y x val'
      DEP file: either 'y x val' or 'DEP 0 y x val'
    """
    reg = np.zeros(reg_shape, dtype=np.float32)
    cls = np.zeros(cls_shape, dtype=np.float32)
    dep = np.zeros(dep_shape, dtype=np.float32)
    filled = {'reg':0,'cls':0,'dep':0}
    # Helper to parse
    def parse_file(fpath, kind):
        if not os.path.exists(fpath):
            return
        # First pass: detect if file is flat list (after optional header comment) or structured lines.
        with open(fpath,'r') as f:
            lines = f.readlines()
        data_lines = [ln.strip() for ln in lines if ln.strip() and not ln.startswith('#')]
        # Detect flat mode: all lines are single float tokens
        flat_mode = all(len(dl.split())==1 for dl in data_lines)
        if flat_mode:
            # Convert all to floats and reshape based on shape meta we already have.
            try:
                vals = np.array([float(dl) for dl in data_lines], dtype=np.float32)
            except:
                return
            if kind=='DFL':
                expected = reg_shape[1]*reg_shape[2]*reg_shape[3]
                if vals.size != expected:
                    print(f"Size mismatch {fpath}: got {vals.size} expected {expected}")
                    return
                reg[0] = vals.reshape(reg_shape[1], reg_shape[2], reg_shape[3])
                filled['reg'] = int(expected)
            elif kind=='CLS':
                expected = cls_shape[1]*cls_shape[2]*cls_shape[3]
                if vals.size != expected:
                    print(f"Size mismatch {fpath}: got {vals.size} expected {expected}")
                    return
                cls[0] = vals.reshape(cls_shape[1], cls_shape[2], cls_shape[3])
                filled['cls'] = int(expected)
            else: # DEP
                expected = dep_shape[2]*dep_shape[3]
                if vals.size != expected:
                    print(f"Size mismatch {fpath}: got {vals.size} expected {expected}")
                    return
                dep[0,0] = vals.reshape(dep_shape[2], dep_shape[3])
                filled['dep'] = int(expected)
            return
        # Structured mode fallback
        for line in data_lines:
            parts = line.split()
            if parts[0] in ('DFL','CLS','DEP'): parts = parts[1:]
            if kind in ('DFL','CLS'):
                if len(parts)!=4: continue
                c,y,x,val = parts
                try:
                    c=int(c); y=int(y); x=int(x); v=float(val)
                except: continue
                if kind=='DFL':
                    if 0<=c<reg_shape[1] and 0<=y<reg_shape[2] and 0<=x<reg_shape[3]:
                        reg[0,c,y,x]=v; filled['reg']+=1
                else:
                    if 0<=c<cls_shape[1] and 0<=y<cls_shape[2] and 0<=x<cls_shape[3]:
                        cls[0,c,y,x]=v; filled['cls']+=1
            else: # DEP
                if len(parts)==3:
                    y,x,val = parts
                elif len(parts)==4:
                    _,y,x,val = parts
                else:
                    continue
                try:
                    y=int(y); x=int(x); v=float(val)
                except: continue
                if 0<=y<dep_shape[2] and 0<=x<dep_shape[3]:
                    dep[0,0,y,x]=v; filled['dep']+=1
    parse_file(os.path.join(head_dir, f"{level_tag}_dfl.txt"), 'DFL')
    parse_file(os.path.join(head_dir, f"{level_tag}_cls.txt"), 'CLS')
    parse_file(os.path.join(head_dir, f"{level_tag}_dep.txt"), 'DEP')
    return reg, cls, dep, filled

# ------------ Diff & diagnostics ------------
def diff_stats(ref, dump, mask=None):
    if ref is None or dump is None: return {}
    if mask is None:
        mask = np.ones_like(dump, dtype=bool)
    # Only positions where dump has non-zero (to focus on provided entries)
    nz = dump!=0
    m = mask & nz
    if not np.any(m):
        return {'count':0}
    d = np.abs(ref - dump)[m]
    return {
        'count': int(d.size),
        'mean': float(d.mean()),
        'median': float(np.median(d)),
        'max': float(d.max()),
        'ref_mean_abs': float(np.mean(np.abs(ref[m]))),
    }

def sample_nonzero_cells(reg_dump, max_samples=5):
    # Find (c,y,x) groups where any channel non-zero -> pick unique (y,x)
    _,C,H,W = reg_dump.shape
    mask_cell = (np.sum(np.abs(reg_dump[0]), axis=0)!=0)  # HxW
    coords = np.argwhere(mask_cell)
    if coords.size==0:
        return []
    if len(coords)>max_samples:
        # prioritize center-ish cells by distance from center
        cy, cx = H/2, W/2
        dists = [((y-cy)**2 + (x-cx)**2, (y,x)) for y,x in coords]
        dists.sort()
        coords = [p for _,p in dists[:max_samples]]
    else:
        coords = [tuple(c) for c in coords]
    return coords

def print_cell_report(level, y, x, onnx_reg, dump_reg):
    C = onnx_reg.shape[1]
    ref_vec = onnx_reg[0,:,y,x]
    dump_vec = dump_reg[0,:,y,x]
    # group by 16
    def group(v):
        return [" ".join(f"{vv:.1f}" for vv in v[g*16:(g+1)*16]) for g in range(4)]
    ref_groups = group(ref_vec)
    dump_groups = group(dump_vec)
    print(f"[L{level}] Cell (y={y},x={x}) DFL groups (ref vs dump)")
    for g in range(4):
        print(f"  Group{g} REF: {ref_groups[g]}")
        print(f"          DMP: {dump_groups[g]}")

# Heuristic test: does dump_reg look like it's sampled from wrong index order (y,x,c)?
# We'll compute correlation between ref_vec and dump_vec, and between ref_vec and alternative mapping attempt.
# Alternative mapping (if bug) roughly random, so correlation low.

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('wide', nargs='?', help='Wide image path')
    ap.add_argument('narrow', nargs='?', help='Narrow image path')
    ap.add_argument('--head-base', type=str, help='Base path or head0.txt')
    ap.add_argument('--head-files', nargs=3, metavar=('H0','H1','H2'))
    ap.add_argument('--max-samples', type=int, default=5)
    ap.add_argument('--save-report', action='store_true')
    ap.add_argument('--dfl-order', type=str, default='ltrb', help='(For annotation only here) order string of ltrb')
    ap.add_argument('--wide-dir', type=str, help='Override wide images directory')
    ap.add_argument('--narrow-dir', type=str, help='Override narrow images directory')
    ap.add_argument('--image-name', type=str, help='Specific image filename to use (present in both dirs)')
    ap.add_argument('--head-dir', type=str, help='Directory containing p3_dfl/cls/dep.txt etc.')
    ap.add_argument('--visualize', action='store_true', help='Decode and visualize detections for ONNX vs dump heads')
    ap.add_argument('--conf-thres', type=float, default=CONF_THRES_DEFAULT)
    ap.add_argument('--iou-thres', type=float, default=IOU_THRES_DEFAULT)
    args = ap.parse_args()

    # Resolve image paths
    wide_dir = args.wide_dir or WIDE_DIR
    narrow_dir = args.narrow_dir or NARROW_DIR
    # Choose image according to priority: explicit positional paths > explicit image-name > default hard-coded > random fallback
    if args.wide and args.narrow and os.path.exists(args.wide) and os.path.exists(args.narrow):
        wide_img = args.wide
        narrow_img = args.narrow
        base_name = Path(wide_img).stem
    else:
        chosen_name = None
        if args.image_name:
            candidate_w = os.path.join(wide_dir, args.image_name)
            candidate_n = os.path.join(narrow_dir, args.image_name)
            if os.path.exists(candidate_w) and os.path.exists(candidate_n):
                chosen_name = args.image_name
        if chosen_name is None:
            # Try default image name
            candidate_w = os.path.join(wide_dir, DEFAULT_IMAGE_NAME)
            candidate_n = os.path.join(narrow_dir, DEFAULT_IMAGE_NAME)
            if os.path.exists(candidate_w) and os.path.exists(candidate_n):
                chosen_name = DEFAULT_IMAGE_NAME
        if chosen_name is None:
            # Fallback random
            try:
                files = [f for f in os.listdir(wide_dir) if f.lower().endswith('.jpg')]
            except FileNotFoundError:
                print(f"Wide dir not found: {wide_dir}")
                return
            if not files:
                print('No images found.')
                return
            chosen_name = random.choice(files)
        wide_img = os.path.join(wide_dir, chosen_name)
        narrow_img = os.path.join(narrow_dir, chosen_name)
        base_name = Path(chosen_name).stem
    print(f"Using images: {wide_img} | {narrow_img}")

    # Head file paths
    use_separated = False
    if args.head_dir:
        if not os.path.isdir(args.head_dir):
            print('Provided --head-dir does not exist')
            return
        use_separated = True
        print(f"Using separated head dir: {args.head_dir}")
    else:
        if args.head_files:
            head_paths = list(args.head_files)
        elif args.head_base:
            base = args.head_base
            if base.endswith('head0.txt'): base = base[:-len('0.txt')]
            elif base.endswith('.txt'):
                for d in ('0','1','2'):
                    suf = d + '.txt'
                    if base.endswith(suf):
                        base = base[:-len(suf)]
                        break
            head_paths = [f"{base}{i}.txt" for i in range(3)]
        else:
            print('Need --head-dir or --head-base or --head-files')
            return
        print('Head files:', head_paths)

    # Load images
    wide_tensor, (orig_h, orig_w) = preprocess_image(wide_img, IMAGE_SIZE)
    narrow_tensor, _ = preprocess_image(narrow_img, IMAGE_SIZE)

    # ONNX inference
    print('Loading ONNX model...')
    sess = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
    start = time.time()
    outputs = sess.run(None, {'images_wide': wide_tensor, 'images_narrow': narrow_tensor})
    inf_time = time.time() - start
    print(f'ONNX inference done in {inf_time:.3f}s')

    # Load head dumps
    loaded = []
    coverage = []
    if use_separated:
        level_tags = ['p3','p4','p5']  # order must align with LEVEL_SHAPES and ONNX outputs
        for i, (tag, shp) in enumerate(zip(level_tags, LEVEL_SHAPES)):
            reg, cls, dep, fill = load_level_separated(args.head_dir, tag, shp['reg'], shp['cls'], shp['dep'])
            coverage.append({'level': tag, **fill})
            loaded.extend([reg, cls, dep])
    else:
        for i, shp in enumerate(LEVEL_SHAPES):
            reg, cls, dep, fill = load_head_text(head_paths[i], shp['reg'], shp['cls'], shp['dep'])
            coverage.append({'file': head_paths[i], **fill})
            if reg is None:
                print(f"Missing file {head_paths[i]} - abort")
                return
            loaded.extend([reg, cls, dep])

    # Compare per level
    report_lines = []
    print('\n=== DIFF STATS (non-zero dump entries) ===')
    report_lines.append('Level,Type,count,mean,median,max,ref_mean_abs')
    for li in range(3):
        ref_reg = outputs[li*3]      # [1,64,H,W]
        ref_cls = outputs[li*3 + 1]  # [1,28,H,W]
        ref_dep = outputs[li*3 + 2]  # [1,1,H,W]
        dump_reg = loaded[li*3]
        dump_cls = loaded[li*3 + 1]
        dump_dep = loaded[li*3 + 2]
        s_reg = diff_stats(ref_reg, dump_reg)
        s_cls = diff_stats(ref_cls, dump_cls)
        s_dep = diff_stats(ref_dep, dump_dep)
        for tag, s in [('reg',s_reg),('cls',s_cls),('dep',s_dep)]:
            if 'count' not in s or s['count']==0:
                print(f"[L{li}] {tag}: no entries")
                report_lines.append(f"{li},{tag},0,,,,")
            else:
                print(f"[L{li}] {tag}: n={s['count']} mean={s['mean']:.4f} median={s['median']:.4f} max={s['max']:.4f}")
                report_lines.append(f"{li},{tag},{s['count']},{s['mean']:.6f},{s['median']:.6f},{s['max']:.6f},{s['ref_mean_abs']:.6f}")
        # Sample cells
        cells = sample_nonzero_cells(dump_reg, max_samples=args.max_samples)
        for (y,x) in cells:
            print_cell_report(li, y, x, ref_reg, dump_reg)

    # Coverage info
    print('\n=== COVERAGE ===')
    for cov in coverage:
        if 'reg' in cov:  # compatibility safeguard
            print(cov)
        else:
            print(cov)

    # Save report
    if args.save_report:
        out_dir = Path(OUTPUT_DIR)
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"diff_report_{base_name}.csv"
        with open(out_path,'w') as f:
            f.write('\n'.join(report_lines))
        print('Report saved:', out_path)

    # Simple heuristic suggestions
    print('\n=== HEURISTIC CHECKS ===')
    # Check if mean diff >> typical magnitude of logits
    for li in range(3):
        ref_reg = outputs[li*3]
        dump_reg = loaded[li*3]
        # Compare correlation for first sampled cell
        cells = sample_nonzero_cells(dump_reg, max_samples=1)
        if not cells:
            print(f"[L{li}] no sample cell for correlation test")
            continue
        y,x = cells[0]
        ref_vec = ref_reg[0,:,y,x]
        dump_vec = dump_reg[0,:,y,x]
        # Pearson correlation
        if np.std(ref_vec)>0 and np.std(dump_vec)>0:
            corr = np.corrcoef(ref_vec, dump_vec)[0,1]
            print(f"[L{li}] sample cell (y={y},x={x}) corr={corr:.4f}")
            if corr < 0.25:
                print(f"  -> Low correlation: indicates likely indexing bug in C dump (e.g., using (y,x,c) order)")
        else:
            print(f"[L{li}] sample cell insufficient variance for corr test")

    print('\nDone.')

    # ================= Visualization (optional) =================
    if args.visualize:
        print('\n=== VISUALIZATION ===')
        # Wide image load (BGR)
        wide_bgr = cv2.imread(str(wide_img))
        if wide_bgr is None:
            print('Cannot load wide image for visualization, skipping.')
            return
        orig_h, orig_w = wide_bgr.shape[:2]

        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))

        def softmax_expect(vals):
            m = np.max(vals)
            e = np.exp(vals - m)
            s = np.sum(e)
            if s <= 0: s = 1e-6
            p = e / s
            return float((p * np.arange(len(vals), dtype=np.float32)).sum())

        def decode(outputs_list, src_tag):
            detections = []
            for li in range(3):
                reg = outputs_list[li*3]
                cls = outputs_list[li*3 + 1]
                dep = outputs_list[li*3 + 2]
                _, Creg, H, W = reg.shape
                stride = IMAGE_SIZE / H
                num_bins = Creg // 4
                for y in range(H):
                    for x in range(W):
                        # classification
                        cls_logits = cls[0, :, y, x]
                        cls_scores = sigmoid(cls_logits)
                        class_id = int(np.argmax(cls_scores))
                        conf = float(cls_scores[class_id])
                        if conf < args.conf_thres:
                            continue
                        # DFL groups order assumed l,t,r,b (ltrb)
                        g0 = reg[0, 0*num_bins:1*num_bins, y, x]
                        g1 = reg[0, 1*num_bins:2*num_bins, y, x]
                        g2 = reg[0, 2*num_bins:3*num_bins, y, x]
                        g3 = reg[0, 3*num_bins:4*num_bins, y, x]
                        d_left = softmax_expect(g0)
                        d_top = softmax_expect(g1)
                        d_right = softmax_expect(g2)
                        d_bottom = softmax_expect(g3)
                        cx = (x + 0.5) * stride
                        cy = (y + 0.5) * stride
                        x1 = cx - d_left * stride
                        y1 = cy - d_top * stride
                        x2 = cx + d_right * stride
                        y2 = cy + d_bottom * stride
                        # clip to model space
                        x1 = max(0.0, min(x1, IMAGE_SIZE))
                        y1 = max(0.0, min(y1, IMAGE_SIZE))
                        x2 = max(0.0, min(x2, IMAGE_SIZE))
                        y2 = max(0.0, min(y2, IMAGE_SIZE))
                        # reverse letterbox
                        gain = min(IMAGE_SIZE / orig_w, IMAGE_SIZE / orig_h)
                        pad_w = (IMAGE_SIZE - orig_w * gain) * 0.5
                        pad_h = (IMAGE_SIZE - orig_h * gain) * 0.5
                        rx1 = (x1 - pad_w) / gain
                        rx2 = (x2 - pad_w) / gain
                        ry1 = (y1 - pad_h) / gain
                        ry2 = (y2 - pad_h) / gain
                        rx1 = max(0.0, min(rx1, orig_w))
                        rx2 = max(0.0, min(rx2, orig_w))
                        ry1 = max(0.0, min(ry1, orig_h))
                        ry2 = max(0.0, min(ry2, orig_h))
                        if (rx2 - rx1) < 2 or (ry2 - ry1) < 2:
                            continue
                        depth_logit = dep[0,0,y,x]
                        depth_norm = sigmoid(depth_logit)
                        detections.append([rx1, ry1, rx2, ry2, conf, class_id, depth_norm, li])
            # NMS
            detections.sort(key=lambda d: d[4], reverse=True)
            keep = []
            for det in detections:
                x1,y1,x2,y2,conf,cls_id,depth_norm,level = det
                suppress = False
                for kept in keep:
                    if kept[5] != cls_id:
                        continue
                    kx1,ky1,kx2,ky2 = kept[0],kept[1],kept[2],kept[3]
                    # IoU
                    ix1 = max(x1,kx1); iy1 = max(y1,ky1)
                    ix2 = min(x2,kx2); iy2 = min(y2,ky2)
                    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
                    inter = iw * ih
                    area = (x2-x1)*(y2-y1)
                    karea = (kx2-kx1)*(ky2-ky1)
                    denom = area + karea - inter + 1e-6
                    iou = inter/denom if denom>0 else 0.0
                    if iou > args.iou_thres:
                        suppress = True
                        break
                if not suppress:
                    keep.append(det)
            print(f"Decoded {len(keep)} detections for {src_tag}")
            return keep

        def draw(image, detections, tag, out_name):
            img = image.copy()
            palette = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]
            for det in detections:
                x1,y1,x2,y2,conf,cls_id,depth_norm,level = det
                color = palette[int(cls_id) % len(palette)]
                cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),color,2)
                depth_orig = depth_norm*(DEP_MAX-DEP_MIN)+DEP_MIN
                label = f"{tag} L{level} C{cls_id} {conf:.2f} D{depth_orig:.1f}"
                (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,0.45,1)
                yy = max(0,int(y1)-th-2)
                cv2.rectangle(img,(int(x1),yy),(int(x1)+tw,yy+th+2),color,-1)
                cv2.putText(img,label,(int(x1),yy+th),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1)
            out_path = Path(OUTPUT_DIR)/out_name
            Path(OUTPUT_DIR).mkdir(exist_ok=True)
            cv2.imwrite(str(out_path), img)
            print('Saved', out_path)

        # Build ONNX outputs list already have 'outputs'
        onnx_dets = decode(outputs, 'ONNX')
        draw(wide_bgr, onnx_dets, 'ONNX', 'onnx_detections.jpg')

        # Build dump outputs (replace tensors with loaded dump arrays if present)
        if len(loaded)==9:  # ensure full
            dump_outputs = []
            for i in range(3):
                dump_outputs.append(loaded[i*3])
                dump_outputs.append(loaded[i*3+1])
                dump_outputs.append(loaded[i*3+2])
            dump_dets = decode(dump_outputs, 'DUMP')
            draw(wide_bgr, dump_dets, 'DUMP', 'dump_detections.jpg')
        else:
            print('Dump tensors incomplete; skipping dump visualization.')

if __name__ == '__main__':
    main()
