#!/usr/bin/env python
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
Run inference with a dual-stream YOLOv8 model.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import cv2
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv8 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics import YOLO
from ultralytics.nn.modules.dual_model import DualStreamWrapper
from ultralytics.utils import LOGGER, colorstr


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'yolo11n.pt', help='model weights path')
    parser.add_argument('--wide', type=str, default=None, help='wide camera image or folder path')
    parser.add_argument('--narrow', type=str, default=None, help='narrow camera image or folder path')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--save', action='store_true', help='save results to files')
    parser.add_argument('--save-dir', type=str, default='runs/detect', help='directory to save results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class, i.e. --classes 0, or --classes 0 2 3')
    parser.add_argument('--verbose', action='store_true', help='verbose output')
    parser.add_argument('--stream', action='store_true', help='stream video inputs')
    return parser.parse_args()


def main(opt):
    """Run inference with specified options."""
    LOGGER.info(colorstr('Arguments: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    
    # Check inputs
    if not opt.wide or not opt.narrow:
        LOGGER.error('Must specify both --wide and --narrow inputs')
        return
    
    # Load model
    model = YOLO(opt.weights)
    LOGGER.info(f"Loaded {opt.weights} model")
    
    # Wrap the model for dual-stream processing
    model.model = DualStreamWrapper(model.model)
    LOGGER.info(f"Wrapped model for dual-stream processing")
    
    # Run inference
    if not opt.stream:
        # Standard dual-stream inference from files/folders
        LOGGER.info(f"Running dual-stream inference with wide={opt.wide}, narrow={opt.narrow}")
        results = model.predict(
            [opt.wide, opt.narrow], 
            conf=opt.conf_thres,
            iou=opt.iou_thres,
            max_det=opt.max_det,
            classes=opt.classes,
            verbose=opt.verbose,
            stream=False,
            save=opt.save,
            save_dir=opt.save_dir if opt.save_dir else None,
        )
        
        # Process and display results
        if opt.show:
            for r in results:
                img = r.plot()
                cv2.imshow("Dual-stream YOLOv8 Inference", img)
                if cv2.waitKey(0) & 0xFF == 27:  # ESC key
                    break
            cv2.destroyAllWindows()
    
    else:
        # Handle streaming video sources
        try:
            # Open video captures
            wide_cap = cv2.VideoCapture(eval(opt.wide) if opt.wide.isnumeric() else opt.wide)
            narrow_cap = cv2.VideoCapture(eval(opt.narrow) if opt.narrow.isnumeric() else opt.narrow)
            
            if not wide_cap.isOpened() or not narrow_cap.isOpened():
                LOGGER.error("Failed to open one or both video sources")
                return
                
            width = int(wide_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(wide_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = wide_cap.get(cv2.CAP_PROP_FPS)
            
            LOGGER.info(f"Streaming from dual video sources at {width}x{height} @ {fps}fps")
            
            # Create output video writer if saving
            writer = None
            if opt.save:
                os.makedirs(opt.save_dir, exist_ok=True)
                output_path = os.path.join(opt.save_dir, 'dual_stream_output.mp4')
                writer = cv2.VideoWriter(
                    output_path, 
                    cv2.VideoWriter_fourcc(*'mp4v'), 
                    fps, 
                    (width, height)
                )
                LOGGER.info(f"Saving output to {output_path}")
            
            # Main processing loop
            while wide_cap.isOpened() and narrow_cap.isOpened():
                ret1, wide_frame = wide_cap.read()
                ret2, narrow_frame = narrow_cap.read()
                
                if not ret1 or not ret2:
                    LOGGER.info("End of video stream")
                    break
                
                # Run inference on the frames
                results = model.predict(
                    [wide_frame, narrow_frame],
                    conf=opt.conf_thres,
                    iou=opt.iou_thres,
                    max_det=opt.max_det,
                    classes=opt.classes,
                    verbose=False,
                )
                
                # Get the processed frame with detections
                img = results[0].plot()
                
                # Display the result
                if opt.show:
                    cv2.imshow("Dual-stream YOLOv8 Inference", img)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                        break
                
                # Save to video if requested
                if writer is not None:
                    writer.write(img)
            
            # Clean up
            if writer is not None:
                writer.release()
            wide_cap.release()
            narrow_cap.release()
            cv2.destroyAllWindows()
            
        except Exception as e:
            LOGGER.error(f"Error during video processing: {e}")
    
    LOGGER.info(f"Inference complete")


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
