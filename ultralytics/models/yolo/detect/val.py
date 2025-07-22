# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
from pathlib import Path

import numpy as np
import torch

from ultralytics.data import build_dataloader, build_yolo_dataset, converter
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.utils.plotting import output_to_target, plot_images


class DetectionValidator(BaseValidator):
    """
    A class extending the BaseValidator class for validation based on a detection model.

    This class implements validation functionality specific to object detection tasks, including metrics calculation,
    prediction processing, and visualization of results.

    Attributes:
        nt_per_class (np.ndarray): Number of targets per class.
        nt_per_image (np.ndarray): Number of targets per image.
        is_coco (bool): Whether the dataset is COCO.
        is_lvis (bool): Whether the dataset is LVIS.
        class_map (list): Mapping from model class indices to dataset class indices.
        metrics (DetMetrics): Object detection metrics calculator.
        iouv (torch.Tensor): IoU thresholds for mAP calculation.
        niou (int): Number of IoU thresholds.
        lb (list): List for storing ground truth labels for hybrid saving.
        jdict (list): List for storing JSON detection results.
        stats (dict): Dictionary for storing statistics during validation.

    Examples:
        >>> from ultralytics.models.yolo.detect import DetectionValidator
        >>> args = dict(model="yolo11n.pt", data="coco8.yaml")
        >>> validator = DetectionValidator(args=args)
        >>> validator()
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """
        Initialize detection validator with necessary variables and settings.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): Dataloader to use for validation.
            save_dir (Path, optional): Directory to save results.
            pbar (Any, optional): Progress bar for displaying progress.
            args (dict, optional): Arguments for the validator.
            _callbacks (list, optional): List of callback functions.
        """
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        
        # 🔧 dual_stream 설정 확인
        # if hasattr(self.args, 'dual_stream') and self.args.dual_stream:
        #     print(f"🔍 VAL: DetectionValidator initialized with dual_stream=True")
        # else:
        #     print(f"🔍 VAL: DetectionValidator initialized with dual_stream=False")
        
        self.nt_per_class = None
        self.nt_per_image = None
        self.is_coco = False
        self.is_lvis = False
        self.class_map = None
        self.args.task = "detect"
        self.metrics = DetMetrics(save_dir=self.save_dir)
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()

    def preprocess(self, batch):
        """
        Preprocess batch of images for YOLO validation.
        """
        # print(f"🔍 VAL: Batch img type: {type(batch['img'])}")
        # print(f"🔍 VAL: Batch img shape: {batch['img'].shape}")
        
        # ✅ Dual stream 처리 개선
        if isinstance(batch["img"], tuple):
            # print("✅ VAL: Dual stream tuple detected!")
            wide_batch, narrow_batch = batch["img"]
            batch["img"] = torch.stack([wide_batch, narrow_batch], dim=1)
        elif isinstance(batch["img"], list) and len(batch["img"]) == 2:
            # print("✅ VAL: Dual stream list detected!")
            wide_batch, narrow_batch = batch["img"]
            batch["img"] = torch.stack([wide_batch, narrow_batch], dim=1)
        elif batch["img"].dim() == 5 and batch["img"].shape[1] == 2:
            # print("✅ VAL: Dual stream tensor already formatted!")
            pass
        else:
            # print("❌ VAL: Single stream detected - this may cause issues!")
            pass
        
        # print(f"🔍 VAL: Final batch img shape: {batch['img'].shape}")
        
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        
        for k in ["batch_idx", "cls", "bboxes"]:
            batch[k] = batch[k].to(self.device)

        return batch

    def init_metrics(self, model):
        """
        Initialize evaluation metrics for YOLO detection validation.

        Args:
            model (torch.nn.Module): Model to validate.
        """
        val = self.data.get(self.args.split, "")  # validation path
        self.is_coco = (
            isinstance(val, str)
            and "coco" in val
            and (val.endswith(f"{os.sep}val2017.txt") or val.endswith(f"{os.sep}test-dev2017.txt"))
        )  # is COCO
        self.is_lvis = isinstance(val, str) and "lvis" in val and not self.is_coco  # is LVIS
        self.class_map = converter.coco80_to_coco91_class() if self.is_coco else list(range(1, len(model.names) + 1))
        self.args.save_json |= self.args.val and (self.is_coco or self.is_lvis) and not self.training  # run final val
        self.names = model.names
        self.nc = len(model.names)
        self.end2end = getattr(model, "end2end", False)
        self.metrics.names = self.names
        self.metrics.plot = self.args.plots
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf)
        self.seen = 0
        self.jdict = []
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])

    def get_desc(self):
        """Return a formatted string summarizing class metrics of YOLO model."""
        return ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)")

    def postprocess(self, preds):
        """
        Apply Non-maximum suppression to prediction outputs.

        Args:
            preds (torch.Tensor): Raw predictions from the model.

        Returns:
            (List[torch.Tensor]): Processed predictions after NMS.
        """
        # print(f"DEBUG: ===== VAL POSTPROCESS DEBUG =====")
        # print(f"DEBUG: Postprocess preds type: {type(preds)}")
        
        # ✅ 표준 YOLO 출력 처리: (inference_output, raw_predictions) 또는 depth 포함된 경우
        if isinstance(preds, tuple):
            if len(preds) == 2:
                # Case 1: 표준 YOLO format (inference_output, raw_predictions)
                # Case 2: Depth estimation format (detection_preds, depth_preds)
                inference_output, second_output = preds
                # print(f"DEBUG: Tuple format with length 2")
                # print(f"DEBUG: Inference output type: {type(inference_output)}")
                # print(f"DEBUG: Second output type: {type(second_output)}")
                
                # inference output을 NMS에 사용 (depth는 mAP 계산에 사용하지 않음)
                preds = inference_output
            elif len(preds) == 3:
                # Case 3: 확장된 경우 (detection_preds, depth_preds, other_output)
                inference_output = preds[0]
                preds = inference_output
            else:
                # Default: 첫 번째 요소를 사용
                preds = preds[0]
        
        # if isinstance(preds, (list, tuple)):
        #     print(f"DEBUG: Postprocess preds length: {len(preds)}")
        #     if len(preds) > 0 and hasattr(preds[0], 'shape'):
        #         print(f"DEBUG: First pred shape: {preds[0].shape}")
        #         print(f"DEBUG: First pred sample values: {preds[0][0, :10] if len(preds[0]) > 0 else 'Empty'}")
        # else:
        #     if hasattr(preds, 'shape'):
                # print(f"DEBUG: Postprocess preds shape: {preds.shape}")
                # print(f"DEBUG: Preds sample values: {preds[0, 0, :10] if preds.numel() > 0 else 'Empty'}")
        
        # print(f"DEBUG: Conf threshold: {self.args.conf}")
        # print(f"DEBUG: IoU threshold: {self.args.iou}")
        
        processed = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            nc=0 if self.args.task == "detect" else self.nc,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            end2end=self.end2end,
            rotated=self.args.task == "obb",
        )
        
        # print(f"DEBUG: NMS output type: {type(processed)}")
        # print(f"DEBUG: NMS output length: {len(processed) if isinstance(processed, (list, tuple)) else 'N/A'}")
        # if isinstance(processed, (list, tuple)) and len(processed) > 0:
        #     print(f"DEBUG: First NMS result shape: {processed[0].shape if hasattr(processed[0], 'shape') else len(processed[0])}")
        #     print(f"DEBUG: First NMS result detections: {len(processed[0]) if hasattr(processed[0], '__len__') else 'N/A'}")
        #     if len(processed[0]) > 0:
        #         print(f"DEBUG: First detection: {processed[0][0] if len(processed[0]) > 0 else 'Empty'}")
        # print(f"DEBUG: ===== END VAL POSTPROCESS DEBUG =====")
        
        return processed

    def _prepare_batch(self, si, batch):
        """Prepare a batch for training or inference."""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        
        # Handle dual stream image dimensions
        if batch['img'].dim() == 5:  # [B, 2, C, H, W]
            imgsz = batch['img'].shape[-2:]
        else:  # [B, C, H, W]  
            imgsz = batch['img'].shape[-2:]
        
        # ✅ 핵심 수정: ratio_pad 정보 사용
        ratio_pad = batch.get("ratio_pad", [None])[si] if "ratio_pad" in batch else None
        
        # Convert ori_shape to list for compatibility
        if isinstance(ori_shape, torch.Tensor):
            ori_shape = ori_shape.tolist()
        
        # ✅ 표준 방식: ground truth 좌표 처리  
        if len(bbox) > 0:
            # 1단계: 정규화된 xywh를 모델 입력 크기의 pixel 좌표로 변환
            bbox = bbox.clone()
            bbox[:, [0, 2]] *= imgsz[1]  # x coordinates
            bbox[:, [1, 3]] *= imgsz[0]  # y coordinates
            
            # 2단계: ✅ 표준 scale_boxes로 원본 이미지 크기로 변환 (패딩 고려)
            bbox = ops.xywh2xyxy(bbox)  # xywh → xyxy 
            ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad, xywh=False)
        
        return {
            "cls": cls,
            "bbox": bbox,
            "ori_shape": ori_shape,
            "imgsz": imgsz,
            "ratio_pad": ratio_pad
        }

    # detect/val.py
    def _prepare_pred(self, pred, pbatch):
        """Prepare predictions for evaluation against ground truth."""
        predn = pred.clone()
        
        # print(f"DEBUG: ===== _prepare_pred DEBUG =====")
        # print(f"DEBUG: Input pred shape: {pred.shape}")
        # print(f"DEBUG: Input pred bbox range: x={pred[:, 0].min():.1f}-{pred[:, 2].max():.1f}, y={pred[:, 1].min():.1f}-{pred[:, 3].max():.1f}")
        # print(f"DEBUG: pbatch keys: {pbatch.keys()}")
        # print(f"DEBUG: pbatch['imgsz']: {pbatch['imgsz']}")
        # print(f"DEBUG: pbatch['ori_shape']: {pbatch['ori_shape']}")
        # print(f"DEBUG: pbatch['ratio_pad']: {pbatch['ratio_pad']}")
        
        # 🔧 scale_boxes 호출 전후 비교
        predn_before = predn[:, :4].clone()
        
        ops.scale_boxes(
            pbatch["imgsz"],      # 현재 이미지 크기 (예: [640, 640])
            predn[:, :4],         # 예측된 bbox 좌표
            pbatch["ori_shape"],  # 원본 이미지 크기
            ratio_pad=None        # 패딩 정보는 자동 계산
        )
        
        # print(f"DEBUG: Before scale_boxes: x={predn_before[:, 0].min():.1f}-{predn_before[:, 2].max():.1f}, y={predn_before[:, 1].min():.1f}-{predn_before[:, 3].max():.1f}")
        # print(f"DEBUG: After scale_boxes:  x={predn[:, 0].min():.1f}-{predn[:, 2].max():.1f}, y={predn[:, 1].min():.1f}-{predn[:, 3].max():.1f}")
        # print(f"DEBUG: ===== END _prepare_pred DEBUG =====")
        
        return predn

    def update_metrics(self, preds, batch):
        """Update metrics with new predictions and ground truth."""
        # print(f"DEBUG: ===== UPDATE METRICS DEBUG =====")
        # print(f"DEBUG: Preds length: {len(preds)}")
        # print(f"DEBUG: Batch keys: {list(batch.keys())}")
        
        total_detections = sum(len(pred) for pred in preds)
        # print(f"DEBUG: Total detections across all images: {total_detections}")
        
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            # print(f"DEBUG: Image {si}: {npr} predictions")
            
            # if npr > 0:
            #     print(f"DEBUG: Image {si} first prediction: {pred[0]}")
            #     print(f"DEBUG: Image {si} confidence range: {pred[:, 4].min():.4f} - {pred[:, 4].max():.4f}")
            #     print(f"DEBUG: Image {si} classes: {pred[:, 5].unique()}")
            
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)
            # print(f"DEBUG: Image {si}: {nl} ground truth objects")
            
            # if nl > 0:
            #     print(f"DEBUG: Image {si} GT classes: {cls}")
            #     print(f"DEBUG: Image {si} GT bbox shape: {bbox.shape}")
            #     print(f"DEBUG: Image {si} GT bbox range: x={bbox[:, [0,2]].min():.1f}-{bbox[:, [0,2]].max():.1f}, y={bbox[:, [1,3]].min():.1f}-{bbox[:, [1,3]].max():.1f}")
            
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = self._prepare_pred(pred, pbatch)
            
            # print(f"DEBUG: Image {si} predn shape: {predn.shape}")
            # if len(predn) > 0:
            #     print(f"DEBUG: Image {si} predn bbox range: x={predn[:, [0,2]].min():.1f}-{predn[:, [0,2]].max():.1f}, y={predn[:, [1,3]].min():.1f}-{predn[:, [1,3]].max():.1f}")
            
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            # Evaluate
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
                # print(f"DEBUG: Image {si} TP shape: {stat['tp'].shape}, TP sum: {stat['tp'].sum()}")
            if self.args.plots:
                self.confusion_matrix.process_batch(predn, bbox, cls)
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            if self.args.save_txt:
                self.save_one_txt(
                    predn,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f"{Path(batch['im_file'][si]).stem}.txt",
                )
        
        # print(f"DEBUG: ===== END UPDATE METRICS DEBUG =====")

    def finalize_metrics(self, *args, **kwargs):
        """
        Set final values for metrics speed and confusion matrix.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.
        """
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix

    def get_stats(self):
        """
        Calculate and return metrics statistics.

        Returns:
            (dict): Dictionary containing metrics results.
        """
        stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}  # to numpy
        self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=self.nc)
        self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=self.nc)
        stats.pop("target_img", None)
        if len(stats):
            self.metrics.process(**stats, on_plot=self.on_plot)
        return self.metrics.results_dict

    def print_results(self):
        """Print training/validation set metrics per class."""
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # print format
        LOGGER.info(pf % ("all", self.seen, self.nt_per_class.sum(), *self.metrics.mean_results()))
        if self.nt_per_class.sum() == 0:
            LOGGER.warning(f"no labels found in {self.args.task} set, can not compute metrics without labels")

        # Print results per class
        if self.args.verbose and not self.training and self.nc > 1 and len(self.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                LOGGER.info(
                    pf % (self.names[c], self.nt_per_image[c], self.nt_per_class[c], *self.metrics.class_result(i))
                )

        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(
                    save_dir=self.save_dir, names=self.names.values(), normalize=normalize, on_plot=self.on_plot
                )

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """Return correct prediction matrix."""
        # print(f"DEBUG: ===== _process_batch IoU DEBUG =====")
        # print(f"DEBUG: detections shape: {detections.shape}")
        # print(f"DEBUG: gt_bboxes shape: {gt_bboxes.shape}")
        
        if len(detections) > 0 and len(gt_bboxes) > 0:
            # print(f"DEBUG: First detection: {detections[0]}")
            # print(f"DEBUG: First GT bbox: {gt_bboxes[0]}")
            # print(f"DEBUG: Detection classes: {detections[:, 5]}")
            # print(f"DEBUG: GT classes: {gt_cls}")
            
            iou = box_iou(gt_bboxes, detections[:, :4])
            # print(f"DEBUG: IoU matrix shape: {iou.shape}")
            # print(f"DEBUG: IoU matrix:\n{iou}")
            # print(f"DEBUG: Max IoU: {iou.max():.4f}")
            
            # ✅ 클래스 매칭 여부 확인 (에러 수정)
            for i, det_cls in enumerate(detections[:, 5]):
                matching_gts = (gt_cls == det_cls).nonzero().squeeze(-1)  # ✅ -1 추가
                # print(f"DEBUG: Detection {i} (class {det_cls}): matching GT indices {matching_gts}")
                
                # ✅ 0-d tensor 문제 해결
                if matching_gts.numel() > 0:  # ✅ len() 대신 numel() 사용
                    if matching_gts.dim() == 0:  # 스칼라인 경우
                        matching_gts = matching_gts.unsqueeze(0)  # 1D로 변환
                    best_iou = iou[matching_gts, i].max() if iou.dim() > 1 else iou[matching_gts]
                    # print(f"DEBUG: Best IoU for det {i}: {best_iou:.4f}")
        
        iou = box_iou(gt_bboxes, detections[:, :4])
        result = self.match_predictions(detections[:, 5], gt_cls, iou)
        # print(f"DEBUG: match_predictions result shape: {result.shape}")
        # print(f"DEBUG: match_predictions TP count: {result.sum()}")
        # print(f"DEBUG: ===== END _process_batch IoU DEBUG =====")
        
        return result

    def build_dataset(self, img_path, mode="val", batch=None):
        """Build YOLO Dataset with proper dual_stream handling."""
        
        # 🔧 dual_stream 설정 확인 및 전달
        has_dual_stream = getattr(self.args, 'dual_stream', False)
        
        # args에서 dual_stream 플래그 확인
        if has_dual_stream:
            self.args.dual_stream = True
        # 데이터셋 YAML에서 dual stream 확인
        elif self.data and any(k.endswith(('_wide', '_narrow')) for k in self.data.keys()):
            self.args.dual_stream = True
            has_dual_stream = True
        
        # multi_modal도 dual_stream일 때 활성화
        original_multi_modal = getattr(self.args, 'multi_modal', False)
        if has_dual_stream:
            self.args.multi_modal = True
        
        dataset = build_yolo_dataset(
            self.args, 
            img_path, 
            batch, 
            self.data, 
            mode=mode, 
            stride=self.stride
        )
        
        # 원래 값 복원
        self.args.multi_modal = original_multi_modal
        
        return dataset

    def get_dataloader(self, dataset_path, batch_size):
        """
        Construct and return dataloader.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int): Size of each batch.

        Returns:
            (torch.utils.data.DataLoader): Dataloader for validation.
        """
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)  # return dataloader

    def plot_val_samples(self, batch, ni):
        """
        Plot validation image samples.

        Args:
            batch (dict): Batch containing images and annotations.
            ni (int): Batch index.
        """
        # print(f"DEBUG: ===== PLOT VAL SAMPLES DEBUG =====")
        # print(f"DEBUG: Plot validation batch img shape: {batch['img'].shape}")
        
        # Dual stream에서 wide stream만 시각화
        img_to_plot = batch["img"]
        if batch["img"].dim() == 5 and batch["img"].shape[1] == 2:
            # print("DEBUG: Using wide stream for validation plot")
            img_to_plot = batch["img"][:, 0]  # Wide stream만 사용
        
        # print(f"DEBUG: Image to plot shape: {img_to_plot.shape}")
        # print(f"DEBUG: Image to plot range: min={img_to_plot.min():.4f}, max={img_to_plot.max():.4f}")
        # print(f"DEBUG: ===== END PLOT VAL SAMPLES DEBUG =====")
        
        plot_images(
            img_to_plot,  # 수정된 부분
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        """
        Plot predicted bounding boxes on input images and save the result.

        Args:
            batch (dict): Batch containing images and annotations.
            preds (List[torch.Tensor]): List of predictions from the model.
            ni (int): Batch index.
        """
        # print(f"DEBUG: ===== PLOT PREDICTIONS DEBUG =====")
        # print(f"DEBUG: Plot predictions batch img shape: {batch['img'].shape}")
        
        # Dual stream에서 wide stream만 시각화
        img_to_plot = batch["img"]
        if batch["img"].dim() == 5 and batch["img"].shape[1] == 2:
            # print("DEBUG: Using wide stream for prediction plot")
            img_to_plot = batch["img"][:, 0]  # Wide stream만 사용
        
        # print(f"DEBUG: Image to plot shape: {img_to_plot.shape}")
        # print(f"DEBUG: ===== END PLOT PREDICTIONS DEBUG =====")
        
        plot_images(
            img_to_plot,  # 수정된 부분
            *output_to_target(preds, max_det=self.args.max_det),
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred

    def save_one_txt(self, predn, save_conf, shape, file):
        """
        Save YOLO detections to a txt file in normalized coordinates in a specific format.

        Args:
            predn (torch.Tensor): Predictions in the format (x1, y1, x2, y2, conf, class).
            save_conf (bool): Whether to save confidence scores.
            shape (tuple): Shape of the original image.
            file (Path): File path to save the detections.
        """
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=predn[:, :6],
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn, filename):
        """
        Serialize YOLO predictions to COCO json format.

        Args:
            predn (torch.Tensor): Predictions in the format (x1, y1, x2, y2, conf, class).
            filename (str): Image filename.
        """
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for p, b in zip(predn.tolist(), box.tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(p[5])],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(p[4], 5),
                }
            )

    def eval_json(self, stats):
        """
        Evaluate YOLO output in JSON format and return performance statistics.

        Args:
            stats (dict): Current statistics dictionary.

        Returns:
            (dict): Updated statistics dictionary with COCO/LVIS evaluation results.
        """
        if self.args.save_json and (self.is_coco or self.is_lvis) and len(self.jdict):
            pred_json = self.save_dir / "predictions.json"  # predictions
            anno_json = (
                self.data["path"]
                / "annotations"
                / ("instances_val2017.json" if self.is_coco else f"lvis_v1_{self.args.split}.json")
            )  # annotations
            pkg = "pycocotools" if self.is_coco else "lvis"
            LOGGER.info(f"\nEvaluating {pkg} mAP using {pred_json} and {anno_json}...")
            try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                for x in pred_json, anno_json:
                    assert x.is_file(), f"{x} file not found"
                check_requirements("pycocotools>=2.0.6" if self.is_coco else "lvis>=0.5.3")
                if self.is_coco:
                    from pycocotools.coco import COCO  # noqa
                    from pycocotools.cocoeval import COCOeval  # noqa

                    anno = COCO(str(anno_json))  # init annotations api
                    pred = anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path)
                    val = COCOeval(anno, pred, "bbox")
                else:
                    from lvis import LVIS, LVISEval

                    anno = LVIS(str(anno_json))  # init annotations api
                    pred = anno._load_json(str(pred_json))  # init predictions api (must pass string, not Path)
                    val = LVISEval(anno, pred, "bbox")
                val.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # images to eval
                val.evaluate()
                val.accumulate()
                val.summarize()
                if self.is_lvis:
                    val.print_results()  # explicitly call print_results
                # update mAP50-95 and mAP50
                stats[self.metrics.keys[-1]], stats[self.metrics.keys[-2]] = (
                    val.stats[:2] if self.is_coco else [val.results["AP"], val.results["AP50"]]
                )
                if self.is_lvis:
                    stats["metrics/APr(B)"] = val.results["APr"]
                    stats["metrics/APc(B)"] = val.results["APc"]
                    stats["metrics/APf(B)"] = val.results["APf"]
                    stats["fitness"] = val.results["AP"]
            except Exception as e:
                LOGGER.warning(f"{pkg} unable to run: {e}")
        return stats
