# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import math
import random
from copy import copy

import numpy as np
import torch.nn as nn
import torch

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.plotting import plot_images, plot_labels, plot_results
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first


class DetectionTrainer(BaseTrainer):
    """
    A class extending the BaseTrainer class for training based on a detection model.

    This trainer specializes in object detection tasks, handling the specific requirements for training YOLO models
    for object detection.

    Attributes:
        model (DetectionModel): The YOLO detection model being trained.
        data (dict): Dictionary containing dataset information including class names and number of classes.
        loss_names (Tuple[str]): Names of the loss components used in training (box_loss, cls_loss, dfl_loss).

    Methods:
        build_dataset: Build YOLO dataset for training or validation.
        get_dataloader: Construct and return dataloader for the specified mode.
        preprocess_batch: Preprocess a batch of images by scaling and converting to float.
        set_model_attributes: Set model attributes based on dataset information.
        get_model: Return a YOLO detection model.
        get_validator: Return a validator for model evaluation.
        label_loss_items: Return a loss dictionary with labeled training loss items.
        progress_string: Return a formatted string of training progress.
        plot_training_samples: Plot training samples with their annotations.
        plot_metrics: Plot metrics from a CSV file.
        plot_training_labels: Create a labeled training plot of the YOLO model.
        auto_batch: Calculate optimal batch size based on model memory requirements.

    Examples:
        >>> from ultralytics.models.yolo.detect import DetectionTrainer
        >>> args = dict(model="yolo11n.pt", data="coco8.yaml", epochs=3)
        >>> trainer = DetectionTrainer(overrides=args)
        >>> trainer.train()
    """

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build YOLO Dataset for training or validation.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`.

        Returns:
            (Dataset): YOLO dataset object configured for the specified mode.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=False, stride=gs)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """
        Construct and return dataloader for the specified mode.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int): Number of images per batch.
            rank (int): Process rank for distributed training.
            mode (str): 'train' for training dataloader, 'val' for validation dataloader.

        Returns:
            (DataLoader): PyTorch dataloader object.
        """
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        dataloader = build_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader
        
        # 🔧 데이터 로더 검증 및 디버깅
        # print(f"DEBUG: ===== {mode.upper()} DATALOADER CHECK =====")
        try:
            # 첫 번째 배치 확인
            first_batch = next(iter(dataloader))
            img = first_batch['img']
            
            # print(f"DEBUG: {mode} batch img shape: {img.shape}")
            # print(f"DEBUG: {mode} batch img range: {img.min():.6f} ~ {img.max():.6f}")
            # print(f"DEBUG: {mode} batch img dtype: {img.dtype}")
            # print(f"DEBUG: {mode} batch size: {len(first_batch['img'])}")
            
            # Dual stream 확인
            if img.dim() == 5 and img.shape[1] == 2:
                # print(f"DEBUG: ✅ {mode} dual stream data detected")
                # print(f"DEBUG: Wide stream range: {img[:, 0].min():.6f} ~ {img[:, 0].max():.6f}")
                # print(f"DEBUG: Narrow stream range: {img[:, 1].min():.6f} ~ {img[:, 1].max():.6f}")
                
                # 채널별 평균값 확인 (정규화 상태 판단)
                wide_mean = img[:, 0].mean(dim=(0, 2, 3))
                narrow_mean = img[:, 1].mean(dim=(0, 2, 3))
                # print(f"DEBUG: Wide stream RGB means: R={wide_mean[0]:.4f}, G={wide_mean[1]:.4f}, B={wide_mean[2]:.4f}")
                # print(f"DEBUG: Narrow stream RGB means: R={narrow_mean[0]:.4f}, G={narrow_mean[1]:.4f}, B={narrow_mean[2]:.4f}")
                
                # 정규화 상태 판단
            #     if img.max() <= 1.0 and img.min() >= 0:
            #         if wide_mean[0] > 0.4 and wide_mean[1] > 0.4 and wide_mean[2] > 0.4:
            #             print(f"DEBUG: ⚠️ Possible ImageNet normalization issue - means too high")
            #         else:
            #             print(f"DEBUG: ✅ Standard 0-1 normalization detected")
            #     elif img.max() > 1.0:
            #         print(f"DEBUG: ⚠️ Images not normalized (0-255 range)")
            #     elif img.min() < 0:
            #         print(f"DEBUG: ⚠️ ImageNet normalization detected (negative values)")
                
            # elif img.dim() == 4:
            #     print(f"DEBUG: ❌ {mode} single stream data (expected dual stream)")
            #     print(f"DEBUG: This may cause issues in dual stream model")
            # else:
            #     print(f"DEBUG: ❌ Unexpected tensor dimensions: {img.dim()}D")
            
            # 라벨 정보 확인
            # if 'cls' in first_batch:
            #     print(f"DEBUG: Classes shape: {first_batch['cls'].shape}")
            #     print(f"DEBUG: Unique classes: {torch.unique(first_batch['cls'])}")
            
            # if 'bboxes' in first_batch:
            #     print(f"DEBUG: Bboxes shape: {first_batch['bboxes'].shape}")
            #     print(f"DEBUG: Bboxes range: {first_batch['bboxes'].min():.4f} ~ {first_batch['bboxes'].max():.4f}")
            
            # 파일 경로 확인
            # if 'im_file' in first_batch:
            #     print(f"DEBUG: Sample file paths:")
            #     for i, path in enumerate(first_batch['im_file'][:3]):  # 첫 3개만
            #         print(f"DEBUG:   [{i}]: {path}")
            #         if '|' in path:
            #             print(f"DEBUG:       ✅ Dual stream path format detected")
            #         else:
            #             print(f"DEBUG:       ❌ Single stream path format")
            
            # # 데이터셋 정보
            # print(f"DEBUG: Dataset size: {len(dataset)}")
            # if hasattr(dataset, 'is_dual_stream'):
            #     print(f"DEBUG: Dataset dual_stream flag: {dataset.is_dual_stream}")
            
        except Exception as e:
            print(f"DEBUG: ❌ Error checking {mode} dataloader: {e}")
            import traceback
            traceback.print_exc()
        
        # print(f"DEBUG: ===== END {mode.upper()} DATALOADER CHECK =====")
        
        return dataloader

    def preprocess_batch(self, batch):
        """
        Preprocess a batch of images by scaling and converting to float.

        Args:
            batch (dict): Dictionary containing batch data with 'img' tensor.

        Returns:
            (dict): Preprocessed batch with normalized images.
        """
        # print(f"DEBUG: ===== DETECTION PREPROCESS_BATCH DEBUG =====")
        # print(f"DEBUG: Input batch img shape: {batch['img'].shape}")
        # print(f"DEBUG: Input batch img range: min={batch['img'].min():.4f}, max={batch['img'].max():.4f}")
        
        # Handle dual stream properly
        if batch["img"].dim() == 5 and batch["img"].shape[1] == 2:  # Dual stream: [B, 2, C, H, W]
            # print("DEBUG: ✅ DUAL STREAM detected in DetectionTrainer.preprocess_batch")
            
            # Move to device
            batch["img"] = batch["img"].to(self.device, non_blocking=True).float()
            
            # Check if already normalized (0-1 range) or needs normalization (0-255 range)
            if batch["img"].max() > 1.0:
                # print("DEBUG: Normalizing dual stream from 0-255 to 0-1")
                batch["img"] = batch["img"] / 255.0
            # else:
                # print("DEBUG: Dual stream already normalized (0-1 range), not dividing by 255")
            
            # print(f"DEBUG: After processing dual stream - shape: {batch['img'].shape}")
            # print(f"DEBUG: After processing dual stream - range: min={batch['img'].min():.4f}, max={batch['img'].max():.4f}")
            
        else:  # Single stream: [B, C, H, W]
            # print("DEBUG: Regular single stream detected")
            batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
            
            # print(f"DEBUG: After processing single stream - shape: {batch['img'].shape}")
            # print(f"DEBUG: After processing single stream - range: min={batch['img'].min():.4f}, max={batch['img'].max():.4f}")
        
        if self.args.multi_scale:
            imgs = batch["img"]
            sz = (
                random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
                // self.stride
                * self.stride
            )  # size
            
            # Handle dual stream multi-scale
            if imgs.dim() == 5 and imgs.shape[1] == 2:  # Dual stream: [B, 2, C, H, W]
                # print("DEBUG: Applying multi-scale to dual stream")
                sf = sz / max(imgs.shape[3:])  # scale factor based on H, W
                if sf != 1:
                    ns = [
                        math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[3:]
                    ]  # new shape
                    # Reshape for interpolation: [B*2, C, H, W]
                    b, streams, c, h, w = imgs.shape
                    imgs = imgs.view(b * streams, c, h, w)
                    imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
                    # Reshape back: [B, 2, C, H, W]
                    imgs = imgs.view(b, streams, c, ns[0], ns[1])
                    # print(f"DEBUG: After multi-scale dual stream - shape: {imgs.shape}")
            else:  # Single stream: [B, C, H, W]
                # print("DEBUG: Applying multi-scale to single stream")
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [
                        math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                    ]  # new shape
                    imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            
            batch["img"] = imgs
        
        # print(f"DEBUG: Final batch img shape: {batch['img'].shape}")
        # print(f"DEBUG: Final batch img range: min={batch['img'].min():.4f}, max={batch['img'].max():.4f}")
        # print(f"DEBUG: ===== END DETECTION PREPROCESS_BATCH DEBUG =====")
        
        return batch

    def set_model_attributes(self):
        """Set model attributes based on dataset information."""
        # Nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data["nc"]  # attach number of classes to model
        self.model.names = self.data["names"]  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model

        # Add dual stream information if available
        if hasattr(self.args, 'dual_stream') and self.args.dual_stream:
            self.model.dual_stream = True
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        Return a YOLO detection model.

        Args:
            cfg (str, optional): Path to model configuration file.
            weights (str, optional): Path to model weights.
            verbose (bool): Whether to display model information.

        Returns:
            (DetectionModel): YOLO detection model.
        """
        # Check if using dual stream configuration
        is_dual_stream = cfg and 'dual' in str(cfg).lower()
        
        # For dual stream, we still use the same number of channels per stream
        # The dual stream handling is done in the model architecture itself
        model = DetectionModel(
            cfg, 
            nc=self.data["nc"], 
            ch=self.data["channels"], 
            verbose=verbose and RANK == -1
        )
        
        if weights:
            model.load(weights)
        
        # Add dual stream flag to model if needed
        if is_dual_stream:
            model.dual_stream = True
        
        return model

    def get_validator(self):
        """Return a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Return a loss dict with labeled training loss items tensor.

        Args:
            loss_items (List[float], optional): List of loss values.
            prefix (str): Prefix for keys in the returned dictionary.

        Returns:
            (Dict | List): Dictionary of labeled loss items if loss_items is provided, otherwise list of keys.
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """Return a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )
    


    def plot_training_samples(self, batch, ni):
        """Plot training samples with their annotations."""
        print(f"DEBUG: ===== PLOT TRAINING SAMPLES DEBUG =====")
        
        images = batch["img"]
        print(f"DEBUG: Input images shape: {images.shape}")
        print(f"DEBUG: Input images range: {images.min():.6f} ~ {images.max():.6f}")
        print(f"DEBUG: Input images dtype: {images.dtype}")
        
        # 🔧 올바른 정규화 복원 함수
        def denormalize_for_plot(img_tensor):
            """이미지를 시각화용으로 올바르게 복원"""
            print(f"DEBUG: Denormalizing - input range: {img_tensor.min():.6f} ~ {img_tensor.max():.6f}")
            
            # 🔧 Case 1: 이미 0-255 범위 (정규화 안됨)
            if img_tensor.max() > 1.0:
                print(f"DEBUG: Already in 0-255 range, clamping only")
                return torch.clamp(img_tensor, 0, 255)
            
            # 🔧 Case 2: 0-1 범위이지만 ImageNet 정규화 없음 (일반적인 YOLO)
            elif img_tensor.min() >= 0 and img_tensor.max() <= 1.0:
                print(f"DEBUG: 0-1 range detected (standard YOLO normalization)")
                # 단순히 0-255로 변환
                result = img_tensor * 255.0
                result = torch.clamp(result, 0, 255)
                print(f"DEBUG: After 0-255 conversion: {result.min():.6f} ~ {result.max():.6f}")
                return result
            
            # 🔧 Case 3: ImageNet 정규화 적용됨 (음수 값 포함)
            elif img_tensor.min() < 0:
                print(f"DEBUG: ImageNet normalization detected (negative values present)")
                # ImageNet 역정규화: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img_tensor.device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img_tensor.device)
                
                # 역정규화
                result = img_tensor * std + mean
                result = torch.clamp(result, 0, 1)
                print(f"DEBUG: After ImageNet denorm: {result.min():.6f} ~ {result.max():.6f}")
                
                # 0-255 변환
                result = result * 255.0
                result = torch.clamp(result, 0, 255)
                print(f"DEBUG: After 0-255 conversion: {result.min():.6f} ~ {result.max():.6f}")
                return result
            
            # 🔧 Case 4: 예상치 못한 범위
            else:
                print(f"DEBUG: ⚠️ Unexpected range, using as-is")
                return torch.clamp(img_tensor * 255.0, 0, 255)
        
        # Handle dual stream visualization
        if images.dim() == 5 and images.shape[1] == 2:  # Dual stream: [B, 2, C, H, W]
            print(f"DEBUG: ✅ Dual stream detected for visualization")
            
            # Plot wide images (first stream)
            wide_images = images[:, 0]  # [B, C, H, W]
            wide_images_plot = denormalize_for_plot(wide_images.clone())
            
            print(f"DEBUG: Wide stream final range: {wide_images_plot.min():.1f} ~ {wide_images_plot.max():.1f}")
            
            plot_images(
                images=wide_images_plot,
                batch_idx=batch["batch_idx"],
                cls=batch["cls"].squeeze(-1),
                bboxes=batch["bboxes"],
                paths=batch["im_file"],
                fname=self.save_dir / f"train_batch{ni}_wide.jpg",
                on_plot=self.on_plot,
            )
            
            # Plot narrow images (second stream)
            narrow_images = images[:, 1]  # [B, C, H, W]
            narrow_images_plot = denormalize_for_plot(narrow_images.clone())
            
            print(f"DEBUG: Narrow stream final range: {narrow_images_plot.min():.1f} ~ {narrow_images_plot.max():.1f}")
            
            # narrow 경로명 수정
            narrow_paths = []
            for path in batch["im_file"]:
                if '|' in path:
                    narrow_paths.append(path.replace('|', '_narrow|'))
                else:
                    from pathlib import Path
                    p = Path(path)
                    narrow_path = str(p.parent / f"{p.stem}_narrow{p.suffix}")
                    narrow_paths.append(narrow_path)
            
            plot_images(
                images=narrow_images_plot,
                batch_idx=batch["batch_idx"],
                cls=batch["cls"].squeeze(-1),
                bboxes=batch["bboxes"],
                paths=narrow_paths,
                fname=self.save_dir / f"train_batch{ni}_narrow.jpg",
                on_plot=self.on_plot,
            )
            
            print(f"DEBUG: Wide plot: {self.save_dir / f'train_batch{ni}_wide.jpg'}")
            print(f"DEBUG: Narrow plot: {self.save_dir / f'train_batch{ni}_narrow.jpg'}")
            
        else:  # Single stream
            print(f"DEBUG: Single stream detected for visualization")
            images_plot = denormalize_for_plot(images.clone())
            
            print(f"DEBUG: Single stream final range: {images_plot.min():.1f} ~ {images_plot.max():.1f}")
            
            plot_images(
                images=images_plot,
                batch_idx=batch["batch_idx"],
                cls=batch["cls"].squeeze(-1),
                bboxes=batch["bboxes"],
                paths=batch["im_file"],
                fname=self.save_dir / f"train_batch{ni}.jpg",
                on_plot=self.on_plot,
            )
            print(f"DEBUG: Single stream plot: {self.save_dir / f'train_batch{ni}.jpg'}")
        
        print(f"DEBUG: ===== END PLOT TRAINING SAMPLES DEBUG =====")

    def plot_metrics(self):
        """Plot metrics from a CSV file."""
        plot_results(file=self.csv, on_plot=self.on_plot)  # save results.png

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

    def auto_batch(self):
        """
        Get optimal batch size by calculating memory occupation of model.

        Returns:
            (int): Optimal batch size.
        """
        train_dataset = self.build_dataset(self.data["train"], mode="train", batch=16)
        max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4  # 4 for mosaic augmentation
        return super().auto_batch(max_num_obj)
