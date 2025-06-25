# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import dataloader, distributed

from ultralytics.data.dataset import GroundingDataset, YOLODataset, YOLOMultiModalDataset, YOLODualStreamDataset
from ultralytics.data.loaders import (
    LOADERS,
    LoadImagesAndVideos,
    LoadPilAndNumpy,
    LoadScreenshots,
    LoadStreams,
    LoadTensor,
    LoadDualImagesAndVideos,
    SourceTypes,
    autocast_list,
)
from ultralytics.data.utils import IMG_FORMATS, PIN_MEMORY, VID_FORMATS
from ultralytics.utils import LOGGER, RANK, colorstr
from ultralytics.utils.checks import check_file


class InfiniteDataLoader(dataloader.DataLoader):
    """
    Dataloader that reuses workers.

    This dataloader extends the PyTorch DataLoader to provide infinite recycling of workers, which improves efficiency
    for training loops that need to iterate through the dataset multiple times.

    Attributes:
        batch_sampler (_RepeatSampler): A sampler that repeats indefinitely.
        iterator (Iterator): The iterator from the parent DataLoader.

    Methods:
        __len__: Returns the length of the batch sampler's sampler.
        __iter__: Creates a sampler that repeats indefinitely.
        __del__: Ensures workers are properly terminated.
        reset: Resets the iterator, useful when modifying dataset settings during training.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the InfiniteDataLoader with the same arguments as DataLoader."""
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        """Return the length of the batch sampler's sampler."""
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        """Create an iterator that yields indefinitely from the underlying iterator."""
        for _ in range(len(self)):
            yield next(self.iterator)

    def __del__(self):
        """Ensure that workers are properly terminated when the dataloader is deleted."""
        try:
            if not hasattr(self.iterator, "_workers"):
                return
            for w in self.iterator._workers:  # force terminate
                if w.is_alive():
                    w.terminate()
            self.iterator._shutdown_workers()  # cleanup
        except Exception:
            pass

    def reset(self):
        """Reset the iterator to allow modifications to the dataset during training."""
        self.iterator = self._get_iterator()


class _RepeatSampler:
    """
    Sampler that repeats forever.

    This sampler wraps another sampler and yields its contents indefinitely, allowing for infinite iteration
    over a dataset.

    Attributes:
        sampler (Dataset.sampler): The sampler to repeat.
    """

    def __init__(self, sampler):
        """Initialize the _RepeatSampler with a sampler to repeat indefinitely."""
        self.sampler = sampler

    def __iter__(self):
        """Iterate over the sampler indefinitely, yielding its contents."""
        while True:
            yield from iter(self.sampler)


def seed_worker(worker_id):  # noqa
    """Set dataloader worker seed for reproducibility across worker processes."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_yolo_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32, multi_modal=False, dual_stream=True):
    """
    Build and return a YOLO dataset based on configuration parameters.
    
    Args:
        cfg (dict): Configuration parameters from model args.
        img_path (str): Path to the images.
        batch (int): Batch size for the dataloader.
        data (dict): Dataset configuration dictionary.
        mode (str): Dataset mode ('train', 'val', 'test').
        rect (bool): Enable rectangular inference.
        stride (int): Model stride.
        multi_modal (bool): Whether to use multi-modal dataset.
        dual_stream (bool): Whether to use dual-stream dataset.
        
    Returns:
        (Dataset): Dataset object for the specified configuration.
    """
    # Check if this is a dual-stream configuration
    is_dual = dual_stream or getattr(cfg, 'dual_stream', False)
    
    # If using dual-stream dataset
    if is_dual:
        LOGGER.info(f"Using dual-stream dataset for {mode}")
        
        # Use the dual-stream dataset class
        dataset = YOLODualStreamDataset(
            img_path=img_path,  # This will contain {} placeholders
            imgsz=cfg.imgsz,
            batch_size=batch,
            augment=mode == "train",  # augmentation
            hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
            rect=cfg.rect or rect,  # rectangular batches
            cache=cfg.cache or None,
            single_cls=cfg.single_cls or False,
            stride=int(stride),
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode} (dual): "),
            task=cfg.task,
            classes=cfg.classes,
            data=data,
            fraction=cfg.fraction if mode == "train" else 1.0,
        )
        # Add attribute to mark as dual stream (for use in trainer)
        dataset.is_dual_stream = True
        
        LOGGER.info(f"Dual stream dataset created successfully")
        return dataset
    else:
        # Standard single-stream dataset loading
        dataset = YOLOMultiModalDataset if multi_modal else YOLODataset
        return dataset(
            img_path=img_path,
            imgsz=cfg.imgsz,
            batch_size=batch,
            augment=mode == "train",  # augmentation
            hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
            rect=cfg.rect or rect,  # rectangular batches
            cache=cfg.cache or None,
            single_cls=cfg.single_cls or False,
            stride=int(stride),
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=cfg.task,
            classes=cfg.classes,
            data=data,
            fraction=cfg.fraction if mode == "train" else 1.0,
        )


def build_grounding(cfg, img_path, json_file, batch, mode="train", rect=False, stride=32):
    """Build and return a GroundingDataset based on configuration parameters."""
    return GroundingDataset(
        img_path=img_path,
        json_file=json_file,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )


def build_dataloader(dataset, batch, workers, shuffle=True, rank=-1):
    """
    Create and return an InfiniteDataLoader or DataLoader for training or validation.

    Args:
        dataset (Dataset): Dataset to load data from.
        batch (int): Batch size for the dataloader.
        workers (int): Number of worker threads for loading data.
        shuffle (bool): Whether to shuffle the dataset.
        rank (int): Process rank in distributed training. -1 for single-GPU training.

    Returns:
        (InfiniteDataLoader): A dataloader that can be used for training or validation.
    """
    # Handle dual-stream dataset
    if hasattr(dataset, 'is_dual_stream') and dataset.is_dual_stream and hasattr(dataset, 'narrow_path'):
        LOGGER.info(f"Creating dual-stream dataloader with narrow path: {dataset.narrow_path}")
        return DualStreamDataLoader(
            dataset=dataset,
            batch_size=min(batch, len(dataset)),
            narrow_path=dataset.narrow_path,
            workers=workers,
            shuffle=shuffle and rank == -1
        )
        
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min((os.cpu_count() or 8) // max(nd, 1), workers)  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return InfiniteDataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        pin_memory=PIN_MEMORY,
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,
    )


def check_source(source):
    """
    Check the type of input source and return corresponding flag values.

    Args:
        source (str | int | Path | List | Tuple | np.ndarray | PIL.Image | torch.Tensor): The input source to check.

    Returns:
        source (str | int | Path | List | Tuple | np.ndarray | PIL.Image | torch.Tensor): The processed source.
        webcam (bool): Whether the source is a webcam.
        screenshot (bool): Whether the source is a screenshot.
        from_img (bool): Whether the source is an image or list of images.
        in_memory (bool): Whether the source is an in-memory object.
        tensor (bool): Whether the source is a torch.Tensor.

    Raises:
        TypeError: If the source type is unsupported.
    """
    webcam, screenshot, from_img, in_memory, tensor = False, False, False, False, False
    if isinstance(source, (str, int, Path)):  # int for local usb camera
        source = str(source)
        is_file = Path(source).suffix[1:] in (IMG_FORMATS | VID_FORMATS)
        is_url = source.lower().startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://"))
        webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
        screenshot = source.lower() == "screen"
        if is_url and is_file:
            source = check_file(source)  # download
    elif isinstance(source, LOADERS):
        in_memory = True
    elif isinstance(source, (list, tuple)):
        source = autocast_list(source)  # convert all list elements to PIL or np arrays
        from_img = True
    elif isinstance(source, (Image.Image, np.ndarray)):
        from_img = True
    elif isinstance(source, torch.Tensor):
        tensor = True
    else:
        raise TypeError("Unsupported image type. For supported types see https://docs.ultralytics.com/modes/predict")

    return source, webcam, screenshot, from_img, in_memory, tensor


def load_inference_source(source=None, batch=1, vid_stride=1, buffer=False, channels=3, source2=None):
    """
    Load an inference source for object detection and apply necessary transformations.

    Args:
        source (str | Path | torch.Tensor | PIL.Image | np.ndarray, optional): The input source for inference.
        batch (int, optional): Batch size for dataloaders.
        vid_stride (int, optional): The frame interval for video sources.
        buffer (bool, optional): Whether stream frames will be buffered.
        channels (int): The number of input channels for the model.
        source2 (str | Path, optional): The second input source for dual-stream inference.

    Returns:
        (Dataset): A dataset object for the specified input source with attached source_type attribute.
    """
    if source2 is not None:
        # Handle dual-stream inputs
        source1, stream1, screenshot1, from_img1, in_memory1, tensor1 = check_source(source)
        source2, stream2, screenshot2, from_img2, in_memory2, tensor2 = check_source(source2)
        
        # Check if both sources are compatible
        if (stream1 and not stream2) or (not stream1 and stream2):
            raise ValueError("Both sources must be of the same type for dual-stream loading")
        if (screenshot1 and not screenshot2) or (not screenshot1 and screenshot2):
            raise ValueError("Both sources must be of the same type for dual-stream loading")
            
        # Create dual-source loader
        dataset = LoadDualImagesAndVideos(source1, source2, batch=batch, vid_stride=vid_stride, channels=channels)
        
        # Create source type for dual-stream (treat as image-type for compatibility)
        source_type = SourceTypes(False, False, True, False)
    else:
        # Standard single-stream processing
        source, stream, screenshot, from_img, in_memory, tensor = check_source(source)
        
        # Create source type
        if in_memory:
            # For in-memory datasets that already have source_type attribute
            source_type = getattr(source, "source_type", SourceTypes(stream, screenshot, from_img, tensor))
        else:
            source_type = SourceTypes(stream, screenshot, from_img, tensor)

        # Dataloader
        if tensor:
            dataset = LoadTensor(source)
        elif in_memory:
            dataset = source
        elif stream:
            dataset = LoadStreams(source, vid_stride=vid_stride, buffer=buffer)
        elif screenshot:
            dataset = LoadScreenshots(source)
        elif from_img:
            dataset = LoadPilAndNumpy(source, channels=channels)
        else:
            dataset = LoadImagesAndVideos(source, batch=batch, vid_stride=vid_stride, channels=channels)

    # Attach source types to the dataset
    setattr(dataset, "source_type", source_type)

    return dataset


# def is_dual_stream_yaml(data):
#     """
#     Check if the data configuration is for a dual-stream setup.
#     
#     Args:
#         data (dict): Dataset configuration dictionary.
#         
#     Returns:
#         (bool): True if data contains dual-stream paths, False otherwise.
#     """
#     # Check for both train_wide/narrow and val_wide/narrow keys
#     has_train_dual = 'train_wide' in data and 'train_narrow' in data
#     has_val_dual = 'val_wide' in data and 'val_narrow' in data
#     
#     # Only return true if both train and val dual keys exist
#     return has_train_dual and has_val_dual


# def get_dual_stream_paths(data, mode='train'):
#     """
#     Extract dual-stream paths from data configuration.
#     
#     Args:
#         data (dict): Dataset configuration dictionary.
#         mode (str): Dataset mode ('train', 'val', 'test').
#         
#     Returns:
#         (tuple): Tuple containing paths for wide and narrow camera images.
#     """
#     wide_key = f"{mode}_wide"
#     narrow_key = f"{mode}_narrow"
#     
#     # Check if the dual-stream keys exist
#     if wide_key in data and narrow_key in data:
#         LOGGER.info(f"Using dual-stream paths: {wide_key}={data[wide_key]}, {narrow_key}={data[narrow_key]}")
#         return data[wide_key], data[narrow_key]
#     
#     # If dual-stream keys don't exist, but we have the standard key, use it for both (as fallback)
#     if mode in data:
#         LOGGER.warning(f"Dual-stream keys not found. Using '{mode}' for both wide and narrow paths.")
#         return data[mode], data[mode]
#     
#     # If we're using purely dual-stream keys without standard keys
#     if mode == 'train' and 'train_wide' in data:
#         # We are using exclusively dual-stream keys without standard keys
#         raise KeyError(f"Using dual-stream mode but '{mode}' key is missing. Make sure to call check_det_dataset first.")
#     
#     raise KeyError(f"Dual-stream configuration requires both '{wide_key}' and '{narrow_key}' keys " 
#                     f"or at least the standard '{mode}' key")


class DualStreamDataLoader(InfiniteDataLoader):
    """
    DataLoader for dual-stream datasets.
    
    This class extends InfiniteDataLoader to handle dual-stream datasets with wide and narrow camera inputs.
    Instead of just loading images from a single source, this dataloader loads paired images from both
    wide and narrow camera sources.
    
    Attributes:
        dataset (YOLODataset): The wide camera dataset with labels.
        narrow_path (str): Path to the narrow camera images.
    """
    
    def __init__(self, dataset, batch_size, narrow_path, workers=8, shuffle=False):
        """
        Initialize the DualStreamDataLoader.
        
        Args:
            dataset (YOLODataset): The dataset for the wide camera with labels.
            batch_size (int): Number of samples per batch.
            narrow_path (str): Path to narrow camera images.
            workers (int): Number of worker threads for loading data.
            shuffle (bool): Whether to shuffle the dataset.
        """
        self.dataset = dataset
        self.narrow_path = narrow_path
        # Create a dual-stream dataloader using LoadDualImagesAndVideos
        self.dual_loader = LoadDualImagesAndVideos(
            path1=dataset.im_files,  # wide camera images
            path2=narrow_path,        # narrow camera images
            batch=batch_size,
            channels=3
        )
        # Initialize with original dataset, but we'll override __iter__
        super().__init__(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=workers, 
            pin_memory=PIN_MEMORY,
            collate_fn=dataset.collate_fn
        )
    
    def __iter__(self):
        """
        Return an iterator over the dual-stream dataset.
        
        This method overrides the default iterator to use the dual-stream loader.
        """
        for (paths_wide, paths_narrow), (imgs_wide, imgs_narrow), info in self.dual_loader:
            # Use original dataset iterator for labels and preprocessing
            original_iter = super().__iter__()
            batch = next(original_iter)
            
            # Replace the images with our dual-stream images
            if isinstance(batch, dict):
                # For dictionary-style batches
                batch['img'] = torch.stack([torch.from_numpy(img).to(batch['img'].device) for img in imgs_wide])
                batch['img2'] = torch.stack([torch.from_numpy(img).to(batch['img'].device) for img in imgs_narrow])
            else:
                # For tuple/list-style batches, assume images are the first element
                images = torch.stack([torch.from_numpy(img) for img in imgs_wide])
                narrow_images = torch.stack([torch.from_numpy(img) for img in imgs_narrow])
                batch = (images, *batch[1:], narrow_images)
            
            yield batch
