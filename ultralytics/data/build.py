# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import dataloader, distributed

from ultralytics.data.dataset import GroundingDataset, YOLODataset, YOLOMultiModalDataset
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


def build_yolo_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32, multi_modal=False):
    """Build and return a YOLO dataset based on configuration parameters."""
    is_dual = is_dual_stream_yaml(data) and getattr(cfg, 'dual_stream', False)
    
    # If using dual-stream dataset
    if is_dual:
        try:
            wide_path, narrow_path = get_dual_stream_paths(data, mode)
            
            dataset = YOLODataset(
                img_path=wide_path,  # Use wide_path instead of img_path
                narrow_path=narrow_path,  # Pass narrow_path to dataset
                imgsz=cfg.imgsz,
                batch_size=batch,
                augment=mode == "train",
                hyp=cfg,
                rect=cfg.rect or rect,
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
            return dataset
            
        except KeyError as e:
            LOGGER.warning(f"Dual stream configuration error: {e}. Defaulting to single stream.")
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
    """Create and return an InfiniteDataLoader or DataLoader for training or validation."""
    
    # Check if it's a dual-stream dataset
    if hasattr(dataset, 'is_dual_stream') and dataset.is_dual_stream:
        # For dual stream, use regular InfiniteDataLoader
        # The dual stream logic is handled in YOLODataset.__getitem__
        batch = min(batch, len(dataset))
        nd = torch.cuda.device_count()
        nw = min((os.cpu_count() or 8) // max(nd, 1), workers)
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
    
    # Standard single-stream dataloader
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
    """
    # Handle dual source input (list of two sources)
    if isinstance(source, list) and len(source) == 2:
        # For dual stream, check the first source type and assume second is similar
        source_to_check = source[0]
    else:
        source_to_check = source
    
    webcam, screenshot, from_img, in_memory, tensor = False, False, False, False, False
    if isinstance(source_to_check, (str, int, Path)):  # int for local usb camera
        source_to_check = str(source_to_check)
        is_file = Path(source_to_check).suffix[1:] in (IMG_FORMATS | VID_FORMATS)
        is_url = source_to_check.lower().startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://"))
        webcam = source_to_check.isnumeric() or source_to_check.endswith(".streams") or (is_url and not is_file)
        screenshot = source_to_check.lower() == "screen"
        if is_url and is_file:
            if isinstance(source, list):
                source[0] = check_file(source[0])  # download first source
                source[1] = check_file(source[1])  # download second source
            else:
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
    Loads an inference source for object detection and applies necessary transformations.

    Args:
        source (str, Path, Tensor, PIL.Image, np.ndarray): The input source for inference.
        batch (int): Batch size for dataloaders.
        vid_stride (int): The frame interval for video sources.
        buffer (bool): Determined whether stream frames will be buffered.
        channels (int): Number of image channels (1=grayscale, 3=color).
        source2 (str, Path, optional): Second source for dual-stream inference.

    Returns:
        dataset (Dataset): A dataset object for the specified source(s).
    """
    source = str(source)
    
    # Handle dual stream input
    if source2 is not None:
        source2 = str(source2)
        source_type = check_source([source, source2])
        
        # For dual stream, both sources should be similar types
        if source_type.stream or source_type.screenshot:
            raise NotImplementedError("Dual stream not supported for live streams or screenshots")
        elif source_type.from_img:
            dataset = LoadDualImagesAndVideos(source, source2, batch=batch, vid_stride=vid_stride, channels=channels)
        else:
            raise ValueError(f"Unsupported dual stream source types")
    else:
        # Original single stream logic
        source_type = check_source(source)
        
        if source_type.stream or source_type.screenshot:
            dataset = LoadStreams(source, vid_stride=vid_stride, buffer=buffer)
        elif source_type.from_img:
            dataset = LoadImagesAndVideos(source, batch=batch, vid_stride=vid_stride, channels=channels)
        elif source_type.tensor:
            dataset = LoadTensor(source)
        else:
            dataset = LoadPilAndNumpy(source, channels=channels)
    
    return dataset


def is_dual_stream_yaml(data):
    """
    Check if the data configuration is for a dual-stream setup.
    
    Args:
        data (dict): Dataset configuration dictionary.
        
    Returns:
        (bool): True if data contains dual-stream paths, False otherwise.
    """
    return any(k.endswith(('_wide', '_narrow')) for k in data.keys())


def get_dual_stream_paths(data, mode='train'):
    """
    Extract dual-stream paths from data configuration.
    
    Args:
        data (dict): Dataset configuration dictionary.
        mode (str): Dataset mode ('train', 'val', 'test').
        
    Returns:
        (tuple): Tuple containing paths for wide and narrow camera images.
    """
    wide_key = f"{mode}_wide"
    narrow_key = f"{mode}_narrow"
    
    # Check if the keys exist
    if wide_key not in data or narrow_key not in data:
        raise KeyError(f"Dual-stream configuration requires both '{wide_key}' and '{narrow_key}' keys")
    
    return data[wide_key], data[narrow_key]


class DualStreamDataLoader(InfiniteDataLoader):
    """DataLoader for dual-stream datasets."""
    
    def __init__(self, dataset, batch_size, narrow_path, workers=8, shuffle=False):
        """Initialize the DualStreamDataLoader."""
        # Standard initialization
        super().__init__(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=workers, 
            pin_memory=PIN_MEMORY,
            collate_fn=dataset.collate_fn
        )
        # Just store narrow_path - the actual dual loading is handled in YOLODataset.__getitem__
        self.narrow_path = narrow_path