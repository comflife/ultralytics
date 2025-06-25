# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import json
from collections import defaultdict
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset

from ultralytics.utils import LOCAL_RANK, LOGGER, NUM_THREADS, TQDM, colorstr
from ultralytics.utils.instance import Instances
from ultralytics.utils.ops import resample_segments, segments2boxes
from ultralytics.utils.torch_utils import TORCHVISION_0_18
import os
import glob
import tqdm

from .augment import (
    Compose,
    Format,
    LetterBox,
    RandomLoadText,
    classify_augmentations,
    classify_transforms,
    v8_transforms,
)
from .base import BaseDataset
from .converter import merge_multi_segment
from .utils import (
    HELP_URL,
    check_file_speeds,
    get_hash,
    img2label_paths,
    load_dataset_cache_file,
    save_dataset_cache_file,
    verify_image,
    verify_image_label,
)

# Ultralytics dataset *.cache version, >= 1.0.0 for Ultralytics YOLO models
DATASET_CACHE_VERSION = "1.0.3"


class YOLODataset(BaseDataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.

    This class supports loading data for object detection, segmentation, pose estimation, and oriented bounding box
    (OBB) tasks using the YOLO format.

    Attributes:
        use_segments (bool): Indicates if segmentation masks should be used.
        use_keypoints (bool): Indicates if keypoints should be used for pose estimation.
        use_obb (bool): Indicates if oriented bounding boxes should be used.
        data (dict): Dataset configuration dictionary.
        is_dual_stream (bool): Indicates if the dataset is for dual-stream models.
        narrow_path (str): Path to narrow camera images for dual-stream models.

    Methods:
        cache_labels: Cache dataset labels, check images and read shapes.
        get_labels: Returns dictionary of labels for YOLO training.
        build_transforms: Builds and appends transforms to the list.
        close_mosaic: Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations.
        update_labels_info: Updates label format for different tasks.
        collate_fn: Collates data samples into batches.

    Examples:
        >>> dataset = YOLODataset(img_path="path/to/images", data={"names": {0: "person"}}, task="detect")
        >>> dataset.get_labels()
    """

    def __init__(self, *args, data=None, task="detect", **kwargs):
        """
        Initialize the YOLODataset.

        Args:
            data (dict, optional): Dataset configuration dictionary.
            task (str): Task type, one of 'detect', 'segment', 'pose', or 'obb'.
            *args (Any): Additional positional arguments for the parent class.
            **kwargs (Any): Additional keyword arguments for the parent class.
        """
        self.use_segments = task == "segment"
        self.use_keypoints = task == "pose"
        self.use_obb = task == "obb"
        self.data = data
        # Initialize dual-stream attributes
        self.is_dual_stream = False
        self.narrow_path = None
        assert not (self.use_segments and self.use_keypoints), "Can not use both segments and keypoints."
        
        # Get channels from data dict if available, else default to 3
        channels = self.data.get("channels", 3) if self.data else 3
        super().__init__(*args, channels=channels, **kwargs)

    def cache_labels(self, path=Path("./labels.cache")):
        """
        Cache dataset labels, check images and read shapes.

        Args:
            path (Path): Path where to save the cache file.

        Returns:
            (dict): Dictionary containing cached labels and related information.
        """
        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in {2, 3}):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(self.use_keypoints),
                    repeat(len(self.data["names"])),
                    repeat(nkpt),
                    repeat(ndim),
                    repeat(self.single_cls),
                ),
            )
            pbar = TQDM(results, desc=desc, total=total)
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x["labels"].append(
                        {
                            "im_file": im_file,
                            "shape": shape,
                            "cls": lb[:, 0:1],  # n, 1
                            "bboxes": lb[:, 1:],  # n, 4
                            "segments": segments,
                            "keypoints": keypoint,
                            "normalized": True,
                            "bbox_format": "xywh",
                        }
                    )
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{self.prefix}No labels found in {path}. {HELP_URL}")
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self):
        """
        Returns dictionary of labels for YOLO training.

        This method loads labels from disk or cache, verifies their integrity, and prepares them for training.

        Returns:
            (List[dict]): List of label dictionaries, each containing information about an image and its annotations.
        """
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)  # display results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels = cache["labels"]
        if not labels:
            raise RuntimeError(
                f"No valid images found in {cache_path}. Images with incorrectly formatted labels are ignored. {HELP_URL}"
            )
        self.im_files = [lb["im_file"] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f"Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
                "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
            )
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            LOGGER.warning(f"Labels are missing or empty in {cache_path}, training may not work correctly. {HELP_URL}")
        return labels

    def build_transforms(self, hyp=None):
        """
        Builds and appends transforms to the list.

        Args:
            hyp (dict, optional): Hyperparameters for transforms.

        Returns:
            (Compose): Composed transforms.
        """
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            hyp.cutmix = hyp.cutmix if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,  # only affect training.
            )
        )
        return transforms

    def close_mosaic(self, hyp):
        """
        Disable mosaic, copy_paste, mixup and cutmix augmentations by setting their probabilities to 0.0.

        Args:
            hyp (dict): Hyperparameters for transforms.
        """
        hyp.mosaic = 0.0
        hyp.copy_paste = 0.0
        hyp.mixup = 0.0
        hyp.cutmix = 0.0
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label):
        """
        Custom your label format here.

        Args:
            label (dict): Label dictionary containing bboxes, segments, keypoints, etc.

        Returns:
            (dict): Updated label dictionary with instances.

        Note:
            cls is not with bboxes now, classification and semantic segmentation need an independent cls label
            Can also support classification and semantic segmentation by adding or removing dict keys there.
        """
        bboxes = label.pop("bboxes")
        segments = label.pop("segments", [])
        keypoints = label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")

        # NOTE: do NOT resample oriented boxes
        segment_resamples = 100 if self.use_obb else 1000
        if len(segments) > 0:
            # make sure segments interpolate correctly if original length is greater than segment_resamples
            max_len = max(len(s) for s in segments)
            segment_resamples = (max_len + 1) if segment_resamples < max_len else segment_resamples
            # list[np.array(segment_resamples, 2)] * num_samples
            segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
        label["instances"] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    @staticmethod
    def collate_fn(batch):
        """
        Collates data samples into batches.

        Args:
            batch (List[dict]): List of dictionaries containing sample data.

        Returns:
            (dict): Collated batch with stacked tensors.
        """
        new_batch = {}
        batch = [dict(sorted(b.items())) for b in batch]  # make sure the keys are in the same order
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k in {"img", "text_feats"}:
                value = torch.stack(value, 0)
            elif k == "visuals":
                value = torch.nn.utils.rnn.pad_sequence(value, batch_first=True)
            if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch


class YOLOMultiModalDataset(YOLODataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format with multi-modal support.

    This class extends YOLODataset to add text information for multi-modal model training, enabling models to
    process both image and text data.

    Methods:
        update_labels_info: Adds text information for multi-modal model training.
        build_transforms: Enhances data transformations with text augmentation.

    Examples:
        >>> dataset = YOLOMultiModalDataset(img_path="path/to/images", data={"names": {0: "person"}}, task="detect")
        >>> batch = next(iter(dataset))
        >>> print(batch.keys())  # Should include 'texts'
    """

    def __init__(self, *args, data=None, task="detect", **kwargs):
        """
        Initialize a YOLOMultiModalDataset.

        Args:
            data (dict, optional): Dataset configuration dictionary.
            task (str): Task type, one of 'detect', 'segment', 'pose', or 'obb'.
            *args (Any): Additional positional arguments for the parent class.
            **kwargs (Any): Additional keyword arguments for the parent class.
        """
        super().__init__(*args, data=data, task=task, **kwargs)

    def update_labels_info(self, label):
        """
        Add texts information for multi-modal model training.

        Args:
            label (dict): Label dictionary containing bboxes, segments, keypoints, etc.

        Returns:
            (dict): Updated label dictionary with instances and texts.
        """
        labels = super().update_labels_info(label)
        # NOTE: some categories are concatenated with its synonyms by `/`.
        # NOTE: and `RandomLoadText` would randomly select one of them if there are multiple words.
        labels["texts"] = [v.split("/") for _, v in self.data["names"].items()]

        return labels

    def build_transforms(self, hyp=None):
        """
        Enhances data transformations with optional text augmentation for multi-modal training.

        Args:
            hyp (dict, optional): Hyperparameters for transforms.

        Returns:
            (Compose): Composed transforms including text augmentation if applicable.
        """
        transforms = super().build_transforms(hyp)
        if self.augment:
            # NOTE: hard-coded the args for now.
            # NOTE: this implementation is different from official yoloe,
            # the strategy of selecting negative is restricted in one dataset,
            # while official pre-saved neg embeddings from all datasets at once.
            transform = RandomLoadText(
                max_samples=min(self.data["nc"], 80),
                padding=True,
                padding_value=self._get_neg_texts(self.category_freq),
            )
            transforms.insert(-1, transform)
        return transforms

    @property
    def category_names(self):
        """
        Return category names for the dataset.

        Returns:
            (Set[str]): List of class names.
        """
        names = self.data["names"].values()
        return {n.strip() for name in names for n in name.split("/")}  # category names

    @property
    def category_freq(self):
        """Return frequency of each category in the dataset."""
        texts = [v.split("/") for v in self.data["names"].values()]
        category_freq = defaultdict(int)
        for label in self.labels:
            for c in label["cls"].squeeze(-1):  # to check
                text = texts[int(c)]
                for t in text:
                    t = t.strip()
                    category_freq[t] += 1
        return category_freq

    @staticmethod
    def _get_neg_texts(category_freq, threshold=100):
        """Get negative text samples based on frequency threshold."""
        return [k for k, v in category_freq.items() if v >= threshold]


class GroundingDataset(YOLODataset):
    """
    Handles object detection tasks by loading annotations from a specified JSON file, supporting YOLO format.

    This dataset is designed for grounding tasks where annotations are provided in a JSON file rather than
    the standard YOLO format text files.

    Attributes:
        json_file (str): Path to the JSON file containing annotations.

    Methods:
        get_img_files: Returns empty list as image files are read in get_labels.
        get_labels: Loads annotations from a JSON file and prepares them for training.
        build_transforms: Configures augmentations for training with optional text loading.

    Examples:
        >>> dataset = GroundingDataset(img_path="path/to/images", json_file="annotations.json", task="detect")
        >>> len(dataset)  # Number of valid images with annotations
    """

    def __init__(self, *args, task="detect", json_file="", **kwargs):
        """
        Initialize a GroundingDataset for object detection.

        Args:
            json_file (str): Path to the JSON file containing annotations.
            task (str): Must be 'detect' or 'segment' for GroundingDataset.
            *args (Any): Additional positional arguments for the parent class.
            **kwargs (Any): Additional keyword arguments for the parent class.
        """
        assert task in {"detect", "segment"}, "GroundingDataset currently only supports `detect` and `segment` tasks"
        self.json_file = json_file
        super().__init__(*args, task=task, data={"channels": 3}, **kwargs)

    def get_img_files(self, img_path):
        """
        The image files would be read in `get_labels` function, return empty list here.

        Args:
            img_path (str): Path to the directory containing images.

        Returns:
            (list): Empty list as image files are read in get_labels.
        """
        return []

    def verify_labels(self, labels):
        """Verify the number of instances in the dataset matches expected counts."""
        instance_count = sum(label["bboxes"].shape[0] for label in labels)
        if "final_mixed_train_no_coco_segm" in self.json_file:
            assert instance_count == 3662344
        elif "final_mixed_train_no_coco" in self.json_file:
            assert instance_count == 3681235
        elif "final_flickr_separateGT_train_segm" in self.json_file:
            assert instance_count == 638214
        elif "final_flickr_separateGT_train" in self.json_file:
            assert instance_count == 640704
        else:
            assert False

    def cache_labels(self, path=Path("./labels.cache")):
        """
        Loads annotations from a JSON file, filters, and normalizes bounding boxes for each image.

        Args:
            path (Path): Path where to save the cache file.

        Returns:
            (dict): Dictionary containing cached labels and related information.
        """
        x = {"labels": []}
        LOGGER.info("Loading annotation file...")
        with open(self.json_file) as f:
            annotations = json.load(f)
        images = {f"{x['id']:d}": x for x in annotations["images"]}
        img_to_anns = defaultdict(list)
        for ann in annotations["annotations"]:
            img_to_anns[ann["image_id"]].append(ann)
        for img_id, anns in TQDM(img_to_anns.items(), desc=f"Reading annotations {self.json_file}"):
            img = images[f"{img_id:d}"]
            h, w, f = img["height"], img["width"], img["file_name"]
            im_file = Path(self.img_path) / f
            if not im_file.exists():
                continue
            self.im_files.append(str(im_file))
            bboxes = []
            segments = []
            cat2id = {}
            texts = []
            for ann in anns:
                if ann["iscrowd"]:
                    continue
                box = np.array(ann["bbox"], dtype=np.float32)
                box[:2] += box[2:] / 2
                box[[0, 2]] /= float(w)
                box[[1, 3]] /= float(h)
                if box[2] <= 0 or box[3] <= 0:
                    continue

                caption = img["caption"]
                cat_name = " ".join([caption[t[0] : t[1]] for t in ann["tokens_positive"]]).lower().strip()
                if not cat_name:
                    continue

                if cat_name not in cat2id:
                    cat2id[cat_name] = len(cat2id)
                    texts.append([cat_name])
                cls = cat2id[cat_name]  # class
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)
                    if ann.get("segmentation") is not None:
                        if len(ann["segmentation"]) == 0:
                            segments.append(box)
                            continue
                        elif len(ann["segmentation"]) > 1:
                            s = merge_multi_segment(ann["segmentation"])
                            s = (np.concatenate(s, axis=0) / np.array([w, h], dtype=np.float32)).reshape(-1).tolist()
                        else:
                            s = [j for i in ann["segmentation"] for j in i]  # all segments concatenated
                            s = (
                                (np.array(s, dtype=np.float32).reshape(-1, 2) / np.array([w, h], dtype=np.float32))
                                .reshape(-1)
                                .tolist()
                            )
                        s = [cls] + s
                        segments.append(s)
            lb = np.array(bboxes, dtype=np.float32) if len(bboxes) else np.zeros((0, 5), dtype=np.float32)

            if segments:
                classes = np.array([x[0] for x in segments], dtype=np.float32)
                segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in segments]  # (cls, xy1...)
                lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
            lb = np.array(lb, dtype=np.float32)

            x["labels"].append(
                {
                    "im_file": im_file,
                    "shape": (h, w),
                    "cls": lb[:, 0:1],  # n, 1
                    "bboxes": lb[:, 1:],  # n, 4
                    "segments": segments,
                    "normalized": True,
                    "bbox_format": "xywh",
                    "texts": texts,
                }
            )
        x["hash"] = get_hash(self.json_file)
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self):
        """
        Load labels from cache or generate them from JSON file.

        Returns:
            (List[dict]): List of label dictionaries, each containing information about an image and its annotations.
        """
        cache_path = Path(self.json_file).with_suffix(".cache")
        try:
            cache, _ = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash(self.json_file)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, _ = self.cache_labels(cache_path), False  # run cache ops
        [cache.pop(k) for k in ("hash", "version")]  # remove items
        labels = cache["labels"]
        # self.verify_labels(labels)
        self.im_files = [str(label["im_file"]) for label in labels]
        if LOCAL_RANK in {-1, 0}:
            LOGGER.info(f"Load {self.json_file} from cache file {cache_path}")
        return labels

    def build_transforms(self, hyp=None):
        """
        Configures augmentations for training with optional text loading.

        Args:
            hyp (dict, optional): Hyperparameters for transforms.

        Returns:
            (Compose): Composed transforms including text augmentation if applicable.
        """
        transforms = super().build_transforms(hyp)
        if self.augment:
            # NOTE: hard-coded the args for now.
            # NOTE: this implementation is different from official yoloe,
            # the strategy of selecting negative is restricted in one dataset,
            # while official pre-saved neg embeddings from all datasets at once.
            transform = RandomLoadText(
                max_samples=80,
                padding=True,
                padding_value=self._get_neg_texts(self.category_freq),
            )
            transforms.insert(-1, transform)
        return transforms

    @property
    def category_names(self):
        """Return unique category names from the dataset."""
        return {t.strip() for label in self.labels for text in label["texts"] for t in text}

    @property
    def category_freq(self):
        """Return frequency of each category in the dataset."""
        category_freq = defaultdict(int)
        for label in self.labels:
            for text in label["texts"]:
                for t in text:
                    t = t.strip()
                    category_freq[t] += 1
        return category_freq

    @staticmethod
    def _get_neg_texts(category_freq, threshold=100):
        """Get negative text samples based on frequency threshold."""
        return [k for k, v in category_freq.items() if v >= threshold]


class YOLODualStreamDataset(YOLODataset):
    """
    Dataset class for loading dual-stream RGBT (RGB-Thermal) data in YOLO format.
    
    This class extends YOLODataset to handle two image modalities (wide and narrow cameras)
    for each training sample, supporting dual-stream model training.
    
    Attributes:
        modalities (tuple): Image modalities ('wide', 'narrow').
    """
    
    cache_version = "1.0.5"  # dataset labels *.cache version for dual stream
    modalities = ('wide', 'narrow')

    def __init__(self, *args, data=None, task="detect", **kwargs):
        """Initialize YOLODualStreamDataset with dual-stream configuration."""
        # Store original arguments
        self.data = data
        self.task = task
        
        # TODO: Disable mosaic for now - need to implement dual-stream mosaic
        if 'hyp' in kwargs and kwargs['hyp'] is not None:
            if hasattr(kwargs['hyp'], 'mosaic'):
                kwargs['hyp'].mosaic = 0.0
        
        # Initialize parent class
        super().__init__(*args, data=data, task=task, **kwargs)
        
        # Set dual-stream flag
        self.is_dual_stream = True
        
        LOGGER.info(f"Dual-stream dataset initialized with {len(self.im_files)} samples")

    def img2label_paths(self, img_paths):
        """Generates label file paths from corresponding image file paths by replacing `/images/{}` with `/labels/` and
        extension with `.txt`. Cleanly removes any {} placeholder in the label path.
        """
        label_paths = []
        for img_path in img_paths:
            if '{}' in img_path:
                # Extract the base filename and root directory
                # Example: from "/datasets/swm-data/train/images/{}/file.jpg" get "file" and "/datasets/swm-data/train/"
                img_parts = img_path.split(os.sep)
                file_name = img_parts[-1]
                
                # Find the 'images' directory index
                try:
                    images_idx = img_parts.index('images')
                    # Build the path to the root directory (before 'images')
                    root_dir = os.sep.join(img_parts[:images_idx])
                    # Get base filename without extension
                    base_name = os.path.splitext(file_name)[0]
                    # Create the label path
                    label_path = os.path.join(root_dir, 'labels', f"{base_name}.txt")
                    label_paths.append(label_path)
                except ValueError:
                    # If 'images' not found, use a fallback method
                    LOGGER.warning(f"Could not parse path correctly: {img_path}")
                    # Use a regex-based replacement as fallback
                    import re
                    label_path = re.sub(r'[/\\]images[/\\]\{\}[/\\]', f'{os.sep}labels{os.sep}', img_path)
                    label_path = os.path.splitext(label_path)[0] + '.txt'
                    label_paths.append(label_path)
            else:
                # Standard path without placeholder - replace /images/ with /labels/
                sa_standard = f"{os.sep}images{os.sep}"
                sb = f"{os.sep}labels{os.sep}"
                label_path = img_path.replace(sa_standard, sb).rsplit(".", 1)[0] + ".txt"
                label_paths.append(label_path)
        return label_paths

    def cache_labels(self, path=Path("./labels.cache"), prefix=""):
        """Caches dataset labels, verifies images, reads shapes, and tracks dataset integrity."""
        x = {}  # dict
        data = []  # use list to keep image orders
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{prefix}Scanning {path.parent / path.stem}..."

        pbar = TQDM(
            zip(self.im_files, self.label_files, repeat(prefix)),
            desc=desc,
            total=len(self.im_files))

        for blob in pbar:
            im_file, lb_file, prefix_str = blob
            
            # For dual-stream, manually verify both images exist
            wide_path = im_file.format('wide')
            narrow_path = im_file.format('narrow')
            
            both_images_exist = os.path.exists(wide_path) and os.path.exists(narrow_path)
            label_exists = os.path.exists(lb_file)
            
            if both_images_exist and label_exists:
                # Both images and label exist
                nf += 1
                # Get shape from wide image
                from PIL import Image
                try:
                    img = Image.open(wide_path)
                    shape = img.size[::-1]  # (h, w)
                    # Read labels
                    with open(lb_file, "r") as f:
                        lb_content = f.read().strip().splitlines()
                        if lb_content:
                            lb = np.array([x.split() for x in lb_content], dtype=np.float32)
                        else:
                            ne += 1
                            lb = np.zeros((0, 5), dtype=np.float32)
                    
                    segments = []  # No segment info
                    # Use original placeholder path
                    im_file_out = im_file
                    msg = ""
                
                except Exception as e:
                    nc += 1
                    msg = f"{prefix_str}WARNING ⚠️ {im_file}: corrupt image/label: {e}"
                    im_file_out, lb, shape, segments = None, None, None, None
            
            else:
                # Missing images or label
                if not label_exists:
                    nm += 1
                    msg = f"{prefix_str}WARNING ⚠️ {im_file}: missing label file {lb_file}"
                else:
                    nm += 1
                    msg = f"{prefix_str}WARNING ⚠️ {im_file}: missing wide or narrow image"
                
                im_file_out, lb, shape, segments = None, None, None, None
            
            if msg:
                msgs.append(msg)
            pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            data.append([im_file_out, lb, shape, segments])
        pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{prefix}WARNING ⚠️ No labels found in {path}.")
        x["data"] = data
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        x["version"] = self.cache_version  # cache version
        try:
            # Convert data structure to a json serializable format before saving
            import pickle
            with open(path, 'wb') as f:
                pickle.dump(x, f)
            LOGGER.info(f"{prefix}New cache created: {path}")
        except Exception as e:
            LOGGER.warning(f"{prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable: {e}")  # not writeable
        return x



    def load_image(self, i, rect_mode=False):
        """
        Loads dual-stream images (wide and narrow) by index.
        
        Args:
            i (int): Index of the image to load.
            rect_mode (bool): Whether to use rectangular mode (for inference).
            
        Returns:
            tuple: (imgs, hw0s, hw1s) where imgs is a tuple of [wide_img, narrow_img]
        """
        # Get image file path with {} placeholder
        im_file = self.im_files[i]
        if im_file is None:
            # Invalid image file (likely corrupt)
            return None, None, None
        
        # Load both modality images
        imgs = []
        
        # Process both modalities (wide and narrow)
        for modality in self.modalities:
            # Replace {} with modality name
            img_path = im_file.format(modality)
            
            try:
                img = cv2.imread(img_path)
                assert img is not None, f"Image not found: {img_path}"
            except Exception as e:
                LOGGER.warning(f"Error loading image {img_path}: {e}")
                # Create a black placeholder image
                img = np.zeros((640, 640, 3), dtype=np.uint8)
            
            # Add to the list
            imgs.append(img)
        
        if len(imgs) != 2:
            LOGGER.warning(f"Expected 2 images (wide, narrow) but got {len(imgs)}")
            return None, None, None
        
        # Get original dimensions from both images
        hw0 = [(img.shape[0], img.shape[1]) for img in imgs]
        
        # All images should use the same resize ratio (based on wide image)
        r = self.imgsz / max(hw0[0])  # ratio from wide image
        
        # Process all images with same ratio
        processed_imgs = []
        hw1 = []
        
        for img, (h0, w0) in zip(imgs, hw0):
            # Resize
            if r != 1:
                interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
                img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
            processed_imgs.append(img)
            hw1.append(img.shape[:2])
        
        return (processed_imgs[0], processed_imgs[1]), (hw0[0], hw0[1]), (hw1[0], hw1[1])

    def __getitem__(self, index):
        """
        Fetch dual-stream dataset item at given index.
        
        Returns:
            tuple: (imgs, labels_out, path, shapes, index) where imgs is tuple of [wide_img, narrow_img]
        """
        # Load dual-stream images
        imgs, hw0s, hw1s = self.load_image(index)
        
        # Process each image with letterbox and convert to tensors
        processed_imgs = []
        shapes = None
        
        for i, (img, (h0, w0), (h, w)) in enumerate(zip(imgs, hw0s, hw1s)):
            # Letterbox
            from ultralytics.data.augment import LetterBox
            letterbox = LetterBox(new_shape=(self.imgsz, self.imgsz), auto=False, scaleup=self.augment)
            transformed = letterbox(image=img)
            img_processed = transformed['image']
            
            # Get transformation info from first image only
            if i == 0:
                shapes = (h0, w0), ((h / h0, w / w0), (0, 0))  # for COCO mAP rescaling
            
            # Convert to torch tensor
            img_processed = img_processed.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img_processed = np.ascontiguousarray(img_processed)
            processed_imgs.append(torch.from_numpy(img_processed).float() / 255.0)
        
        # Get labels
        labels = self.labels[index].copy()
        nl = len(labels)  # number of labels
        labels_out = torch.zeros((nl, 6))  # 6 = batch_idx + cls + xywh
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)
        
        return tuple(processed_imgs), labels_out, self.im_files[index], shapes, index

    @staticmethod
    def collate_fn(batch):
        """Collates batch into model input tuple"""
        imgs, labels, paths, shapes, indices = zip(*batch)  # transposed
        for i, lb in enumerate(labels):
            lb[:, 0] = i  # add target image index for build_targets()
        
        wide_imgs = torch.stack([img[0] for img in imgs], 0)
        narrow_imgs = torch.stack([img[1] for img in imgs], 0)

        return (wide_imgs, narrow_imgs), torch.cat(labels, 0), paths, shapes, indices

    def get_img_files(self, img_path):
        """
        Override the get_img_files method to handle image paths with {} placeholders.
        
        Args:
            img_path (str | List[str]): Path or list of paths to image directories or files.
            
        Returns:
            (List[str]): List of image file paths, possibly with {} placeholders.
        """
        try:
            f = []  # image files
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                elif p.is_file():  # file
                    with open(p, encoding="utf-8") as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace("./", parent) if x.startswith("./") else x for x in t]
                else:
                    # For dual-stream paths with {}, check if both wide and narrow versions exist
                    wide_path = str(p).format('wide') if '{}' in str(p) else None
                    narrow_path = str(p).format('narrow') if '{}' in str(p) else None
                    
                    if wide_path and narrow_path and os.path.exists(wide_path) and os.path.exists(narrow_path):
                        # Path with {} placeholder is valid if both modalities exist
                        f.append(str(p))
                    else:
                        raise FileNotFoundError(f"{self.prefix}{p} does not exist or missing modalities")
            
            # Filter for valid image formats
            from ultralytics.data.utils import IMG_FORMATS
            im_files = sorted(x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS 
                              or ('{}' in x and any(x.format(m).split(".")[-1].lower() in IMG_FORMATS for m in self.modalities)))
            
            assert im_files, f"{self.prefix}No images found in {img_path}"
            
            # For dual-stream inputs with {}, we need to skip the speed check or modify it
            # We'll opt to skip it for paths containing {}
            valid_im_files = []
            placeholder_im_files = []
            
            for im_file in im_files:
                if '{}' in im_file:
                    # Verify both versions exist before adding
                    if all(os.path.exists(im_file.format(m)) for m in self.modalities):
                        placeholder_im_files.append(im_file)
                else:
                    valid_im_files.append(im_file)
            
            # Only run speed check on regular image files
            if valid_im_files:
                from ultralytics.data.utils import check_file_speeds
                check_file_speeds(valid_im_files, prefix=self.prefix)
            
            # Combine both lists, prioritizing placeholder files
            final_im_files = placeholder_im_files + valid_im_files
            
            if self.fraction < 1:
                final_im_files = final_im_files[: round(len(final_im_files) * self.fraction)]
                
            return final_im_files
            
        except Exception as e:
            # For debugging
            LOGGER.error(f"Error in get_img_files: {e}")
            import traceback
            LOGGER.error(traceback.format_exc())
            
            raise FileNotFoundError(f"{self.prefix}Error loading data from {img_path}\n") from e


class YOLOConcatDataset(ConcatDataset):
    """
    Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets for YOLO training, ensuring they use the same
    collation function.

    Methods:
        collate_fn: Static method that collates data samples into batches using YOLODataset's collation function.

    Examples:
        >>> dataset1 = YOLODataset(...)
        >>> dataset2 = YOLODataset(...)
        >>> combined_dataset = YOLOConcatDataset([dataset1, dataset2])
    """

    @staticmethod
    def collate_fn(batch):
        """
        Collates data samples into batches.

        Args:
            batch (List[dict]): List of dictionaries containing sample data.

        Returns:
            (dict): Collated batch with stacked tensors.
        """
        return YOLODataset.collate_fn(batch)

    def close_mosaic(self, hyp):
        """
        Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations.

        Args:
            hyp (dict): Hyperparameters for transforms.
        """
        for dataset in self.datasets:
            if not hasattr(dataset, "close_mosaic"):
                continue
            dataset.close_mosaic(hyp)


# TODO: support semantic segmentation
class SemanticDataset(BaseDataset):
    """Semantic Segmentation Dataset."""

    def __init__(self):
        """Initialize a SemanticDataset object."""
        super().__init__()


class ClassificationDataset:
    """
    Extends torchvision ImageFolder to support YOLO classification tasks.

    This class offers functionalities like image augmentation, caching, and verification. It's designed to efficiently
    handle large datasets for training deep learning models, with optional image transformations and caching mechanisms
    to speed up training.

    Attributes:
        cache_ram (bool): Indicates if caching in RAM is enabled.
        cache_disk (bool): Indicates if caching on disk is enabled.
        samples (list): A list of tuples, each containing the path to an image, its class index, path to its .npy cache
                        file (if caching on disk), and optionally the loaded image array (if caching in RAM).
        torch_transforms (callable): PyTorch transforms to be applied to the images.
        root (str): Root directory of the dataset.
        prefix (str): Prefix for logging and cache filenames.

    Methods:
        __getitem__: Returns subset of data and targets corresponding to given indices.
        __len__: Returns the total number of samples in the dataset.
        verify_images: Verifies all images in dataset.
    """

    def __init__(self, root, args, augment=False, prefix=""):
        """
        Initialize YOLO object with root, image size, augmentations, and cache settings.

        Args:
            root (str): Path to the dataset directory where images are stored in a class-specific folder structure.
            args (Namespace): Configuration containing dataset-related settings such as image size, augmentation
                parameters, and cache settings.
            augment (bool, optional): Whether to apply augmentations to the dataset.
            prefix (str, optional): Prefix for logging and cache filenames, aiding in dataset identification.
        """
        import torchvision  # scope for faster 'import ultralytics'

        # Base class assigned as attribute rather than used as base class to allow for scoping slow torchvision import
        if TORCHVISION_0_18:  # 'allow_empty' argument first introduced in torchvision 0.18
            self.base = torchvision.datasets.ImageFolder(root=root, allow_empty=True)
        else:
            self.base = torchvision.datasets.ImageFolder(root=root)
        self.samples = self.base.samples
        self.root = self.base.root

        # Initialize attributes
        if augment and args.fraction < 1.0:  # reduce training fraction
            self.samples = self.samples[: round(len(self.samples) * args.fraction)]
        self.prefix = colorstr(f"{prefix}: ") if prefix else ""
        self.cache_ram = args.cache is True or str(args.cache).lower() == "ram"  # cache images into RAM
        if self.cache_ram:
            LOGGER.warning(
                "Classification `cache_ram` training has known memory leak in "
                "https://github.com/ultralytics/ultralytics/issues/9824, setting `cache_ram=False`."
            )
            self.cache_ram = False
        self.cache_disk = str(args.cache).lower() == "disk"  # cache images on hard drive as uncompressed *.npy files
        self.samples = self.verify_images()  # filter out bad images
        self.samples = [list(x) + [Path(x[0]).with_suffix(".npy"), None] for x in self.samples]  # file, index, npy, im
        scale = (1.0 - args.scale, 1.0)  # (0.08, 1.0)
        self.torch_transforms = (
            classify_augmentations(
                size=args.imgsz,
                scale=scale,
                hflip=args.fliplr,
                vflip=args.flipud,
                erasing=args.erasing,
                auto_augment=args.auto_augment,
                hsv_h=args.hsv_h,
                hsv_s=args.hsv_s,
                hsv_v=args.hsv_v,
            )
            if augment
            else classify_transforms(size=args.imgsz)
        )

    def __getitem__(self, i):
        """
        Returns subset of data and targets corresponding to given indices.

        Args:
            i (int): Index of the sample to retrieve.

        Returns:
            (dict): Dictionary containing the image and its class index.
        """
        f, j, fn, im = self.samples[i]  # filename, index, filename.with_suffix('.npy'), image
        if self.cache_ram:
            if im is None:  # Warning: two separate if statements required here, do not combine this with previous line
                im = self.samples[i][3] = cv2.imread(f)
        elif self.cache_disk:
            if not fn.exists():  # load npy
                np.save(fn.as_posix(), cv2.imread(f), allow_pickle=False)
            im = np.load(fn)
        else:  # read image
            im = cv2.imread(f)  # BGR
        # Convert NumPy array to PIL image
        im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        sample = self.torch_transforms(im)
        return {"img": sample, "cls": j}

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.samples)

    def verify_images(self):
        """
        Verify all images in dataset.

        Returns:
            (list): List of valid samples after verification.
        """
        desc = f"{self.prefix}Scanning {self.root}..."
        path = Path(self.root).with_suffix(".cache")  # *.cache file path

        try:
            check_file_speeds([file for (file, _) in self.samples[:5]], prefix=self.prefix)  # check image read speeds
            cache = load_dataset_cache_file(path)  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash([x[0] for x in self.samples])  # identical hash
            nf, nc, n, samples = cache.pop("results")  # found, missing, empty, corrupt, total
            if LOCAL_RANK in {-1, 0}:
                d = f"{desc} {nf} images, {nc} corrupt"
                TQDM(None, desc=d, total=n, initial=n)
                if cache["msgs"]:
                    LOGGER.info("\n".join(cache["msgs"]))  # display warnings
            return samples

        except (FileNotFoundError, AssertionError, AttributeError):
            # Run scan if *.cache retrieval failed
            nf, nc, msgs, samples, x = 0, 0, [], [], {}
            with ThreadPool(NUM_THREADS) as pool:
                results = pool.imap(func=verify_image, iterable=zip(self.samples, repeat(self.prefix)))
                pbar = TQDM(results, desc=desc, total=len(self.samples))
                for sample, nf_f, nc_f, msg in pbar:
                    if nf_f:
                        samples.append(sample)
                    if msg:
                        msgs.append(msg)
                    nf += nf_f
                    nc += nc_f
                    pbar.desc = f"{desc} {nf} images, {nc} corrupt"
                pbar.close()
            if msgs:
                LOGGER.info("\n".join(msgs))
            x["hash"] = get_hash([x[0] for x in self.samples])
            x["results"] = nf, nc, len(samples), samples
            x["msgs"] = msgs  # warnings
            save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
            return samples
