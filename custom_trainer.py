"""Custom Ultralytics YOLO detection trainer with pre-mosaic Albumentations."""

from __future__ import annotations

import copy
import os
import random
from pathlib import Path
from typing import Any

import numpy as np


class AlbumentationsPreTransform:
    """Apply Albumentations to Ultralytics labels before the normal YOLO transform chain."""

    def __init__(self, transforms: list[Any] | None = None, p: float = 1.0) -> None:
        self.transforms = transforms or []
        self.p = p
        self.transform = None
        self.contains_spatial = False

        if not self.transforms:
            return

        os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
        import albumentations as A

        spatial_transforms = {
            "Affine",
            "BBoxSafeRandomCrop",
            "CenterCrop",
            "Crop",
            "CropAndPad",
            "D4",
            "ElasticTransform",
            "Flip",
            "GridDistortion",
            "HorizontalFlip",
            "LongestMaxSize",
            "NoOp",
            "OpticalDistortion",
            "PadIfNeeded",
            "Perspective",
            "PiecewiseAffine",
            "RandomCrop",
            "RandomCropFromBorders",
            "RandomResizedCrop",
            "RandomRotate90",
            "RandomScale",
            "RandomSizedBBoxSafeCrop",
            "RandomSizedCrop",
            "Resize",
            "Rotate",
            "SafeRotate",
            "ShiftScaleRotate",
            "SmallestMaxSize",
            "Transpose",
            "VerticalFlip",
        }
        self.contains_spatial = any(t.__class__.__name__ in spatial_transforms for t in self.transforms)
        if self.contains_spatial:
            self.transform = A.Compose(
                self.transforms,
                bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.0),
            )
        else:
            self.transform = A.Compose(self.transforms)

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        if self.transform is None or random.random() > self.p:
            return labels

        image = labels["img"]
        if image.ndim != 3 or image.shape[2] != 3:
            return labels

        if not self.contains_spatial:
            labels["img"] = self.transform(image=image)["image"]
            labels["resized_shape"] = labels["img"].shape[:2]
            return labels

        instances = labels["instances"]
        cls = labels["cls"].reshape(-1, 1).astype(np.float32)
        if instances.segments is None:
            instances.segments = np.zeros((len(cls), 0, 2), dtype=np.float32)
        instances.convert_bbox("xywh")
        instances.normalize(*image.shape[:2][::-1])
        boxes = instances.bboxes.astype(np.float32)

        transformed = self.transform(image=image, bboxes=boxes, class_labels=cls)
        labels["img"] = transformed["image"]
        labels["resized_shape"] = labels["img"].shape[:2]

        new_boxes = np.asarray(transformed["bboxes"], dtype=np.float32).reshape(-1, 4)
        new_cls = np.asarray(transformed["class_labels"], dtype=np.float32).reshape(-1, 1)

        instances.update(bboxes=new_boxes)
        instances.convert_bbox("xyxy")
        instances.denormalize(*labels["img"].shape[:2][::-1])
        instances.clip(*labels["img"].shape[:2][::-1])
        keep = instances.remove_zero_area_boxes()
        instances.normalize(*labels["img"].shape[:2][::-1])
        instances.convert_bbox("xywh")

        labels["instances"] = instances
        labels["cls"] = new_cls[keep] if len(new_cls) else new_cls
        return labels


class CustomYOLODatasetMixin:
    """Mixin that inserts pretransforms after image loading and before YOLO augmentation."""

    def __init__(
        self,
        *args: Any,
        pretransform_augmentations: list[Any] | None = None,
        debug_pretransform_samples: int = 0,
        debug_pretransform_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        self.pretransform = AlbumentationsPreTransform(pretransform_augmentations)
        self.debug_pretransform_samples = int(debug_pretransform_samples or 0)
        self.debug_pretransform_dir = Path(debug_pretransform_dir) if debug_pretransform_dir else None
        self._debug_pretransform_count = 0
        super().__init__(*args, **kwargs)

    def get_image_and_label(self, index: int) -> dict[str, Any]:
        labels = super().get_image_and_label(index)
        if self.augment:
            before = copy.deepcopy(labels) if self._should_debug_pretransform() else None
            labels = self.pretransform(labels)
            if before is not None:
                self._save_pretransform_debug_pair(index, before, labels)
        return labels

    def _should_debug_pretransform(self) -> bool:
        return (
            self.debug_pretransform_samples > 0
            and self.debug_pretransform_dir is not None
            and self._debug_pretransform_count < self.debug_pretransform_samples
        )

    def _save_pretransform_debug_pair(self, index: int, before: dict[str, Any], after: dict[str, Any]) -> None:
        self._debug_pretransform_count += 1
        sample_id = f"{self._debug_pretransform_count:03d}_idx{index}"
        self.debug_pretransform_dir.mkdir(parents=True, exist_ok=True)
        _save_debug_overlay(before, self.debug_pretransform_dir / f"{sample_id}_before.jpg")
        _save_debug_overlay(after, self.debug_pretransform_dir / f"{sample_id}_after.jpg")


def _save_debug_overlay(labels: dict[str, Any], path: Path) -> None:
    import cv2

    image = labels["img"].copy()
    h, w = image.shape[:2]
    boxes = _normalized_xywh_boxes(labels)
    for x, y, bw, bh in boxes:
        x1 = int(round((x - bw / 2) * w))
        y1 = int(round((y - bh / 2) * h))
        x2 = int(round((x + bw / 2) * w))
        y2 = int(round((y + bh / 2) * h))
        cv2.rectangle(image, (max(x1, 0), max(y1, 0)), (min(x2, w - 1), min(y2, h - 1)), (0, 0, 255), 2)
    cv2.imwrite(str(path), image)


def _normalized_xywh_boxes(labels: dict[str, Any]) -> np.ndarray:
    instances = copy.deepcopy(labels["instances"])
    if instances.segments is None:
        instances.segments = np.zeros((len(instances.bboxes), 0, 2), dtype=np.float32)
    instances.convert_bbox("xywh")
    instances.normalize(*labels["img"].shape[:2][::-1])
    return instances.bboxes.astype(np.float32)


def _custom_dataset_class():
    from ultralytics.data.dataset import YOLODataset

    class CustomYOLODataset(CustomYOLODatasetMixin, YOLODataset):
        pass

    return CustomYOLODataset


class CustomerTrainer:
    """Detection trainer factory subclass loaded lazily so imports work before Pixi install."""

    def __new__(cls, *args: Any, **kwargs: Any):
        from ultralytics.models.yolo.detect import DetectionTrainer

        class _CustomerTrainer(DetectionTrainer):
            def __init__(self, cfg=None, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
                from ultralytics.utils import DEFAULT_CFG

                overrides = copy.copy(overrides) if overrides else {}
                self.pretransform_augmentations = overrides.pop("pretransform_augmentations", None)
                self.debug_pretransform_samples = int(overrides.pop("debug_pretransform_samples", 0) or 0)
                regular = overrides.pop("regular_augmentations", None)
                if regular is not None:
                    overrides["augmentations"] = regular
                cfg = DEFAULT_CFG if cfg is None else cfg
                super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)

            def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
                from ultralytics.utils import colorstr
                from ultralytics.utils.torch_utils import unwrap_model

                dataset_class = _custom_dataset_class() if mode == "train" else None
                if dataset_class is None:
                    return super().build_dataset(img_path, mode=mode, batch=batch)

                gs = max(int(unwrap_model(self.model).stride.max()), 32)
                return dataset_class(
                    img_path=img_path,
                    imgsz=self.args.imgsz,
                    batch_size=batch,
                    augment=True,
                    hyp=self.args,
                    rect=self.args.rect,
                    cache=self.args.cache or None,
                    single_cls=self.args.single_cls or False,
                    stride=gs,
                    pad=0.0,
                    prefix=colorstr(f"{mode}: "),
                    task=self.args.task,
                    classes=self.args.classes,
                    data=self.data,
                    fraction=self.args.fraction,
                    pretransform_augmentations=self.pretransform_augmentations,
                    debug_pretransform_samples=self.debug_pretransform_samples,
                    debug_pretransform_dir=self.save_dir / "pretransform_debug",
                )

        return _CustomerTrainer(*args, **kwargs)


def select_device_from_env() -> str:
    """Return the requested YOLO device, requiring explicit GPU opt-in."""

    requested = os.environ.get("YOLO_DEVICE", "cpu").strip().lower()
    if requested in {"", "cpu"}:
        return "cpu"

    import torch

    if requested in {"cuda", "cuda:0", "0"}:
        if not torch.cuda.is_available():
            raise RuntimeError("YOLO_DEVICE=cuda was requested, but CUDA is not available in this Pixi environment.")
        return 0

    if requested == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("YOLO_DEVICE=mps was requested, but MPS is not available in this Pixi environment.")
        return "mps"

    raise ValueError("YOLO_DEVICE must be one of: cpu, cuda, mps.")
