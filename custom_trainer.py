"""Custom Ultralytics YOLO detection trainer with pre-mosaic Albumentations.

Ultralytics detection training has two important layers:

1. A trainer (`DetectionTrainer`) decides which dataset class to instantiate.
2. A dataset (`YOLODataset`) loads one image/label pair with `get_image_and_label()`, then
   the dataset's transform chain applies mosaic, mixup, perspective, regular Albumentations,
   flips, formatting, and batching.

This module customizes those two seams only. The trainer swaps in a training-only dataset
subclass, and that dataset applies optional Albumentations immediately after Ultralytics has
loaded the image and created its `Instances` annotation object. Because Ultralytics' own mosaic,
mixup, and cutmix transforms fetch their source images through `dataset.get_image_and_label()`,
placing the transform there means every image they see has already been pretransformed.
"""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import numpy as np


class AlbumentationsPreTransform:
    """Apply Albumentations to Ultralytics labels before the normal YOLO transform chain.

    Ultralytics already has a regular Albumentations hook later in `v8_transforms()`. That hook
    runs after mosaic/perspective-style transforms. This class is for the earlier hook requested
    here: transforms that change the raw loaded image before mosaic has a chance to compose it.
    """

    def __init__(self, transforms: list[Any] | None = None, p: float = 1.0) -> None:
        # Store user-supplied Albumentations transform objects as-is. The example scripts pass
        # normal Albumentations instances, e.g. `A.CenterCrop(...)` or `A.RandomBrightnessContrast(...)`.
        self.transforms = transforms or []
        self.p = p
        self.transform = None
        self.contains_spatial = False

        if not self.transforms:
            return

        import albumentations as A
        from albumentations.core.transforms_interface import DualTransform

        # Albumentations only needs bbox metadata when a transform can move pixels spatially.
        # Image-only transforms can run without bboxes, which avoids unnecessary conversions and
        # keeps behavior close to Ultralytics' built-in `Albumentations` transform. The primary
        # check below uses Albumentations' own `DualTransform` base class so newly added spatial
        # transforms are detected automatically; the name set is a fallback for wrappers/custom
        # objects whose class names match the official targets matrix.
        spatial_transforms = {
            "Affine",
            "AtLeastOneBBoxRandomCrop",
            "BBoxSafeRandomCrop",
            "CenterCrop",
            "CoarseDropout",
            "ConstrainedCoarseDropout",
            "CopyAndPaste",
            "Crop",
            "CropAndPad",
            "CropNonEmptyMaskIfExists",
            "D4",
            "ElasticTransform",
            "Erasing",
            "FrequencyMasking",
            "Flip",
            "GridDistortion",
            "GridDropout",
            "GridElasticDeform",
            "GridMask",
            "HorizontalFlip",
            "LetterBox",
            "LongestMaxSize",
            "MaskDropout",
            "Morphological",
            "Mosaic",
            "NoOp",
            "OpticalDistortion",
            "OverlayElements",
            "Pad",
            "PadIfNeeded",
            "Perspective",
            "PiecewiseAffine",
            "PixelDropout",
            "PixelSpread",
            "RandomCrop",
            "RandomCropFromBorders",
            "RandomCropNearBBox",
            "RandomGridShuffle",
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
            "SquareSymmetry",
            "ThinPlateSpline",
            "TimeMasking",
            "TimeReverse",
            "Transpose",
            "VerticalFlip",
            "WaterRefraction",
            "XYMasking",
        }
        self.contains_spatial = any(
            isinstance(t, DualTransform) or t.__class__.__name__ in spatial_transforms for t in self.transforms
        )
        if self.contains_spatial:
            # Ultralytics stores detection boxes as YOLO-style xywh normalized boxes at this
            # point, which matches Albumentations' "yolo" bbox format. `label_fields` keeps class
            # ids paired with boxes when Albumentations drops boxes outside a crop.
            self.transform = A.Compose(
                self.transforms,
                bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.0),
            )
        else:
            self.transform = A.Compose(self.transforms)

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        # Ultralytics transform objects all receive and return a mutable `labels` dict. The
        # important keys here are:
        #   img       -> HWC numpy image
        #   cls       -> class ids shaped (N, 1)
        #   instances -> Ultralytics `Instances` object containing boxes/segments/keypoints
        if self.transform is None or random.random() > self.p:
            return labels

        image = labels["img"]
        # Albumentations image transforms expect a regular 3-channel image. If a future dataset
        # uses another channel count, leave it untouched instead of guessing how to adapt boxes.
        if image.ndim != 3 or image.shape[2] != 3:
            return labels

        if not self.contains_spatial:
            # Image-only transforms do not affect geometry, so `instances` and `cls` stay valid.
            # Only update `resized_shape` because downstream Ultralytics code may consult it.
            labels["img"] = self.transform(image=image)["image"]
            labels["resized_shape"] = labels["img"].shape[:2]
            return labels

        instances = labels["instances"]
        cls = labels["cls"].reshape(-1, 1).astype(np.float32)
        # `Instances` methods assume `segments` is an ndarray. Pure detection labels may not
        # carry segments in synthetic tests or future Ultralytics changes, so create an empty
        # segment array to keep bbox-only operations safe.
        if instances.segments is None:
            instances.segments = np.zeros((len(cls), 0, 2), dtype=np.float32)
        # Make the box representation exactly what Albumentations expects:
        # normalized YOLO xywh, i.e. [x_center, y_center, width, height] in 0..1.
        instances.convert_bbox("xywh")
        instances.normalize(*image.shape[:2][::-1])
        boxes = instances.bboxes.astype(np.float32)

        # Albumentations returns only boxes that survive the spatial transform. Because `cls` was
        # passed as `class_labels`, it returns the matching class labels in the same filtered order.
        transformed = self.transform(image=image, bboxes=boxes, class_labels=cls)
        labels["img"] = transformed["image"]
        labels["resized_shape"] = labels["img"].shape[:2]

        # Normalize array shapes for the zero-box case. Without the reshape, numpy can produce
        # shape `(0,)`, which later code cannot treat as an `(N, 4)` bbox matrix.
        new_boxes = np.asarray(transformed["bboxes"], dtype=np.float32).reshape(-1, 4)
        new_cls = np.asarray(transformed["class_labels"], dtype=np.float32).reshape(-1, 1)

        instances.update(bboxes=new_boxes)
        # Albumentations may return boxes that touch or cross the new image border after crop-like
        # transforms. Ultralytics later expects valid boxes, so convert to absolute xyxy, clip to
        # the new image, drop zero-area boxes, then return to normalized xywh.
        instances.convert_bbox("xyxy")
        instances.denormalize(*labels["img"].shape[:2][::-1])
        instances.clip(*labels["img"].shape[:2][::-1])
        keep = instances.remove_zero_area_boxes()
        instances.normalize(*labels["img"].shape[:2][::-1])
        instances.convert_bbox("xywh")

        labels["instances"] = instances
        # `keep` accounts for an extra safety filter after clipping. Albumentations already
        # removes many invalid boxes, but clipping can still turn a border-touching box into zero
        # area, so class labels must be filtered one last time to stay aligned with boxes.
        labels["cls"] = new_cls[keep] if len(new_cls) else new_cls
        return labels


class CustomYOLODatasetMixin:
    """Mixin that inserts pretransforms after image loading and before YOLO augmentation.

    `YOLODataset.__getitem__()` is effectively:

        return self.transforms(self.get_image_and_label(index))

    Overriding `get_image_and_label()` is the narrowest useful hook because the label dict already
    has `img`, `cls`, and `instances`, but the normal transform chain has not started yet. That is
    why this runs before mosaic, and why it also affects the extra images that mosaic/mixup/cutmix
    request internally.
    """

    def __init__(
        self,
        *args: Any,
        pretransform_augmentations: list[Any] | None = None,
        debug_pretransform_samples: int = 0,
        debug_pretransform_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        # This object is created once per dataset. It owns the Albumentations compose pipeline and
        # is reused every time Ultralytics asks the dataset for an image/label sample.
        self.pretransform = AlbumentationsPreTransform(pretransform_augmentations)
        self.debug_pretransform_samples = int(debug_pretransform_samples or 0)
        self.debug_pretransform_dir = Path(debug_pretransform_dir) if debug_pretransform_dir else None
        self._debug_pretransform_count = 0
        super().__init__(*args, **kwargs)

    def get_image_and_label(self, index: int) -> dict[str, Any]:
        labels = super().get_image_and_label(index)
        if self.augment:
            # Debug captures are intentionally taken here, not after `__getitem__()`, so the saved
            # images prove what the training transform chain receives as pre-mosaic inputs.
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
        # Include both capture order and dataset index. With shuffling or mix transforms, the same
        # index may appear more than once, and this keeps every debug pair distinct.
        sample_id = f"{self._debug_pretransform_count:03d}_idx{index}"
        self.debug_pretransform_dir.mkdir(parents=True, exist_ok=True)
        _save_debug_overlay(before, self.debug_pretransform_dir / f"{sample_id}_before.jpg")
        _save_debug_overlay(after, self.debug_pretransform_dir / f"{sample_id}_after.jpg")


def _save_debug_overlay(labels: dict[str, Any], path: Path) -> None:
    import cv2

    # These overlays are diagnostic only. They are not used by training, but they make it easy to
    # inspect whether spatial pretransforms kept boxes aligned with visible objects.
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
    # Work on a copy because debug drawing should never mutate the training label object. The
    # training pipeline will continue using the original `labels` dict after debug images are saved.
    instances = copy.deepcopy(labels["instances"])
    if instances.segments is None:
        instances.segments = np.zeros((len(instances.bboxes), 0, 2), dtype=np.float32)
    instances.convert_bbox("xywh")
    instances.normalize(*labels["img"].shape[:2][::-1])
    return instances.bboxes.astype(np.float32)


def _custom_dataset_class():
    from ultralytics.data.dataset import YOLODataset

    # Build the concrete subclass lazily so importing this module does not require Ultralytics to
    # be installed yet. That keeps basic tooling and tests around helper code lighter.
    class CustomYOLODataset(CustomYOLODatasetMixin, YOLODataset):
        pass

    return CustomYOLODataset


class CustomerTrainer:
    """Factory that returns a lazily defined `DetectionTrainer` subclass.

    Ultralytics accepts `YOLO(...).train(trainer=SomeTrainerClass)`. It then instantiates that
    class with an `overrides` dict. We remove our custom keys before calling the base trainer
    because Ultralytics validates override names against its config schema.
    """

    def __new__(cls, *args: Any, **kwargs: Any):
        from ultralytics.models.yolo.detect import DetectionTrainer

        # Define the real trainer subclass inside `__new__` so Ultralytics is imported only when a
        # training run actually requests this trainer. The returned object is still a normal
        # `DetectionTrainer` instance from Ultralytics' point of view.
        class _CustomerTrainer(DetectionTrainer):
            def __init__(self, cfg=None, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
                from ultralytics.utils import DEFAULT_CFG

                overrides = copy.copy(overrides) if overrides else {}
                # Keep the custom options as trainer attributes. They are later used when
                # `build_dataset()` creates the training dataset.
                self.pretransform_augmentations = overrides.pop("pretransform_augmentations", None)
                self.debug_pretransform_samples = int(overrides.pop("debug_pretransform_samples", 0) or 0)
                regular = overrides.pop("regular_augmentations", None)
                if regular is not None:
                    # Current Ultralytics already supports custom post-mosaic Albumentations under
                    # the `augmentations` hyperparameter. `regular_augmentations` is just a clearer
                    # public alias so callers can distinguish pre- and regular transforms.
                    overrides["augmentations"] = regular
                # `YOLO.train(trainer=...)` may instantiate the trainer with `cfg=None`; the base
                # trainer expects a config dict/path, so use Ultralytics' default config explicitly.
                cfg = DEFAULT_CFG if cfg is None else cfg
                super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)

            def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
                from ultralytics.utils import colorstr
                from ultralytics.utils.torch_utils import unwrap_model

                # Only training gets the custom dataset. Validation uses the parent implementation
                # so validation remains comparable with normal Ultralytics training.
                dataset_class = _custom_dataset_class() if mode == "train" else None
                if dataset_class is None:
                    # Validation/test data should stay standard. Pretransforms are a training
                    # augmentation, and changing validation images would corrupt metrics.
                    return super().build_dataset(img_path, mode=mode, batch=batch)

                # Ultralytics datasets need the model stride to build rectangular/multi-scale
                # shapes safely. This is copied from the stock `DetectionTrainer.build_dataset()`.
                gs = max(int(unwrap_model(self.model).stride.max()), 32)
                # This mirrors Ultralytics' `build_yolo_dataset()` call, but swaps in our dataset
                # subclass for training so `get_image_and_label()` can do the pre-mosaic work.
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
