"""One-epoch smoke training example for CustomTrainer."""

from __future__ import annotations

import albumentations as A
from ultralytics import YOLO

from custom_trainer import CustomTrainer
from runtime import select_device_from_env


def main() -> None:
    device = select_device_from_env()
    print(f"Using YOLO device: {device}")

    model = YOLO("yolo26n.pt")
    model.train(
        data="coco8.yaml",
        trainer=CustomTrainer,
        epochs=1,
        imgsz=320,
        batch=2,
        workers=0,
        mosaic=0.0,
        # coco8 is tiny and does not contain COCO class 2 (car), so keep all classes for the smoke test.
        # For a COCO-style dataset with cars, pass classes=[2].
        device=device,
        debug_pretransform_samples=4,
        pretransform_augmentations=[
            A.RandomSizedBBoxSafeCrop(height=288, width=288, erosion_rate=0.1, p=1.0),
        ],
        regular_augmentations=[
            A.RandomBrightnessContrast(p=0.2),
        ],
    )


if __name__ == "__main__":
    main()
