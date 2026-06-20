"""Verify pre-mosaic Albumentations before running real YOLO training."""

from __future__ import annotations

import shutil
from pathlib import Path

import albumentations as A
import cv2
import numpy as np

from custom_trainer import CustomYOLODatasetMixin
from runtime import select_device_from_env


ROOT = Path("runs/pretransform_debug")


def _write_dataset(root: Path) -> Path:
    if root.exists():
        shutil.rmtree(root)
    image_dir = root / "images" / "train"
    label_dir = root / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    image = np.full((100, 100, 3), 245, dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (60, 60), (0, 160, 255), -1)
    cv2.rectangle(image, (80, 80), (96, 96), (255, 80, 0), -1)
    cv2.imwrite(str(image_dir / "sample.jpg"), image)
    (label_dir / "sample.txt").write_text("0 0.40 0.40 0.40 0.40\n0 0.88 0.88 0.16 0.16\n", encoding="utf-8")

    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        f"path: {root.resolve()}\ntrain: images/train\nval: images/train\nnames:\n  0: car\n",
        encoding="utf-8",
    )
    return data_yaml


def _draw_boxes(image: np.ndarray, boxes: np.ndarray, path: Path) -> None:
    out = image.copy()
    h, w = out.shape[:2]
    for x, y, bw, bh in boxes:
        x1 = int((x - bw / 2) * w)
        y1 = int((y - bh / 2) * h)
        x2 = int((x + bw / 2) * w)
        y2 = int((y + bh / 2) * h)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite(str(path), out)


def main() -> None:
    device = select_device_from_env()
    data_yaml = _write_dataset(ROOT)

    from ultralytics.cfg import get_cfg
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.data.dataset import YOLODataset

    class InspectDataset(CustomYOLODatasetMixin, YOLODataset):
        pass

    data = check_det_dataset(str(data_yaml))
    base_overrides = {
            "task": "detect",
            "mode": "train",
            "imgsz": 100,
            "batch": 1,
            "rect": False,
            "cache": False,
            "single_cls": False,
            "classes": None,
            "fraction": 1.0,
            "mosaic": 0.0,
            "mixup": 0.0,
            "cutmix": 0.0,
            "copy_paste": 0.0,
        }
    hyp = get_cfg(overrides=base_overrides)

    dataset = InspectDataset(
        img_path=data["train"],
        imgsz=100,
        batch_size=1,
        augment=True,
        hyp=hyp,
        rect=False,
        cache=None,
        single_cls=False,
        stride=32,
        pad=0.0,
        prefix="inspect: ",
        task="detect",
        classes=None,
        data=data,
        fraction=1.0,
        pretransform_augmentations=[A.Crop(x_min=10, y_min=10, x_max=70, y_max=70, p=1.0)],
    )

    original = cv2.imread(str(ROOT / "images" / "train" / "sample.jpg"))
    _draw_boxes(original, np.array([[0.40, 0.40, 0.40, 0.40], [0.88, 0.88, 0.16, 0.16]], dtype=np.float32), ROOT / "before.jpg")

    transformed = dataset.get_image_and_label(0)
    boxes = transformed["instances"].bboxes
    expected = np.array([[0.5, 0.5, 2 / 3, 2 / 3]], dtype=np.float32)

    assert transformed["img"].shape[:2] == (60, 60), transformed["img"].shape
    assert boxes.shape == (1, 4), boxes
    assert np.allclose(boxes, expected, atol=1e-4), boxes
    assert np.all((boxes >= 0.0) & (boxes <= 1.0)), boxes

    _draw_boxes(transformed["img"], boxes, ROOT / "after_pretransform.jpg")
    sample = dataset[0]
    assert sample["bboxes"].numel() > 0
    assert bool(((sample["bboxes"] >= 0.0) & (sample["bboxes"] <= 1.0)).all())

    mosaic_hyp = get_cfg(overrides={**base_overrides, "mosaic": 1.0})
    mosaic_dataset = InspectDataset(
        img_path=data["train"],
        imgsz=100,
        batch_size=1,
        augment=True,
        hyp=mosaic_hyp,
        rect=False,
        cache=None,
        single_cls=False,
        stride=32,
        pad=0.0,
        prefix="inspect-mosaic: ",
        task="detect",
        classes=None,
        data=data,
        fraction=1.0,
        pretransform_augmentations=[A.Crop(x_min=10, y_min=10, x_max=70, y_max=70, p=1.0)],
    )
    mosaic_sample = mosaic_dataset[0]
    assert mosaic_sample["bboxes"].numel() > 0
    assert bool(((mosaic_sample["bboxes"] >= 0.0) & (mosaic_sample["bboxes"] <= 1.0)).all())

    print(f"Pretransform inspection passed on device setting '{device}'.")
    print(f"Wrote debug overlays to {ROOT.resolve()}")


if __name__ == "__main__":
    main()
