"""Run baseline vs regular_augmentations training and compare generated batch images."""

from __future__ import annotations

import shutil
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from ultralytics import YOLO

from custom_trainer import CustomerTrainer
from runtime import select_device_from_env


ROOT = Path("runs/compare_regular_augmentations")


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _shared_train_kwargs(device: str | int) -> dict:
    return {
        "data": "coco8.yaml",
        "epochs": 1,
        "imgsz": 320,
        "batch": 2,
        "workers": 0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "fliplr": 0.0,
        "flipud": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "seed": 0,
        "deterministic": True,
        "device": device,
        "project": str(ROOT.resolve()),
        "exist_ok": True,
    }


def _train_baseline(device: str | int) -> Path:
    model = YOLO("yolo26n.pt")
    model.train(**_shared_train_kwargs(device), name="baseline")
    return ROOT / "baseline"


def _train_regular(device: str | int) -> Path:
    model = YOLO("yolo26n.pt")
    model.train(
        **_shared_train_kwargs(device),
        trainer=CustomerTrainer,
        name="regular",
        regular_augmentations=[
            A.ToGray(p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.45, contrast_limit=0.0, p=1.0),
        ],
    )
    return ROOT / "regular"


def _compare_batches(baseline_dir: Path, regular_dir: Path) -> None:
    comparison_dir = ROOT / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    lines = ["batch,mean_abs_diff,max_abs_diff,changed_pixels"]

    baseline_batches = sorted(baseline_dir.glob("train_batch*.jpg"))
    if not baseline_batches:
        raise FileNotFoundError(f"No baseline train_batch*.jpg files found in {baseline_dir}")

    for baseline_path in baseline_batches:
        regular_path = regular_dir / baseline_path.name
        if not regular_path.exists():
            raise FileNotFoundError(f"Missing regular batch image: {regular_path}")

        baseline = cv2.imread(str(baseline_path))
        regular = cv2.imread(str(regular_path))
        if baseline is None or regular is None:
            raise RuntimeError(f"Could not read {baseline_path} or {regular_path}")
        if baseline.shape != regular.shape:
            raise AssertionError(f"Shape mismatch for {baseline_path.name}: {baseline.shape} vs {regular.shape}")

        diff = cv2.absdiff(baseline, regular)
        mean_abs_diff = float(diff.mean())
        max_abs_diff = int(diff.max())
        changed_pixels = int(np.count_nonzero(diff.any(axis=2)))
        if changed_pixels == 0:
            raise AssertionError(f"{baseline_path.name} is identical; expected regular augmentations to change it.")

        side_by_side = np.concatenate([baseline, regular, diff], axis=1)
        cv2.imwrite(str(comparison_dir / f"{baseline_path.stem}_baseline_regular_diff.jpg"), side_by_side)
        lines.append(f"{baseline_path.name},{mean_abs_diff:.4f},{max_abs_diff},{changed_pixels}")

    (comparison_dir / "summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    device = select_device_from_env()
    print(f"Using YOLO device: {device}")
    _reset_dir(ROOT)
    baseline_dir = _train_baseline(device)
    regular_dir = _train_regular(device)
    _compare_batches(baseline_dir, regular_dir)
    print(f"Baseline run: {baseline_dir.resolve()}")
    print(f"Regular augmentation run: {regular_dir.resolve()}")
    print(f"Comparison output: {(ROOT / 'comparison').resolve()}")


if __name__ == "__main__":
    main()
