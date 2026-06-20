"""Run baseline vs CustomerTrainer training and compare generated batch images."""

from __future__ import annotations

import shutil
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from ultralytics import YOLO

from custom_trainer import CustomerTrainer, select_device_from_env


ROOT = Path("runs/compare_pretransform")


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _train_baseline(device: str | int) -> Path:
    project = ROOT.resolve()
    model = YOLO("yolo26n.pt")
    model.train(
        data="coco8.yaml",
        epochs=1,
        imgsz=320,
        batch=2,
        workers=0,
        mosaic=0.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        fliplr=0.0,
        flipud=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        seed=0,
        deterministic=True,
        device=device,
        project=str(project),
        name="baseline",
        exist_ok=True,
    )
    return ROOT / "baseline"


def _train_custom(device: str | int) -> Path:
    project = ROOT.resolve()
    model = YOLO("yolo26n.pt")
    model.train(
        data="coco8.yaml",
        trainer=CustomerTrainer,
        epochs=1,
        imgsz=320,
        batch=2,
        workers=0,
        mosaic=0.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        fliplr=0.0,
        flipud=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        seed=0,
        deterministic=True,
        device=device,
        project=str(project),
        name="custom",
        exist_ok=True,
        debug_pretransform_samples=4,
        pretransform_augmentations=[
            A.CenterCrop(height=200, width=200, p=1.0),
        ],
    )
    return ROOT / "custom"


def _compare_batches(baseline_dir: Path, custom_dir: Path) -> None:
    comparison_dir = ROOT / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    lines = ["batch,mean_abs_diff,max_abs_diff,changed_pixels"]

    baseline_batches = sorted(baseline_dir.glob("train_batch*.jpg"))
    if not baseline_batches:
        raise FileNotFoundError(f"No baseline train_batch*.jpg files found in {baseline_dir}")

    for baseline_path in baseline_batches:
        custom_path = custom_dir / baseline_path.name
        if not custom_path.exists():
            raise FileNotFoundError(f"Missing custom batch image: {custom_path}")

        baseline = cv2.imread(str(baseline_path))
        custom = cv2.imread(str(custom_path))
        if baseline is None or custom is None:
            raise RuntimeError(f"Could not read {baseline_path} or {custom_path}")
        if baseline.shape != custom.shape:
            raise AssertionError(f"Shape mismatch for {baseline_path.name}: {baseline.shape} vs {custom.shape}")

        diff = cv2.absdiff(baseline, custom)
        mean_abs_diff = float(diff.mean())
        max_abs_diff = int(diff.max())
        changed_pixels = int(np.count_nonzero(diff.any(axis=2)))
        if changed_pixels == 0:
            raise AssertionError(f"{baseline_path.name} is identical; expected crop pretransform to change it.")

        side_by_side = np.concatenate([baseline, custom, diff], axis=1)
        cv2.imwrite(str(comparison_dir / f"{baseline_path.stem}_baseline_custom_diff.jpg"), side_by_side)
        lines.append(f"{baseline_path.name},{mean_abs_diff:.4f},{max_abs_diff},{changed_pixels}")

    (comparison_dir / "summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    device = select_device_from_env()
    print(f"Using YOLO device: {device}")
    _reset_dir(ROOT)
    baseline_dir = _train_baseline(device)
    custom_dir = _train_custom(device)
    _compare_batches(baseline_dir, custom_dir)
    print(f"Baseline run: {baseline_dir.resolve()}")
    print(f"Custom run: {custom_dir.resolve()}")
    print(f"Comparison output: {(ROOT / 'comparison').resolve()}")


if __name__ == "__main__":
    main()
