from __future__ import annotations

import os
from types import SimpleNamespace
from unittest import mock

import albumentations as A
import numpy as np
import pytest
from ultralytics.utils.instance import Instances

from custom_trainer import AlbumentationsPreTransform, CustomerTrainer, select_device_from_env


def labels(boxes: np.ndarray) -> dict:
    return {
        "img": np.full((100, 100, 3), 128, dtype=np.uint8),
        "cls": np.zeros((len(boxes), 1), dtype=np.float32),
        "instances": Instances(boxes.copy(), bbox_format="xywh", normalized=True),
        "resized_shape": (100, 100),
    }


def test_precrop_updates_partial_box_and_removes_outside_box() -> None:
    transform = AlbumentationsPreTransform([A.Crop(x_min=10, y_min=10, x_max=70, y_max=70, p=1.0)])
    out = transform(labels(np.array([[0.40, 0.40, 0.40, 0.40], [0.88, 0.88, 0.16, 0.16]], dtype=np.float32)))

    assert out["img"].shape[:2] == (60, 60)
    assert out["instances"].bboxes.shape == (1, 4)
    np.testing.assert_allclose(out["instances"].bboxes, np.array([[0.5, 0.5, 2 / 3, 2 / 3]], dtype=np.float32), atol=1e-4)
    assert out["cls"].shape == (1, 1)


def test_precrop_allows_empty_labels() -> None:
    transform = AlbumentationsPreTransform([A.Crop(x_min=0, y_min=0, x_max=30, y_max=30, p=1.0)])
    out = transform(labels(np.array([[0.88, 0.88, 0.16, 0.16]], dtype=np.float32)))

    assert out["img"].shape[:2] == (30, 30)
    assert out["instances"].bboxes.shape == (0, 4)
    assert out["cls"].shape == (0, 1)


def test_image_only_pretransform_preserves_boxes() -> None:
    transform = AlbumentationsPreTransform([A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0, p=1.0)])
    box = np.array([[0.4, 0.4, 0.4, 0.4]], dtype=np.float32)
    out = transform(labels(box))

    np.testing.assert_allclose(out["instances"].bboxes, box)
    assert out["img"].shape[:2] == (100, 100)


def test_regular_augmentations_alias_maps_to_ultralytics_augmentations() -> None:
    regular = [A.Blur(p=1.0)]
    captured = {}

    with mock.patch("ultralytics.models.yolo.detect.DetectionTrainer.__init__", return_value=None) as base_init:
        CustomerTrainer(
            overrides={
                "model": "yolo26n.pt",
                "regular_augmentations": regular,
                "debug_pretransform_samples": 3,
            }
        )
        captured.update(base_init.call_args.kwargs["overrides"])

    assert captured["augmentations"] is regular
    assert "regular_augmentations" not in captured
    assert "debug_pretransform_samples" not in captured


def test_select_device_defaults_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("YOLO_DEVICE", raising=False)
    assert select_device_from_env() == "cpu"


def test_select_device_rejects_unavailable_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YOLO_DEVICE", "cuda")
    with mock.patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            select_device_from_env()


def test_select_device_rejects_unavailable_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YOLO_DEVICE", "mps")
    fake_mps = SimpleNamespace(is_available=lambda: False)
    with mock.patch("torch.backends.mps", fake_mps):
        with pytest.raises(RuntimeError, match="MPS is not available"):
            select_device_from_env()


def test_select_device_accepts_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YOLO_DEVICE", "cuda")
    with mock.patch("torch.cuda.is_available", return_value=True):
        assert select_device_from_env() == 0


def test_select_device_accepts_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YOLO_DEVICE", "mps")
    fake_mps = SimpleNamespace(is_available=lambda: True)
    with mock.patch("torch.backends.mps", fake_mps):
        assert select_device_from_env() == "mps"
