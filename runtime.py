"""Runtime helpers for Pixi-based example scripts."""

from __future__ import annotations

import os


def select_device_from_env() -> str | int:
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
