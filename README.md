# YOLO Custom Trainer

This project adds `CustomTrainer`, a custom Ultralytics YOLO detection trainer that can run Albumentations before Ultralytics mosaic/mixup/cutmix and can also pass regular Albumentations into Ultralytics' built-in post-mosaic hook.

All commands are Pixi commands. Do not install or run this project with global `python` or `pip`.

Runnable entrypoints live in `scripts/`; use the Pixi tasks below instead of calling them directly.

## Environments

CPU is the default:

```bash
pixi install
pixi run inspect-pretransforms
pixi run test
pixi run train-example
pixi run compare-pretransform
pixi run compare-regular
```

macOS Apple Silicon MPS must be selected explicitly:

```bash
pixi install -e mps
pixi run -e mps inspect-pretransforms
pixi run -e mps test
pixi run -e mps train-example
pixi run -e mps compare-pretransform
pixi run -e mps compare-regular
```

Linux NVIDIA CUDA must be selected explicitly:

```bash
pixi install -e cuda
pixi run -e cuda inspect-pretransforms
pixi run -e cuda test
pixi run -e cuda train-example
pixi run -e cuda compare-pretransform
pixi run -e cuda compare-regular
```

The CUDA environment installs `pytorch-cuda=12.4.*`, which is appropriate for CUDA-enabled machines such as A100 hosts when the driver supports that runtime.

## Usage

```python
import albumentations as A
from ultralytics import YOLO

from custom_trainer import CustomTrainer

model = YOLO("yolo26n.pt")
model.train(
    data="coco8.yaml",
    trainer=CustomTrainer,
    pretransform_augmentations=[
        A.RandomSizedBBoxSafeCrop(height=288, width=288, erosion_rate=0.1, p=1.0),
    ],
    regular_augmentations=[
        A.RandomBrightnessContrast(p=0.2),
    ],
    debug_pretransform_samples=4,
)
```

`pretransform_augmentations` run right after a sample image and YOLO boxes are loaded, before mosaic and other Ultralytics training augmentations. `regular_augmentations` is a friendly alias for Ultralytics' current `augmentations` hyperparameter and runs in the normal post-mosaic Albumentations slot.

When pretransforms are configured, the trainer logs them with a `pretransform albumentations:` line, similar to Ultralytics' existing regular `albumentations:` log output.

`coco8.yaml` is only a tiny smoke-test dataset and does not include the COCO `car` class. For a COCO-style dataset that contains cars, add `classes=[2]` to train on cars only.

Run `pixi run inspect-pretransforms` before training. It creates a synthetic detection sample, applies a deterministic crop, numerically checks the updated boxes, and writes before/after overlays to `runs/pretransform_debug/`.

When `debug_pretransform_samples` is greater than zero, training saves before/after overlays from the actual training dataset calls to `runs/detect/<run-name>/pretransform_debug/`. These images are captured before Ultralytics mosaic/mixup/cutmix receives the sample, so they show the real pretransform inputs used by the training loop.

Run `pixi run compare-pretransform` to train once without `CustomTrainer`, once with `CustomTrainer` and a deterministic `CenterCrop(200, 200)`, then compare the generated Ultralytics `train_batch*.jpg` images. It writes baseline/custom/diff panels and a numeric summary to `runs/compare_pretransform/comparison/`.

Run `pixi run compare-regular` to train once without `CustomTrainer`, once with `CustomTrainer` and strong `regular_augmentations`, then compare the generated Ultralytics `train_batch*.jpg` images. It writes baseline/regular/diff panels and a numeric summary to `runs/compare_regular_augmentations/comparison/`.
