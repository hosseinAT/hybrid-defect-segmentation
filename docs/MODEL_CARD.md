# Model and experiment card

## Intended use

Binary surface defect segmentation. This is a compact research and engineering study for technical review and reproducibility.

## Architecture

Four small residual CNN variants; optional local shifted-window attention and context fusion.

## Recorded data and fitting

128 fit / 48 validation / 80 test procedural images; three initialization/order seeds.

Focal plus per-image Dice; validation Dice selects checkpoint and threshold.

## Evaluation and observed evidence

Pixel Dice, IoU, precision/recall, normal-image alarms, defect-kind breakdown and CPU latency.

CNN mean Dice 0.6046; attention 0.6294; fusion 0.5339; combined 0.5425. Variation across starts is substantial.

Raw counts and configuration are in `results/demo/metrics.json`; a readable comparison is in `results/demo/EXPERIMENTS.md`. The recorded runtime is CPU PyTorch 2.5.1+cpu with Python 3.12.14. Runtime timestamps identify when the recorded measurements were taken.

## Verification coverage

Attention boundary masking, padding gradients, non-square tensors, empty targets, threshold selection and split leakage.

The supplied GitHub workflow is prepared configuration. Local checks do not imply a remote workflow has already run.

## Material limitations

No production qualification, parameter-matched ablation, native-resolution tiling, GPU profiling or exported inference backend.

The procedural generator is narrow and has exact generated masks. It cannot represent the full variation in materials, optics, defect severity, annotation quality or production batches. Real deployment requires independent representative data and an operating point tied to inspection costs. No proprietary/customer dataset is bundled.

## Data handling

Private datasets stay outside version control via `data/` and generated private runs via `runs/`. The manifest audit rejects structural leaks but cannot validate the meaning of manually assigned batch groups or detect all near-duplicates. Do not publish private run folders that contain sensitive sample IDs or paths.
