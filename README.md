# Few-label indoor scene segmentation with frozen foundation features

How far do frozen DINOv2 features with a lightweight head get on indoor semantic
segmentation when labels are scarce, compared with a fully fine-tuned CNN?

**Setup:** NYU Depth v2 (1,449 labelled RGB-D frames) · DeepLabV3-ResNet50 (fine-tuned)
vs. DINOv2 (frozen) + linear head · label budgets from 10 % to 100 %, 3 seeds each.
Planned extensions: cross-validation, inference latency on a laptop GPU.

## Status

| Milestone | Tag | State |
|---|---|---|
| M0 Repo, environment, CI | `v0.0` | done |
| M1 Data loader + CNN baseline (own training loop) | `v0.1` | not started |
| M2 Frozen DINOv2 heads + label-budget sweep | `v0.2` | not started |
| M3 Latency benchmark + write-up | `v0.3` | not started |

Each tag is a fixed state: the numbers in this README at that tag are reproduced by
the command listed next to them.

## Results

_No results yet._ This table is filled only from files in `results/` written by the
scripts in `scripts/`; nothing is typed in by hand.

| Model | Trainable params | mIoU @10 % labels | mIoU @100 % labels | Latency (ms / frame) | Command |
|---|---|---|---|---|---|
| DeepLabV3-ResNet50, fine-tuned | – | – | – | – | – |
| DINOv2 ViT-S/14 frozen + linear head | – | – | – | – | – |

## Reproduce

```bash
conda env create -f environment.yml   # or: conda activate dl
pip install -e .
python scripts/download_nyuv2.py      # ~2.8 GB, stored in data/ (not in git)
pytest                                # CPU-only checks
```

Hardware used: RTX 5070 Laptop GPU (8 GB), CUDA 12.8, Ubuntu 22.04 on WSL2.
Seeds are fixed in `configs/`; every result file records the config and git commit.

## Layout

```
src/fewlabel/   library code (data, models, train, eval)
scripts/        entry points: one command per result
configs/        YAML configs, one per experiment
results/        small result files (JSON/CSV) that back the README table
docs/           notes, figures, decisions
tests/          CPU-only tests, run in CI
```

## What I wrote, what I reused

Own code: data loading, training loop, evaluation, sweep and benchmark scripts.
Reused: pretrained weights (torchvision DeepLabV3, DINOv2), the NYUv2 dataset.

## Data and licence

Code: MIT. NYU Depth v2 is not redistributed here; download it from the authors
(Silberman et al., ECCV 2012) and cite their paper.
