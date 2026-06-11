# CERA-IAD

[English](README.md) | [中文](README-zh.md)

**CERA-IAD: Conformal Evidence-Rarity Agent for Zero-shot Industrial Anomaly Detection**

CERA-IAD is a research implementation for zero-shot industrial anomaly detection. The method combines semantic anomaly proposals, normal-memory rarity estimation, conformal calibration, mask-level localization, and optional multimodal evidence reflection into a unified evaluation pipeline.

## Highlights

- Evidence-centric anomaly scoring with semantic, rarity, disagreement, and uncertainty signals.
- Normal-memory retrieval for category-agnostic visual rarity estimation.
- Conformal calibration for thresholding and review-band analysis.
- Mask refinement with promptable segmentation models.
- Configurable ablation protocol for isolating each component of the method.

## Installation

```bash
cd /workspace/IAD/code/CERA-IAD

conda create -y -n cera python=3.10
conda activate cera
bash scripts/setup_env.sh
```

The dependency list is maintained in `requirements.txt`. PyTorch is installed by `scripts/setup_env.sh` with the CUDA wheel selected for the target machine.

## Pretrained Models

Pretrained checkpoints are downloaded on demand and stored under the configured weight root in `configs/cera_iad_config.yaml`.

Inspect the registry and local status:

```bash
bash weights/download_weights.sh --list
```

Preview a download plan without network access:

```bash
bash weights/download_weights.sh --dry-run --basic
```

| Model | Purpose | Required for | Group | Source |
| --- | --- | --- | --- | --- |
| CLIP ViT-L/14 | Semantic image/patch encoder | A0-A6 | `basic` | `openai/clip-vit-large-patch14` |
| SAM ViT-H | Box-prompt mask refinement | A4-A6 | `basic` | `facebook/sam-vit-huge` |
| DINOv2 Large | Language-free encoder ablation | encoder swap | `optional` | `facebook/dinov2-large` |
| SAM2.1 Hiera Large | Mask refiner ablation | mask swap | `optional` | `facebook/sam2.1-hiera-large` |
| Qwen2.5-VL-7B-Instruct | Evidence reflection | A5-A6 | `mllm` | `Qwen/Qwen2.5-VL-7B-Instruct` |
| InternVL2.5-8B | Reasoner ablation | reasoner swap | `mllm` | `OpenGVLab/InternVL2_5-8B` |
| AnomalyVFM checkpoints | Encoder ablation | manual swap | `manual` | Follow upstream checkpoint instructions |

Download checkpoints by group:

```bash
bash weights/download_weights.sh --basic
bash weights/download_weights.sh --optional
bash weights/download_weights.sh --mllm
```

Each completed model directory contains `.cera_weight.json` with the source repository, download time, usage tags, and local file count.

## Data Preparation

Dataset paths are configured in `configs/cera_iad_config.yaml`. The default root is `../../data`:

```text
data/
  MVTec-AD/
  MVTec-AD-2/
  VisA/
  Real-IAD/
  MMAD/
  M3-AD/
```

## Running Experiments

Dry-run a single ablation:

```bash
python scripts/run_cera.py \
  --config configs/cera_iad_config.yaml \
  --ablation A0_clip_only \
  --dry-run
```

Generate all ablation plans:

```bash
bash scripts/run_ablation_matrix.sh
```

The A0-A6 protocol is defined in `configs/ablation_matrix.yaml`:

| ID | Description |
| --- | --- |
| A0 | CLIP semantic prior |
| A1 | Anomaly-aware candidate generation |
| A2 | Normal-memory rarity evidence |
| A3 | Conformal calibration |
| A4 | SAM-based mask refinement |
| A5 | MLLM evidence reflection |
| A6 | Full CERA-IAD |

## Results

Benchmark results will be reported after the full evaluation run.

| Dataset | Image AUROC | Pixel AUROC | AU-PRO | ECE | AURC |
| --- | --- | --- | --- | --- | --- |
| MVTec AD 2 | TBD | TBD | TBD | TBD | TBD |
| Real-IAD | TBD | TBD | TBD | TBD | TBD |
| MMAD | TBD | TBD | TBD | TBD | TBD |
| M3-AD | TBD | TBD | TBD | TBD | TBD |

## Project Layout

- `cera_iad/`: core package and method components.
- `configs/`: experiment configuration and ablation matrix.
- `scripts/`: environment setup and experiment entrypoints.
- `weights/`: pretrained model registry and download utilities.
- `tests/`: static and dry-run checks.

## Acknowledgements

CERA-IAD uses public foundation models and may interoperate with external anomaly detection baselines for comparison or optional adapters. Please cite the corresponding upstream works when using their models, checkpoints, or code.
