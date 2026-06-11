# CERA-IAD

[English](README.md) | [中文](README-zh.md)

**CERA-IAD: Conformal Evidence-Rarity Agent for Zero-shot Industrial Anomaly Detection**

CERA-IAD is a modular research codebase for zero-shot industrial anomaly detection on Ubuntu. It integrates reusable VFM/VLM backbones, top-conference anomaly detection baselines, normal-memory retrieval, conformal calibration, mask refinement, and optional MLLM evidence reflection.

## Project Layout

- `cera_iad/`: the CERA-IAD Python package. The core data flow remains `RegionProposal -> NormalMemory -> CeraIAD -> ImageEvidence`.
- `cera_iad/modules/`: module registry and experiment plan builder.
- `configs/cera_iad_config.yaml`: main experiment configuration.
- `configs/ablation_matrix.yaml`: fixed A0-A6 ablation matrix.
- `weights/`: model weight manifest and Hugging Face download script. Large weights are not committed.
- `scripts/setup_env.sh`: Ubuntu environment bootstrap script.
- `scripts/run_cera.py`: experiment entrypoint. It currently supports dry-run and adapter plan output.
- `scripts/run_ablation_matrix.sh`: batch launcher for ablation plans.
- `../baselines/`: cloned and sanitized top-conference repositories reused through adapters.

## Setup

Run these commands from the project root:

```bash
cd /workspace/IAD/code/CERA-IAD

bash scripts/setup_env.sh
```

Paths are configured in `configs/cera_iad_config.yaml`. Dependencies are listed in the single `requirements.txt` file. `setup_env.sh` installs PyTorch first with the CUDA-specific wheel index, then installs `requirements.txt`.

Download the basic weights:

```bash
bash weights/download_weights.sh --basic
```

Download MLLM reflection weights:

```bash
bash weights/download_weights.sh --mllm
```

Download optional swap-module weights:

```bash
bash weights/download_weights.sh --optional
```

## Module Swaps

The default modules are configured in `configs/cera_iad_config.yaml`:

```yaml
modules:
  encoder: clip_openai
  candidate_generator: aa_clip
  memory_backend: exact_knn
  mask_refiner: sam_box
  reasoner: no_reasoner
  calibrator: conformal_global
  fusion_scorer: linear_fusion
```

Available modules are registered in `cera_iad/modules/registry.py`:

- `encoder`: `clip_openai`, `dinov2_large`, `anomaly_vfm`
- `candidate_generator`: `clip_only`, `aa_clip`, `anomaly_clip`, `mrad`
- `memory_backend`: `exact_knn`, `faiss_knn`
- `mask_refiner`: `no_mask`, `sam_box`, `sam2_box`
- `reasoner`: `no_reasoner`, `qwen25_vl`, `internvl25`
- `calibrator`: `fixed_threshold`, `conformal_global`
- `fusion_scorer`: `linear_fusion`

The design principle is to reuse foundation models and top-conference code from `../baselines/` or Hugging Face/package APIs. CERA-IAD should only provide adapters that convert upstream outputs into `RegionProposal`, `ImageEvidence`, and metrics JSON.

## Ablation Experiments

Ablations are defined in `configs/ablation_matrix.yaml`.

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

Ablation items:

- `A0_clip_only`: CLIP semantic prior only.
- `A1_aa_clip_candidate`: replace plain CLIP proposals with AA-CLIP candidates.
- `A2_memory_rarity`: add normal memory retrieval and rarity evidence.
- `A3_conformal_gate`: add conformal gate and review band.
- `A4_sam_mask`: add SAM box-prompt mask geometry evidence.
- `A5_mllm_reflection`: add Qwen2.5-VL evidence reflection.
- `A6_full_cera_iad`: full CERA-IAD.

To replace an intermediate module for an ablation, edit the corresponding `modules` block in `configs/ablation_matrix.yaml`. Avoid hard-coding module choices in scripts.

## Weights and Data

Weights are managed by `weights/weights_manifest.yaml`. Downloads prefer Hugging Face:

- CLIP: `openai/clip-vit-large-patch14`
- DINOv2: `facebook/dinov2-large`
- SAM: `facebook/sam-vit-huge`
- SAM2.1: `facebook/sam2.1-hiera-large`
- Qwen2.5-VL: `Qwen/Qwen2.5-VL-7B-Instruct`
- InternVL2.5: `OpenGVLab/InternVL2_5-8B`

Datasets are not downloaded by the scripts. The default dataset root is configured as `../../data` in `configs/cera_iad_config.yaml`:

```text
/data/iad/
  MVTec-AD/
  MVTec-AD-2/
  VisA/
  Real-IAD/
  MMAD/
  M3-AD/
```

## Outputs

The default output root is configured in `configs/cera_iad_config.yaml`:

```bash
../../outputs/cera_experiments
```

Recommended artifacts:

- `features/normal_memory.npy`
- `calibration/normal_scores.json`
- `proposals/*.json`
- `evidence/*.json`
- `metrics/*.json`
- `plans/A0_clip_only.json` through `plans/A6_full_cera_iad.json`

The final paper method should be selected according to `../../reviews/cera_iad_method_validation_zh.html`: if a module only improves the old MVTec benchmark but not MVTec AD 2 / Real-IAD, or if it significantly hurts calibration, downgrade it to an optional module rather than including it in the main method.
