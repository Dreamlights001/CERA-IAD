# CERA-IAD Implementation Notes

This directory contains a runnable v0 skeleton for:

**CERA-IAD: Conformal Evidence-Rarity Agent for Zero-shot Industrial Anomaly Detection**

The implementation intentionally starts with feature-level inputs so the research idea can be tested before binding the project to a specific CLIP/SAM/MLLM stack.

## What is implemented

- Exact kNN normal memory over frozen image/patch features.
- Region-level score fusion:

  `S(r) = alpha * Semantic + beta * Rarity + gamma * Disagreement + delta * Uncertainty`

- Conformal calibration gate with `normal / anomaly / review` decisions.
- Evidence JSON output for region-level and image-level decisions.
- Lightweight metrics for ECE, abstention gain, and retrieval purity.
- Synthetic smoke demo and tests.

## Plug-in points

- Candidate generators: AA-CLIP, Bayes-PFL, MRAD-style retrieval, or any external detector that emits `RegionProposal`.
- Encoders: CLIP, SigLIP, DINOv2, AnomalyVFM, or extracted features from existing baselines.
- Mask tools: SAM/SAM2 masks can be attached to `RegionProposal.mask`.
- MLLM reasoner: consume `ImageEvidence.to_json_dict()` plus visual crops/masks to produce final explanation/reflection.

## Recommended benchmark order

1. Sanity: MVTec AD, VisA.
2. Main: MVTec AD 2, Real-IAD.
3. Reasoning: MMAD or MMR-AD.
4. Reflection/multimodal: M3-AD, Real-IAD D3.

## Run

```powershell
cd D:\Users\Dlts\Documents\GitHub\IAD
python scripts\run_cera_demo.py
python -m pytest code\tests
```
