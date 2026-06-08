#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/cera_iad_cloud.yaml}"
MATRIX="${MATRIX:-configs/ablation_matrix.yaml}"
OUTPUT_ROOT="${IAD_OUTPUTS:-outputs/cera_cloud}/plans"

ABLATIONS=(
  A0_clip_only
  A1_aa_clip_candidate
  A2_memory_rarity
  A3_conformal_gate
  A4_sam_mask
  A5_mllm_reflection
  A6_full_cera_iad
)

mkdir -p "${OUTPUT_ROOT}"

for ablation in "${ABLATIONS[@]}"; do
  echo "[CERA-IAD] resolving ${ablation}"
  python scripts/run_cera_cloud.py \
    --config "${CONFIG}" \
    --ablation-matrix "${MATRIX}" \
    --ablation "${ablation}" \
    --dry-run \
    --output-plan "${OUTPUT_ROOT}/${ablation}.json"
done

echo "[CERA-IAD] dry-run plans written to ${OUTPUT_ROOT}"
