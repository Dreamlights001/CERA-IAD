#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/cera_iad_config.yaml}"
MATRIX="${MATRIX:-configs/ablation_matrix.yaml}"
OUTPUT_ROOT="$(python - "${CONFIG}" <<'PY'
from pathlib import Path
import sys
import yaml

config_path = Path(sys.argv[1])
config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
project_root = config_path.resolve().parent.parent
output_root = Path(config.get("paths", {}).get("output_root", "../../outputs/cera_experiments"))
if not output_root.is_absolute():
    output_root = (project_root / output_root).resolve()
print(output_root / "plans")
PY
)"

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
  python scripts/run_cera.py \
    --config "${CONFIG}" \
    --ablation-matrix "${MATRIX}" \
    --ablation "${ablation}" \
    --dry-run \
    --output-plan "${OUTPUT_ROOT}/${ablation}.json"
done

echo "[CERA-IAD] plans written to ${OUTPUT_ROOT}"
