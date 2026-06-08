#!/usr/bin/env bash
set -euo pipefail

MODE="basic"
WEIGHTS_ROOT="${IAD_WEIGHTS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cache}"

usage() {
  cat <<'EOF'
Usage: bash weights/download_weights.sh [--basic|--optional|--mllm|--all]

Groups:
  --basic     CLIP + SAM weights needed by A0-A4 and non-MLLM evidence runs.
  --optional  DINOv2 + SAM2 optional swaps.
  --mllm      Qwen2.5-VL + InternVL reasoner weights. Large downloads.
  --all       basic + optional + mllm.

Environment:
  IAD_WEIGHTS    Target cache directory. Default: weights/cache
  HF_TOKEN       Optional Hugging Face token for gated/private repos.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --basic) MODE="basic" ;;
    --optional) MODE="optional" ;;
    --mllm) MODE="mllm" ;;
    --all) MODE="all" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
  shift
done

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "[ERROR] huggingface-cli not found. Install with: python -m pip install -U huggingface_hub" >&2
  exit 1
fi

mkdir -p "${WEIGHTS_ROOT}"
echo "[CERA-IAD] weights root: ${WEIGHTS_ROOT}"
echo "[CERA-IAD] mode: ${MODE}"

download_hf() {
  local group="$1"
  local key="$2"
  local repo="$3"
  local target="$4"

  if [[ "${MODE}" != "all" && "${MODE}" != "${group}" ]]; then
    return 0
  fi

  local local_dir="${WEIGHTS_ROOT}/${target}"
  mkdir -p "${local_dir}"
  echo "[CERA-IAD] downloading ${key}: ${repo} -> ${local_dir}"
  huggingface-cli download "${repo}" \
    --local-dir "${local_dir}" \
    --local-dir-use-symlinks False
}

download_hf basic clip_vit_l_14 openai/clip-vit-large-patch14 clip/openai-clip-vit-large-patch14
download_hf basic sam_vit_h facebook/sam-vit-huge sam/facebook-sam-vit-huge

download_hf optional dinov2_large facebook/dinov2-large dinov2/facebook-dinov2-large
download_hf optional sam2_1_hiera_large facebook/sam2.1-hiera-large sam2/facebook-sam2.1-hiera-large

download_hf mllm qwen25_vl_7b Qwen/Qwen2.5-VL-7B-Instruct mllm/Qwen2.5-VL-7B-Instruct
download_hf mllm internvl25_8b OpenGVLab/InternVL2_5-8B mllm/InternVL2_5-8B

echo "[CERA-IAD] done."
