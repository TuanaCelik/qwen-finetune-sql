#!/usr/bin/env bash
# Fast local check of train_sql_sft.py (no Vertex): tiny subset, few steps, no merge, no DLC pip bootstrap.
# Requires: `uv sync` so HF dependency **ranges** match pyproject.toml / train_sql_sft.HF_PIP_SPECS; HF_TOKEN;
# enough RAM for MODEL_ID (Gemma 4 is heavy on CPU). Install PyTorch 2.4+ separately for CUDA/local parity with Vertex.
#
# Usage from repo root:
#   uv sync
#   export HF_TOKEN=hf_...
#   ./scripts/local_smoke_train.sh
#
# Optional: MODEL_ID=google/gemma-2-2b-it ./scripts/local_smoke_train.sh   # smaller model for CPU

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Preserve explicit shell overrides (e.g. MODEL_ID=... ./script) before .env overwrites them.
_cli_model_id=""
if [[ -n "${MODEL_ID+x}" ]]; then _cli_model_id="${MODEL_ID}"; fi
_cli_output_dir=""
if [[ -n "${OUTPUT_DIR+x}" ]]; then _cli_output_dir="${OUTPUT_DIR}"; fi
_cli_max_steps=""
if [[ -n "${MAX_STEPS+x}" ]]; then _cli_max_steps="${MAX_STEPS}"; fi
_cli_smoke_max=""
if [[ -n "${SMOKE_MAX_EXAMPLES+x}" ]]; then _cli_smoke_max="${SMOKE_MAX_EXAMPLES}"; fi

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ -n "${_cli_model_id}" ]]; then export MODEL_ID="${_cli_model_id}"; fi
if [[ -n "${_cli_output_dir}" ]]; then export OUTPUT_DIR="${_cli_output_dir}"; fi
if [[ -n "${_cli_max_steps}" ]]; then export MAX_STEPS="${_cli_max_steps}"; fi
if [[ -n "${_cli_smoke_max}" ]]; then export SMOKE_MAX_EXAMPLES="${_cli_smoke_max}"; fi

export VERTEX_SKIP_HF_BOOTSTRAP=1
export MERGE_AFTER_TRAIN=false
export BF16=false
export MAX_STEPS="${MAX_STEPS:-2}"
export SMOKE_MAX_EXAMPLES="${SMOKE_MAX_EXAMPLES:-32}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/.smoke-train-out}"
export MODEL_ID="${MODEL_ID:-google/gemma-4-E2B-it}"

mkdir -p "${OUTPUT_DIR}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "Set HF_TOKEN (or add it to .env)." >&2
  exit 1
fi

echo "MODEL_ID=${MODEL_ID}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "Running ${MAX_STEPS} steps on ${SMOKE_MAX_EXAMPLES} examples (CPU-friendly settings)..."

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required for this script (same toolchain as the repo). Install: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

uv run python train_sql_sft.py

echo "Smoke run finished. Artifacts under ${OUTPUT_DIR}"
