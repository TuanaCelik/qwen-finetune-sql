#!/usr/bin/env bash
# Fast local check of train_qwen_sql_sft.py (no Vertex): tiny subset, few steps, no merge, no DLC pip bootstrap.
# Same layout as scripts/local_smoke_train.sh (Gemma): same env vars, defaults, and default OUTPUT_DIR (.smoke-train-out).
# Use a different OUTPUT_DIR if you alternate Gemma vs Qwen smoke runs in the same clone.
#
# Usage from repo root (HF_TOKEN in `.env` when Hub needs it):
#   uv sync
#   ./scripts/local_smoke_train_qwen.sh
#
# Fastest sanity check (1 step, 8 rows, short seq — good “will Vertex train script run?” proxy):
#   FAST_SMOKE=1 ./scripts/local_smoke_train_qwen.sh
#
# Optional: MODEL_ID=Qwen/Qwen3.5-0.8B ./scripts/local_smoke_train_qwen.sh

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
if [[ "${FAST_SMOKE:-}" == "1" ]]; then
  if [[ -n "${_cli_max_steps}" ]]; then export MAX_STEPS="${_cli_max_steps}"; else export MAX_STEPS=1; fi
  if [[ -n "${_cli_smoke_max}" ]]; then export SMOKE_MAX_EXAMPLES="${_cli_smoke_max}"; else export SMOKE_MAX_EXAMPLES=8; fi
  export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-128}"
else
  export MAX_STEPS="${MAX_STEPS:-2}"
  export SMOKE_MAX_EXAMPLES="${SMOKE_MAX_EXAMPLES:-32}"
fi
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/.smoke-train-out}"
export MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-0.8B}"

mkdir -p "${OUTPUT_DIR}"

echo "MODEL_ID=${MODEL_ID}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "Running ${MAX_STEPS} steps on ${SMOKE_MAX_EXAMPLES} examples (CPU-friendly settings)..."

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required for this script (same toolchain as the repo). Install: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

uv run python train_qwen_sql_sft.py

echo "Smoke run finished. Artifacts under ${OUTPUT_DIR}"
