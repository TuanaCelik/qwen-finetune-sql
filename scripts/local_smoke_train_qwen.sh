#!/usr/bin/env bash
# Fast local check of scripts/train_qwen_sql_sft.py (no Vertex): tiny subset, few steps, no merge, no DLC pip bootstrap.
#
# Usage from repo root (HF_TOKEN in `.env` when Hub needs it):
#   uv sync
#   ./scripts/local_smoke_train_qwen.sh
#
# Fastest sanity check (8 rows — good “will Vertex train script run?” proxy):
#   FAST_SMOKE=true ./scripts/local_smoke_train_qwen.sh
#
# Optional: QWEN_MODEL_ID=Qwen/Qwen3.5-0.8B ./scripts/local_smoke_train_qwen.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

_cli_model_id=""
if [[ -n "${QWEN_MODEL_ID+x}" ]]; then _cli_model_id="${QWEN_MODEL_ID}"; fi
_cli_output_dir=""
if [[ -n "${QWEN_OUTPUT_DIR+x}" ]]; then _cli_output_dir="${QWEN_OUTPUT_DIR}"; fi
_cli_smoke_max=""
if [[ -n "${SMOKE_MAX_EXAMPLES+x}" ]]; then _cli_smoke_max="${SMOKE_MAX_EXAMPLES}"; fi

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ -n "${_cli_model_id}" ]]; then export QWEN_MODEL_ID="${_cli_model_id}"; fi
if [[ -n "${_cli_output_dir}" ]]; then export QWEN_OUTPUT_DIR="${_cli_output_dir}"; fi
if [[ -n "${_cli_smoke_max}" ]]; then export SMOKE_MAX_EXAMPLES="${_cli_smoke_max}"; fi

export VERTEX_SKIP_HF_BOOTSTRAP=true
if [[ "${FAST_SMOKE:-}" == "true" ]]; then
  if [[ -n "${_cli_smoke_max}" ]]; then export SMOKE_MAX_EXAMPLES="${_cli_smoke_max}"; else export SMOKE_MAX_EXAMPLES=8; fi
else
  export SMOKE_MAX_EXAMPLES="${SMOKE_MAX_EXAMPLES:-32}"
fi
export QWEN_OUTPUT_DIR="${QWEN_OUTPUT_DIR:-${ROOT}/.smoke-train-out}"
export QWEN_MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen3.5-0.8B}"

mkdir -p "${QWEN_OUTPUT_DIR}"

echo "QWEN_MODEL_ID=${QWEN_MODEL_ID}"
echo "QWEN_OUTPUT_DIR=${QWEN_OUTPUT_DIR}"
echo "Running smoke train on ${SMOKE_MAX_EXAMPLES} examples (CPU-friendly settings)..."

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required for this script (same toolchain as the repo). Install: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

uv run python scripts/train_qwen_sql_sft.py

echo "Smoke run finished. Artifacts under ${QWEN_OUTPUT_DIR}"
