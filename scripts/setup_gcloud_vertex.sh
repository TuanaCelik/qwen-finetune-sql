#!/usr/bin/env bash
# Interactive / idempotent Vertex AI + GCS setup (CustomJob + GCS bucket + default SA).
# Usage: from repo root, bash scripts/setup_gcloud_vertex.sh
# Loads .env if present (same variables as .env.example).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

prompt() {
  local var="$1"
  local text="$2"
  local default="${3:-}"
  local current="${!var:-}"
  if [[ -n "$current" ]]; then
    read -r -p "${text} [${current}]: " input || true
    if [[ -n "${input:-}" ]]; then
      printf -v "$var" '%s' "$input"
    fi
  else
    if [[ -n "$default" ]]; then
      read -r -p "${text} [default: ${default}]: " input || true
      if [[ -z "${input:-}" ]]; then
        printf -v "$var" '%s' "$default"
      else
        printf -v "$var" '%s' "$input"
      fi
    else
      read -r -p "${text}: " input || true
      printf -v "$var" '%s' "$input"
    fi
  fi
}

echo "=== Google Cloud + Vertex setup ==="
echo ""

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud CLI not found. Install: https://cloud.google.com/sdk/docs/install"
  exit 1
fi

if ! command -v gsutil >/dev/null 2>&1; then
  echo "gsutil not found (install Google Cloud SDK)."
  exit 1
fi

prompt GCP_PROJECT_ID "GCP project ID"
if [[ -z "${GCP_PROJECT_ID:-}" ]]; then
  echo "GCP_PROJECT_ID is required."
  exit 1
fi

prompt GCP_REGION "Vertex / bucket region" "us-central1"
prompt GCS_BUCKET "GCS bucket name (no gs:// prefix)"

if [[ -z "${GCS_BUCKET:-}" ]]; then
  echo "GCS_BUCKET is required."
  exit 1
fi

echo ""
read -r -p "Run 'gcloud auth login' now? [y/N]: " do_login || true
if [[ "${do_login:-}" =~ ^[Yy]$ ]]; then
  gcloud auth login
fi

read -r -p "Run 'gcloud auth application-default login' (recommended for local Python SDK)? [Y/n]: " do_adc || true
if [[ ! "${do_adc:-}" =~ ^[Nn]$ ]]; then
  gcloud auth application-default login
fi

gcloud config set project "$GCP_PROJECT_ID"
gcloud config set ai/region "$GCP_REGION"

echo ""
echo "Enabling APIs..."
gcloud services enable \
  aiplatform.googleapis.com \
  compute.googleapis.com \
  storage.googleapis.com \
  artifactregistry.googleapis.com \
  iam.googleapis.com

if gsutil ls -b "gs://${GCS_BUCKET}" >/dev/null 2>&1; then
  echo "Bucket gs://${GCS_BUCKET} already exists."
else
  echo "Creating bucket gs://${GCS_BUCKET}..."
  gsutil mb -l "$GCP_REGION" -p "$GCP_PROJECT_ID" "gs://${GCS_BUCKET}"
fi

PROJECT_NUMBER="$(gcloud projects describe "$GCP_PROJECT_ID" --format="value(projectNumber)")"
CE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

echo "Granting Vertex training default SA bucket + Vertex User..."
gsutil iam ch "serviceAccount:${CE_SA}:objectAdmin" "gs://${GCS_BUCKET}"

gcloud projects add-iam-policy-binding "$GCP_PROJECT_ID" \
  --member="serviceAccount:${CE_SA}" \
  --role="roles/aiplatform.user" \
  --condition=None \
  --quiet

echo ""
echo "=== Sanity check ==="
python3 - <<PY || true
try:
    from google.cloud import aiplatform
    import os
    pid = os.environ.get("GCP_PROJECT_ID")
    region = os.environ.get("GCP_REGION", "us-central1")
    bucket = os.environ.get("GCS_BUCKET")
    if pid and bucket:
        aiplatform.init(project=pid, location=region, staging_bucket=f"gs://{bucket}")
        print("Vertex SDK init OK")
    else:
        print("Skip SDK init (missing env after shell export)")
except Exception as e:
    print("Vertex SDK check skipped or failed:", e)
PY

echo ""
echo "Done."
echo "  Project : $GCP_PROJECT_ID"
echo "  Region  : $GCP_REGION"
echo "  Bucket  : gs://$GCS_BUCKET"
echo ""
echo "Quota: request NVIDIA_L4_GPUS in this region if training/deploy fails with quota errors."
