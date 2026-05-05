# Qwen SQL fine-tune on Vertex AI

This repository wires **Google Cloud Vertex AI** with Hugging Face PyTorch Deep Learning Containers to fine-tune `Qwen/Qwen3.5-0.8B` on SQL instruction data, merge LoRA weights into a full checkpoint on Cloud Storage. 

The trained model is hosted on HuggingFace under `Tuana/qwen35-08b-text2sql`

There are 2 parts to this project:
1. Fine-tuning the model on vertex, for which scripts are in `scripts`
2. Using the fine-tuned model in a Gradio app, comparing its results vs the base model

Training and serving are separate resources: each training run is a one-shot CustomJob. 

## Prerequisites

1. **Google Cloud project** with billing enabled.
2. **`gcloud` CLI** ([Install the Google Cloud SDK](https://cloud.google.com/sdk/docs/install)).
3. **`uv`** for Python ([Install uv](https://docs.astral.sh/uv/getting-started/installation/)) or `pip install -r requirements.txt`.
4. **Hugging Face token** for model and private dataset access.
5. **Quota** for your chosen GPU, for example NVIDIA L4 in `us-central1`.

## Repository Layout

| Path | Purpose |
|------|---------|
| `.env.example` | Template for Vertex, Qwen training, serving, and local compare settings. |
| `scripts/train_qwen_sql_sft.py` | Runs inside the Vertex training container: dataset load, LoRA SFT, optional merge. |
| `scripts/finetune_deploy_qwen.py` | Submits training and optionally deploys merged weights to Vertex. |
| `scripts/query_finetuned_qwen.py` | Syncs merged weights from GCS and runs local generation. |
| `scripts/quick_sql_validate.py` | Executes a small SQLite benchmark against local fine-tuned and/or Hub base Qwen. |
| `scripts/generate_sql_sft_data_gemini.py` | Generates validated SQL SFT JSONL examples. |
| `scripts/upload_generated_sql_sft_dataset.py` | Uploads generated JSONL to a Hugging Face dataset. |
| `sql_compare_ui_qwen/` | Gradio compare UI for fine-tuned Qwen vs Hub base Qwen. |

## Setup

```bash
uv venv
uv sync
cp .env.example .env
```

Edit `.env` at minimum:

- `GCP_PROJECT_ID`
- `GCP_REGION`
- `GCS_BUCKET`
- `HF_TOKEN`
- `QWEN_OUTPUT_GCS_PREFIX`

PyTorch is not pinned because CPU and CUDA wheels differ. For local model loading, install a torch build for your machine. On Vertex, `scripts/train_qwen_sql_sft.py` upgrades the torch stack inside the training container.

## Google Cloud Setup

Run the interactive setup script:

```bash
./scripts/setup_gcloud_vertex.sh
```

It enables required APIs, creates the bucket if needed, and grants the default training service account the IAM roles used by the Vertex CustomJob flow.

## Dataset Workflow

The trainer expects a Hugging Face dataset split with these columns:

- `context`
- `question`
- `answer`

The default dataset is `b-mc2/sql-create-context`. To upload generated local JSONL:

```bash
uv run python scripts/upload_generated_sql_sft_dataset.py Tuana/synthetic-sql-dataset
```

Then select it for training:

```bash
QWEN_DATASET_ID=Tuana/synthetic-sql-dataset \
QWEN_DATASET_SPLIT=train \
uv run python scripts/finetune_deploy_qwen.py --train --training-async
```

## Fine-Tune

Train only:

```bash
uv run python scripts/finetune_deploy_qwen.py --train
```

Train and deploy:

```bash
uv run python scripts/finetune_deploy_qwen.py --train --deploy
```

Common small-dataset run:

```bash
QWEN_MODEL_ID=Qwen/Qwen3.5-0.8B \
QWEN_DATASET_ID=Tuana/synthetic-sql-dataset \
QWEN_DATASET_SPLIT=train \
QWEN_OUTPUT_GCS_PREFIX=qwen-sql/qwen35-synthetic-1k-ep4 \
SKIP_DEPLOY=true \
uv run python scripts/finetune_deploy_qwen.py --train --training-async
```

The training recipe for this run lives in `scripts/train_qwen_sql_sft.py`; environment variables select resources, dataset, and artifact names.

## Local Smoke Test

```bash
./scripts/local_smoke_train_qwen.sh
FAST_SMOKE=true ./scripts/local_smoke_train_qwen.sh
```

The smoke script uses a tiny dataset slice, disables merge, and forces CPU-friendly settings.

## Sync And Validate

Sync a merged checkpoint from GCS:

```bash
QWEN_OUTPUT_GCS_PREFIX=qwen-sql/qwen35-synthetic-1k-ep4 \
LOCAL_QWEN_MERGED_CACHE_NAME=qwen35-synthetic-1k-ep4 \
uv run python scripts/query_finetuned_qwen.py --sync
```

Run the SQLite benchmark:

```bash
LOCAL_QWEN_MERGED_CACHE_NAME=qwen35-synthetic-1k-ep4 \
uv run python scripts/quick_sql_validate.py --local --max-new-tokens 128
```

Compare fine-tuned vs base:

```bash
LOCAL_QWEN_MERGED_CACHE_NAME=qwen35-synthetic-1k-ep4 \
uv run python scripts/quick_sql_validate.py --local --hub --max-new-tokens 128
```

## Compare UI

```bash
uv run python sql_compare_ui_qwen/app.py
```

The UI compares the Hub base model from `QWEN_COMPARE_HUB_MODEL_ID` with `Tuana/qwen35-08b-text2sql`.

## Troubleshooting

| Symptom | What to check |
|--------|----------------|
| `403` on Hugging Face | Confirm `HF_TOKEN` has access to the model and dataset. |
| Quota / L4 errors | Increase GPU quota in `GCP_REGION`. |
| Training OOM | Lower the batch size or sequence length constants in `scripts/train_qwen_sql_sft.py`. |
| Deploy looks for wrong path | Merged weights must exist at `gs://<bucket>/<QWEN_OUTPUT_GCS_PREFIX>/merged`. |
| Endpoint cannot read GCS weights | Set `VERTEX_PREDICTION_SERVICE_ACCOUNT` to a service account with `roles/storage.objectViewer` on the bucket. |
| Mixed torch / torchvision errors | Use the default Vertex bootstrap or install a matching local torch stack. |
