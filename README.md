# Gemma 4 (E2B IT) SQL fine-tune on Vertex AI

This repository wires **Google Cloud Vertex AI** with the **Hugging Face PyTorch training DLC** to fine-tune `google/gemma-4-E2B-it` on the [`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context) dataset, merge LoRA weights into a full checkpoint on Cloud Storage, then deploy with the **vLLM** Model Garden container behind a Vertex **Endpoint**.

Training and serving are separate resources: each training run is a one-shot CustomJob; the endpoint stays up until you delete or scale it down.

---

## Prerequisites

1. **Google Cloud project** with billing enabled.
2. **`gcloud` CLI** and **`gsutil`** ([Install the Google Cloud SDK](https://cloud.google.com/sdk/docs/install)).
3. **`uv`** for Python ([Install uv](https://docs.astral.sh/uv/getting-started/installation/)) — or use `pip install -r requirements.txt` instead.
4. **Hugging Face token** (recommended for model/dataset downloads inside the training container). Create one under [Hugging Face settings](https://huggingface.co/settings/tokens) if you do not already use Hub auth.
5. **Quota** for **NVIDIA L4** GPUs in your chosen region (for example `us-central1`). If the training or deploy step fails with a quota error, request an increase in **IAM & Admin → Quotas** (filter for `NVIDIA_L4_GPUS`). This cannot be fixed from scripts alone.

---

## Repository layout

| Path | Purpose |
|------|---------|
| `.env.example` | Template for all Vertex / training / serving settings — copy to `.env`. |
| `pyproject.toml` | Dependencies for local tooling (`uv sync`). |
| `scripts/setup_gcloud_vertex.sh` | Interactive GCP setup: APIs, bucket, default service account IAM. |
| `train_sql_sft.py` | Runs **inside** the Vertex training container (dataset → LoRA SFT → optional merge). |
| `scripts/finetune_deploy_gemma4.py` | From your laptop: submit training, wait, upload model, deploy vLLM endpoint. |
| `scripts/query_finetuned_gemma.py` | Sync `gs://…/merged` from `.env` to disk and run **Transformers** generation (same flow as [HF Vertex Gemma 4 “use fine-tune”](https://huggingface.co/docs/google-cloud/examples/vertex-ai-notebooks-fine-tune-gemma-4#use-fine-tune-with-transformers)). |
| `sql_compare_ui/` | Gradio compare UI (local merged fine-tuned vs Hub base via Transformers). |
| `scripts/local_smoke_train.sh` | Optional: run `train_sql_sft.py` locally (CPU, tiny subset) to validate the script before Vertex. |

---

## Step 1 — Python environment (uv)

From the repository root:

```bash
uv venv
uv sync
```

Activating the virtualenv (optional):

```bash
source .venv/bin/activate   # macOS / Linux
```

If you do **not** use `uv`, install dependencies with:

```bash
pip install -r requirements.txt
```

**PyTorch** is not pinned in `pyproject.toml` because CPU vs CUDA wheels differ. For **local** runs of `train_sql_sft.py`, install **torch ≥ 2.4** so it matches **Transformers 5.x** (same expectation as Vertex after bootstrap). Examples:

```bash
# CUDA 12.1 (closest to the default HF Training DLC / TORCH_PIP_INDEX_URL default)
uv add torch torchvision torchaudio --index https://download.pytorch.org/whl/cu121
# Or CUDA 12.4
uv add torch torchvision torchaudio --index https://download.pytorch.org/whl/cu124
```

On **Vertex**, `train_sql_sft.py` upgrades **torch / torchvision / torchaudio** from **`TORCH_PIP_INDEX_URL`** (default **cu121**, same family as `TRAINING_CONTAINER_URI`), then installs the HF packages listed in **`HF_PIP_SPECS`** (aligned with **`pyproject.toml`**). Tune **`VERTEX_UPGRADE_TORCH`** / **`TORCH_PIP_INDEX_URL`** in `.env` if needed — see `.env.example`.

### Local smoke test (no cloud GPU)

Use this when you only want to verify that **`train_sql_sft.py`** runs (imports, dataset load, tokenizer, a few training steps) without submitting a Vertex job. Full Gemma 4 may not fit in RAM on a laptop; for smoke runs you can point **`MODEL_ID`** at a smaller instruct model your machine can load.

Run **`uv sync`** once so dependency **ranges** match **`pyproject.toml`** / **`train_sql_sft.HF_PIP_SPECS`**. Add **PyTorch 2.4+** for your machine if you want parity with the Vertex container (see above). **`sentencepiece`** is included for Gemma tokenizers.

```bash
uv sync
export HF_TOKEN=hf_...
./scripts/local_smoke_train.sh
# Optional override:
# MODEL_ID=google/gemma-2-2b-it ./scripts/local_smoke_train.sh
```

The script runs **`train_sql_sft.py`** via **`uv run python`** and sets **`VERTEX_SKIP_HF_BOOTSTRAP=1`** (skips the DLC `pip` bootstrap), **`BF16=false`**, **`MERGE_AFTER_TRAIN=false`**, **`MAX_STEPS`**, and **`SMOKE_MAX_EXAMPLES`**. That **`SMOKE_MAX_EXAMPLES`** flag also forces **`use_cpu=True`** so Apple **MPS** does not hit Metal buffer limits on large models. Artifacts go to **`.smoke-train-out/`** by default.

---

## Step 2 — Environment file

Copy the template and fill in secrets and project values:

```bash
cp .env.example .env
```

Edit `.env` at minimum:

- **`GCP_PROJECT_ID`** — Your GCP project ID.
- **`GCP_REGION`** — Same region for bucket, training, and serving (default `us-central1` is a common choice for L4 DLCs).
- **`GCS_BUCKET`** — Bucket **name only** (no `gs://` prefix). Must live in or near the same region as `GCP_REGION` for sane latency and cost.
- **`HF_TOKEN`** — Hugging Face token passed into the training job for Hub downloads (recommended).

Fill in the rest of **`TRAINING_*`**, **`SERVING_*`**, **`DISPLAY_NAME_*`**, hyperparameters, and container URIs from `.env.example` as needed. The orchestration script reads everything from the environment except **`--train`**, **`--deploy`**, and **`--training-async`** (see Step 4).

---

## Step 3 — Google Cloud setup

Run the interactive setup script (it loads `.env` if present):


```bash
brew install --cask gcloud-cli
./scripts/setup_gcloud_vertex.sh
```

It will:

1. Prompt for project ID, region, and bucket name (pre-filled from `.env` when set).
2. Optionally run `gcloud auth login` and **`gcloud auth application-default login`** (recommended so `google-cloud-aiplatform` on your machine uses Application Default Credentials).
3. Enable Vertex AI, Compute, Storage, Artifact Registry, and IAM APIs.
4. Create the GCS bucket if it does not exist.
5. Grant the **Compute Engine default service account** `Storage Object Admin` on the bucket and **Vertex AI User** on the project — matching how Vertex CustomJobs often run by default.

Verify from your shell:

```bash
gcloud config get-value project
gcloud config get-value ai/region
```

---

## Step 4 — Fine-tune and deploy

Ensure `.env` is filled out (`python-dotenv` loads it automatically).

From the **repository root**:

```bash
uv run python scripts/finetune_deploy_gemma4.py
```

Train only, then deploy in a second invocation:

```bash
uv run python scripts/finetune_deploy_gemma4.py --train
uv run python scripts/finetune_deploy_gemma4.py --deploy
```

`--help` only documents **`--train`**, **`--deploy`**, and **`--training-async`**; all other options live in `.env`.

What this does:

1. **`CustomJob`** — Packages `train_sql_sft.py`, runs it in the HF PyTorch training DLC. At startup the script upgrades **PyTorch** (when **`VERTEX_UPGRADE_TORCH=1`**) from **`TORCH_PIP_INDEX_URL`**, then installs the HF stack (**`HF_PIP_SPECS`**, same ranges as **`pyproject.toml`**). Environment passes **`MODEL_ID`**, **`OUTPUT_DIR`** (`/gcs/<bucket>/<prefix>` via GCS Fuse), **`HF_TOKEN`**, and hyperparameters from **`.env`**.
2. **Training script** — Loads `b-mc2/sql-create-context`, formats turns as chat, runs **LoRA** SFT with TRL, saves adapters under `.../final`, optionally **merges** to full weights under `.../merged` when **`MERGE_AFTER_TRAIN`** is true.
3. **Model upload** — Registers a Vertex model pointing at **`gs://<bucket>/<prefix>/merged`** using **`VLLM_SERVING_CONTAINER_URI`**.
4. **Deploy** — Deploys an endpoint using **`SERVING_*`**, **`DISPLAY_NAME_*`**, and related `.env` variables.

The script prints the **endpoint resource name** when deploy succeeds. Use the [Vertex AI prediction docs](https://cloud.google.com/vertex-ai/docs/predictions/overview) to send requests to the HTTPS prediction API (payload shape depends on the vLLM container’s route — this setup uses predict route **`/generate`** on port **7080** in the model definition).

Because **`--model`** points at **`gs://…/merged`**, prediction replicas must read Cloud Storage. Set **`VERTEX_PREDICTION_SERVICE_ACCOUNT`** in **`.env`** (passed to **`model.deploy`**) to a service account with **`roles/storage.objectViewer`** on that bucket (see Google’s [Cloud Storage access for custom vLLM](https://cloud.google.com/vertex-ai/generative-ai/docs/open-models/deploy-custom-vllm#create_an_iam_service_account_for_cloud_storage_access)). The deploy script also requests **16&nbsp;GiB** container shared memory (**`SERVING_SHARED_MEMORY_MB`**) by default — low **`/dev/shm`** often crashes vLLM on Vertex.

### Which steps to run

- **`--train`** — Run the Vertex training CustomJob only (no deploy).
- **`--deploy`** — Upload from GCS and deploy the endpoint only (expects merged weights at `gs://<bucket>/<prefix>/merged`).
- **Neither flag** — Run the full pipeline (**train** then **deploy**). Same as **`--train --deploy`**.

**`--training-async`** — Submit training and return without waiting (do not combine with a deploy step in the same invocation unless artifacts already exist; run **`--deploy`** again afterward).

---

## Step 5 — Cost and cleanup

- **Training** is billed for the wall-clock duration of the CustomJob.
- **Endpoints** are billed **continuously** while `min_replica_count` is at least 1. To avoid idle GPU cost, delete the endpoint in the [Vertex AI console](https://console.cloud.google.com/vertex-ai) or use `gcloud` / the SDK to undeploy or delete resources when you are done.

---

## Troubleshooting

| Symptom | What to check |
|--------|----------------|
| `403` on Hugging Face | Confirm **`HF_TOKEN`** is valid and has access to the model and dataset. |
| Quota / L4 errors | Increase **`NVIDIA_L4_GPUS`** quota for **`GCP_REGION`**. |
| Training OOM | Lower **`PER_DEVICE_TRAIN_BATCH_SIZE`**, raise **`GRADIENT_ACCUMULATION_STEPS`**, or shorten **`MAX_SEQ_LENGTH`** in `.env`. |
| Deploy looks for wrong path | Merged weights must exist at `gs://<bucket>/<prefix>/merged` when **`MERGE_AFTER_TRAIN=true`**. |
| **`400 Model server exited unexpectedly`** (deploy LRO fails) | Open **Cloud Logging** from the error URL and read **vLLM** stderr. Common fixes: set **`VERTEX_PREDICTION_SERVICE_ACCOUNT`** to an SA with **`roles/storage.objectViewer`** on **`GCS_BUCKET`** (required for **`gs://`** weights per [custom vLLM deploy](https://cloud.google.com/vertex-ai/generative-ai/docs/open-models/deploy-custom-vllm)); raise **`SERVING_SHARED_MEMORY_MB`** (default **16384** in the script — low **`/dev/shm`** breaks vLLM workers); ensure **`HF_TOKEN`** reaches the container for gated Gemma; lower **`DEPLOY_MAX_MODEL_LEN`** / **`VLLM_GPU_MEMORY_UTILIZATION`**; try **`VLLM_EXTRA_ARGS`** (e.g. **`--trust-remote-code`**); scale GPUs (**`TENSOR_PARALLEL_SIZE`** / **`SERVING_ACCELERATOR_COUNT`**); pin **`VLLM_SERVING_CONTAINER_URI`** if logs show an outdated stack without **Gemma 4**. |
| Container tag drift | Pin **`TRAINING_CONTAINER_URI`** and **`VLLM_SERVING_CONTAINER_URI`** to known-good digests or tags instead of `:latest`. |
| `AttributeError: 'list' object has no attribute 'keys'` / tokenizer errors on Gemma 4 | **`train_sql_sft.py`** normalizes tokenizer loading; Gemma 4 Hub configs can list **`extra_special_tokens`** as a list. |
| `KeyError: 'gemma4'` / unknown **`gemma4`** architecture | **Gemma 4** needs **Transformers ≥ 5.5** (native **`gemma4`**); run **`uv sync`** locally and ensure Vertex bootstrap installs **`transformers>=5.5`** (see **`pyproject.toml`**). |
| `RuntimeError: operator torchvision::nms does not exist` / mixed torch versions | Do **not** use **`pip install -t`** for HF libs. With **`VERTEX_UPGRADE_TORCH=1`** (default), the entrypoint reinstalls **torch / torchvision / torchaudio** together from **`TORCH_PIP_INDEX_URL`**, then the HF stack. With **`VERTEX_UPGRADE_TORCH=0`**, only HF libs are upgraded and **`-c`** pins the DLC’s existing torch family (legacy). |
| **`Disabling PyTorch because PyTorch >= 2.4 is required`** / **`PyTorch was not found`** | Leave **`VERTEX_UPGRADE_TORCH=1`** so the job pulls **PyTorch 2.4+** from **`TORCH_PIP_INDEX_URL`**, or install a matching **torch** locally / in a custom image. |
| **`SFTConfig.__init__() got an unexpected keyword argument 'max_seq_length'`** | TRL renamed this field to **`max_length`** (see TRL `SFTConfig`). |
| **`Cannot copy out of meta tensor`** (local / no GPU) | TRL uses **`device_map=auto`** by default; on **CPU** that can leave the model on **meta**. **`train_sql_sft.py`** sets **`device_map=None`** when **CUDA** is absent. On **Vertex with a GPU**, CUDA is present so **`auto`** is unchanged — this is mostly a **local smoke** issue. |
| **`Invalid buffer size: … GiB`** (often **Apple Silicon**) | The Trainer was using **MPS**; large weights exceed Metal buffer limits. **`SMOKE_MAX_EXAMPLES`** (local smoke) forces **`use_cpu=True`** and **`packing=False`**. For non-smoke local runs, set **`TRAINING_USE_CPU=1`** if needed. **Vertex GPU** jobs use CUDA, not MPS. |

---

## Further reading

- In-repo notes: [`gemma-vertex-finetune.md`](gemma-vertex-finetune.md) (training vs serving lifecycle, LoRA vs merged deploy).
- Original Colab-style checklist: [`gcloud-setup.txt`](gcloud-setup.txt).
