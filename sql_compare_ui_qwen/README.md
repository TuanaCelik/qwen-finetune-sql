---
title: Qwen Text-to-SQL Demo
emoji: 🦄
colorFrom: indigo
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
---

# Qwen Text-to-SQL Demo

Gradio app for comparing the fine-tuned Hub model **`Tuana/qwen35-08b-text2sql`** against the **Hub base** model, then executing the first extracted `SELECT` / `WITH` on a read-only SQLite database.

There is **no** internal smolagents tab (compare only).

## Prerequisites

- Local development: Python env with `uv` (or pip) and dependencies from `requirements.txt`.
- Repo `.env` with at least **`HF_TOKEN`** if you pull gated assets. The Hub comparison defaults to **`Qwen/Qwen3.5-0.8B`**.
- Fine-tuned model available on the Hub as **`Tuana/qwen35-08b-text2sql`**.

## Environment variables

| Variable | Purpose |
|----------|---------|
| **`QWEN_COMPARE_HUB_MODEL_ID`** | Hub model id (default: **`Qwen/Qwen3.5-0.8B`**). |
| **`QWEN_COMPARE_HF_TOKEN`** | Optional; falls back to **`HF_TOKEN`**. |
| **`QWEN_COMPARE_MAX_NEW_TOKENS`** | Generation cap (default **512**; also respects **`MAX_NEW_TOKENS`**). |
| **`QWEN_COMPARE_SEQUENTIAL_UNLOAD`** | Default **true**: unload the fine-tuned model before loading Hub base, then unload Hub base after **Run both** (lower peak memory). |
| **`QWEN_COMPARE_SKIP_HUB`** / **`QWEN_COMPARE_SKIP_FINETUNED`** | Set to **true** to skip that column. |
| **`QWEN_COMPARE_DB_PATH`** | Read-only SQLite (default **`data/spider_eval_synthetic/synthetic.db`**). |
| **`QWEN_COMPARE_DB_SCHEMA`** | Optional full DDL string injected into the prompt. |
| **`QWEN_COMPARE_DEVICE_MAP`** / **`QWEN_COMPARE_HUB_DEVICE_MAP`** | **`auto`**, **`mps`**, **`cpu`**, **`none`**, or a CUDA device map for model loading. |
| **`QWEN_COMPARE_GRADIO_HOST`** / **`QWEN_COMPARE_GRADIO_PORT`** | Bind address (default **127.0.0.1**:**7861**). |

Copy `.env.example` to `.env` for UI-only overrides. When running inside the larger repo, the repo-root `.env` is also loaded.

## Run the UI

From this folder:

```bash
uv run python app.py
```

From the repository root:

```bash
uv run python sql_compare_ui_qwen/app.py
```

Open the printed URL (default **http://127.0.0.1:7861** if the port is free). Use **Run both** to compare the fine-tuned model vs Hub base and show SQLite results for each side.

## Push As A Space

Push the contents of this folder as the Space root, not the whole training repo:

```bash
uv run hf repo create Tuana/qwen-text2sql-demo \
  --type space \
  --space-sdk gradio \
  --exist-ok

uv run hf upload Tuana/qwen-text2sql-demo sql_compare_ui_qwen . \
  --repo-type space \
  --exclude ".env" \
  --exclude "data/spider_eval_synthetic/synthetic.db" \
  --exclude "**/__pycache__/**" \
  --exclude "local_inference.py" \
  --exclude "inference_device.py"
```

Prompting matches **`scripts/train_qwen_sql_sft.py`** and **`sql_compare_ui_qwen.prompting`** (user message body + chat template on the model side).
