# SQL compare (Qwen)

Gradio app that matches the **Compare** tab from `sql_compare_ui/`: run the same text-to-SQL prompt against **your merged fine-tuned Qwen** (local disk) and the **Hub base** model, then execute the first extracted `SELECT` / `WITH` on a read-only SQLite database.

There is **no** internal smolagents tab (compare only).

## Prerequisites

- Repository root: Python env with `uv` (or pip) and dependencies from `sql_compare_ui_qwen/requirements.txt` plus **PyTorch** for your machine (see [pytorch.org](https://pytorch.org/get-started/locally/)).
- Repo `.env` with at least **`HF_TOKEN`** if you pull gated assets; **`QWEN_MODEL_ID`** (e.g. `Qwen/Qwen3.5-0.8B`) is used as the default Hub model when **`QWEN_COMPARE_HUB_MODEL_ID`** is unset.
- Merged weights under **`<repo>/.cache/<name>/`** with `config.json`, tokenizer, and weights. The default folder name is **`qwen-sql-merged`**, matching **`LOCAL_QWEN_MERGED_CACHE_NAME`** in the repo `.env`.

## Sync merged weights from GCS

From the repository root (uses **`GCS_BUCKET`**, **`QWEN_OUTPUT_GCS_PREFIX`**, and **`LOCAL_QWEN_MERGED_CACHE_NAME`** from `.env`; optional override **`QWEN_GCS_MERGED_URI`**):

```bash
uv run python scripts/query_finetuned_qwen.py --sync
```

Smoke-check generation against the same cache:

```bash
uv run python scripts/query_finetuned_qwen.py --prompt "List all department names."
```

## Environment variables

| Variable | Purpose |
|----------|---------|
| **`QWEN_COMPARE_HUB_MODEL_ID`** | Hub model id (default: **`QWEN_MODEL_ID`**, then `Qwen/Qwen3.5-0.8B`). |
| **`QWEN_COMPARE_LOCAL_MERGED_NAME`** | Subdirectory under `<repo>/.cache/`. If unset, **`LOCAL_QWEN_MERGED_CACHE_NAME`** is used (then **`qwen-sql-merged`**). |
| **`QWEN_COMPARE_HF_TOKEN`** | Optional; falls back to **`HF_TOKEN`**. |
| **`QWEN_COMPARE_MAX_NEW_TOKENS`** | Generation cap (default **512**; also respects **`MAX_NEW_TOKENS`**). |
| **`QWEN_COMPARE_SEQUENTIAL_UNLOAD`** | Default **1**: unload local before loading Hub, then unload Hub after **Run both** (lower peak memory). |
| **`QWEN_COMPARE_SKIP_HUB`** / **`QWEN_COMPARE_SKIP_LOCAL`** | Set to **1** to skip that column. |
| **`QWEN_COMPARE_DB_PATH`** | Read-only SQLite (default **`data/spider_eval_synthetic/synthetic.db`**). |
| **`QWEN_COMPARE_DB_SCHEMA`** | Optional full DDL string injected into the prompt. |
| **`QWEN_COMPARE_LOCAL_DEVICE_MAP`** | **`auto`**, **`none`**, or a device map string for local load. |
| **`QWEN_COMPARE_GRADIO_HOST`** / **`QWEN_COMPARE_GRADIO_PORT`** | Bind address (default **127.0.0.1**:**7861**). |

Copy `sql_compare_ui_qwen/.env.example` to `sql_compare_ui_qwen/.env` for UI-only overrides; **`sql_compare_ui_qwen/.env`** is loaded first, then **`<repo>/.env`**.

## Run the UI

From the **repository root**:

```bash
uv run python sql_compare_ui_qwen/app.py
```

Open the printed URL (default **http://127.0.0.1:7861** if the port is free). Use **Run both** to compare local merged vs Hub and show SQLite results for each side.

## How it relates to the Gemma app

| Gemma (`sql_compare_ui`) | Qwen (`sql_compare_ui_qwen`) |
|--------------------------|------------------------------|
| `MODEL_ID` / `HF_MODEL_ID` | `QWEN_COMPARE_HUB_MODEL_ID` / `QWEN_MODEL_ID` |
| `LOCAL_FT_MERGED_CACHE_NAME` | `QWEN_COMPARE_LOCAL_MERGED_NAME` / `LOCAL_QWEN_MERGED_CACHE_NAME` |
| `SQL_COMPARE_*` | `QWEN_COMPARE_*` |
| `query_finetuned_gemma.py --sync` | `query_finetuned_qwen.py --sync` |

Prompting matches **`train_qwen_sql_sft.py`** / **`sql_compare_ui.prompting`** (user message body + chat template on the model side).
