# SQL Compare UI

Gradio app for comparing the fine-tuned Hub model **`Tuana/qwen35-08b-text2sql`** against the **Hub base** model, then executing the first extracted `SELECT` / `WITH` on a read-only SQLite database.

There is **no** internal smolagents tab (compare only).

## Prerequisites

- Repository root: Python env with `uv` (or pip) and dependencies from `sql_compare_ui_qwen/requirements.txt` plus **PyTorch** for your machine (see [pytorch.org](https://pytorch.org/get-started/locally/)).
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

Copy `sql_compare_ui_qwen/.env.example` to `sql_compare_ui_qwen/.env` for UI-only overrides; **`sql_compare_ui_qwen/.env`** is loaded first, then **`<repo>/.env`**.

## Run the UI

From the **repository root**:

```bash
uv run python sql_compare_ui_qwen/app.py
```

Open the printed URL (default **http://127.0.0.1:7861** if the port is free). Use **Run both** to compare the fine-tuned model vs Hub base and show SQLite results for each side.

Prompting matches **`scripts/train_qwen_sql_sft.py`** and **`sql_compare_ui_qwen.prompting`** (user message body + chat template on the model side).
