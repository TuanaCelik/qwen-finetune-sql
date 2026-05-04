# SQL compare UI (Gradio)

From repo root: **`uv sync --extra sql-compare-ui`** (declared in root `pyproject.toml`; includes **`smolagents[transformers]`** for the second tab). Alternatively: **`pip install -r sql_compare_ui/requirements.txt`**.

- **Tab “Compare”:** **Local** merged weights under `<repo>/.cache/<LOCAL_FT_MERGED_CACHE_NAME>/` (repo `.env`: `GCS_BUCKET`, `OUTPUT_GCS_PREFIX`, optional `LOCAL_FT_MERGED_CACHE_NAME`) vs **Hub** base (`HF_TOKEN`, `MODEL_ID` / `HF_MODEL_ID`).
- **Tab “Internal SQL agent”:** [smolagents](https://huggingface.co/docs/smolagents) **CodeAgent** uses the **Hub** model to write Python that calls **`text2sql`** (local FT) and **`run_sql`** (read-only SQLite at **`SQL_AGENT_DB_PATH`**, default `data/spider_eval_synthetic/synthetic.db`). Optional: **`SQL_AGENT_HUB_MODEL_ID`**, **`SQL_AGENT_MAX_STEPS`**, **`SQL_AGENT_MAX_NEW_TOKENS`**.

## Point at a different training run (e.g. run4-l4 instead of run7-1h)

1. In **repo root** `.env`, set **`OUTPUT_GCS_PREFIX`** to the prefix used when that Vertex job wrote artifacts (merged lives at `gs://<GCS_BUCKET>/<OUTPUT_GCS_PREFIX>/merged`). Confirm in Cloud Console or:
   ```bash
   gcloud storage ls "gs://YOUR_BUCKET/YOUR_PREFIX/merged/"
   ```
2. Optional: set **`LOCAL_FT_MERGED_CACHE_NAME`** (e.g. `gemma-sql-merged-run4-l4`) so weights go to `<repo>/.cache/<name>/` and old runs are not overwritten.
3. Re-download merged weights:
   ```bash
   uv run python scripts/query_finetuned_gemma.py --sync
   ```
4. Run Gradio: `uv run python sql_compare_ui/app.py`

By default (**`SQL_COMPARE_SEQUENTIAL_UNLOAD=1`**) the app loads the local merged model, runs it, **unloads** it, then loads the Hub model and unloads after—so peak memory is about **one** full model at a time. Set **`SQL_COMPARE_SEQUENTIAL_UNLOAD=0`** only if you have enough VRAM/RAM to keep both cached for faster repeat runs. **`SQL_COMPARE_SKIP_HUB=1`** / **`SQL_COMPARE_SKIP_LOCAL=1`** (exactly `1`) run a single column only.
