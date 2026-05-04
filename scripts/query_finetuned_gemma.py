#!/usr/bin/env python3
"""
Sync merged fine-tuned weights from GCS and run SQL-style generation with Transformers locally.

Follows the same pattern as Hugging Face’s Vertex Gemma 4 example (copy from GCS, then ``from_pretrained``):
https://huggingface.co/docs/google-cloud/examples/vertex-ai-notebooks-fine-tune-gemma-4#use-fine-tune-with-transformers

Paths and base model id come from the repo ``.env`` (``GCS_BUCKET``, ``OUTPUT_GCS_PREFIX``, ``MODEL_ID``,
optional ``LOCAL_FT_MERGED_CACHE_NAME`` for ``<repo>/.cache/<name>/``).
The **small** Gemma 4 variant in this project is ``google/gemma-4-E2B-it`` (see ``MODEL_ID`` in ``.env``).

Examples (from repository root):

  uv run python scripts/query_finetuned_gemma.py --sync
  uv run python scripts/query_finetuned_gemma.py --prompt "List all department names."
  uv run python scripts/query_finetuned_gemma.py --sync --prompt "Count employees per department."
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--sync",
        action="store_true",
        help="Download gs://<GCS_BUCKET>/<OUTPUT_GCS_PREFIX>/merged into <repo>/.cache/<LOCAL_FT_MERGED_CACHE_NAME>/ (gcloud storage rsync).",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default="",
        help="User request for the SQL agent (same template as sql_compare_ui). Omit for --sync-only.",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override MAX_NEW_TOKENS for this call.",
    )
    args = p.parse_args(argv)

    from sql_compare_ui import local_inference
    from sql_compare_ui.prompting import build_prompt

    model_id = os.environ.get("MODEL_ID", "google/gemma-4-E2B-it").strip()
    gcs_uri = local_inference.gcs_merged_uri()
    local_dir = local_inference.default_local_merged_dir()

    print(f"MODEL_ID (base checkpoint family): {model_id}")
    print(f"GCS merged: {gcs_uri}")
    print(f"Local dir: {local_dir}")

    if args.sync:
        print("Syncing from GCS …")
        local_inference.sync_merged_from_gcs(local_dir)
        print("Sync finished.")

    if not args.prompt.strip():
        if not args.sync:
            p.error("Provide --prompt and/or --sync (nothing to do).")
        return 0

    if not (local_dir / "config.json").is_file():
        print(
            f"No local checkpoint at {local_dir}. Run with --sync first.",
            file=sys.stderr,
        )
        return 1

    prompt = build_prompt(args.prompt)
    print("Generating …")
    text = local_inference.generate_local(
        prompt,
        model_dir=local_dir,
        max_new_tokens=args.max_new_tokens,
    )
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
