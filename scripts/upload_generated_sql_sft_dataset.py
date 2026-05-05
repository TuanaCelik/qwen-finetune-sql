#!/usr/bin/env python3
"""Upload the generated Gemini SQL SFT JSONL as a Hugging Face dataset.

Expected JSONL schema per row:
  {"context": "...", "question": "...", "answer": "..."}

Usage:
  uv run python scripts/upload_generated_sql_sft_dataset.py YOUR_HF_USER/gemini-sql-sft
  uv run python scripts/upload_generated_sql_sft_dataset.py --private YOUR_HF_USER/gemini-sql-sft
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/generated_sql_sft/gemini_generated.jsonl"

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload generated SQL SFT JSONL to Hugging Face Hub.")
    p.add_argument(
        "repo_id",
        nargs="?",
        default=(os.environ.get("HF_DATASET_REPO_ID") or os.environ.get("QWEN_DATASET_ID") or "").strip(),
        help="Target dataset repo, e.g. your-user/gemini-sql-sft. Can also use HF_DATASET_REPO_ID.",
    )
    p.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Source JSONL file. Default: {DEFAULT_INPUT.relative_to(ROOT)}",
    )
    p.add_argument("--split", default="train", help="Hub split name to upload. Default: train.")
    p.add_argument("--private", action="store_true", help="Create the dataset repo as private.")
    p.add_argument(
        "--commit-message",
        default="Upload generated SQL SFT data",
        help="Commit message for the Hub upload.",
    )
    return p.parse_args(argv)


def _clean_text(value: object) -> str:
    return "" if value is None else str(value).strip()


def load_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    required = ("context", "question", "answer")
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            missing = [key for key in required if not _clean_text(obj.get(key))]
            if missing:
                raise ValueError(f"{path}:{line_no}: missing/empty fields: {', '.join(missing)}")
            rows.append({key: _clean_text(obj[key]) for key in required})
    if not rows:
        raise ValueError(f"{path}: no usable rows found")
    return rows


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.repo_id:
        print(
            "Missing target dataset repo. Pass repo_id or set HF_DATASET_REPO_ID, "
            "for example: your-user/gemini-sql-sft",
            file=sys.stderr,
        )
        sys.exit(2)
    if not args.input.is_file():
        print(f"Missing input JSONL: {args.input}", file=sys.stderr)
        sys.exit(1)

    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if not token:
        print("Missing HF_TOKEN or HUGGING_FACE_HUB_TOKEN for Hugging Face upload.", file=sys.stderr)
        sys.exit(1)

    from datasets import Dataset  # noqa: PLC0415

    rows = load_rows(args.input)
    ds = Dataset.from_list(rows)
    print(f"Uploading {len(ds)} rows to {args.repo_id} split={args.split!r} ...", flush=True)
    ds.push_to_hub(
        args.repo_id,
        split=args.split,
        private=args.private,
        token=token,
        commit_message=args.commit_message,
    )
    print(f"Uploaded dataset: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
