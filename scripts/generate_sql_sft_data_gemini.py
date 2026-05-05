#!/usr/bin/env python3
"""
Generate text-to-SQL SFT rows (context / question / answer) with **Google Gemini** (cloud),
then validate each ``answer`` with the same SQLite rules as ``scripts/quick_sql_validate.py``.

Auth (pick one):

- **Vertex AI** — set ``GCP_PROJECT_ID`` (or ``GOOGLE_CLOUD_PROJECT``). Uses ``google.genai``
  with ``vertexai=True`` (ADC: ``gcloud auth application-default login``). **Gemini 3** preview
  models are routed to ``location=global`` by default (they are not published under
  ``us-central1``). Override with ``GEMINI_VERTEX_LOCATION`` (e.g. ``us-central1`` for Gemini 2.x).
- **Gemini Developer API** — set ``GEMINI_API_KEY`` or ``GOOGLE_API_KEY`` (AI Studio).

Default model is **Gemini 3 Flash (preview)** — ``gemini-3-flash-preview`` ([model card](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/3-flash)).
On Vertex, call the **global** endpoint (see ``GEMINI_VERTEX_LOCATION``). Override model with
``GEMINI_SQL_GEN_MODEL`` (e.g. ``gemini-2.0-flash``) if preview access is unavailable.

Examples (repo root)::

  uv run python scripts/generate_sql_sft_data_gemini.py --n 3
  uv run python scripts/generate_sql_sft_data_gemini.py --n 5 --schema-file my.sql --spec-file spec.md
  uv run python scripts/generate_sql_sft_data_gemini.py --n 2 --dry-run
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def _env(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    return str(v).strip()


def _load_quick_sql_validate():
    path = ROOT / "scripts" / "quick_sql_validate.py"
    spec = importlib.util.spec_from_file_location("_quick_sql_validate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_DEFAULT_SPEC = """\
Generate training pairs for **SQLite** text-to-SQL supervised fine-tuning.

Requirements:
- Questions must be natural English (no SQL in the question).
- Answers must be a **single** ``SELECT`` or ``WITH … SELECT`` statement only.
- No DDL/DML, no ``PRAGMA``, no multiple statements, no string-escaped tricks.
- Use only tables and columns that appear in the schema.
- Prefer variety: simple filters, JOINs, ``GROUP BY`` / ``HAVING``, ``EXISTS`` / ``NOT EXISTS``,
  ``DISTINCT``, aggregates, ordering when it helps disambiguate.

The validation step runs each ``answer`` read-only against the same database file used by
``scripts/quick_sql_validate.py`` (see ``--db`` / ``QWEN_COMPARE_DB_PATH``).
"""


def _response_schema_exactly_n(n: int, types: Any) -> Any:
    return types.Schema(
        type=types.Type.ARRAY,
        min_items=n,
        max_items=n,
        items=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "id": types.Schema(
                    type=types.Type.STRING,
                    description="snake_case unique id for this example",
                ),
                "question": types.Schema(type=types.Type.STRING),
                "answer": types.Schema(
                    type=types.Type.STRING,
                    description="Raw SQLite SQL only, no markdown",
                ),
            },
            required=["id", "question", "answer"],
            property_ordering=["id", "question", "answer"],
        ),
    )


def _strip_sql_fence(s: str) -> str:
    t = (s or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:sql)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def _build_user_prompt(*, schema: str, specification: str, n: int) -> str:
    return f"""You are an expert SQLite database instructor.

## Schema

{schema.strip()}

## Specification

{specification.strip()}

## Task

Produce exactly {n} training examples. Each example must have:
- ``id``: short snake_case identifier (unique across the list).
- ``question``: clear user question in English.
- ``answer``: one correct SQLite query answering the question (executable on the described schema).

Output **only** a JSON array of length {n} matching the required schema. No prose before or after.
"""


def _vertex_location_for_model(model: str) -> str:
    """Gemini 3 preview models use the Vertex *global* publisher endpoint; regional 404s otherwise."""
    explicit = _env("GEMINI_VERTEX_LOCATION")
    if explicit:
        return explicit
    if "gemini-3" in model.lower():
        return "global"
    return (
        _env("GCP_REGION") or _env("GOOGLE_CLOUD_REGION") or _env("GOOGLE_CLOUD_LOCATION") or "us-central1"
    ).strip()


def _make_genai_client(*, model: str):
    from google import genai

    api_key = (_env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY")).strip()
    project = (_env("GCP_PROJECT_ID") or _env("GOOGLE_CLOUD_PROJECT")).strip()

    if api_key:
        return genai.Client(api_key=api_key), "developer_api"
    if project:
        location = _vertex_location_for_model(model)
        return (
            genai.Client(vertexai=True, project=project, location=location),
            f"vertex({project}/{location})",
        )
    raise SystemExit(
        "Missing credentials: set GEMINI_API_KEY (or GOOGLE_API_KEY), or set "
        "GCP_PROJECT_ID / GOOGLE_CLOUD_PROJECT for Vertex AI, and run "
        "`gcloud auth application-default login` if needed."
    )


def _default_model() -> str:
    return (_env("GEMINI_SQL_GEN_MODEL") or "gemini-3-flash-preview").strip() or "gemini-3-flash-preview"


def _max_output_tokens() -> int:
    raw = _env("GEMINI_SQL_MAX_OUTPUT_TOKENS", "32768")
    try:
        value = int(raw)
    except ValueError:
        raise SystemExit(f"GEMINI_SQL_MAX_OUTPUT_TOKENS must be an integer, got {raw!r}") from None
    if value < 1:
        raise SystemExit("GEMINI_SQL_MAX_OUTPUT_TOKENS must be >= 1")
    return value


def _empty_response_details(resp: Any) -> str:
    parts: list[str] = []
    prompt_feedback = getattr(resp, "prompt_feedback", None)
    if prompt_feedback is not None:
        parts.append(f"prompt_feedback={prompt_feedback!r}")
    candidates = getattr(resp, "candidates", None) or []
    for i, candidate in enumerate(candidates):
        finish_reason = getattr(candidate, "finish_reason", None)
        finish_message = getattr(candidate, "finish_message", None)
        parts.append(f"candidate[{i}].finish_reason={finish_reason!r}")
        if finish_message:
            parts.append(f"candidate[{i}].finish_message={finish_message!r}")
    return "; ".join(parts) or "no candidates returned"


def _call_gemini(*, user_prompt: str, n: int, model: str) -> str:
    from google.genai import types

    client, mode = _make_genai_client(model=model)
    cfg = types.GenerateContentConfig(
        temperature=1.0,
        max_output_tokens=_max_output_tokens(),
        response_mime_type="application/json",
        response_schema=_response_schema_exactly_n(n, types),
    )
    print(f"Calling Gemini model={model!r} via {mode} …", flush=True)
    resp = client.models.generate_content(model=model, contents=user_prompt, config=cfg)
    text = (getattr(resp, "text", None) or "").strip()
    if not text:
        raise RuntimeError(f"empty response from Gemini ({_empty_response_details(resp)})")
    return text


def _parse_examples(raw_json: str) -> list[dict[str, Any]]:
    data = json.loads(raw_json)
    if not isinstance(data, list):
        raise ValueError("top-level JSON must be an array")
    out: list[dict[str, Any]] = []
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"item {i} must be an object")
        for k in ("id", "question", "answer"):
            if k not in row:
                raise ValueError(f"item {i} missing {k!r}")
        out.append(row)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=5, help="Number of examples to generate (small for smoke tests).")
    p.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "data" / "generated_sql_sft" / "gemini_generated.jsonl"),
        help="Output JSONL path (parent dirs created). Each line: context, question, answer.",
    )
    p.add_argument("--db", type=str, default="", help="SQLite path (default: same as quick_sql_validate).")
    p.add_argument("--schema-file", type=str, default="", help="DDL text file; default: QWEN_COMPARE_DB_SCHEMA or compare UI default.")
    p.add_argument("--spec-file", type=str, default="", help="Extra specification markdown/text; else built-in default.")
    p.add_argument("--model", type=str, default="", help="Override GEMINI_SQL_GEN_MODEL.")
    p.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Maximum examples per Gemini call. Large batches can hit Vertex schema or output-token limits.",
    )
    p.add_argument("--dry-run", action="store_true", help="Print prompt only; no API or DB calls.")
    p.add_argument(
        "--allow-partial",
        action="store_true",
        help="Exit 0 if at least one row validates, even if others fail (still writes only valid rows).",
    )
    args = p.parse_args(argv)

    n = args.n
    if n < 1:
        p.error("--n must be >= 1")
    if args.batch_size < 1:
        p.error("--batch-size must be >= 1")

    if args.schema_file.strip():
        schema = Path(args.schema_file).expanduser().read_text(encoding="utf-8")
    else:
        from sql_compare_ui_qwen.prompting import database_schema

        schema = database_schema()

    if args.spec_file.strip():
        specification = Path(args.spec_file).expanduser().read_text(encoding="utf-8")
    else:
        specification = _DEFAULT_SPEC

    qsv = _load_quick_sql_validate()
    db = Path(args.db).expanduser().resolve() if args.db.strip() else qsv.default_db_path()
    if not args.dry_run and not db.is_file():
        print(f"ERROR: SQLite database not found: {db}", file=sys.stderr)
        return 1

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = (args.model.strip() or _default_model()).strip()
    batch_size = min(args.batch_size, n)
    user_prompt = _build_user_prompt(schema=schema, specification=specification, n=batch_size)

    if args.dry_run:
        print("--- DRY RUN: prompt (no API call) ---\n")
        print(user_prompt)
        if n > batch_size:
            batches = (n + batch_size - 1) // batch_size
            print(f"\n--- would run {batches} batch(es) of up to {batch_size} example(s) ---")
        print("\n--- end prompt ---")
        return 0

    valid_count = 0
    failures: list[tuple[str, str]] = []

    generated = 0
    batch_idx = 0
    total_batches = (n + batch_size - 1) // batch_size
    existing_rows = sum(1 for _ in out_path.open("r", encoding="utf-8")) if out_path.is_file() else 0
    print(
        f"Generating {n} example(s) in {total_batches} batch(es) of up to {batch_size}; "
        f"appending to {out_path} ({existing_rows} existing row(s)).",
        flush=True,
    )
    while generated < n:
        batch_idx += 1
        batch_n = min(batch_size, n - generated)
        batch_prompt = _build_user_prompt(schema=schema, specification=specification, n=batch_n)
        print(
            f"\nBatch {batch_idx}/{total_batches}: requesting {batch_n} example(s) "
            f"({generated + 1}-{generated + batch_n} of {n})",
            flush=True,
        )

        try:
            raw = _call_gemini(user_prompt=batch_prompt, n=batch_n, model=model)
        except Exception as e:
            print(f"ERROR: Gemini call failed in batch {batch_idx}: {e}", file=sys.stderr)
            return 1

        try:
            examples = _parse_examples(raw)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"ERROR: invalid JSON from model in batch {batch_idx}: {e}\n--- raw ---\n{raw[:4000]}", file=sys.stderr)
            return 1

        if len(examples) != batch_n:
            print(f"ERROR: batch {batch_idx} expected {batch_n} examples, got {len(examples)}", file=sys.stderr)
            return 1

        batch_rows: list[dict[str, str]] = []
        valid_before = valid_count
        failures_before = len(failures)
        for ex in examples:
            eid = str(ex["id"]).strip()
            question = str(ex["question"]).strip()
            answer = _strip_sql_fence(str(ex["answer"]))
            err, _fp = qsv.execute_fingerprint(db, answer)
            if err:
                failures.append((eid, err))
                print(f"INVALID {eid}: {err}\n  SQL: {answer[:500]!s}\n", flush=True)
            else:
                row = {"context": schema.strip(), "question": question, "answer": answer}
                batch_rows.append(row)
                valid_count += 1
                print(f"OK      {eid}: executes on {db}", flush=True)

        if batch_rows:
            with out_path.open("a", encoding="utf-8") as f:
                for row in batch_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())

        generated += batch_n
        print(
            f"Batch {batch_idx}/{total_batches} complete: "
            f"{valid_count - valid_before} valid, {len(failures) - failures_before} invalid "
            f"(cumulative: {valid_count} valid, {len(failures)} invalid).",
            flush=True,
        )

    print(f"\nAppended {valid_count} valid row(s) to {out_path}")
    if failures:
        print(f"Failed validation: {len(failures)} example(s).")
        if args.allow_partial and valid_count:
            return 0
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
