#!/usr/bin/env python3
"""
Quick execution benchmark for the compare schema + ``synthetic.db``.

- **Default / ``--gold-only``**: runs bundled gold SQL on the DB (~instant). No ML deps beyond sqlite3.
- **``--local``**: loads merged Qwen once, scores extracted ``SELECT``/``WITH`` vs gold row-sets.
- **``--hub``**: loads Hub Qwen **once** per run, reuses for every case (no per-question reload).
  **``--local``** reuses ``sql_compare_ui_qwen.local_inference``’s in-process cache the same way.
  With **``--local --hub``** you still get **two** full loads (different checkpoints), unless both fit in memory and you later add a “keep both” mode.

Row comparison is **order-insensitive** (sorted row tuples) so gold SQL need not mirror model ``ORDER BY``.

Examples (repo root)::

  uv run python scripts/quick_sql_validate.py
  uv run python scripts/quick_sql_validate.py --local
  uv run python scripts/quick_sql_validate.py --local --hub
  uv run python scripts/quick_sql_validate.py --limit 5 --local
"""
from __future__ import annotations

import argparse
import gc
import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any


def _fmt_s(seconds: float) -> str:
    return f"{seconds:.3f}s"

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

# (id, question, gold_sql) — gold verified against ``data/spider_eval_synthetic/synthetic.db``.
BENCHMARK: list[tuple[str, str, str]] = [
    (
        "groupby_temp_acting",
        "For each value of temporary_acting in management, how many rows have that value?",
        "SELECT temporary_acting, COUNT(*) AS n FROM management GROUP BY temporary_acting ORDER BY temporary_acting",
    ),
    (
        "dept_yes_counts",
        "Departments with at least one temporary_acting = 'Yes' management row; show department_id and count of such rows, highest count first.",
        """SELECT d.department_id, COUNT(*) AS c
FROM department d JOIN management m ON d.department_id = m.department_id
WHERE m.temporary_acting = 'Yes'
GROUP BY d.department_id
ORDER BY c DESC""",
    ),
    (
        "dept_no_management",
        "Which departments have no management row at all? Return department_id and name.",
        """SELECT d.department_id, d.name FROM department d
WHERE NOT EXISTS (SELECT 1 FROM management m WHERE m.department_id = d.department_id)""",
    ),
    (
        "head_never_managed",
        "Which head_ids from head never appear in management as head_id?",
        """SELECT h.head_id FROM head h
WHERE NOT EXISTS (SELECT 1 FROM management m WHERE m.head_id = h.head_id)""",
    ),
    (
        "join_three",
        "For each management row, show department name and head name.",
        """SELECT d.name AS dept_name, h.name AS head_name
FROM management m
JOIN department d ON m.department_id = d.department_id
JOIN head h ON m.head_id = h.head_id
ORDER BY d.name, h.name""",
    ),
    (
        "heads_per_state",
        "How many heads per born_state? List born_state and count, order by count descending.",
        """SELECT born_state, COUNT(*) AS n FROM head GROUP BY born_state ORDER BY n DESC""",
    ),
    (
        "mgmt_per_dept",
        "How many management rows per department_id? Show department_id and count ordered by count descending.",
        """SELECT department_id, COUNT(*) AS n FROM management GROUP BY department_id ORDER BY n DESC""",
    ),
    (
        "having_multi_mgmt",
        "Which department_ids have more than one management row? Show department_id and row count.",
        """SELECT department_id, COUNT(*) AS n FROM management GROUP BY department_id HAVING COUNT(*) > 1 ORDER BY n DESC""",
    ),
    (
        "yes_heads_distinct",
        "Distinct head_ids that have at least one management row with temporary_acting = 'Yes'.",
        """SELECT DISTINCT m.head_id FROM management m WHERE m.temporary_acting = 'Yes' ORDER BY m.head_id""",
    ),
    (
        "california_heads",
        "Names of heads born in state 'California'.",
        """SELECT name FROM head WHERE born_state = 'California' ORDER BY name""",
    ),
    (
        "dept_count",
        "How many departments are there?",
        "SELECT COUNT(*) AS n FROM department",
    ),
    (
        "mgmt_yes_filter",
        "List department_id and head_id from management where temporary_acting is 'Yes', ordered by department_id.",
        """SELECT department_id, head_id FROM management WHERE temporary_acting = 'Yes' ORDER BY department_id, head_id""",
    ),
]


def _env(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    return str(v).strip()


def default_db_path() -> Path:
    raw = _env(
        "QWEN_COMPARE_DB_PATH",
        str(ROOT / "data" / "spider_eval_synthetic" / "synthetic.db"),
    )
    return Path(raw).expanduser().resolve()


def _compare_validate_select(sql: str) -> tuple[bool, str]:
    s = sql.strip()
    if not s:
        return False, "empty SQL"
    parts = [p.strip() for p in s.split(";") if p.strip()]
    if len(parts) != 1:
        return False, "exactly one SQL statement (no multiple statements)"
    one = parts[0]
    low = one.lower()
    if not low.startswith("select") and not low.startswith("with"):
        return False, "only SELECT (or WITH … SELECT) are allowed"
    for b in (
        "attach",
        "pragma",
        "delete",
        "insert",
        "update",
        "drop",
        "create",
        "alter",
        "replace",
        "truncate",
        "vacuum",
        "detach",
    ):
        if re.search(rf"\b{b}\b", low):
            return False, f"forbidden keyword: {b}"
    return True, one


def _last_select_statement(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    anchors = [
        m.start()
        for m in re.finditer(r"(?:^|\n)\s*\b(WITH|SELECT)\b", s, re.MULTILINE | re.IGNORECASE)
    ]
    if not anchors:
        return ""
    frag = s[anchors[-1] :].strip()
    if ";" in frag:
        primary = frag.split(";", 1)[0].strip()
        if re.match(r"(?is)^\s*(?:with|select)\b", primary):
            return primary.rstrip(";").strip()
    return frag.rstrip(";").strip()


def extract_sql(text: str) -> str:
    if not text or not str(text).strip():
        return ""
    t = str(text).strip()
    if t.lower().startswith("no local checkpoint") or "skipped" in t.lower():
        return ""
    blocks = re.findall(r"```(?:sql)?\s*([\s\S]*?)```", t, re.IGNORECASE)
    for raw in reversed(blocks):
        stmt = _last_select_statement(raw)
        if stmt:
            return stmt
    return _last_select_statement(t)


def _cell_key(x: Any) -> str:
    if x is None:
        return "NULL"
    return str(x)


def execute_fingerprint(db: Path, sql: str) -> tuple[str | None, tuple[tuple[str, ...], ...]]:
    """Return (error or None, row multiset as sorted tuple of cell-string tuples — order-insensitive)."""
    ok, stmt = _compare_validate_select(sql)
    if not ok:
        return stmt, tuple()
    if not db.is_file():
        return f"database not found: {db}", tuple()
    conn = None
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        cur = conn.cursor()
        cur.execute(stmt)
        rows = [tuple(_cell_key(c) for c in r) for r in cur.fetchall()]
    except sqlite3.Error as e:
        return str(e), tuple()
    finally:
        if conn is not None:
            conn.close()
    rows.sort()
    return None, tuple(rows)


def run_gold_only(db: Path, cases: list[tuple[str, str, str]]) -> tuple[int, list[tuple[str, str]]]:
    """Return (failure_count, [(case_id, outcome), ...])."""
    bad = 0
    rows: list[tuple[str, str]] = []
    t_block = time.perf_counter()
    for cid, _q, gold in cases:
        t0 = time.perf_counter()
        err, fp = execute_fingerprint(db, gold)
        dt = time.perf_counter() - t0
        if err:
            print(f"FAIL {cid}: gold SQL error: {err}  [TIMING exec {_fmt_s(dt)}]")
            bad += 1
            rows.append((cid, f"FAIL  {err[:120]}"))
        else:
            print(
                f"OK   {cid}: gold returned {len(fp)} rows (fingerprint order-insensitive)  "
                f"[TIMING exec {_fmt_s(dt)}]"
            )
            rows.append((cid, f"OK    {len(fp)} rows"))
    print(f"TIMING gold_sanity_total: {_fmt_s(time.perf_counter() - t_block)}")
    return bad, rows


def _print_final_summary(
    *,
    gold: list[tuple[str, str]],
    local: list[tuple[str, str, str]] | None,
    hub: list[tuple[str, str, str]] | None,
) -> None:
    def _tally(rows: list[tuple[str, str, str]]) -> str:
        m = sum(1 for *_, s in rows if s == "MATCH")
        mis = sum(1 for *_, s in rows if s.startswith("MISMATCH"))
        f = sum(1 for *_, s in rows if s.startswith("FAIL"))
        return f"MATCH={m}  MISMATCH={mis}  FAIL={f}  (total {len(rows)})"

    print()
    print("=" * 72)
    print("RESULT SUMMARY (all cases)")
    print("=" * 72)
    print("\n-- Gold SQL (reference queries on DB) --")
    g_ok = sum(1 for _, s in gold if s.startswith("OK"))
    g_bad = len(gold) - g_ok
    print(f"  (OK={g_ok}  FAIL={g_bad}  total {len(gold)})")
    for cid, st in gold:
        print(f"  {cid:30}  {st}")
    if local is not None:
        print("\n-- Local merged --")
        print(f"  {_tally(local)}")
        for prog, cid, st in local:
            print(f"  {prog:8}  {cid:26}  {st}")
    if hub is not None:
        print("\n-- Hub base --")
        print(f"  {_tally(hub)}")
        for prog, cid, st in hub:
            print(f"  {prog:8}  {cid:26}  {st}")
    print("=" * 72)
    print()


class _HubGenerator:
    """Load Hub Qwen **once**; each ``__call__(prompt)`` only tokenizes + generates."""

    def __init__(self, *, max_new_tokens: int) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

        t_init = time.perf_counter()
        self._max_new_tokens = max_new_tokens
        from sql_compare_ui_qwen.inference_device import log_qwen_device, pick_hub_device_spec

        mid = _env("QWEN_COMPARE_HUB_MODEL_ID") or _env("QWEN_MODEL_ID") or "Qwen/Qwen3.5-0.8B"
        token = (_env("QWEN_COMPARE_HF_TOKEN") or _env("HF_TOKEN", "")).strip() or None
        spec = pick_hub_device_spec()
        tok_kw: dict = {"trust_remote_code": True, "use_fast": True}
        if token:
            tok_kw["token"] = token
        t_tok0 = time.perf_counter()
        self._tokenizer = AutoTokenizer.from_pretrained(mid, **tok_kw)
        t_tok1 = time.perf_counter()
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        kw: dict = {
            "trust_remote_code": True,
            "torch_dtype": spec.torch_dtype,
            "low_cpu_mem_usage": spec.device_map is None,
        }
        if token:
            kw["token"] = token
        if spec.device_map is not None:
            kw["device_map"] = spec.device_map
        t_w0 = time.perf_counter()
        try:
            self._model = AutoModelForImageTextToText.from_pretrained(mid, **kw)
        except (OSError, ValueError, TypeError):
            self._model = AutoModelForCausalLM.from_pretrained(mid, **kw)
        if spec.to_device:
            self._model = self._model.to(spec.to_device)
        t_w1 = time.perf_counter()
        self._model.eval()
        self._torch = torch
        self._last_timing: dict[str, float] = {}
        log_qwen_device("hub(quick_sql_validate)", self._model, spec)
        print(
            "TIMING hub_init: "
            f"tokenizer={_fmt_s(t_tok1 - t_tok0)} "
            f"weights={_fmt_s(t_w1 - t_w0)} "
            f"total={_fmt_s(time.perf_counter() - t_init)}"
        )

    def __call__(self, prompt: str) -> str:
        tok = self._tokenizer
        model = self._model
        t0 = time.perf_counter()
        messages = [{"role": "user", "content": prompt}]
        try:
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        inputs = tok(text, return_tensors="pt")
        dev = next(model.parameters()).device
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        t_pre = time.perf_counter()
        with self._torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        t_gen = time.perf_counter()
        in_len = inputs["input_ids"].shape[-1]
        gen_ids = out[0, in_len:]
        decoded = tok.decode(gen_ids, skip_special_tokens=True).strip()
        t_end = time.perf_counter()
        # Tokenize+template is usually tiny; split prep vs generate for hub hot path.
        self._last_timing = {
            "prep": t_pre - t0,
            "generate": t_gen - t_pre,
            "decode": t_end - t_gen,
            "total_forward": t_end - t0,
        }
        return decoded

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        gc.collect()
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        if getattr(self._torch.backends, "mps", None) is not None and self._torch.backends.mps.is_available():
            try:
                self._torch.mps.empty_cache()
            except Exception:
                pass


def run_model_column(
    label: str,
    db: Path,
    cases: list[tuple[str, str, str]],
    *,
    max_new_tokens: int,
    generate_fn,
) -> tuple[int, list[tuple[str, str, str]]]:
    """Return (failure_count, [(prog, case_id, outcome), ...]). failure = non-MATCH."""
    ok_n = 0
    exec_fail = 0
    mismatch = 0
    t_column = time.perf_counter()
    sum_generate = 0.0
    sum_gold_exec = 0.0
    sum_pred_exec = 0.0
    total_n = len(cases)
    outcomes: list[tuple[str, str, str]] = []
    print(f"{label}: {total_n} test case(s)", flush=True)
    for idx, (cid, question, gold) in enumerate(cases, start=1):
        from sql_compare_ui_qwen.prompting import build_prompt

        prog = f"[{idx}/{total_n}]"
        print(f"PROGRESS {label} {prog} {cid} …", flush=True)
        prompt = build_prompt(question)
        t0 = time.perf_counter()
        try:
            raw = generate_fn(prompt)
        except Exception as e:
            print(
                f"FAIL {label} {prog} {cid}: generate raised {e!r}  "
                f"[TIMING generate {_fmt_s(time.perf_counter() - t0)}]"
            )
            outcomes.append((prog, cid, f"FAIL generate {e!r}"))
            exec_fail += 1
            continue
        t1 = time.perf_counter()
        sum_generate += t1 - t0

        hub_extra = ""
        lt = getattr(generate_fn, "_last_timing", None)
        if isinstance(lt, dict) and lt:
            hub_extra = (
                f" hub_prep={lt.get('prep', 0):.4f}s hub_generate={lt.get('generate', 0):.3f}s "
                f"hub_decode={lt.get('decode', 0):.4f}s"
            )

        pred_sql = extract_sql(raw)
        t_g0 = time.perf_counter()
        g_err, gold_fp = execute_fingerprint(db, gold)
        t_g1 = time.perf_counter()
        sum_gold_exec += t_g1 - t_g0
        if g_err:
            print(
                f"FAIL {label} {prog} {cid}: internal gold error {g_err}  "
                f"[TIMING generate={_fmt_s(t1 - t0)} gold_exec={_fmt_s(t_g1 - t_g0)}{hub_extra}]"
            )
            outcomes.append((prog, cid, f"FAIL gold_internal {g_err[:100]}"))
            exec_fail += 1
            continue
        if not pred_sql:
            print(
                f"FAIL {label} {prog} {cid}: no SQL extracted (output head: {raw[:200]!r}...)  "
                f"[TIMING generate={_fmt_s(t1 - t0)} gold_exec={_fmt_s(t_g1 - t_g0)}{hub_extra}]"
            )
            outcomes.append((prog, cid, "FAIL no_sql_extracted"))
            mismatch += 1
            continue
        t_p0 = time.perf_counter()
        p_err, pred_fp = execute_fingerprint(db, pred_sql)
        t_p1 = time.perf_counter()
        sum_pred_exec += t_p1 - t_p0
        if p_err:
            print(
                f"FAIL {label} {prog} {cid}: pred SQL exec: {p_err} | SQL: {pred_sql[:300]}...  "
                f"[TIMING generate={_fmt_s(t1 - t0)} gold_exec={_fmt_s(t_g1 - t_g0)} "
                f"pred_exec={_fmt_s(t_p1 - t_p0)}{hub_extra}]"
            )
            outcomes.append((prog, cid, f"FAIL pred_exec {p_err[:100]}"))
            exec_fail += 1
            continue
        case_wall = t_p1 - t0
        timing_common = (
            f"TIMING {label} {prog} {cid}: generate={_fmt_s(t1 - t0)} gold_exec={_fmt_s(t_g1 - t_g0)} "
            f"pred_exec={_fmt_s(t_p1 - t_p0)} case_total={_fmt_s(case_wall)}{hub_extra}"
        )
        if gold_fp == pred_fp:
            print(f"MATCH {label} {prog} {cid}  |  {timing_common}")
            outcomes.append((prog, cid, "MATCH"))
            ok_n += 1
        else:
            print(
                f"MISMATCH {label} {prog} {cid}: gold_rows={len(gold_fp)} pred_rows={len(pred_fp)} "
                f"| pred: {pred_sql[:220]}...  |  {timing_common}"
            )
            outcomes.append(
                (prog, cid, f"MISMATCH gold_rows={len(gold_fp)} pred_rows={len(pred_fp)}")
            )
            mismatch += 1
    total = len(cases)
    wall = time.perf_counter() - t_column
    print(
        f"\n--- {label} summary: match={ok_n}/{total} exec_or_gen_fail={exec_fail} mismatch={mismatch} ---"
    )
    print(
        f"TIMING {label}_column_total: {_fmt_s(wall)} "
        f"(sum_generate={_fmt_s(sum_generate)} sum_gold_exec={_fmt_s(sum_gold_exec)} "
        f"sum_pred_exec={_fmt_s(sum_pred_exec)})"
    )
    print()
    return total - ok_n, outcomes


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gold-only", action="store_true", help="Only verify gold SQL runs (default if no --local/--hub).")
    p.add_argument("--local", action="store_true", help="Score merged local Qwen checkpoint.")
    p.add_argument("--hub", action="store_true", help="Score Hub Qwen base.")
    p.add_argument("--limit", type=int, default=0, help="Max benchmark cases (0 = all).")
    p.add_argument("--db", type=str, default="", help="Override SQLite path (default: QWEN_COMPARE_DB_PATH or synthetic.db).")
    p.add_argument("--max-new-tokens", type=int, default=512, help="Generation cap for --local / --hub.")
    args = p.parse_args(argv)

    db = Path(args.db).expanduser().resolve() if args.db.strip() else default_db_path()
    cases = list(BENCHMARK)
    if args.limit and args.limit > 0:
        cases = cases[: args.limit]

    print(f"DB: {db} ({len(cases)} cases)\n")

    run_models = args.local or args.hub
    if not run_models:
        print("== Gold SQL sanity ==")
        t0 = time.perf_counter()
        bad, gold_summary = run_gold_only(db, cases)
        print(f"TIMING run_total (gold-only): {_fmt_s(time.perf_counter() - t0)}")
        _print_final_summary(gold=gold_summary, local=None, hub=None)
        return 1 if bad else 0

    print("== Gold SQL sanity (required before model runs) ==")
    t_gold = time.perf_counter()
    gold_bad, gold_summary = run_gold_only(db, cases)
    print(f"TIMING pre_model_gold_block: {_fmt_s(time.perf_counter() - t_gold)}")
    if gold_bad:
        _print_final_summary(gold=gold_summary, local=None, hub=None)
        return 1

    failures = 0
    summary_local: list[tuple[str, str, str]] | None = None
    summary_hub: list[tuple[str, str, str]] | None = None
    t_run = time.perf_counter()
    if args.local:

        def gen_local(prompt: str) -> str:
            from sql_compare_ui_qwen import local_inference

            return local_inference.generate_local(
                prompt, max_new_tokens=args.max_new_tokens
            )

        print("== Local merged ==")
        t_loc = time.perf_counter()
        fl, summary_local = run_model_column(
            "local", db, cases, max_new_tokens=args.max_new_tokens, generate_fn=gen_local
        )
        failures += fl
        from sql_compare_ui_qwen import local_inference

        t_unl0 = time.perf_counter()
        local_inference.unload_local_model()
        print(f"TIMING local_unload: {_fmt_s(time.perf_counter() - t_unl0)}")
        print(f"TIMING local_section_wall: {_fmt_s(time.perf_counter() - t_loc)}")

    if args.hub:
        print("== Hub base (single load for all cases) ==")
        t_hub_ctor = time.perf_counter()
        hub_gen = _HubGenerator(max_new_tokens=args.max_new_tokens)
        print(f"TIMING hub_ctor_wall: {_fmt_s(time.perf_counter() - t_hub_ctor)}")
        try:
            t_hub_loop = time.perf_counter()
            fh, summary_hub = run_model_column(
                "hub", db, cases, max_new_tokens=args.max_new_tokens, generate_fn=hub_gen
            )
            failures += fh
            print(f"TIMING hub_eval_loop_wall: {_fmt_s(time.perf_counter() - t_hub_loop)}")
        finally:
            t_cl = time.perf_counter()
            hub_gen.close()
            print(f"TIMING hub_close: {_fmt_s(time.perf_counter() - t_cl)}")

    print(f"TIMING script_model_sections_total: {_fmt_s(time.perf_counter() - t_run)}")
    _print_final_summary(
        gold=gold_summary,
        local=summary_local if args.local else None,
        hub=summary_hub if args.hub else None,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
