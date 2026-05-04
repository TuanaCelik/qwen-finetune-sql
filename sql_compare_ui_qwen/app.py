#!/usr/bin/env python3
"""
Gradio: **SQL compare** — locally merged Qwen SQL fine-tune vs Hub base (Transformers).

No smolagents tab (compare only).

Env (see ``sql_compare_ui_qwen/.env.example`` and README): ``QWEN_COMPARE_*`` for UI; repo ``.env``
for ``HF_TOKEN``, ``QWEN_MODEL_ID``, ``LOCAL_QWEN_MERGED_CACHE_NAME``, ``QWEN_OUTPUT_GCS_PREFIX``, etc.

Sync merged weights: ``uv run python scripts/query_finetuned_qwen.py --sync`` from repo root.
"""
from __future__ import annotations

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"

if os.environ.get("QWEN_COMPARE_SHOW_RESOURCE_TRACKER_WARNINGS", "").strip() != "1":
    _pw = os.environ.get("PYTHONWARNINGS", "").strip()
    _rt = "ignore:resource_tracker:UserWarning"
    os.environ["PYTHONWARNINGS"] = f"{_pw},{_rt}" if _pw else _rt

import gc
import io
import re
import socket
import sqlite3
import sys
import warnings
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv

    for env_path in (ROOT / ".env", REPO_ROOT / ".env"):
        if env_path.is_file():
            load_dotenv(env_path)
except ImportError:
    pass


def _install_resource_tracker_warning_silencer() -> None:
    if os.environ.get("QWEN_COMPARE_SHOW_RESOURCE_TRACKER_WARNINGS", "").strip() == "1":
        return
    warnings.filterwarnings(
        "ignore",
        message=r".*resource_tracker:.*[Ll]eaked.*semaphore.*",
        category=UserWarning,
    )
    _orig = warnings.showwarning

    def _showwarning(message, category, filename, lineno, file=None, line=None):
        try:
            text = str(message)
        except Exception:
            text = ""
        if (
            "resource_tracker" in text
            and "leaked" in text
            and "semaphore" in text
            and "clean up at shutdown" in text
        ):
            return
        _orig(message, category, filename, lineno, file=file, line=line)

    warnings.showwarning = _showwarning  # type: ignore[assignment]


_install_resource_tracker_warning_silencer()

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

from sql_compare_ui_qwen.prompting import build_prompt

_hf_model = None
_hf_tokenizer = None
_hf_model_id: str | None = None


def _env(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    return str(v).strip()


def _hub_model_id() -> str:
    return (
        _env("QWEN_COMPARE_HUB_MODEL_ID")
        or _env("QWEN_MODEL_ID")
        or "Qwen/Qwen3.5-0.8B"
    )


def _hf_token() -> str | None:
    t = (_env("QWEN_COMPARE_HF_TOKEN") or _env("HF_TOKEN", "")).strip()
    return t or None


def _first_free_port(host: str, start: int, *, max_tries: int = 40) -> int:
    for p in range(start, start + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, p))
                return p
            except OSError:
                continue
    raise RuntimeError(f"No free TCP port in {start}..{start + max_tries - 1} on {host!r}")


def unload_hf_model() -> None:
    global _hf_model, _hf_tokenizer, _hf_model_id
    _hf_model = None
    _hf_tokenizer = None
    _hf_model_id = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def predict_hf(prompt: str) -> str:
    global _hf_model, _hf_tokenizer, _hf_model_id

    if _env("QWEN_COMPARE_SKIP_HUB") == "1":
        return (
            "Hub column skipped (`QWEN_COMPARE_SKIP_HUB=1`). Remove or set to anything other than "
            "`1` to load the Hub model again."
        )

    mid = _hub_model_id()
    token = _hf_token()
    max_new = int(
        _env("QWEN_COMPARE_MAX_NEW_TOKENS", _env("MAX_NEW_TOKENS", "512")) or "512"
    )

    try:
        if _hf_model is None or _hf_model_id != mid:
            from sql_compare_ui_qwen.inference_device import log_qwen_device, pick_hub_device_spec
            from sql_compare_ui_qwen.local_inference import _apply_hf_load_env

            _apply_hf_load_env()
            spec = pick_hub_device_spec()
            tok_kw: dict = {"trust_remote_code": True, "use_fast": True}
            if token:
                tok_kw["token"] = token
            try:
                tokenizer = AutoTokenizer.from_pretrained(mid, **tok_kw)
            except (AttributeError, TypeError) as e:
                err = str(e)
                if "'list' object has no attribute 'keys'" in err or "not a string" in err.lower():
                    tokenizer = AutoTokenizer.from_pretrained(
                        mid, **tok_kw, extra_special_tokens={}
                    )
                else:
                    raise
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            kw: dict = {
                "trust_remote_code": True,
                "torch_dtype": spec.torch_dtype,
                "low_cpu_mem_usage": spec.device_map is None,
            }
            if token:
                kw["token"] = token
            if spec.device_map is not None:
                kw["device_map"] = spec.device_map
            try:
                model = AutoModelForImageTextToText.from_pretrained(mid, **kw)
            except (OSError, ValueError, TypeError):
                model = AutoModelForCausalLM.from_pretrained(mid, **kw)
            if spec.to_device:
                model = model.to(spec.to_device)
            model.eval()
            log_qwen_device("hub", model, spec)
            _hf_model, _hf_tokenizer, _hf_model_id = model, tokenizer, mid

        assert _hf_tokenizer is not None and _hf_model is not None
        messages = [{"role": "user", "content": prompt}]
        try:
            text = _hf_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = _hf_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = _hf_tokenizer(text, return_tensors="pt")
        dev = next(_hf_model.parameters()).device
        inputs = {k: v.to(dev) for k, v in inputs.items()}

        with torch.inference_mode():
            out = _hf_model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=_hf_tokenizer.pad_token_id,
                eos_token_id=_hf_tokenizer.eos_token_id,
            )
        in_len = inputs["input_ids"].shape[-1]
        gen_ids = out[0, in_len:]
        return _hf_tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    except Exception as ex:
        return f"Hub base: {ex!r}"


def _compare_sqlite_db_path() -> Path:
    raw = _env(
        "QWEN_COMPARE_DB_PATH",
        str(REPO_ROOT / "data" / "spider_eval_synthetic" / "synthetic.db"),
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
        return False, "only SELECT (or WITH … SELECT) queries are allowed"
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


def _compare_format_rows(cols: list[str], rows: list[tuple[Any, ...]], *, limit: int) -> str:
    if not cols:
        return "(no columns)"
    buf = io.StringIO()
    buf.write(" | ".join(cols) + "\n")
    buf.write("-" * min(120, 8 * len(cols)) + "\n")
    for row in rows[:limit]:
        buf.write(" | ".join(str(x) if x is not None else "NULL" for x in row) + "\n")
    if len(rows) > limit:
        buf.write(f"\n… truncated to {limit} rows ({len(rows)} returned)\n")
    return buf.getvalue()


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


def _extract_sql(text: str) -> str:
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


def _execute_compare_sql(sql: str, *, row_limit: int = 150) -> str:
    if not (sql or "").strip():
        return "(no SELECT / WITH extracted — nothing to run)"
    ok, stmt = _compare_validate_select(sql)
    if not ok:
        return f"Error: {stmt}"
    db = _compare_sqlite_db_path()
    if not db.is_file():
        return f"Error: database file not found: {db}"
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error as e:
        return f"Error opening database: {e!r}"
    try:
        cur = conn.cursor()
        cur.execute(stmt)
        rows = [tuple(r) for r in cur.fetchall()]
        cols = [d[0] for d in cur.description] if cur.description else []
        return _compare_format_rows(list(cols), rows, limit=row_limit)
    except sqlite3.Error as e:
        return f"Error executing SQL: {e!r}"
    finally:
        conn.close()


def predict_local_ft(prompt: str) -> str:
    from sql_compare_ui_qwen import local_inference

    if _env("QWEN_COMPARE_SKIP_LOCAL") == "1":
        return (
            "Local column skipped (`QWEN_COMPARE_SKIP_LOCAL=1`). Remove or set to anything other "
            "than `1` to load local again."
        )

    local_dir = local_inference.default_local_merged_dir()
    if not (local_dir / "config.json").is_file():
        return (
            f"No local checkpoint at {local_dir}. From repo root run:\n"
            "  uv run python scripts/query_finetuned_qwen.py --sync"
        )
    try:
        return local_inference.generate_local(prompt, model_dir=local_dir)
    except Exception as ex:
        return f"Local: {ex!r}"


def run_compare(user_request: str):
    prompt = build_prompt(user_request)
    out_local = predict_local_ft(prompt)
    if _env("QWEN_COMPARE_SEQUENTIAL_UNLOAD", "1") == "1" and _env("QWEN_COMPARE_SKIP_LOCAL") != "1":
        from sql_compare_ui_qwen import local_inference

        local_inference.unload_local_model()
    out_hf = predict_hf(prompt)
    if _env("QWEN_COMPARE_SEQUENTIAL_UNLOAD", "1") == "1" and _env("QWEN_COMPARE_SKIP_HUB") != "1":
        unload_hf_model()
    sql_local = _extract_sql(out_local)
    sql_hf = _extract_sql(out_hf)
    res_local = _execute_compare_sql(sql_local)
    res_hf = _execute_compare_sql(sql_hf)
    return out_local, res_local, out_hf, res_hf


def main() -> None:
    from sql_compare_ui_qwen import local_inference as _li

    hub = _hub_model_id()
    cache = _li.default_local_merged_dir().name
    title = "SQL compare (Qwen) — local merged vs Hub base"
    desc = (
        f"**Hub**: `{hub}` (override with **`QWEN_COMPARE_HUB_MODEL_ID`** or repo **`QWEN_MODEL_ID`**). "
        f"**Local merged**: **`<repo>/.cache/{cache}/`** (**`QWEN_COMPARE_LOCAL_MERGED_NAME`** or "
        "**`LOCAL_QWEN_MERGED_CACHE_NAME`**; same basename as **`query_finetuned_qwen.py --sync`**). "
        "Each **Run both** unloads local before loading Hub, then unloads Hub (see "
        "**`QWEN_COMPARE_SEQUENTIAL_UNLOAD`**). **`QWEN_COMPARE_SKIP_HUB=1`** / **`QWEN_COMPARE_SKIP_LOCAL=1`** "
        "for single-column mode.\n\n"
        "SQLite runs the first extracted `SELECT`/`WITH` against **`QWEN_COMPARE_DB_PATH`** "
        "(default: `data/spider_eval_synthetic/synthetic.db`)."
    )

    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}")
        gr.Markdown(desc)
        inp = gr.Textbox(
            label="User request",
            placeholder="e.g. List all department names.",
            lines=4,
        )
        btn = gr.Button("Run both", variant="primary")
        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                gr.Markdown("#### Local fine-tuned (merged)")
                out_local = gr.Textbox(label="Model output", lines=12)
                out_local_result = gr.Textbox(label="SQLite query result", lines=14)
            with gr.Column(scale=1):
                gr.Markdown("#### Hub base (Transformers)")
                out_hf = gr.Textbox(label="Model output", lines=12)
                out_hf_result = gr.Textbox(label="SQLite query result", lines=14)
        btn.click(
            fn=run_compare,
            inputs=[inp],
            outputs=[out_local, out_local_result, out_hf, out_hf_result],
        )

    host = _env("QWEN_COMPARE_GRADIO_HOST", "127.0.0.1")
    preferred = int(_env("QWEN_COMPARE_GRADIO_PORT", "7861") or "7861")
    port = _first_free_port(host, preferred)
    if port != preferred:
        print(f"Port {preferred} busy; using {port}.", file=sys.stderr)
    demo.launch(server_name=host, server_port=port)


if __name__ == "__main__":
    main()
