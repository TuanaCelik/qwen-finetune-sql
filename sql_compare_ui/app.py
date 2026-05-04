#!/usr/bin/env python3
"""
Gradio: (1) SQL compare — local fine-tuned vs Hub base; (2) Internal SQL agent — smolagents CodeAgent
(Hub planner) with ``text2sql`` (local FT) + ``run_sql`` (SQLite).

Env: ``HF_TOKEN``, ``MODEL_ID``; optional ``HF_MODEL_ID``. Sync local weights with
``query_finetuned_gemma.py`` (repo ``.env``: ``GCS_BUCKET``, ``OUTPUT_GCS_PREFIX``,
optional ``LOCAL_FT_MERGED_CACHE_NAME`` → ``<repo>/.cache/<name>/``).

By default each **Run both** unloads the local checkpoint before loading Hub (and unloads Hub
after), so peak memory is roughly **one** model at a time. Set ``SQL_COMPARE_SEQUENTIAL_UNLOAD=0``
to keep both cached in memory (faster repeat runs if VRAM/RAM allows).

Optional one-column mode (value must be exactly ``1``): ``SQL_COMPARE_SKIP_HUB=1`` /
``SQL_COMPARE_SKIP_LOCAL=1``.
"""
from __future__ import annotations

import os

# Before ``torch`` / ``transformers``: weight loaders and tokenizers consult these during import/load.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"

# The leaked-semaphore line is emitted by a *spawned* ``multiprocessing.resource_tracker`` child
# (``resource_tracker.main``), not the Gradio process — ``warnings.showwarning`` hooks here never run
# there. Those children inherit ``PYTHONWARNINGS`` (matched on the warning *message* substring).
if os.environ.get("SQL_COMPARE_SHOW_RESOURCE_TRACKER_WARNINGS", "").strip() != "1":
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


def _install_resource_tracker_warning_silencer() -> None:
    """``filterwarnings`` alone often misses this; Gradio/threads can still print it."""
    if os.environ.get("SQL_COMPARE_SHOW_RESOURCE_TRACKER_WARNINGS", "").strip() == "1":
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
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv

    for env_path in (ROOT / ".env", ROOT.parent / ".env"):
        if env_path.is_file():
            load_dotenv(env_path)
except ImportError:
    pass

from sql_compare_ui.prompting import build_prompt

_hf_model = None
_hf_processor = None
_hf_model_id: str | None = None


def _env(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    return str(v).strip()


def _first_free_port(host: str, start: int, *, max_tries: int = 40) -> int:
    for p in range(start, start + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, p))
                return p
            except OSError:
                continue
    raise RuntimeError(f"No free TCP port in {start}..{start + max_tries - 1} on {host!r}")


def predict_hf(prompt: str) -> str:
    global _hf_model, _hf_processor, _hf_model_id

    if _env("SQL_COMPARE_SKIP_HUB") == "1":
        return (
            "Hub column skipped (`SQL_COMPARE_SKIP_HUB=1`). Remove or set to anything other than "
            "`1` to load the Hub model again."
        )

    token = os.environ["HF_TOKEN"].strip()
    mid = os.environ.get("HF_MODEL_ID", "").strip() or os.environ["MODEL_ID"].strip()
    max_new = int(os.environ.get("MAX_NEW_TOKENS", "512"))

    try:
        if _hf_model is None or _hf_model_id != mid:
            from sql_compare_ui.local_inference import _apply_hf_load_env

            _apply_hf_load_env()
            device_map = "auto" if torch.cuda.is_available() else None
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            processor = AutoProcessor.from_pretrained(mid, token=token, trust_remote_code=True)
            tok = getattr(processor, "tokenizer", None)
            if tok is not None and tok.pad_token is None:
                tok.pad_token = tok.eos_token
            kw: dict = {
                "trust_remote_code": True,
                "dtype": dtype,
                "token": token,
                "low_cpu_mem_usage": device_map is None,
            }
            if device_map is not None:
                kw["device_map"] = device_map
            model = AutoModelForImageTextToText.from_pretrained(mid, **kw)
            model.eval()
            _hf_model, _hf_processor, _hf_model_id = model, processor, mid

        assert _hf_processor is not None and _hf_model is not None
        messages = [{"role": "user", "content": prompt}]
        try:
            text = _hf_processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = _hf_processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = _hf_processor(text=text, return_tensors="pt")
        dev = next(_hf_model.parameters()).device
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        tok = getattr(_hf_processor, "tokenizer", _hf_processor)
        pad_id = getattr(tok, "pad_token_id", None)
        eos_id = getattr(tok, "eos_token_id", None)

        with torch.inference_mode():
            out = _hf_model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )
        in_len = inputs["input_ids"].shape[-1]
        gen_ids = out[0, in_len:]
        return _hf_processor.decode(gen_ids, skip_special_tokens=True).strip()
    except Exception as ex:
        return f"Hub base: {ex!r}"


def unload_hf_model() -> None:
    global _hf_model, _hf_processor, _hf_model_id
    _hf_model = None
    _hf_processor = None
    _hf_model_id = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def _compare_sqlite_db_path() -> Path:
    raw = _env(
        "SQL_AGENT_DB_PATH",
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
    """Last top-level ``WITH``/``SELECT`` line in *s* (model may prepend reasoning)."""
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
    """Prefer the **last** ```sql … ``` block, else last ``WITH``/``SELECT`` tail (after reasoning)."""
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
    from sql_compare_ui import local_inference

    if _env("SQL_COMPARE_SKIP_LOCAL") == "1":
        return "Local column skipped (`SQL_COMPARE_SKIP_LOCAL=1`). Remove or set to anything other than `1` to load local again."

    local_dir = local_inference.default_local_merged_dir()
    if not (local_dir / "config.json").is_file():
        return (
            f"No local checkpoint at {local_dir}. From repo root run:\n"
            "  uv run python scripts/query_finetuned_gemma.py --sync"
        )
    try:
        return local_inference.generate_local(prompt, model_dir=local_dir)
    except Exception as ex:
        return f"Local: {ex!r}"


def run_compare(user_request: str):
    prompt = build_prompt(user_request)
    out_local = predict_local_ft(prompt)
    if _env("SQL_COMPARE_SEQUENTIAL_UNLOAD", "1") == "1" and _env("SQL_COMPARE_SKIP_LOCAL") != "1":
        from sql_compare_ui import local_inference

        local_inference.unload_local_model()
    out_hf = predict_hf(prompt)
    if _env("SQL_COMPARE_SEQUENTIAL_UNLOAD", "1") == "1" and _env("SQL_COMPARE_SKIP_HUB") != "1":
        unload_hf_model()
    sql_local = _extract_sql(out_local)
    sql_hf = _extract_sql(out_hf)
    res_local = _execute_compare_sql(sql_local)
    res_hf = _execute_compare_sql(sql_hf)
    return out_local, res_local, out_hf, res_hf


def _agent_chat_messages(history: list | None) -> list[dict]:
    """Gradio Chatbot (messages format) expects ``{"role", "content"}`` per entry."""
    out: list[dict] = []
    for item in list(history or []):
        if isinstance(item, dict) and "role" in item and "content" in item:
            out.append({"role": item["role"], "content": item["content"]})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            user_txt, asst_txt = item[0], item[1]
            if user_txt is not None and str(user_txt).strip():
                out.append({"role": "user", "content": str(user_txt)})
            if asst_txt is not None and str(asst_txt).strip():
                out.append({"role": "assistant", "content": str(asst_txt)})
    return out


def _run_internal_agent(message: str, history: list):
    if not (message or "").strip():
        return _agent_chat_messages(history), ""
    try:
        from sql_compare_ui.prompting import database_schema
        from sql_compare_ui import agent_smol

        schema = database_schema()
        reply = agent_smol.run_agent_task(message, schema)
    except ImportError as e:
        reply = (
            f"Missing dependency: {e!r}\nInstall with: pip install 'smolagents[transformers]' "
            "(see sql_compare_ui/requirements.txt)."
        )
    except Exception as e:
        reply = f"Error: {e!r}"
    msgs = _agent_chat_messages(history)
    msgs.append({"role": "user", "content": message.strip()})
    msgs.append({"role": "assistant", "content": reply})
    return msgs, ""


def _reset_internal_agent():
    try:
        from sql_compare_ui import agent_smol

        agent_smol.reset_agent()
    except Exception:
        pass
    return [], ""


def main() -> None:
    base = os.environ.get("MODEL_ID", "google/gemma-4-E2B-it").strip()
    title = "SQL tools — compare & internal agent"
    desc_compare = (
        f"**`MODEL_ID`**: `{base}`. Local merged dir: **`<repo>/.cache/<LOCAL_FT_MERGED_CACHE_NAME>/`** (or sync via `query_finetuned_gemma.py --sync`). "
        "Each **Run both** loads local, unloads it, then loads Hub and unloads (see "
        "**`SQL_COMPARE_SEQUENTIAL_UNLOAD`**). Use **`SQL_COMPARE_SKIP_HUB=1`** / **`SQL_COMPARE_SKIP_LOCAL=1`** "
        "to run a single column only.\n\n"
        "**SQLite results** run the first extracted `SELECT`/`WITH` from each model output against "
        "**`SQL_AGENT_DB_PATH`** (default: `data/spider_eval_synthetic/synthetic.db`), same as the agent tab."
    )
    desc_agent = (
        "**Internal company DB (read-only).** The **planner** is the Hub base model (`MODEL_ID` or "
        "**`SQL_AGENT_HUB_MODEL_ID`**). It writes short Python that calls:\n"
        "- **`text2sql`** — your **merged fine-tuned** Gemma under **`<repo>/.cache/<LOCAL_FT_MERGED_CACHE_NAME>/`**.\n"
        "- **`run_sql`** — **SQLite** at **`SQL_AGENT_DB_PATH`** (default: `data/spider_eval_synthetic/synthetic.db`).\n\n"
        "First message loads the Hub agent model (slow). Tools may load the local FT model — **two large models** "
        "can briefly coexist in memory (unlike **Compare**, there is no automatic unload between them). That costs "
        "RAM/VRAM but is **not** the root cause of macOS ``resource_tracker`` / semaphore console noise (threaded "
        "load + Python 3.11). Set **`SQL_COMPARE_SHOW_RESOURCE_TRACKER_WARNINGS=1`** to see those warnings if needed."
    )

    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}")
        with gr.Tabs():
            with gr.Tab("Compare (local vs Hub)"):
                gr.Markdown(desc_compare)
                inp = gr.Textbox(
                    label="User request",
                    placeholder="e.g. List employees in Engineering hired after 2020-01-01 with salary above 90000.",
                    lines=4,
                )
                btn = gr.Button("Run both", variant="primary")
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        gr.Markdown("#### Local fine-tuned")
                        out_local = gr.Textbox(
                            label="Model output",
                            lines=12,
                            elem_classes=["sql-compare-model-out"],
                        )
                        out_local_result = gr.Textbox(
                            label="SQLite query result",
                            lines=14,
                            elem_classes=["sql-compare-exec"],
                        )
                    with gr.Column(scale=1):
                        gr.Markdown("#### Hub base (Transformers)")
                        out_hf = gr.Textbox(
                            label="Model output",
                            lines=12,
                            elem_classes=["sql-compare-model-out"],
                        )
                        out_hf_result = gr.Textbox(
                            label="SQLite query result",
                            lines=14,
                            elem_classes=["sql-compare-exec"],
                        )
                btn.click(
                    fn=run_compare,
                    inputs=[inp],
                    outputs=[out_local, out_local_result, out_hf, out_hf_result],
                )

            with gr.Tab("Internal SQL agent (smolagents)"):
                gr.Markdown(desc_agent)
                agent_chat = gr.Chatbot(label="Conversation", height=400)
                agent_inp = gr.Textbox(
                    label="Your question",
                    placeholder="e.g. How many departments have no management row?",
                    lines=2,
                )
                with gr.Row():
                    agent_send = gr.Button("Send", variant="primary")
                    agent_clear = gr.Button("Clear chat & reset agent")
                agent_send.click(
                    _run_internal_agent,
                    inputs=[agent_inp, agent_chat],
                    outputs=[agent_chat, agent_inp],
                )
                agent_clear.click(_reset_internal_agent, outputs=[agent_chat, agent_inp])

    host = _env("GRADIO_SERVER_NAME", "127.0.0.1")
    preferred = int(_env("GRADIO_SERVER_PORT", "7860") or "7860")
    port = _first_free_port(host, preferred)
    if port != preferred:
        print(f"Port {preferred} busy; using {port}.", file=sys.stderr)
    demo.launch(server_name=host, server_port=port)


if __name__ == "__main__":
    main()
