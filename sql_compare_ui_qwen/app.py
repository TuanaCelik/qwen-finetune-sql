#!/usr/bin/env python3
"""
Gradio: **SQL compare** — fine-tuned Qwen SQL demo model vs Hub base (Transformers).

No smolagents tab (compare only).

Env (see ``sql_compare_ui_qwen/.env.example`` and README): ``QWEN_COMPARE_*`` for UI;
repo ``.env`` for ``HF_TOKEN`` and shared project settings.
"""
from __future__ import annotations

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"

if os.environ.get("QWEN_COMPARE_SHOW_RESOURCE_TRACKER_WARNINGS", "").strip().lower() != "true":
    _pw = os.environ.get("PYTHONWARNINGS", "").strip()
    _rt = "ignore:resource_tracker:UserWarning"
    os.environ["PYTHONWARNINGS"] = f"{_pw},{_rt}" if _pw else _rt

import gc
import csv
import html
import io
import re
import socket
import sqlite3
import sys
import warnings
from pathlib import Path

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
    if os.environ.get("QWEN_COMPARE_SHOW_RESOURCE_TRACKER_WARNINGS", "").strip().lower() == "true":
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

try:
    from sql_compare_ui_qwen.prompting import build_prompt
except ModuleNotFoundError:
    from prompting import build_prompt

_hf_model = None
_hf_tokenizer = None
_hf_model_id: str | None = None
_ft_hf_model = None
_ft_hf_tokenizer = None
_ft_hf_model_id: str | None = None
FINETUNED_HUB_MODEL_ID = "Tuana/qwen35-08b-text2sql"
BASE_MODEL_ID = "Qwen/Qwen3.5-0.8B"

def _env(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    return str(v).strip()

def _hf_token() -> str | None:
    t = (_env("QWEN_COMPARE_HF_TOKEN") or _env("HF_TOKEN", "")).strip()
    return t or None


def _demo_data_dir() -> Path:
    for path in (
        ROOT / "data" / "spider_eval_synthetic",
        REPO_ROOT / "data" / "spider_eval_synthetic",
    ):
        if (path / "department.csv").is_file():
            return path
    return ROOT / "data" / "spider_eval_synthetic"


def _first_free_port(host: str, start: int, *, max_tries: int = 40) -> int:
    for p in range(start, start + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, p))
                return p
            except OSError:
                continue
    raise RuntimeError(f"No free TCP port in {start}..{start + max_tries - 1} on {host!r}")


def _mps_is_available() -> bool:
    b = getattr(torch.backends, "mps", None)
    return b is not None and b.is_available()


def _mps_load_dtype() -> torch.dtype:
    raw = _env("QWEN_COMPARE_MPS_DTYPE").lower()
    if raw in ("bf16", "bfloat16"):
        return torch.bfloat16
    if raw in ("fp16", "float16", "16"):
        return torch.float16
    return torch.float32


def _model_load_spec() -> tuple[torch.dtype, str | None, str | None, str]:
    raw = (
        _env("QWEN_COMPARE_HUB_DEVICE_MAP")
        or _env("QWEN_COMPARE_DEVICE_MAP")
    ).lower()
    if raw in ("none", "null", "cpu"):
        return torch.float32, None, "cpu", raw or "cpu"
    if raw == "mps":
        if _mps_is_available():
            return _mps_load_dtype(), None, "mps", "mps"
        return torch.float32, None, "cpu", "mps_unavailable"
    if raw.startswith("cuda") or raw == "auto":
        if torch.cuda.is_available():
            return torch.bfloat16, ("auto" if raw == "auto" else raw), None, raw
        if _mps_is_available():
            return _mps_load_dtype(), None, "mps", f"{raw}_cuda_missing"
        return torch.float32, None, "cpu", f"{raw}_no_accel"
    if raw:
        if torch.cuda.is_available():
            return torch.bfloat16, raw, None, raw
        if _mps_is_available():
            return _mps_load_dtype(), None, "mps", f"{raw}_mps_fallback"
        return torch.float32, None, "cpu", f"{raw}_cpu_fallback"
    if torch.cuda.is_available():
        return torch.bfloat16, "auto", None, "cuda_auto"
    if _mps_is_available():
        return _mps_load_dtype(), None, "mps", "mps_default"
    return torch.float32, None, "cpu", "cpu_default"


def _log_model_device(kind: str, model: torch.nn.Module, reason: str, dtype: torch.dtype, device_map: str | None, to_device: str | None) -> None:
    p = next(model.parameters())
    print(
        f"QWEN_DEVICE {kind}: reason={reason} | param_device={p.device} | "
        f"param_dtype={p.dtype} | load_dtype={dtype} | device_map={device_map!r} | "
        f"post_to={to_device!r}",
        flush=True,
    )


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


def unload_ft_hf_model() -> None:
    global _ft_hf_model, _ft_hf_tokenizer, _ft_hf_model_id
    _ft_hf_model = None
    _ft_hf_tokenizer = None
    _ft_hf_model_id = None
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

    if _env("QWEN_COMPARE_SKIP_HUB").lower() == "true":
        return (
            "Hub column skipped (`QWEN_COMPARE_SKIP_HUB=true`). Set `QWEN_COMPARE_SKIP_HUB=false` "
            "to load the Hub model again."
        )

    mid = BASE_MODEL_ID
    token = _hf_token()
    max_new = int(
        _env("QWEN_COMPARE_MAX_NEW_TOKENS", _env("MAX_NEW_TOKENS", "512")) or "512"
    )

    try:
        if _hf_model is None or _hf_model_id != mid:
            dtype, device_map, to_device, device_reason = _model_load_spec()
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
                "torch_dtype": dtype,
                "low_cpu_mem_usage": device_map is None,
            }
            if token:
                kw["token"] = token
            if device_map is not None:
                kw["device_map"] = device_map
            try:
                model = AutoModelForImageTextToText.from_pretrained(mid, **kw)
            except (OSError, ValueError, TypeError):
                model = AutoModelForCausalLM.from_pretrained(mid, **kw)
            if to_device:
                model = model.to(to_device)
            model.eval()
            _log_model_device("hub", model, device_reason, dtype, device_map, to_device)
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


def predict_finetuned_hf(prompt: str) -> str:
    global _ft_hf_model, _ft_hf_tokenizer, _ft_hf_model_id

    if _env("QWEN_COMPARE_SKIP_FINETUNED").lower() == "true":
        return (
            "Fine-tuned column skipped (`QWEN_COMPARE_SKIP_FINETUNED=true`). "
            "Set `QWEN_COMPARE_SKIP_FINETUNED=false` to load it again."
        )

    mid = FINETUNED_HUB_MODEL_ID
    token = _hf_token()
    max_new = int(
        _env("QWEN_COMPARE_MAX_NEW_TOKENS", _env("MAX_NEW_TOKENS", "512")) or "512"
    )

    try:
        if _ft_hf_model is None or _ft_hf_model_id != mid:
            dtype, device_map, to_device, device_reason = _model_load_spec()
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
                "torch_dtype": dtype,
                "low_cpu_mem_usage": device_map is None,
            }
            if token:
                kw["token"] = token
            if device_map is not None:
                kw["device_map"] = device_map
            try:
                model = AutoModelForImageTextToText.from_pretrained(mid, **kw)
            except (OSError, ValueError, TypeError):
                model = AutoModelForCausalLM.from_pretrained(mid, **kw)
            if to_device:
                model = model.to(to_device)
            model.eval()
            _log_model_device("fine-tuned-hf", model, device_reason, dtype, device_map, to_device)
            _ft_hf_model, _ft_hf_tokenizer, _ft_hf_model_id = model, tokenizer, mid

        assert _ft_hf_tokenizer is not None and _ft_hf_model is not None
        messages = [{"role": "user", "content": prompt}]
        try:
            text = _ft_hf_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = _ft_hf_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = _ft_hf_tokenizer(text, return_tensors="pt")
        dev = next(_ft_hf_model.parameters()).device
        inputs = {k: v.to(dev) for k, v in inputs.items()}

        with torch.inference_mode():
            out = _ft_hf_model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=_ft_hf_tokenizer.pad_token_id,
                eos_token_id=_ft_hf_tokenizer.eos_token_id,
            )
        in_len = inputs["input_ids"].shape[-1]
        gen_ids = out[0, in_len:]
        return _ft_hf_tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    except Exception as ex:
        return f"Fine-tuned HF: {ex!r}"


def _compare_sqlite_db_path() -> Path:
    raw = _env(
        "QWEN_COMPARE_DB_PATH",
        str(_demo_data_dir() / "synthetic.db"),
    )
    return Path(raw).expanduser().resolve()


def _load_csv_rows(data_dir: Path) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    with (data_dir / "department.csv").open(newline="", encoding="utf-8") as f:
        departments = list(csv.DictReader(f))
    with (data_dir / "head.csv").open(newline="", encoding="utf-8") as f:
        heads = list(csv.DictReader(f))
    with (data_dir / "management.csv").open(newline="", encoding="utf-8") as f:
        management = list(csv.DictReader(f))
    return departments, heads, management


def _ensure_compare_sqlite_db() -> Path:
    db = _compare_sqlite_db_path()
    if db.is_file():
        return db

    data_dir = _demo_data_dir()
    departments, heads, management = _load_csv_rows(data_dir)
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    try:
        conn.executescript(
            """
            DROP TABLE IF EXISTS department;
            DROP TABLE IF EXISTS management;
            DROP TABLE IF EXISTS head;

            CREATE TABLE department (
              department_id VARCHAR,
              name VARCHAR,
              creation VARCHAR
            );
            CREATE TABLE management (
              department_id VARCHAR,
              head_id VARCHAR,
              temporary_acting VARCHAR
            );
            CREATE TABLE head (
              head_id VARCHAR,
              name VARCHAR,
              born_state VARCHAR
            );
            """
        )
        conn.executemany(
            "INSERT INTO department (department_id, name, creation) VALUES (?, ?, ?)",
            [(r["department_id"], r["name"], r["creation"]) for r in departments],
        )
        conn.executemany(
            "INSERT INTO head (head_id, name, born_state) VALUES (?, ?, ?)",
            [(r["head_id"], r["name"], r["born_state"]) for r in heads],
        )
        conn.executemany(
            "INSERT INTO management (department_id, head_id, temporary_acting) VALUES (?, ?, ?)",
            [(r["department_id"], r["head_id"], r["temporary_acting"]) for r in management],
        )
        conn.commit()
    finally:
        conn.close()
    return db


def _database_preview_rows(limit: int = 5) -> list[dict[str, str]]:
    db = _ensure_compare_sqlite_db()
    if db.is_file():
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT
                  d.department_id,
                  d.name AS department,
                  d.creation,
                  h.name AS department_head,
                  h.born_state,
                  m.temporary_acting
                FROM department AS d
                JOIN management AS m ON m.department_id = d.department_id
                JOIN head AS h ON h.head_id = m.head_id
                ORDER BY d.department_id, h.head_id
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    data_dir = _demo_data_dir()
    dept_rows, head_rows, management = _load_csv_rows(data_dir)
    departments = {r["department_id"]: r for r in dept_rows}
    heads = {r["head_id"]: r for r in head_rows}

    preview: list[dict[str, str]] = []
    for rel in management:
        dept = departments.get(rel["department_id"])
        head = heads.get(rel["head_id"])
        if not dept or not head:
            continue
        preview.append(
            {
                "department_id": dept["department_id"],
                "department": dept["name"],
                "creation": dept["creation"],
                "department_head": head["name"],
                "born_state": head["born_state"],
                "temporary_acting": rel["temporary_acting"],
            }
        )
        if len(preview) >= limit:
            break
    return preview


def _database_preview_html() -> str:
    rows = _database_preview_rows()
    headers = [
        "department_id",
        "department",
        "creation",
        "department_head",
        "born_state",
        "temporary_acting",
    ]
    body = "\n".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(row.get(h, '')))}</td>" for h in headers)
        + "</tr>"
        for row in rows
    )
    header = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    return f"""
    <section class="db-preview">
        <div>
            <p class="eyebrow">Dummy database preview</p>
            <h2>What the model is querying</h2>
            <p>
                The demo database has three related tables:
                <code>department</code>, <code>management</code>, and <code>head</code>.
                These five rows are real examples from the local synthetic database.
            </p>
        </div>
        <table>
            <thead><tr>{header}</tr></thead>
            <tbody>{body}</tbody>
        </table>
    </section>
    """


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
    db = _ensure_compare_sqlite_db()
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


def run_compare(user_request: str):
    prompt = build_prompt(user_request)
    out_local = predict_finetuned_hf(prompt)
    if _env("QWEN_COMPARE_SEQUENTIAL_UNLOAD", "true").lower() == "true" and _env("QWEN_COMPARE_SKIP_FINETUNED").lower() != "true":
        unload_ft_hf_model()
    out_hf = predict_hf(prompt)
    if _env("QWEN_COMPARE_SEQUENTIAL_UNLOAD", "true").lower() == "true" and _env("QWEN_COMPARE_SKIP_HUB").lower() != "true":
        unload_hf_model()
    sql_local = _extract_sql(out_local)
    sql_hf = _extract_sql(out_hf)
    res_local = _execute_compare_sql(sql_local)
    res_hf = _execute_compare_sql(sql_hf)
    return out_local, res_local, out_hf, res_hf


def main() -> None:
    hub = BASE_MODEL_ID
    fine_tuned_hub = FINETUNED_HUB_MODEL_ID
    title = "Small Text-to-SQL LLM Demo"
    hero = f"""
    <div class="hero">
        <h1>{title}</h1>
        <p>
            Ask a natural-language question and compare how a small fine-tuned model performs
            against the untouched Hugging Face base model, <strong>{hub}</strong>.
        </p>
        <p>
            The fine-tuned model starts from <strong>{hub}</strong> and is trained for
            <strong>Text-to-SQL on your database</strong> with Vertex AI on Google Cloud,
            using Hugging Face PyTorch Deep Learning Containers.
        </p>
        <p>
            The app extracts each model's generated SQL, runs it against a read-only
            <strong>dummy SQLite database</strong>, and shows the query results side by side.
        </p>
        <p class="hero-meta">
            Fine-tuned model: <b>{fine_tuned_hub}</b>
            &nbsp; Training container family: <b>Hugging Face PyTorch Training DLC</b>
        </p>
    </div>
    """
    theme = gr.themes.Monochrome(
        primary_hue="violet",
        secondary_hue="cyan",
        neutral_hue="slate",
    ).set(
        body_background_fill="#07111f",
        body_text_color="#e5edf8",
        block_background_fill="#0f1b2d",
        block_border_color="#23324a",
        block_label_background_fill="#17243a",
        block_label_text_color="#c7d2fe",
        button_primary_background_fill="#7c3aed",
        button_primary_background_fill_hover="#06b6d4",
        button_primary_text_color="#ffffff",
        input_background_fill="#0b1628",
        input_border_color="#2d3f5f",
    )
    css = """
    .gradio-container {
        background:
            radial-gradient(circle at top left, rgba(124, 58, 237, 0.24), transparent 28rem),
            radial-gradient(circle at top right, rgba(6, 182, 212, 0.18), transparent 24rem),
            #07111f;
    }
    .hero {
        padding: 1.2rem 1.4rem;
        border: 1px solid #25314a;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(15, 27, 45, 0.95), rgba(30, 41, 59, 0.72));
    }
    .hero h1 {
        margin-bottom: 0.4rem;
    }
    .hero p {
        color: #dbeafe;
        font-size: 1.02rem;
        line-height: 1.55;
        margin: 0.45rem 0;
    }
    .hero code {
        color: #a5f3fc;
        background: rgba(8, 47, 73, 0.6);
        border-radius: 6px;
        padding: 0.12rem 0.3rem;
    }
    .hero-meta {
        color: #b6c7e3 !important;
        font-size: 0.92rem !important;
    }
    .db-preview {
        margin-top: 1rem;
        padding: 1.1rem 1.25rem;
        border: 1px solid #25314a;
        border-radius: 18px;
        background: rgba(11, 22, 40, 0.78);
        box-shadow: 0 18px 55px rgba(0, 0, 0, 0.22);
    }
    .db-preview .eyebrow {
        color: #67e8f9;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        margin: 0;
        text-transform: uppercase;
    }
    .db-preview h2 {
        color: #eef4ff;
        margin: 0.15rem 0 0.35rem;
    }
    .db-preview p {
        color: #cbd5e1;
        margin: 0 0 0.85rem;
    }
    .db-preview code {
        color: #a5f3fc;
        background: rgba(8, 47, 73, 0.65);
        border-radius: 6px;
        padding: 0.08rem 0.28rem;
    }
    .db-preview table {
        width: 100%;
        border-collapse: collapse;
        overflow: hidden;
        border-radius: 12px;
        font-size: 0.9rem;
    }
    .db-preview th,
    .db-preview td {
        border-bottom: 1px solid #23324a;
        padding: 0.62rem 0.7rem;
        text-align: left;
    }
    .db-preview th {
        color: #bfdbfe;
        background: rgba(30, 41, 59, 0.92);
        font-weight: 700;
    }
    .db-preview td {
        color: #e2e8f0;
        background: rgba(15, 23, 42, 0.62);
    }
    """

    with gr.Blocks(title=title, theme=theme, css=css) as demo:
        gr.Markdown(hero)
        gr.HTML(_database_preview_html())
        inp = gr.Textbox(
            label="Ask the database",
            placeholder="e.g. List all department names.",
            lines=4,
        )
        btn = gr.Button("Generate and compare SQL", variant="primary")
        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                gr.Markdown("#### Fine-tuned model from Hugging Face")
                out_local = gr.Textbox(label="Generated SQL / model output", lines=12)
                out_local_result = gr.Textbox(label="Dummy database result", lines=14)
            with gr.Column(scale=1):
                gr.Markdown("#### Hub base (Transformers)")
                out_hf = gr.Textbox(label="Generated SQL / model output", lines=12)
                out_hf_result = gr.Textbox(label="Dummy database result", lines=14)
        btn.click(
            fn=run_compare,
            inputs=[inp],
            outputs=[out_local, out_local_result, out_hf, out_hf_result],
        )

    in_space = bool(os.environ.get("SPACE_ID"))
    host = _env("QWEN_COMPARE_GRADIO_HOST", "0.0.0.0" if in_space else "127.0.0.1")
    preferred = int(_env("QWEN_COMPARE_GRADIO_PORT", os.environ.get("PORT", "7860") if in_space else "7861"))
    port = preferred if in_space else _first_free_port(host, preferred)
    if not in_space and port != preferred:
        print(f"Port {preferred} busy; using {port}.", file=sys.stderr)
    demo.launch(server_name=host, server_port=port)


if __name__ == "__main__":
    main()
