"""
Smolagents CodeAgent: Hub model plans Python that calls ``text2sql`` (local FT) and ``run_sql`` (SQLite).

Env: ``HF_TOKEN``, ``MODEL_ID`` (Hub brain); optional ``SQL_AGENT_HUB_MODEL_ID``, ``SQL_AGENT_DB_PATH``,
``SQL_AGENT_MAX_STEPS``, ``SQL_AGENT_MAX_NEW_TOKENS``.
"""
from __future__ import annotations

import io
import logging
import os
import re
import sqlite3
import warnings
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _env(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    return str(v).strip()


def default_sqlite_path() -> Path:
    raw = _env("SQL_AGENT_DB_PATH", str(_REPO_ROOT / "data" / "spider_eval_synthetic" / "synthetic.db"))
    return Path(raw).expanduser().resolve()


def _validate_select_only(sql: str) -> tuple[bool, str]:
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
    banned = (
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
    )
    for b in banned:
        if re.search(rf"\b{b}\b", low):
            return False, f"forbidden keyword: {b}"
    return True, one


def _format_rows(cols: list[str], rows: list[tuple[Any, ...]], *, limit: int) -> str:
    if not cols:
        return "(no columns)"
    buf = io.StringIO()
    buf.write(" | ".join(cols) + "\n")
    buf.write("-" * min(120, 8 * len(cols)) + "\n")
    for i, row in enumerate(rows[:limit]):
        buf.write(" | ".join(str(x) if x is not None else "NULL" for x in row) + "\n")
    if len(rows) > limit:
        buf.write(f"\n… truncated to {limit} rows ({len(rows)} returned)\n")
    return buf.getvalue()


def build_company_instructions(schema_ddl: str) -> str:
    return f"""You are an internal assistant that helps non-technical colleagues explore a read-only company database.

The database schema (tables and columns you may reference) is:

{schema_ddl}

Rules:
- Always use the provided tools. Do not invent table or column names that are not in the schema.
- Use ``text2sql`` with the user's question in plain English to obtain a **single** SQL query (no markdown fences in the tool output).
- Use ``run_sql`` with that SQL to fetch real results from the SQLite database. If the tool reports an error, revise the SQL and try again.
- Prefer simple, correct SQL over clever SQL. Use clear column aliases when helpful.
- Summarize results in plain language for a business reader after you have executed the query.
- Never ask the user to run SQL themselves; you run it via tools.
"""


def make_sql_tools(db_path: Path) -> list[Any]:
    """Two smolagents tools: local FT text2sql + SQLite SELECT executor."""
    from smolagents import Tool

    db = db_path.resolve()

    class Text2SQLTool(Tool):
        name = "text2sql"
        description = (
            "Converts a plain-English question about the company database into a single SQL statement. "
            "Uses the internally fine-tuned Gemma model (same prompt layout as training: Schema + Question). "
            "Returns raw SQL text only—no markdown."
        )
        inputs = {
            "natural_language_question": {
                "type": "string",
                "description": "What the user wants to know, in everyday language.",
            }
        }
        output_type = "string"

        def forward(self, natural_language_question: str) -> str:
            from sql_compare_ui import local_inference
            from sql_compare_ui.prompting import build_prompt

            q = (natural_language_question or "").strip()
            if not q:
                return "Error: empty question."
            prompt = build_prompt(q)
            try:
                return local_inference.generate_local(prompt).strip()
            except Exception as e:
                return f"Error from text2sql model: {e!r}"

    class RunSQLTool(Tool):
        name = "run_sql"
        description = (
            "Runs a **single** read-only SQL query against the internal SQLite database and returns a "
            "text table of results (or an error message). Only SELECT / WITH…SELECT is allowed."
        )
        inputs = {
            "sql": {
                "type": "string",
                "description": "One SELECT (or WITH … SELECT) statement.",
            }
        }
        output_type = "string"

        def forward(self, sql: str) -> str:
            ok, msg = _validate_select_only(sql)
            if not ok:
                return f"Error: {msg}"
            if not db.is_file():
                return f"Error: database file not found: {db}"
            try:
                conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
                conn.row_factory = sqlite3.Row
            except sqlite3.Error as e:
                return f"Error opening database: {e!r}"
            try:
                cur = conn.cursor()
                cur.execute(msg)
                rows = [tuple(r) for r in cur.fetchall()]
                cols = [d[0] for d in cur.description] if cur.description else []
                return _format_rows(list(cols), rows, limit=150)
            except sqlite3.Error as e:
                return f"Error executing SQL: {e!r}"
            finally:
                conn.close()

    return [Text2SQLTool(), RunSQLTool()]


_agent: Any = None
_agent_schema: str | None = None

_hub_transformers_model_cls: type | None = None


def _hub_transformers_model_cls_get() -> type:
    """Smolagents ``TransformersModel`` always passes ``torch_dtype=`` into ``from_pretrained``; Transformers 5.x deprecates that. Load with ``dtype`` only (same behavior, quieter logs)."""
    global _hub_transformers_model_cls
    if _hub_transformers_model_cls is not None:
        return _hub_transformers_model_cls

    from smolagents.models import Model, TransformersModel

    _log = logging.getLogger("smolagents.models")

    class HubTransformersModel(TransformersModel):
        def __init__(
            self,
            model_id: str | None = None,
            device_map: str | None = None,
            torch_dtype: Any = None,
            trust_remote_code: bool = False,
            model_kwargs: dict[str, Any] | None = None,
            max_new_tokens: int = 4096,
            max_tokens: int | None = None,
            **kwargs: Any,
        ):
            try:
                import torch
                from transformers import (
                    AutoModelForCausalLM,
                    AutoModelForImageTextToText,
                    AutoProcessor,
                    AutoTokenizer,
                    TextIteratorStreamer,
                )
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    "Please install 'transformers' extra to use 'TransformersModel': `pip install 'smolagents[transformers]'`"
                ) from e

            if not model_id:
                warnings.warn(
                    "The 'model_id' parameter will be required in version 2.0.0. "
                    "Please update your code to pass this parameter to avoid future errors. "
                    "For now, it defaults to 'HuggingFaceTB/SmolLM2-1.7B-Instruct'.",
                    FutureWarning,
                    stacklevel=2,
                )
                model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

            max_new_tokens = max_tokens if max_tokens is not None else max_new_tokens

            if device_map is None:
                device_map = "cuda" if torch.cuda.is_available() else "cpu"
            _log.info("Using device: %s", device_map)

            from sql_compare_ui.local_inference import _apply_hf_load_env

            _apply_hf_load_env()

            self._is_vlm = False
            mk = dict(model_kwargs or {})
            if "dtype" not in mk:
                if torch_dtype is not None:
                    mk["dtype"] = torch_dtype
                else:
                    mk["dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            self.model_kwargs = mk
            hub_token = mk.get("token")
            load_kw: dict[str, Any] = {
                "device_map": device_map,
                "trust_remote_code": trust_remote_code,
                **mk,
            }

            try:
                self.model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kw)
                proc_kw: dict[str, Any] = {"trust_remote_code": trust_remote_code}
                if hub_token is not None:
                    proc_kw["token"] = hub_token
                self.processor = AutoProcessor.from_pretrained(model_id, **proc_kw)
                self._is_vlm = True
                self.streamer = TextIteratorStreamer(
                    self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True
                )  # type: ignore[arg-type]

            except ValueError as e:
                if "Unrecognized configuration class" not in str(e):
                    raise
                self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kw)
                tok_kw: dict[str, Any] = {"trust_remote_code": trust_remote_code}
                if hub_token is not None:
                    tok_kw["token"] = hub_token
                self.tokenizer = AutoTokenizer.from_pretrained(model_id, **tok_kw)
                self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

            except Exception as e:
                raise ValueError(f"Failed to load tokenizer and model for {model_id=}: {e}") from e

            Model.__init__(
                self,
                flatten_messages_as_text=not self._is_vlm,
                model_id=model_id,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )

    _hub_transformers_model_cls = HubTransformersModel
    return HubTransformersModel


def get_code_agent(schema_ddl: str) -> Any:
    global _agent, _agent_schema
    if _agent is not None and _agent_schema == schema_ddl:
        return _agent
    reset_agent()

    from smolagents import CodeAgent

    from sql_compare_ui.local_inference import _apply_hf_load_env

    _apply_hf_load_env()

    token = os.environ["HF_TOKEN"].strip()
    hub_id = _env("SQL_AGENT_HUB_MODEL_ID") or os.environ.get("MODEL_ID", "google/gemma-4-E2B-it").strip()
    db_path = default_sqlite_path()

    import torch

    dm = "auto" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    HubTM = _hub_transformers_model_cls_get()
    model = HubTM(
        model_id=hub_id,
        trust_remote_code=True,
        max_new_tokens=int(_env("SQL_AGENT_MAX_NEW_TOKENS", "512")),
        device_map=dm,
        model_kwargs={"token": token, "dtype": dtype},
    )

    tools = make_sql_tools(db_path)
    max_steps = int(_env("SQL_AGENT_MAX_STEPS", "12"))
    instructions = build_company_instructions(schema_ddl)

    _agent = CodeAgent(
        tools=tools,
        model=model,
        instructions=instructions,
        max_steps=max_steps,
        additional_authorized_imports=[],
        stream_outputs=False,
    )
    _agent_schema = schema_ddl
    return _agent


def reset_agent() -> None:
    """Drop cached agent (e.g. after changing env)."""
    global _agent, _agent_schema
    if _agent is not None:
        try:
            _agent.cleanup()
        except Exception:
            pass
    _agent = None
    _agent_schema = None


def run_agent_task(user_message: str, schema_ddl: str) -> str:
    agent = get_code_agent(schema_ddl)
    try:
        return str(agent.run(user_message.strip())).strip()
    except Exception as e:
        return f"Agent error: {e!r}"
