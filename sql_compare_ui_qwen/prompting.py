"""Text-to-SQL prompt shared by training and inference."""
from __future__ import annotations

import os

DEFAULT_DATABASE_SCHEMA = """
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


def _env(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    return str(v).strip()


def database_schema() -> str:
    return _env("QWEN_COMPARE_DB_SCHEMA", DEFAULT_DATABASE_SCHEMA)


def build_prompt(user_request: str) -> str:
    schema = database_schema()
    req = (user_request or "").strip()
    return (
        f"Given this database schema:\n{schema}\n\n"
        f"Write a SQL query for:\n{req}\n\nSQL:\n"
    )
