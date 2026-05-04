"""
Load merged fine-tuned Gemma 4 (E2B) from disk with Transformers — same architecture as train_sql_sft merge.

Weights live under ``<repo>/.cache/<LOCAL_FT_MERGED_CACHE_NAME>/`` (see ``default_local_merged_dir``);
sync from GCS with ``scripts/query_finetuned_gemma.py``.
"""
from __future__ import annotations

import gc
import os
import shutil
import subprocess
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[1]
_state: dict = {"model": None, "tokenizer": None, "path": None}  # path: resolved str or None


def _apply_hf_load_env() -> None:
    """Avoid macOS ``resource_tracker`` leaked-semaphore noise during large loads.

    - ``TOKENIZERS_PARALLELISM=false``: fast tokenizers must not fork worker processes.
    - ``HF_DEACTIVATE_ASYNC_LOAD=1``: Transformers skips threaded async safetensors load
      (see ``transformers/core_model_loading.py``); teardown of that pool often triggers the warning.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"


def _env(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    return str(v).strip()


def gcs_merged_uri() -> str:
    bucket = _env("GCS_BUCKET").replace("gs://", "").strip("/")
    prefix = _env("OUTPUT_GCS_PREFIX", "gemma-sql/gemma-4-e2b-run-1").strip("/")
    if not bucket:
        raise ValueError("GCS_BUCKET is required to build gs:// merged URI.")
    return f"gs://{bucket}/{prefix}/merged"


def _merged_cache_basename() -> str:
    """Single directory name under ``<repo>/.cache/`` (no path separators)."""
    name = _env("LOCAL_FT_MERGED_CACHE_NAME", "").strip()
    if not name:
        legacy = _env("LOCAL_FT_MERGED_DIR", "").strip()
        if legacy:
            name = Path(legacy.replace("\\", "/")).name
    if not name:
        name = "gemma-sql-merged"
    if name in (".", "..") or "/" in name or "\\" in name:
        name = "gemma-sql-merged"
    return name


def default_local_merged_dir() -> Path:
    """Always ``<repo>/.cache/<name>/`` so cwd and ``sql_compare_ui/`` vs repo root do not matter."""
    return (_REPO_ROOT / ".cache" / _merged_cache_basename()).resolve()


def sync_merged_from_gcs(
    local_dir: Path | None = None,
    *,
    gcs_uri: str | None = None,
) -> Path:
    """Copy gs://.../merged into local_dir using ``gcloud storage cp`` (same idea as HF Vertex docs)."""
    local_dir = local_dir or default_local_merged_dir()
    uri = gcs_uri or gcs_merged_uri()
    if shutil.which("gcloud") is None:
        raise RuntimeError("gcloud CLI not found. Install Google Cloud SDK and run gcloud auth login.")

    local_dir.parent.mkdir(parents=True, exist_ok=True)
    if local_dir.exists():
        shutil.rmtree(local_dir)
    local_dir.mkdir(parents=True)

    # Match HF Vertex doc pattern: mirror gs://.../merged into a local folder for from_pretrained().
    subprocess.run(
        ["gcloud", "storage", "rsync", "--recursive", uri.rstrip("/"), str(local_dir)],
        check=True,
        cwd=str(_REPO_ROOT),
    )
    return local_dir


def _load_tokenizer(model_dir: Path) -> AutoTokenizer:
    common: dict = {"trust_remote_code": True, "use_fast": True}
    try:
        return AutoTokenizer.from_pretrained(str(model_dir), **common)
    except (AttributeError, TypeError) as e:
        err = str(e)
        if "'list' object has no attribute 'keys'" in err or "not a string" in err.lower():
            return AutoTokenizer.from_pretrained(str(model_dir), **common, extra_special_tokens={})
        raise


def load_local_model(model_dir: Path | None = None, *, device_map: str | None = None) -> None:
    """Load tokenizer + AutoModelForImageTextToText into module-level cache."""
    _apply_hf_load_env()
    model_dir = model_dir or default_local_merged_dir()
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(
            f"No config.json under {model_dir}. Run: uv run python scripts/query_finetuned_gemma.py --sync"
        )

    dm = device_map
    if dm is None:
        raw = _env("LOCAL_FT_DEVICE_MAP", "").strip().lower()
        if raw in ("none", "null"):
            dm = None
        elif raw:
            dm = raw
        else:
            dm = "auto" if torch.cuda.is_available() else None

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tok = _load_tokenizer(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs: dict = {
        "trust_remote_code": True,
        "dtype": dtype,
        "low_cpu_mem_usage": dm is None,
    }
    if dm is not None:
        kwargs["device_map"] = dm

    model = AutoModelForImageTextToText.from_pretrained(str(model_dir), **kwargs)
    model.eval()

    _state["tokenizer"] = tok
    _state["model"] = model
    _state["path"] = str(model_dir.resolve())


def unload_local_model() -> None:
    """Drop cached local model/tokenizer so only one Gemma-sized checkpoint stays resident when alternating with Hub."""
    model = _state.get("model")
    tok = _state.get("tokenizer")
    _state["model"] = None
    _state["tokenizer"] = None
    _state["path"] = None
    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def generate_local(
    prompt: str,
    *,
    model_dir: Path | None = None,
    max_new_tokens: int | None = None,
) -> str:
    """Run greedy generation for a single user-message style prompt (plain text, same as Gradio build_prompt)."""
    model_dir = model_dir or default_local_merged_dir()
    resolved = str(model_dir.resolve())
    if _state.get("model") is None or _state.get("path") != resolved:
        load_local_model(model_dir)

    tok = _state["tokenizer"]
    model = _state["model"]
    assert tok is not None and model is not None

    mnt = max_new_tokens if max_new_tokens is not None else int(_env("MAX_NEW_TOKENS", "512") or "512")

    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tok(text, return_tensors="pt")
    dev = next(model.parameters()).device
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=mnt,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )

    in_len = inputs["input_ids"].shape[-1]
    gen_ids = out[0, in_len:]
    return tok.decode(gen_ids, skip_special_tokens=True).strip()
