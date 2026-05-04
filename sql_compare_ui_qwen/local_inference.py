"""
Load merged Qwen SQL fine-tune from ``<repo>/.cache/<name>/`` (same layout as
``scripts/query_finetuned_qwen.py``). Basename: **QWEN_COMPARE_LOCAL_MERGED_NAME**, else
**LOCAL_QWEN_MERGED_CACHE_NAME**, else ``qwen-sql-merged``.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

_REPO_ROOT = Path(__file__).resolve().parents[1]
_state: dict = {"model": None, "tokenizer": None, "path": None}


def _apply_hf_load_env() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")


def _env(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    return str(v).strip()


def _merged_cache_basename() -> str:
    name = (
        _env("QWEN_COMPARE_LOCAL_MERGED_NAME")
        or _env("LOCAL_QWEN_MERGED_CACHE_NAME", "qwen-sql-merged")
    ).strip()
    if not name or "/" in name or "\\" in name:
        name = "qwen-sql-merged"
    return name


def default_local_merged_dir() -> Path:
    return (_REPO_ROOT / ".cache" / _merged_cache_basename()).resolve()


def _load_tokenizer(model_dir: Path) -> PreTrainedTokenizerBase:
    common: dict = {"trust_remote_code": True, "use_fast": True}
    tok_kw: dict = {}
    t = (_env("QWEN_COMPARE_HF_TOKEN") or _env("HF_TOKEN", "")).strip()
    if t:
        tok_kw["token"] = t
    try:
        return AutoTokenizer.from_pretrained(str(model_dir), **common, **tok_kw)
    except (AttributeError, TypeError) as e:
        err = str(e)
        if "'list' object has no attribute 'keys'" in err or "not a string" in err.lower():
            return AutoTokenizer.from_pretrained(
                str(model_dir), **common, extra_special_tokens={}, **tok_kw
            )
        raise


def _load_model(model_dir: Path, *, device_map: str | None, dtype: torch.dtype):
    tok_kw: dict = {}
    t = (_env("QWEN_COMPARE_HF_TOKEN") or _env("HF_TOKEN", "")).strip()
    if t:
        tok_kw["token"] = t
    kwargs: dict = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": device_map is None,
        **tok_kw,
    }
    if device_map is not None:
        kwargs["device_map"] = device_map
    try:
        return AutoModelForImageTextToText.from_pretrained(str(model_dir), **kwargs)
    except (OSError, ValueError, TypeError):
        return AutoModelForCausalLM.from_pretrained(str(model_dir), **kwargs)


def load_local_model(model_dir: Path | None = None) -> None:
    from sql_compare_ui_qwen.inference_device import log_qwen_device, pick_qwen_device_spec

    _apply_hf_load_env()
    model_dir = model_dir or default_local_merged_dir()
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(
            f"No config.json under {model_dir}. Sync merged weights (see sql_compare_ui_qwen/README.md)."
        )

    spec = pick_qwen_device_spec(env_device_map="QWEN_COMPARE_LOCAL_DEVICE_MAP")
    tok = _load_tokenizer(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = _load_model(model_dir, device_map=spec.device_map, dtype=spec.torch_dtype)
    if spec.to_device:
        model = model.to(spec.to_device)
    model.eval()
    log_qwen_device("local", model, spec)

    _state["tokenizer"] = tok
    _state["model"] = model
    _state["path"] = str(model_dir.resolve())


def unload_local_model() -> None:
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
    model_dir = model_dir or default_local_merged_dir()
    resolved = str(model_dir.resolve())
    if _state.get("model") is None or _state.get("path") != resolved:
        load_local_model(model_dir)

    tok = _state["tokenizer"]
    model = _state["model"]
    assert tok is not None and model is not None

    mnt = max_new_tokens if max_new_tokens is not None else int(_env("QWEN_COMPARE_MAX_NEW_TOKENS", "512") or "512")

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
