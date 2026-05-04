#!/usr/bin/env python3
"""
Sync merged **Qwen3.5** SQL SFT weights from GCS and run greedy generation locally (Transformers).

Uses the same **user** message body as ``train_qwen_sql_sft.py`` / ``sql_compare_ui.prompting.build_prompt``,
then ``apply_chat_template`` on ``[user, assistant]`` with ``add_generation_prompt=True`` (same idea as
Gemma local inference).

Paths (override in ``.env`` in a later iteration):

- **GCS_BUCKET** + **QWEN_OUTPUT_GCS_PREFIX** → ``gs://…/<prefix>/merged``
- Optional **QWEN_GCS_MERGED_URI** overrides that URI
- **LOCAL_QWEN_MERGED_CACHE_NAME** → ``<repo>/.cache/<name>/``

Examples (repository root):

  uv run python scripts/query_finetuned_qwen.py --sync
  uv run python scripts/query_finetuned_qwen.py --prompt "List all department names."
"""
from __future__ import annotations

import argparse
import gc
import os
import shutil

import torch
import subprocess
import sys
from pathlib import Path

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


def gcs_merged_uri() -> str:
    override = _env("QWEN_GCS_MERGED_URI")
    if override:
        return override.rstrip("/")
    bucket = _env("GCS_BUCKET").replace("gs://", "").strip("/")
    prefix = _env("QWEN_OUTPUT_GCS_PREFIX", "qwen-sql/qwen3.5-0.8b-run-1").strip("/")
    if not bucket:
        raise ValueError("Set GCS_BUCKET or QWEN_GCS_MERGED_URI.")
    return f"gs://{bucket}/{prefix}/merged"


def default_local_merged_dir() -> Path:
    name = _env("LOCAL_QWEN_MERGED_CACHE_NAME", "qwen-sql-merged")
    if not name or "/" in name or "\\" in name:
        name = "qwen-sql-merged"
    return (ROOT / ".cache" / name).resolve()


def sync_merged_from_gcs(local_dir: Path, *, gcs_uri: str | None = None) -> Path:
    uri = (gcs_uri or gcs_merged_uri()).rstrip("/")
    if shutil.which("gcloud") is None:
        raise RuntimeError("gcloud CLI not found. Install Google Cloud SDK and authenticate.")
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    if local_dir.exists():
        shutil.rmtree(local_dir)
    local_dir.mkdir(parents=True)
    subprocess.run(
        ["gcloud", "storage", "rsync", "--recursive", uri, str(local_dir)],
        check=True,
        cwd=str(ROOT),
    )
    return local_dir


def _apply_hf_load_env() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")


def _load_tokenizer(model_dir: Path):
    from transformers import AutoTokenizer

    common: dict = {"trust_remote_code": True, "use_fast": True}
    try:
        return AutoTokenizer.from_pretrained(str(model_dir), **common)
    except (AttributeError, TypeError) as e:
        err = str(e)
        if "'list' object has no attribute 'keys'" in err or "not a string" in err.lower():
            return AutoTokenizer.from_pretrained(str(model_dir), **common, extra_special_tokens={})
        raise


def _load_model(model_dir: Path, *, device_map: str | None, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    kwargs: dict = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": device_map is None,
    }
    if device_map is not None:
        kwargs["device_map"] = device_map
    try:
        return AutoModelForImageTextToText.from_pretrained(str(model_dir), **kwargs)
    except (OSError, ValueError, TypeError):
        return AutoModelForCausalLM.from_pretrained(str(model_dir), **kwargs)


def generate_local(
    user_body: str,
    *,
    model_dir: Path,
    max_new_tokens: int | None = None,
) -> str:
    import torch

    _apply_hf_load_env()

    from sql_compare_ui_qwen.inference_device import log_qwen_device, pick_qwen_device_spec

    spec = pick_qwen_device_spec(env_device_map="LOCAL_QWEN_DEVICE_MAP")

    tok = _load_tokenizer(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = _load_model(model_dir, device_map=spec.device_map, dtype=spec.torch_dtype)
    if spec.to_device:
        model = model.to(spec.to_device)
    model.eval()
    log_qwen_device("query_finetuned_qwen", model, spec)

    messages = [{"role": "user", "content": user_body}]
    text = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tok(text, return_tensors="pt")
    dev = next(model.parameters()).device
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    mnt = max_new_tokens if max_new_tokens is not None else int(_env("MAX_NEW_TOKENS", "512") or "512")

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
    decoded = tok.decode(gen_ids, skip_special_tokens=True).strip()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    return decoded


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--sync",
        action="store_true",
        help="rsync gs://…/merged into the local Qwen cache directory.",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Natural-language question (same template as sql_compare_ui / train_qwen_sql_sft).",
    )
    p.add_argument("--max-new-tokens", type=int, default=None, help="Override MAX_NEW_TOKENS (default 512).")
    args = p.parse_args(argv)

    try:
        gcs = gcs_merged_uri()
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1

    local_dir = default_local_merged_dir()
    print(f"GCS merged: {gcs}")
    print(f"Local dir:  {local_dir}")

    if args.sync:
        print("Syncing from GCS …")
        sync_merged_from_gcs(local_dir, gcs_uri=gcs)
        print("Sync finished.")

    if not args.prompt.strip():
        if not args.sync:
            p.error("Provide --prompt and/or --sync.")
        return 0

    if not (local_dir / "config.json").is_file():
        print(f"No checkpoint at {local_dir}. Run with --sync first.", file=sys.stderr)
        return 1

    from sql_compare_ui.prompting import build_prompt

    user_body = build_prompt(args.prompt.strip())
    print("Generating …")
    text = generate_local(user_body, model_dir=local_dir, max_new_tokens=args.max_new_tokens)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
