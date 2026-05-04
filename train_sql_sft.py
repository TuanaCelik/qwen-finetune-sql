"""
Vertex CustomJob entrypoint: SFT on ``b-mc2/sql-create-context``, optional LoRA merge.

User prompt format (must match ``sql_compare_ui/prompting.build_prompt`` and inference):

  Given this database schema:\\n{context}\\n\\nWrite a SQL query for:\\n{question}\\n\\nSQL:\\n

Then the assistant turn is the gold SQL (no extra labels).

Env: MODEL_ID, OUTPUT_DIR, HF_TOKEN, MERGE_AFTER_TRAIN; bootstrap: VERTEX_SKIP_HF_BOOTSTRAP,
VERTEX_UPGRADE_TORCH, TORCH_PIP_INDEX_URL; optional MODEL_DEVICE_MAP, GRADIENT_CHECKPOINTING,
MAX_STEPS, MAX_TRAIN_EXAMPLES (shuffle seed=42; ignored when SMOKE_MAX_EXAMPLES is set),
TRAIN_EVAL_STEPS (eval/checkpoint interval when a held-out split exists), PACKING.

Non-smoke runs: 5% train/test split (seed=42) for ``eval_loss`` and ``load_best_model_at_end``.
Smoke / laptop: **SMOKE_MAX_EXAMPLES** forces ``use_cpu=True``; eval split is skipped.
Under **torchrun** / DDP, only **global rank 0** writes ``final/`` and runs merge to ``merged/``.
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
import time

# Pip specs for Hugging Face stack — **keep aligned with pyproject.toml** `[project.dependencies]`
# (every HF-related line except google-cloud / python-dotenv).
HF_PIP_SPECS: tuple[str, ...] = (
    "transformers>=5.5.0",
    "datasets>=2.16.0",
    "peft>=0.11.0",
    "trl>=0.9.0",
    "accelerate>=1.1.0",
    "huggingface_hub>=0.23.0",
    "sentencepiece>=0.1.99",
)

# Default CUDA wheel index for PyTorch runtime upgrades (matches HF Training DLC tag `cu121...`).
DEFAULT_TORCH_PIP_INDEX_URL = "https://download.pytorch.org/whl/cu121"


def _prefer_pip_user_packages() -> None:
    """If pip installed deps into user site, prefer them over /usr/local."""
    try:
        import site

        us = site.getusersitepackages()
        for p in ([us] if isinstance(us, str) else list(us)):
            if p and p not in sys.path:
                sys.path.insert(0, p)
    except Exception:
        pass


def _dlc_torch_constraint_path() -> str | None:
    """Pin torch/torchvision/torchaudio to the DLC build so `pip install` does not pull a
    newer torch wheel (mixing torch 2.11 in one path with torchvision 2.3 in another
    breaks CUDA ops: `operator torchvision::nms does not exist`).
    """
    detect = r"""import sys
for mod, name in (
    ("torch", "TORCH"),
    ("torchvision", "TORCHVISION"),
    ("torchaudio", "TORCHAUDIO"),
):
    try:
        m = __import__(mod)
        v = m.__version__.split("+")[0].strip()
        print(name, v)
    except Exception:
        print(name, "")
"""
    try:
        out = subprocess.check_output(
            [sys.executable, "-c", detect],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=60,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None
    pins: list[str] = []
    for line in out.strip().splitlines():
        parts = line.split(None, 1)
        if len(parts) != 2 or not parts[1]:
            continue
        label, ver = parts
        if label == "TORCH":
            pins.append(f"torch=={ver}")
        elif label == "TORCHVISION":
            pins.append(f"torchvision=={ver}")
        elif label == "TORCHAUDIO":
            pins.append(f"torchaudio=={ver}")
    if not pins:
        return None
    path = "/tmp/vertex_dlc_torch_constraints.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(pins) + "\n")
    return path


def _env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _bootstrap_hf_packages() -> None:
    """Install / upgrade training dependencies inside the container.

    Do **not** use `pip install -t`: an isolated target pulls its own torch while torchvision
    stays in `/usr/local`, which breaks CUDA ops.

    - **VERTEX_SKIP_HF_BOOTSTRAP=1** — skip everything (local `uv run` / smoke tests).
    - **VERTEX_UPGRADE_TORCH=1** (default) — `pip install -U torch torchvision torchaudio` from
      **TORCH_PIP_INDEX_URL** (default **cu121** to match the HF Training DLC), then upgrade the
      HF stack (**HF_PIP_SPECS**, aligned with **pyproject.toml**).
    - **VERTEX_UPGRADE_TORCH=0** — keep the DLC’s existing torch/torchvision/torchaudio versions and
      only upgrade HF libs using **`-c`** pins (legacy; Transformers 5.x usually needs torch ≥ 2.4).

    Env:
        TORCH_PIP_INDEX_URL — PyTorch wheel index (default: DEFAULT_TORCH_PIP_INDEX_URL).
    """
    if os.environ.get("VERTEX_SKIP_HF_BOOTSTRAP") == "1":
        return

    pip_base = [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "--no-cache-dir"]

    if _env_flag("VERTEX_UPGRADE_TORCH", True):
        index = (
            os.environ.get("TORCH_PIP_INDEX_URL", "").strip() or DEFAULT_TORCH_PIP_INDEX_URL
        )
        print(f"[train_sql_sft] Bootstrapping: upgrading torch stack from {index}", flush=True)
        subprocess.check_call(
            pip_base
            + [
                "--index-url",
                index,
                "torch",
                "torchvision",
                "torchaudio",
            ]
        )
        print("[train_sql_sft] Bootstrapping: upgrading HF stack (see HF_PIP_SPECS / pyproject.toml)", flush=True)
        subprocess.check_call(pip_base + list(HF_PIP_SPECS))
        return

    print(
        "[train_sql_sft] VERTEX_UPGRADE_TORCH=0: upgrading HF libs only (pinned to existing torch). "
        "Gemma 4 + Transformers 5.x typically requires torch >= 2.4.",
        flush=True,
    )
    cmd = pip_base.copy()
    cf = _dlc_torch_constraint_path()
    if cf:
        cmd.extend(["-c", cf])
    cmd.extend(HF_PIP_SPECS)
    subprocess.check_call(cmd)


def _bootstrap_coordination_marker() -> str:
    return "/tmp/vertex_train_sql_sft_bootstrap_done"


def _run_bootstrap_single_writer() -> None:
    """``torchrun`` imports this module in every process; concurrent ``pip install`` on one VM
    corrupts shared ``site-packages`` (races deleting torch/triton files). Only **LOCAL_RANK 0**
    runs bootstrap; other ranks block until the marker file appears.
    """
    if os.environ.get("VERTEX_SKIP_HF_BOOTSTRAP") == "1":
        return
    world = int(os.environ.get("WORLD_SIZE", "1") or "1")
    local_rank = int(os.environ.get("LOCAL_RANK", "0") or "0")
    marker = _bootstrap_coordination_marker()

    if world <= 1:
        _bootstrap_hf_packages()
        return

    if local_rank == 0:
        try:
            if os.path.isfile(marker):
                os.remove(marker)
        except OSError:
            pass
        _bootstrap_hf_packages()
        with open(marker, "w", encoding="utf-8") as f:
            f.write("ok")
    else:
        deadline = time.time() + 7200.0
        while not os.path.isfile(marker):
            if time.time() > deadline:
                raise RuntimeError(
                    "[train_sql_sft] Timed out waiting for LOCAL_RANK 0 to finish pip bootstrap."
                )
            time.sleep(1.0)


_run_bootstrap_single_writer()
_prefer_pip_user_packages()

import gc

from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v is not None and v.strip() != "" else default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v is not None and v.strip() != "" else default


def _load_tokenizer(model_id: str) -> AutoTokenizer:
    """Load tokenizer for Gemma (incl. Gemma 4 E2B on Hub).

    - Newer Hub snapshots may ship only ``tokenizer.json``; the slow SentencePiece
      path then fails (no ``tokenizer.model`` / ``vocab_file`` not a string) — use fast.
    - If ``tokenizer_config.json`` sets ``extra_special_tokens`` as a *list* (Gemma 4),
      some Transformers builds expect a *dict* and raise ``'list' object has no attribute 'keys'``;
      retry with an empty dict override (text SFT does not need those named attributes).
    """
    common: dict = {"trust_remote_code": True, "use_fast": True}
    try:
        return AutoTokenizer.from_pretrained(model_id, **common)
    except (AttributeError, TypeError) as e:
        err = str(e)
        if "'list' object has no attribute 'keys'" in err or "not a string" in err.lower():
            return AutoTokenizer.from_pretrained(
                model_id, **common, extra_special_tokens={}
            )
        raise


def _model_init_kwargs_for_trl() -> dict:
    """TRL sets ``device_map='auto'`` when omitted; on CPU that can leave weights on **meta**, then
    ``Trainer`` fails with ``Cannot copy out of meta tensor``. Force ``device_map=None`` when no CUDA,
    unless ``MODEL_DEVICE_MAP`` overrides (use ``none`` / ``null`` for explicit ``None``).
    """
    import torch

    out: dict = {"trust_remote_code": True}
    raw = os.environ.get("MODEL_DEVICE_MAP", "").strip()
    if raw:
        if raw.lower() in ("none", "null"):
            out["device_map"] = None
        else:
            out["device_map"] = raw
    elif not torch.cuda.is_available():
        out["device_map"] = None
    return out


def main() -> None:
    model_id = os.environ["MODEL_ID"]
    output_dir = os.environ["OUTPUT_DIR"].rstrip("/")
    merge_after = _env_bool("MERGE_AFTER_TRAIN", True)
    smoke_mode = bool(os.environ.get("SMOKE_MAX_EXAMPLES", "").strip())

    num_epochs = _env_float("NUM_TRAIN_EPOCHS", 2.0)
    per_device_bs = _env_int("PER_DEVICE_TRAIN_BATCH_SIZE", 2)
    grad_accum = _env_int("GRADIENT_ACCUMULATION_STEPS", 16)
    lr = _env_float("LEARNING_RATE", 5e-5)
    max_seq_len = _env_int("MAX_SEQ_LENGTH", 1024)
    train_eval_steps = _env_int("TRAIN_EVAL_STEPS", 50)
    # Local CPU smoke: set BF16=false, MAX_STEPS=2, SMOKE_MAX_EXAMPLES=32, MERGE_AFTER_TRAIN=false
    use_bf16 = _env_bool("BF16", True)
    max_steps_env = os.environ.get("MAX_STEPS", "").strip()
    max_steps = int(max_steps_env) if max_steps_env else -1
    lora_r = _env_int("LORA_R", 16)
    lora_alpha = _env_int("LORA_ALPHA", 32)
    lora_dropout = _env_float("LORA_DROPOUT", 0.05)
    # Recompute trade-off: slower steps, much lower activation memory (critical on L4 + long context).
    grad_ckpt_default = not smoke_mode
    gradient_checkpointing = _env_bool("GRADIENT_CHECKPOINTING", grad_ckpt_default)

    ds = load_dataset("b-mc2/sql-create-context", split="train")
    full_train_rows = len(ds)
    smoke_n = os.environ.get("SMOKE_MAX_EXAMPLES", "").strip()
    if smoke_n:
        take = min(int(smoke_n), len(ds))
        ds = ds.select(range(take))
        print(f"[train_sql_sft] SMOKE_MAX_EXAMPLES={take} (subset for local testing)", flush=True)
    else:
        mte = os.environ.get("MAX_TRAIN_EXAMPLES", "").strip()
        if mte:
            take = min(int(mte), len(ds))
            ds = ds.shuffle(seed=42).select(range(take))
            print(
                f"[train_sql_sft] MAX_TRAIN_EXAMPLES={take} of {full_train_rows} (shuffle seed=42, then head)",
                flush=True,
            )

    import transformers as transformers_mod

    print(
        f"[train_sql_sft] transformers {transformers_mod.__version__} from {transformers_mod.__file__}",
        flush=True,
    )

    tok = _load_tokenizer(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def row_to_messages(ex):
        user_content = (
            f"Given this database schema:\n{ex['context']}\n\n"
            f"Write a SQL query for:\n{ex['question']}\n\nSQL:\n"
        )
        return {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": (ex["answer"] or "").strip()},
            ]
        }

    def messages_to_text(batch: dict) -> dict:
        texts = []
        for messages in batch["messages"]:
            texts.append(
                tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
        return {"text": texts}

    use_eval_split = (not smoke_mode) and len(ds) >= 10
    if use_eval_split:
        n_test = max(1, min(len(ds) - 1, math.ceil(len(ds) * 0.05)))
        split = ds.train_test_split(test_size=n_test, seed=42)
        train_raw, eval_raw = split["train"], split["test"]
        print(
            f"[train_sql_sft] train/eval split: train={len(train_raw)} eval={len(eval_raw)} (test_size={n_test})",
            flush=True,
        )
    else:
        train_raw, eval_raw = ds, None
        if smoke_mode:
            print("[train_sql_sft] smoke: no eval split", flush=True)

    cols = train_raw.column_names
    train_raw = train_raw.map(row_to_messages, remove_columns=cols)
    train_ds = train_raw.map(messages_to_text, batched=True, remove_columns=["messages"])
    eval_ds = None
    if eval_raw is not None:
        ec = eval_raw.column_names
        eval_raw = eval_raw.map(row_to_messages, remove_columns=ec)
        eval_ds = eval_raw.map(messages_to_text, batched=True, remove_columns=["messages"])

    print(f"[train_sql_sft] prepared rows train={len(train_ds)} eval={len(eval_ds or [])}", flush=True)

    # Local smoke (`SMOKE_MAX_EXAMPLES`): force plain CPU training. On Apple Silicon the Trainer
    # otherwise picks **MPS**, which rejects huge single-buffer copies for models like Gemma 4
    # (`RuntimeError: Invalid buffer size: … GiB`). Vertex jobs do not set this env.
    train_on_cpu = smoke_mode or _env_bool("TRAINING_USE_CPU", False)
    packing_enabled = False if smoke_mode else _env_bool("PACKING", False)

    world_size = max(int(os.environ.get("WORLD_SIZE", "1") or "1"), 1)
    distributed = world_size > 1

    cfg_kwargs: dict = dict(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        bf16=use_bf16,
        logging_steps=20,
        max_length=max_seq_len,
        packing=packing_enabled,
        dataset_text_field="text",
        report_to="none",
        model_init_kwargs=_model_init_kwargs_for_trl(),
        gradient_checkpointing=gradient_checkpointing,
    )
    if train_on_cpu:
        cfg_kwargs["use_cpu"] = True
    if max_steps > 0:
        cfg_kwargs["max_steps"] = max_steps

    if eval_ds is not None and len(eval_ds) > 0:
        es = train_eval_steps
        if max_steps > 0:
            es = max(1, min(es, max_steps))
        cfg_kwargs.update(
            eval_strategy="steps",
            eval_steps=es,
            save_strategy="steps",
            save_steps=es,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
    else:
        cfg_kwargs["save_strategy"] = "epoch"
    # LoRA + DDP: default ``find_unused_parameters=True`` often breaks or stalls training.
    if distributed:
        cfg_kwargs["ddp_find_unused_parameters"] = False
    # Reentrant checkpointing + DDP frequently triggers incorrect backward / sync errors.
    if distributed and gradient_checkpointing:
        cfg_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    trainer = SFTTrainer(
        model=model_id,
        processing_class=tok,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=SFTConfig(**cfg_kwargs),
        peft_config=LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules="all-linear",
        ),
    )
    trainer.train()
    trainer.accelerator.wait_for_everyone()

    final_dir = f"{output_dir}/final"
    if trainer.is_world_process_zero():
        model_to_save = trainer.accelerator.unwrap_model(trainer.model)
        model_to_save.save_pretrained(final_dir)
        tok.save_pretrained(final_dir)

        if merge_after:
            del trainer
            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            merge_dtype = torch.bfloat16 if use_bf16 else torch.float32
            base = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=merge_dtype,
                trust_remote_code=True,
            )
            merged = PeftModel.from_pretrained(base, final_dir).merge_and_unload()
            merged_dir = f"{output_dir}/merged"
            merged.save_pretrained(merged_dir, safe_serialization=True)
            tok.save_pretrained(merged_dir)


if __name__ == "__main__":
    main()
