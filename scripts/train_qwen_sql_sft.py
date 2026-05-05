"""Vertex CustomJob entrypoint for Qwen SQL SFT."""
from __future__ import annotations

import math
import os
import subprocess
import sys
import time

HF_PIP_SPECS: tuple[str, ...] = (
    "transformers>=5.5.0",
    "datasets>=2.16.0",
    "peft>=0.11.0",
    "trl>=0.9.0",
    "accelerate>=1.1.0",
    "huggingface_hub>=0.23.0",
    "sentencepiece>=0.1.99",
)

DEFAULT_TORCH_PIP_INDEX_URL = "https://download.pytorch.org/whl/cu121"
VERTEX_UPGRADE_TORCH = True

NUM_TRAIN_EPOCHS = 4.0
PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 5e-5
MAX_SEQ_LENGTH = 768
TRAIN_EVAL_STEPS = 50
LR_SCHEDULER_TYPE = "cosine"
WARMUP_RATIO = 0.03
BF16 = True
MAX_STEPS: int | None = None
SMOKE_MAX_STEPS = 2
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
GRADIENT_CHECKPOINTING = True
MERGE_AFTER_TRAIN = True
PACKING = False
MAX_TRAIN_EXAMPLES: int | None = None
DEFAULT_DATASET_ID = "b-mc2/sql-create-context"
DEFAULT_DATASET_SPLIT = "train"


def _prefer_pip_user_packages() -> None:
    try:
        import site

        us = site.getusersitepackages()
        for p in ([us] if isinstance(us, str) else list(us)):
            if p and p not in sys.path:
                sys.path.insert(0, p)
    except Exception:
        pass


def _dlc_torch_constraint_path() -> str | None:
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


def _bootstrap_hf_packages() -> None:
    if os.environ.get("VERTEX_SKIP_HF_BOOTSTRAP", "").strip().lower() == "true":
        return

    pip_base = [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "--no-cache-dir"]

    if VERTEX_UPGRADE_TORCH:
        index = (
            os.environ.get("TORCH_PIP_INDEX_URL", "").strip() or DEFAULT_TORCH_PIP_INDEX_URL
        )
        print(f"[train_qwen_sql_sft] Bootstrapping: upgrading torch stack from {index}", flush=True)
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
        print("[train_qwen_sql_sft] Bootstrapping: upgrading HF stack (see HF_PIP_SPECS / pyproject.toml)", flush=True)
        subprocess.check_call(pip_base + list(HF_PIP_SPECS))
        return

    print(
        "[train_qwen_sql_sft] Upgrading HF libs only (pinned to existing torch). "
        "Transformers 5.x typically requires torch >= 2.4.",
        flush=True,
    )
    cmd = pip_base.copy()
    cf = _dlc_torch_constraint_path()
    if cf:
        cmd.extend(["-c", cf])
    cmd.extend(HF_PIP_SPECS)
    subprocess.check_call(cmd)


def _bootstrap_coordination_marker() -> str:
    return "/tmp/vertex_train_qwen_sql_sft_bootstrap_done"


def _run_bootstrap_single_writer() -> None:
    if os.environ.get("VERTEX_SKIP_HF_BOOTSTRAP", "").strip().lower() == "true":
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
                    "[train_qwen_sql_sft] Timed out waiting for LOCAL_RANK 0 to finish pip bootstrap."
                )
            time.sleep(1.0)


_run_bootstrap_single_writer()
_prefer_pip_user_packages()

import gc

from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def _load_tokenizer(model_id: str) -> AutoTokenizer:
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
    model_id = os.environ["QWEN_MODEL_ID"]
    output_dir = os.environ["QWEN_OUTPUT_DIR"].rstrip("/")
    smoke_mode = bool(os.environ.get("SMOKE_MAX_EXAMPLES", "").strip())

    max_steps = SMOKE_MAX_STEPS if smoke_mode else (MAX_STEPS if MAX_STEPS is not None else -1)
    max_seq_len = 128 if smoke_mode else MAX_SEQ_LENGTH
    per_device_bs = 1 if smoke_mode else PER_DEVICE_TRAIN_BATCH_SIZE
    grad_accum = 1 if smoke_mode else GRADIENT_ACCUMULATION_STEPS
    use_bf16 = False if smoke_mode else BF16
    gradient_checkpointing = False if smoke_mode else GRADIENT_CHECKPOINTING
    merge_after_train = False if smoke_mode else MERGE_AFTER_TRAIN

    dataset_id = (os.environ.get("QWEN_DATASET_ID") or DEFAULT_DATASET_ID).strip()
    dataset_split = (os.environ.get("QWEN_DATASET_SPLIT") or DEFAULT_DATASET_SPLIT).strip()
    print(f"[train_qwen_sql_sft] loading dataset {dataset_id!r} split={dataset_split!r}", flush=True)
    ds = load_dataset(dataset_id, split=dataset_split)
    required_columns = {"context", "question", "answer"}
    missing_columns = sorted(required_columns.difference(ds.column_names))
    if missing_columns:
        raise ValueError(
            f"Dataset {dataset_id!r} split {dataset_split!r} is missing columns: "
            f"{', '.join(missing_columns)}"
        )
    full_train_rows = len(ds)
    smoke_n = os.environ.get("SMOKE_MAX_EXAMPLES", "").strip()
    if smoke_n:
        take = min(int(smoke_n), len(ds))
        ds = ds.select(range(take))
        print(f"[train_qwen_sql_sft] SMOKE_MAX_EXAMPLES={take} (subset for local testing)", flush=True)
    elif MAX_TRAIN_EXAMPLES is not None:
        take = min(MAX_TRAIN_EXAMPLES, len(ds))
        ds = ds.shuffle(seed=42).select(range(take))
        print(
            f"[train_qwen_sql_sft] MAX_TRAIN_EXAMPLES={take} of {full_train_rows} (shuffle seed=42, then head)",
            flush=True,
        )

    import transformers as transformers_mod

    print(
        f"[train_qwen_sql_sft] transformers {transformers_mod.__version__} from {transformers_mod.__file__}",
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
            f"[train_qwen_sql_sft] train/eval split: train={len(train_raw)} eval={len(eval_raw)} (test_size={n_test})",
            flush=True,
        )
    else:
        train_raw, eval_raw = ds, None
        if smoke_mode:
            print("[train_qwen_sql_sft] smoke: no eval split", flush=True)

    cols = train_raw.column_names
    train_raw = train_raw.map(row_to_messages, remove_columns=cols)
    train_ds = train_raw.map(messages_to_text, batched=True, remove_columns=["messages"])
    eval_ds = None
    if eval_raw is not None:
        ec = eval_raw.column_names
        eval_raw = eval_raw.map(row_to_messages, remove_columns=ec)
        eval_ds = eval_raw.map(messages_to_text, batched=True, remove_columns=["messages"])

    print(f"[train_qwen_sql_sft] prepared rows train={len(train_ds)} eval={len(eval_ds or [])}", flush=True)

    train_on_cpu = smoke_mode or (os.environ.get("QWEN_TRAINING_USE_CPU") or "false").strip().lower() == "true"
    packing_enabled = False if smoke_mode else PACKING

    world_size = max(int(os.environ.get("WORLD_SIZE", "1") or "1"), 1)
    distributed = world_size > 1

    cfg_kwargs: dict = dict(
        output_dir=output_dir,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        warmup_ratio=WARMUP_RATIO,
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
        es = TRAIN_EVAL_STEPS
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
    if distributed:
        cfg_kwargs["ddp_find_unused_parameters"] = False
    if distributed and gradient_checkpointing:
        cfg_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    trainer = SFTTrainer(
        model=model_id,
        processing_class=tok,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=SFTConfig(**cfg_kwargs),
        peft_config=LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
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

        if merge_after_train:
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
