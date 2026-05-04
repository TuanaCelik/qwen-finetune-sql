#!/usr/bin/env python3
"""
Submit Vertex CustomJob (HF PyTorch training DLC), wait, then upload + deploy with vLLM.

Configuration comes from .env / environment variables. CLI flags only select steps:
  --train / --deploy (omit both to run train then deploy), and --training-async.

Run from repository root:
  uv run python scripts/finetune_deploy_gemma4.py --help
  uv run python scripts/finetune_deploy_gemma4.py --train
  uv run python scripts/finetune_deploy_gemma4.py --train --deploy   # same as no flags (full pipeline)
"""
from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

# See https://huggingface.co/docs/google-cloud/containers/available (PyTorch Training)
_DEFAULT_TRAINING_IMAGE = (
    "us-docker.pkg.dev/deeplearning-platform-release/gcr.io/"
    "huggingface-pytorch-training-cu121.2-3.transformers.4-48.ubuntu2204.py311:latest"
)
_DEFAULT_VLLM_IMAGE = (
    "us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:latest"
)


def _env_str(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    return v.strip()


def _bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune Gemma SQL on Vertex (CustomJob) and/or deploy merged weights with vLLM. "
        "All resource and hyperparameter settings come from .env — see .env.example.",
    )
    p.add_argument(
        "--train",
        action="store_true",
        help="Run the training CustomJob (omit both --train and --deploy to run train + deploy).",
    )
    p.add_argument(
        "--deploy",
        action="store_true",
        help="Upload merged weights from GCS and deploy the endpoint (omit both flags for full pipeline).",
    )
    p.add_argument(
        "--training-async",
        action="store_true",
        help="Submit training and return without waiting (unsafe with deploy in same run).",
    )

    ns = p.parse_args(argv)
    if not ns.train and not ns.deploy:
        ns.run_train = True
        ns.run_deploy = True
    else:
        ns.run_train = ns.train
        ns.run_deploy = ns.deploy
    return ns


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if _bool("SKIP_TRAINING", False):
        args.run_train = False
    if _bool("SKIP_DEPLOY", False):
        args.run_deploy = False
    if not args.run_train and not args.run_deploy:
        print(
            "Nothing to do: both training and deploy are disabled "
            "(SKIP_TRAINING / SKIP_DEPLOY or CLI flags).",
            file=sys.stderr,
        )
        sys.exit(0)
    from google.cloud import aiplatform  # noqa: PLC0415

    project = _env_str("GCP_PROJECT_ID")
    region = _env_str("GCP_REGION", "us-central1")
    bucket = _env_str("GCS_BUCKET").replace("gs://", "").strip("/")
    hf_token = _env_str("HF_TOKEN")
    model_id = _env_str("MODEL_ID", "google/gemma-4-E2B-it")
    prefix = _env_str("OUTPUT_GCS_PREFIX", "gemma-sql/gemma-4-e2b-control-exp1").strip("/")

    train_uri = _env_str("TRAINING_CONTAINER_URI", _DEFAULT_TRAINING_IMAGE)
    vllm_uri = _env_str("VLLM_SERVING_CONTAINER_URI", _DEFAULT_VLLM_IMAGE)

    train_machine = _env_str("TRAINING_MACHINE_TYPE", "g2-standard-12")
    train_accel = _env_str("TRAINING_ACCELERATOR_TYPE", "NVIDIA_L4")
    train_accel_n = int(_env_str("TRAINING_ACCELERATOR_COUNT", "1") or "1")

    serve_machine = _env_str("SERVING_MACHINE_TYPE", "g2-standard-12")
    serve_accel = _env_str("SERVING_ACCELERATOR_TYPE", "NVIDIA_L4")
    serve_accel_n = int(_env_str("SERVING_ACCELERATOR_COUNT", "1") or "1")
    serve_min = int(_env_str("SERVING_MIN_REPLICAS", "1") or "1")
    serve_max = int(_env_str("SERVING_MAX_REPLICAS", "1") or "1")

    display_job = _env_str("DISPLAY_NAME_TRAINING", "gemma4-sql-control-exp1")
    display_model = _env_str("DISPLAY_NAME_MODEL", "gemma4-sql-ft-control-exp1")
    display_endpoint = _env_str("DISPLAY_NAME_ENDPOINT", "gemma4-sql-endpoint-control-exp1")

    tensor_parallel = int(_env_str("TENSOR_PARALLEL_SIZE", "1") or "1")
    max_model_len = int(
        _env_str("DEPLOY_MAX_MODEL_LEN", _env_str("MAX_SEQ_LENGTH", "4096")) or "4096"
    )

    merge_after = _bool("MERGE_AFTER_TRAIN", False)
    merge_str = "true" if merge_after else "false"
    packing_str = "true" if _bool("PACKING", False) else "false"

    if args.run_train and args.run_deploy and not merge_after:
        print(
            "MERGE_AFTER_TRAIN=false but deploy was requested: vLLM upload uses gs://…/merged. "
            "Use --train only or set SKIP_DEPLOY=true in .env, evaluate the adapter under final/, merge offline "
            "if it helps, then set MERGE_AFTER_TRAIN=true or upload merged weights yourself.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not project:
        print("Missing GCP_PROJECT_ID in environment.", file=sys.stderr)
        sys.exit(1)
    if not bucket:
        print("Missing GCS_BUCKET in environment.", file=sys.stderr)
        sys.exit(1)
    if args.run_train and not hf_token:
        print("Training requires HF_TOKEN in environment.", file=sys.stderr)
        sys.exit(1)

    output_dir = f"/gcs/{bucket}/{prefix}"
    gcs_merged = f"gs://{bucket}/{prefix}/merged"

    script_path = ROOT / "train_sql_sft.py"
    if not script_path.is_file():
        print(f"Missing {script_path}", file=sys.stderr)
        sys.exit(1)

    aiplatform.init(project=project, location=region, staging_bucket=f"gs://{bucket}")

    train_env = {
        "MODEL_ID": model_id,
        "OUTPUT_DIR": output_dir,
        "HF_TOKEN": hf_token,
        "NUM_TRAIN_EPOCHS": _env_str("NUM_TRAIN_EPOCHS", "1"),
        "PER_DEVICE_TRAIN_BATCH_SIZE": _env_str("PER_DEVICE_TRAIN_BATCH_SIZE", "1"),
        "GRADIENT_ACCUMULATION_STEPS": _env_str("GRADIENT_ACCUMULATION_STEPS", "16"),
        "LEARNING_RATE": _env_str("LEARNING_RATE", "5e-5"),
        "MAX_SEQ_LENGTH": _env_str("MAX_SEQ_LENGTH", "1024"),
        "LORA_R": _env_str("LORA_R", "16"),
        "LORA_ALPHA": _env_str("LORA_ALPHA", "32"),
        "LORA_DROPOUT": _env_str("LORA_DROPOUT", "0.05"),
        "MERGE_AFTER_TRAIN": merge_str,
        "PACKING": packing_str,
        "BF16": "true" if _bool("BF16", True) else "false",
        "GRADIENT_CHECKPOINTING": "true" if _bool("GRADIENT_CHECKPOINTING", True) else "false",
        # Mirrors train_sql_sft.py defaults; set in .env to override (e.g. different CUDA wheel index).
        "VERTEX_UPGRADE_TORCH": "1" if _bool("VERTEX_UPGRADE_TORCH", True) else "0",
    }
    max_steps = _env_str("MAX_STEPS", "").strip()
    if max_steps:
        train_env["MAX_STEPS"] = max_steps
    max_train_ex = _env_str("MAX_TRAIN_EXAMPLES", "").strip()
    if max_train_ex:
        train_env["MAX_TRAIN_EXAMPLES"] = max_train_ex
    train_eval_steps = _env_str("TRAIN_EVAL_STEPS", "").strip()
    if train_eval_steps:
        train_env["TRAIN_EVAL_STEPS"] = train_eval_steps
    torch_idx = _env_str("TORCH_PIP_INDEX_URL")
    if torch_idx:
        train_env["TORCH_PIP_INDEX_URL"] = torch_idx

    cuda_alloc = _env_str("PYTORCH_CUDA_ALLOC_CONF")
    if cuda_alloc:
        train_env["PYTORCH_CUDA_ALLOC_CONF"] = cuda_alloc

    # Single-process `train_sql_sft.py` + HF Trainer is not launched with torchrun/DDP. Multiple
    # visible GPUs → "Expected all tensors to be on the same device" (cuda:0 vs cuda:1). Pin one GPU
    # until multi-GPU launch is implemented (torchrun + Trainer).
    if train_accel_n > 1:
        train_env["CUDA_VISIBLE_DEVICES"] = "0"

    if args.run_train:
        job = aiplatform.CustomJob.from_local_script(
            display_name=display_job,
            script_path=str(script_path),
            container_uri=train_uri,
            machine_type=train_machine,
            accelerator_type=train_accel,
            accelerator_count=train_accel_n,
            environment_variables=train_env,
        )
        print("Submitting training job...")
        job.run(sync=not args.training_async)
        if args.training_async:
            try:
                job_rid = job.resource_name
            except RuntimeError:
                job_rid = (
                    f"(resolve in console: Custom jobs → {display_job!r} → copy resource name)"
                )
            print(f"Training submitted (async). Job resource: {job_rid}")
            if args.run_deploy:
                print(
                    "--training-async with --deploy: merged weights may not exist yet; run again with --deploy only.",
                    file=sys.stderr,
                )
            return
        print("Training finished.")
    else:
        print("Skipping training (not requested).")

    if not args.run_deploy:
        print("Skipping deploy (not requested). Done.")
        return

    if not hf_token:
        print(
            "Deploy requires HF_TOKEN: gated Gemma tokenizer/config resolution in the "
            "vLLM container uses Hugging Face Hub auth even when weights load from GCS.",
            file=sys.stderr,
        )
        sys.exit(1)

    shared_mem_mb = int(_env_str("SERVING_SHARED_MEMORY_MB", "16384") or "16384")
    deploy_timeout_sec = int(_env_str("SERVING_DEPLOYMENT_TIMEOUT_SEC", "3600") or "3600")
    prediction_sa = _env_str("VERTEX_PREDICTION_SERVICE_ACCOUNT")
    gpu_mu = _env_str("VLLM_GPU_MEMORY_UTILIZATION", "0.90").strip() or "0.90"

    if gcs_merged.startswith("gs://") and not prediction_sa:
        print(
            "Warning: VERTEX_PREDICTION_SERVICE_ACCOUNT is not set. Replicas that load "
            "--model from GCS need a service account with roles/storage.objectViewer on "
            "your artifact bucket (and Vertex AI User on the project). Deploy often fails "
            "without this — see README.",
            file=sys.stderr,
        )

    serving_args = [
        f"--model={gcs_merged}",
        f"--tensor-parallel-size={tensor_parallel}",
        f"--max-model-len={max_model_len}",
        f"--gpu-memory-utilization={gpu_mu}",
    ]
    extra_args = _env_str("VLLM_EXTRA_ARGS").strip()
    if extra_args:
        serving_args.extend(shlex.split(extra_args))

    # Let Hub libraries authenticate inside the serving container (gated models).
    serving_env = {
        "MODEL_ID": gcs_merged,
        "HF_TOKEN": hf_token,
        "HUGGING_FACE_HUB_TOKEN": hf_token,
    }

    print(f"Uploading model from {gcs_merged} ...")
    model = aiplatform.Model.upload(
        display_name=display_model,
        serving_container_image_uri=vllm_uri,
        serving_container_args=serving_args,
        serving_container_ports=[7080],
        serving_container_predict_route="/generate",
        serving_container_health_route="/ping",
        serving_container_environment_variables=serving_env,
        serving_container_shared_memory_size_mb=shared_mem_mb,
        serving_container_deployment_timeout=deploy_timeout_sec,
    )

    print("Deploying endpoint (continuous billing while replicas > 0)...")
    deploy_endpoint = aiplatform.Endpoint.create(display_name=display_endpoint)
    deploy_kw = {
        "endpoint": deploy_endpoint,
        "machine_type": serve_machine,
        "accelerator_type": serve_accel,
        "accelerator_count": serve_accel_n,
        "min_replica_count": serve_min,
        "max_replica_count": serve_max,
        "deploy_request_timeout": float(deploy_timeout_sec),
    }
    if prediction_sa:
        deploy_kw["service_account"] = prediction_sa
    endpoint = model.deploy(**deploy_kw)

    print("Deployed.")
    print("  Endpoint:", endpoint.resource_name)
    print("  Predict route: /generate — use Vertex prediction client or raw HTTP per Vertex docs.")


if __name__ == "__main__":
    main()
