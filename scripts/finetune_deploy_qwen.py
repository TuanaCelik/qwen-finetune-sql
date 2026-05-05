#!/usr/bin/env python3
"""
Submit Vertex CustomJob (HF PyTorch training DLC) for **Qwen3.5** SQL SFT using
``scripts/train_qwen_sql_sft.py``, then optionally deploy merged weights with a Hugging Face
PyTorch inference DLC.

Run from repository root:

  uv run python scripts/finetune_deploy_qwen.py --help
  uv run python scripts/finetune_deploy_qwen.py --train
  uv run python scripts/finetune_deploy_qwen.py --train --deploy
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

_DEFAULT_TRAINING_IMAGE = (
    "us-docker.pkg.dev/deeplearning-platform-release/gcr.io/"
    "huggingface-pytorch-training-cu121.2-3.transformers.4-48.ubuntu2204.py311:latest"
)
_DEFAULT_INFERENCE_IMAGE = (
    "us-docker.pkg.dev/deeplearning-platform-release/gcr.io/"
    "huggingface-pytorch-inference-cu121.2-3.transformers.4-48.ubuntu2204.py311:latest"
)

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune Qwen3.5 SQL on Vertex (CustomJob) and/or deploy merged weights.",
    )
    p.add_argument("--train", action="store_true", help="Run the training CustomJob.")
    p.add_argument("--deploy", action="store_true", help="Upload merged weights from GCS and deploy.")
    p.add_argument(
        "--training-async",
        action="store_true",
        help="Submit training without waiting (do not combine with --deploy in one run).",
    )
    p.add_argument(
        "--dataset-id",
        help=(
            "Hugging Face dataset repo to train on. Defaults to QWEN_DATASET_ID, "
            "then b-mc2/sql-create-context."
        ),
    )
    p.add_argument(
        "--dataset-split",
        help="Dataset split to train on. Defaults to QWEN_DATASET_SPLIT, then train.",
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
    if (os.environ.get("SKIP_TRAINING") or "").strip().lower() == "true":
        args.run_train = False
    if (os.environ.get("SKIP_DEPLOY") or "").strip().lower() == "true":
        args.run_deploy = False
    if not args.run_train and not args.run_deploy:
        print(
            "Nothing to do: both training and deploy are disabled "
            "(SKIP_TRAINING / SKIP_DEPLOY or CLI flags).",
            file=sys.stderr,
        )
        sys.exit(0)

    from google.cloud import aiplatform  # noqa: PLC0415

    project = os.environ.get("GCP_PROJECT_ID", "")
    region = os.environ.get("GCP_REGION", "us-central1")
    bucket = os.environ.get("GCS_BUCKET", "").removeprefix("gs://").strip("/")
    hf_token = os.environ.get("HF_TOKEN", "")
    model_id = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen3.5-0.8B")
    prefix = os.environ.get("QWEN_OUTPUT_GCS_PREFIX", "qwen-sql/qwen35-synthetic-1k-ep4").strip("/")

    train_uri = os.environ.get("TRAINING_CONTAINER_URI", _DEFAULT_TRAINING_IMAGE)
    dlc_uri = os.environ.get("INFERENCE_CONTAINER_URI", _DEFAULT_INFERENCE_IMAGE)

    train_machine = os.environ.get("QWEN_TRAINING_MACHINE_TYPE", "g2-standard-12")
    train_accel = os.environ.get("QWEN_TRAINING_ACCELERATOR_TYPE", "NVIDIA_L4")
    train_accel_n = int(os.environ.get("QWEN_TRAINING_ACCELERATOR_COUNT", "1"))

    serve_machine = os.environ.get("QWEN_SERVING_MACHINE_TYPE", "g2-standard-12")
    serve_accel = os.environ.get("QWEN_SERVING_ACCELERATOR_TYPE", "NVIDIA_L4")
    serve_accel_n = int(os.environ.get("QWEN_SERVING_ACCELERATOR_COUNT", "1"))
    serve_min = int(os.environ.get("QWEN_SERVING_MIN_REPLICAS", "1"))
    serve_max = int(os.environ.get("QWEN_SERVING_MAX_REPLICAS", "1"))

    display_job = os.environ.get("QWEN_DISPLAY_NAME_TRAINING", "qwen35-sql-sft")
    display_model = os.environ.get("QWEN_DISPLAY_NAME_MODEL", "qwen35-sql-ft")
    display_endpoint = os.environ.get("QWEN_DISPLAY_NAME_ENDPOINT", "qwen35-sql-endpoint")

    if not project:
        print("Missing GCP_PROJECT_ID in environment.", file=sys.stderr)
        sys.exit(1)
    if not bucket:
        print("Missing GCS_BUCKET in environment.", file=sys.stderr)
        sys.exit(1)

    output_dir = f"/gcs/{bucket}/{prefix}"
    gcs_merged = f"gs://{bucket}/{prefix}/merged"
    dataset_id = (
        (args.dataset_id or "").strip()
        or (os.environ.get("QWEN_DATASET_ID") or "").strip()
        or "b-mc2/sql-create-context"
    )
    dataset_split = (
        (args.dataset_split or "").strip()
        or (os.environ.get("QWEN_DATASET_SPLIT") or "").strip()
        or "train"
    )

    script_path = ROOT / "scripts" / "train_qwen_sql_sft.py"
    if not script_path.is_file():
        print(f"Missing {script_path}", file=sys.stderr)
        sys.exit(1)

    aiplatform.init(project=project, location=region, staging_bucket=f"gs://{bucket}")

    train_env: dict[str, str] = {
        "QWEN_MODEL_ID": model_id,
        "QWEN_OUTPUT_DIR": output_dir,
        "QWEN_DATASET_ID": dataset_id,
        "QWEN_DATASET_SPLIT": dataset_split,
    }
    if hf_token:
        train_env["HF_TOKEN"] = hf_token
        train_env["HUGGING_FACE_HUB_TOKEN"] = hf_token
    torch_idx = (os.environ.get("TORCH_PIP_INDEX_URL") or "").strip()
    if torch_idx:
        train_env["TORCH_PIP_INDEX_URL"] = torch_idx
    cuda_alloc = (os.environ.get("PYTORCH_CUDA_ALLOC_CONF") or "").strip()
    if cuda_alloc:
        train_env["PYTORCH_CUDA_ALLOC_CONF"] = cuda_alloc

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
        print(f"Submitting training job (dataset={dataset_id}:{dataset_split}, merged → {gcs_merged}) …")
        job.run(sync=not args.training_async)
        if args.training_async:
            print("Training submitted (async). Re-run with --deploy only when merged/ exists.")
            if args.run_deploy:
                print(
                    "--training-async with --deploy: merged weights may not exist yet.",
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
            "Warning: HF_TOKEN is empty. The inference container may still need Hub auth for tokenizers; "
            "set HF_TOKEN if deploy or tokenizer resolution fails.",
            file=sys.stderr,
        )

    deploy_timeout_sec = int(((os.environ.get("SERVING_DEPLOYMENT_TIMEOUT_SEC") or "").strip() or "3600"))
    prediction_sa = (os.environ.get("VERTEX_PREDICTION_SERVICE_ACCOUNT") or "").strip()

    if gcs_merged.startswith("gs://") and not prediction_sa:
        print(
            "Warning: VERTEX_PREDICTION_SERVICE_ACCOUNT is not set. Replicas loading model artifacts from GCS "
            "usually need storage.objectViewer on the bucket.",
            file=sys.stderr,
        )

    serving_env = {
        "HF_TASK": "text-generation",
        "HF_TOKEN": hf_token,
        "HUGGING_FACE_HUB_TOKEN": hf_token,
        "AIP_HTTP_PORT": "8080",
        "AIP_PREDICT_ROUTE": "/predict",
        "AIP_HEALTH_ROUTE": "/health",
    }

    print(f"Uploading model from {gcs_merged} …")
    model = aiplatform.Model.upload(
        display_name=display_model,
        artifact_uri=gcs_merged,
        serving_container_image_uri=dlc_uri,
        serving_container_ports=[8080],
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
        serving_container_environment_variables=serving_env,
        serving_container_deployment_timeout=deploy_timeout_sec,
    )

    print("Deploying endpoint …")
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
    print(f"  Endpoint: {endpoint.resource_name}")
    print("  Predict route: /predict")


if __name__ == "__main__":
    main()
