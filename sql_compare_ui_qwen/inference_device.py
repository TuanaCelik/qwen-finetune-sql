"""Device selection for Qwen inference."""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch


def mps_is_available() -> bool:
    b = getattr(torch.backends, "mps", None)
    return b is not None and b.is_available()


def _mps_load_dtype() -> torch.dtype:
    raw = (os.environ.get("QWEN_COMPARE_MPS_DTYPE") or "").strip().lower()
    if raw in ("bf16", "bfloat16"):
        return torch.bfloat16
    if raw in ("fp16", "float16", "16"):
        return torch.float16
    return torch.float32


@dataclass(frozen=True)
class QwenDeviceSpec:
    torch_dtype: torch.dtype
    device_map: str | None
    to_device: str | None
    reason: str


def pick_qwen_device_spec(*, env_device_map: str) -> QwenDeviceSpec:
    raw = (os.environ.get(env_device_map) or "").strip().lower()
    tag = env_device_map

    if raw in ("none", "null"):
        return QwenDeviceSpec(torch.float32, None, "cpu", f"{tag}=none->cpu")
    if raw == "cpu":
        return QwenDeviceSpec(torch.float32, None, "cpu", f"{tag}=cpu")
    if raw == "mps":
        if not mps_is_available():
            return QwenDeviceSpec(torch.float32, None, "cpu", f"{tag}=mps_unavailable->cpu")
        return QwenDeviceSpec(_mps_load_dtype(), None, "mps", f"{tag}=mps")

    if raw.startswith("cuda") or raw == "auto":
        want = "auto" if raw == "auto" else raw
        if torch.cuda.is_available():
            return QwenDeviceSpec(torch.bfloat16, want, None, f"{tag}={raw!r}->cuda")
        if mps_is_available():
            return QwenDeviceSpec(_mps_load_dtype(), None, "mps", f"{tag}={raw!r}->mps_cuda_missing")
        return QwenDeviceSpec(torch.float32, None, "cpu", f"{tag}={raw!r}->cpu_no_accel")

    if raw:
        if torch.cuda.is_available():
            return QwenDeviceSpec(torch.bfloat16, raw, None, f"{tag}={raw!r}->cuda")
        if mps_is_available():
            return QwenDeviceSpec(_mps_load_dtype(), None, "mps", f"{tag}={raw!r}->mps_unknown_map")
        return QwenDeviceSpec(torch.float32, None, "cpu", f"{tag}={raw!r}->cpu")

    if torch.cuda.is_available():
        return QwenDeviceSpec(torch.bfloat16, "auto", None, f"{tag}unset->cuda_auto")
    if mps_is_available():
        return QwenDeviceSpec(_mps_load_dtype(), None, "mps", f"{tag}unset->mps_default")
    return QwenDeviceSpec(torch.float32, None, "cpu", f"{tag}unset->cpu")


def log_qwen_device(kind: str, model: torch.nn.Module, spec: QwenDeviceSpec) -> None:
    p = next(model.parameters())
    print(
        f"QWEN_DEVICE {kind}: reason={spec.reason} | param_device={p.device} | "
        f"param_dtype={p.dtype} | load_dtype={spec.torch_dtype} | device_map={spec.device_map!r} | "
        f"post_to={spec.to_device!r}",
        flush=True,
    )


def pick_hub_device_spec() -> QwenDeviceSpec:
    if (os.environ.get("QWEN_COMPARE_HUB_DEVICE_MAP") or "").strip():
        return pick_qwen_device_spec(env_device_map="QWEN_COMPARE_HUB_DEVICE_MAP")
    return pick_qwen_device_spec(env_device_map="QWEN_COMPARE_LOCAL_DEVICE_MAP")
