from __future__ import annotations

import os
import time


def source_date_epoch() -> int | None:
    """Return SOURCE_DATE_EPOCH as an int if it is set to a valid value."""
    value = os.environ.get("SOURCE_DATE_EPOCH")
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def reproducible_ctime() -> str:
    """ctime string using SOURCE_DATE_EPOCH when requested."""
    epoch = source_date_epoch()
    return time.ctime(epoch) if epoch is not None else time.ctime()


def reproducible_timestamp(fmt: str) -> str | None:
    """Formatted timestamp using SOURCE_DATE_EPOCH, or None if not requested."""
    epoch = source_date_epoch()
    if epoch is None:
        return None
    return time.strftime(fmt, time.localtime(epoch))


def configure_torch_determinism(device=None) -> None:
    """Configure PyTorch inference backends for deterministic algorithm choices."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import torch

    device_type = getattr(device, "type", device)
    force_algorithms = device_type != "cpu"
    torch.use_deterministic_algorithms(force_algorithms, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False


def cpu_torch_threads(requested: int | None, device=None) -> int | None:
    """Cap CPU inference threads to physical cores when more threads were requested."""
    device_type = getattr(device, "type", device)
    if device_type != "cpu" or requested is None or requested < 1:
        return requested

    override = os.environ.get("FASTSURFER_CPU_TORCH_THREADS")
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass

    cpu_count = os.cpu_count()
    if cpu_count is None or cpu_count < 2:
        return requested
    return min(requested, max(1, cpu_count // 2))
