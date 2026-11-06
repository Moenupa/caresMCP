import os
from contextlib import nullcontext
from importlib import import_module
from pathlib import Path


def get_torch_device() -> str:
    """Return CUDA when available, otherwise CPU."""
    try:
        torch = import_module("torch")
    except Exception:  # noqa: BLE001
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


DEVICE = get_torch_device()
CACHE_ROOT = Path(os.getenv("CACHE_ROOT", ".cache"))
MODEL_CACHE_DIR = CACHE_ROOT / "models"

LRU_MAX_SIZE = 8
HTTP_TIMEOUT = 15

os.environ.setdefault("CELLPOSE_LOCAL_MODELS_PATH", str(MODEL_CACHE_DIR))


def torch_autocast(dtype=None):
    if DEVICE == "cpu":
        return nullcontext()
    else:
        from torch import autocast

        return autocast(device_type=DEVICE, dtype=dtype)
