import os
from contextlib import nullcontext
from pathlib import Path

from caresmcp.util.moduleloader import get_torch_device

DEVICE = get_torch_device()
CACHE_ROOT = Path(os.getenv("CACHE_ROOT", ".cache"))
MODEL_CACHE_DIR = CACHE_ROOT / "models"

LRU_MAX_SIZE = 8

os.environ.setdefault("CELLPOSE_LOCAL_MODELS_PATH", str(MODEL_CACHE_DIR))


def torch_autocast(dtype=None):
    if DEVICE == "cpu":
        return nullcontext()
    else:
        from torch import autocast

        return autocast(device_type=DEVICE, dtype=dtype)
