import os
from pathlib import Path

from caresmcp.util.moduleloader import get_torch_device

DEVICE = get_torch_device()
CACHE_ROOT = Path(".cache")
MODEL_CACHE_DIR = CACHE_ROOT / "models"

LRU_MAX_SIZE = 8

os.environ.setdefault("CELLPOSE_LOCAL_MODELS_PATH", str(MODEL_CACHE_DIR))
