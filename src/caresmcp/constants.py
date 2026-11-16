import os
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
CACHE_ROOT = Path(os.getenv("CARESMCP_CACHE", "~/.cache/caresmcp")).expanduser()
MODEL_CACHE_DIR = CACHE_ROOT / "models"
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

LRU_MAX_SIZE = int(os.getenv("CARESMCP_LRU_SIZE", "8"))
HTTP_TIMEOUT = int(os.getenv("CARESMCP_TIMEOUT", "15"))

# update model sources with our cache path for central management
os.environ.setdefault(
    "CELLPOSE_LOCAL_MODELS_PATH", (MODEL_CACHE_DIR / "cellpose").absolute().as_posix()
)
os.environ.setdefault("HF_HUB_CACHE", (MODEL_CACHE_DIR / "hf").absolute().as_posix())
# torch will land in $TORCH_HOME/hub
os.environ.setdefault("TORCH_HOME", (MODEL_CACHE_DIR).absolute().as_posix())
