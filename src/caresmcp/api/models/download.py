"""Checksum-verified automatic downloads shared by checkpoint loaders."""

import hashlib
from pathlib import Path

from caresmcp.constants import MODEL_CACHE_DIR


def get_checkpoint_path(url: str, filename: str, sha256: str) -> Path:
    """Download a fixed checkpoint and verify new or cached content once."""
    from torch.hub import download_url_to_file

    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = MODEL_CACHE_DIR / filename
    if not checkpoint.exists():
        download_url_to_file(url, str(checkpoint), hash_prefix=sha256, progress=False)
    with checkpoint.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != sha256:
        raise ValueError("Cached model checkpoint checksum does not match the release")
    return checkpoint
