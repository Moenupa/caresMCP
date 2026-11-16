"""ONNX Runtime sessions for checksum-pinned checkpoints."""

from typing import TYPE_CHECKING

from caresmcp.constants import DEVICE

from .download import get_checkpoint_path

if TYPE_CHECKING:
    from onnxruntime import InferenceSession

_CPU_PROVIDERS = ("CPUExecutionProvider",)
_CUDA_PROVIDERS = ("CUDAExecutionProvider", *_CPU_PROVIDERS)


def _device_providers() -> tuple[str, ...]:
    """Map the globally managed device to ONNX Runtime execution providers."""
    import onnxruntime

    available = onnxruntime.get_available_providers()
    if DEVICE == "cuda" and "CUDAExecutionProvider" in available:
        return _CUDA_PROVIDERS
    return _CPU_PROVIDERS


def get_onnx_session(url: str, filename: str, sha256: str) -> "InferenceSession":
    """Download a checksum-pinned ONNX checkpoint and load its inference session.

    Execution providers follow ``caresmcp.constants.DEVICE``: CUDA with CPU
    fallback when both the device and ONNX Runtime support it, otherwise CPU.

    Args:
        url: Direct download URL for the ONNX checkpoint.
        filename: Name to store the checkpoint under in the model cache.
        sha256: Expected SHA-256 digest of the checkpoint.

    Returns:
        An inference session for the checkpoint. Each call builds a new
        session, so callers that reuse a model should keep and reuse the
        returned instance.
    """
    # this should not be lru cached, because it may return a CUDA session
    import onnxruntime

    checkpoint = get_checkpoint_path(url, filename, sha256)
    return onnxruntime.InferenceSession(
        str(checkpoint), providers=list(_device_providers())
    )
