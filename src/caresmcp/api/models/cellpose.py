"""Load pretrained Cellpose segmentation models."""

from typing import TYPE_CHECKING, Literal

from caresmcp.constants import DEVICE

if TYPE_CHECKING:
    from cellpose.models import CellposeModel

CellposeModels = Literal["cpsam", "cpsam_v2", "cpdino", "cpdino-vitb"]


def get_cellpose_model(pretrained_model: CellposeModels = "cpsam") -> "CellposeModel":
    """Load a pretrained Cellpose model for inference.

    The model is placed on ``caresmcp.constants.DEVICE``. Cellpose downloads
    weights on first use and reuses them from :envvar:`CELLPOSE_LOCAL_MODELS_PATH`
    afterwards. Each call builds a new model, so callers that segment repeatedly
    should keep and reuse the returned instance.

    Args:
        pretrained_model: Pretrained Cellpose model to load. ``"cpsam"`` is
            Cellpose-SAM, ``"cpsam_v2"`` its updated release, and ``"cpdino"``
            and ``"cpdino-vitb"`` select the Cellpose-DINO ViT-L and ViT-B
            models.

    Returns:
        A Cellpose model ready for inference.
    """
    from cellpose.models import CellposeModel

    return CellposeModel(
        gpu=DEVICE != "cpu",
        pretrained_model=pretrained_model,
    )
