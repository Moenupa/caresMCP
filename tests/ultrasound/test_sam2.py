"""SAM 2 prompted segmentation coverage using a real ultrasound frame."""

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from caresmcp.api.decorators import save_returns
from caresmcp.registry import TASK_SEG, ULTRASOUND
from caresmcp.ultrasound.sam2 import segment_ultrasound_sam2

# Prompts on the 630 x 408 hand-ultrasound fixture.
POINT = [[250.0, 165.0]]
BOX = [150.0, 120.0, 400.0, 260.0]


def test_sam2_tool_is_registered() -> None:
    """Expose prompted ultrasound segmentation through its catalogues."""
    assert "segment_ultrasound_sam2" in ULTRASOUND.get_all()
    assert "segment_ultrasound_sam2" in TASK_SEG.get_all()


@pytest.mark.slow
def test_sam2_requires_a_prompt(ultrasound_image: bytes) -> None:
    """Reject a call without points or a box before running inference."""
    with pytest.raises(ValueError, match="provide a box or at least one"):
        segment_ultrasound_sam2(ultrasound_image)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("points", "point_labels", "box", "message"),
    [
        ([[250.0, 165.0]], [1, 0], None, "equal, non-zero lengths"),
        ([[250.0, 165.0]], [2], None, "only 0 or 1"),
        ([[700.0, 165.0]], [1], None, "within the image"),
        (None, [1], None, "point_labels requires points"),
        (None, None, [400.0, 120.0, 150.0, 260.0], "positive area"),
    ],
)
def test_sam2_rejects_invalid_prompts(
    ultrasound_image: bytes,
    points: list[list[float]] | None,
    point_labels: list[int] | None,
    box: list[float] | None,
    message: str,
) -> None:
    """Reject malformed prompt geometry before loading the model."""
    with pytest.raises(ValueError, match=message):
        segment_ultrasound_sam2(
            ultrasound_image,
            points=points,
            point_labels=point_labels,
            box=box,
        )


@pytest.mark.slow
@pytest.mark.parametrize("prompt", ["point", "box"])
@save_returns("ultrasound_sam2")
def test_sam2_segments_a_prompted_structure(
    ultrasound_image: bytes,
    prompt: str,
):
    """Return a non-empty binary palette PNG at the original image dimensions."""
    try:
        if prompt == "point":
            result = segment_ultrasound_sam2(
                ultrasound_image, points=POINT, point_labels=[1]
            )
        else:
            result = segment_ultrasound_sam2(ultrasound_image, box=BOX)
    except ImportError as exc:
        pytest.skip(str(exc))

    assert isinstance(result, bytes)
    with (
        Image.open(BytesIO(ultrasound_image)) as source,
        Image.open(BytesIO(result)) as mask,
    ):
        assert mask.format == "PNG"
        assert mask.mode == "P"
        assert mask.size == source.size
        assert mask.info["transparency"] == 0
        labels = np.asarray(mask)
        assert set(np.unique(labels)) <= {0, 1}
        assert labels.any()
    return result
