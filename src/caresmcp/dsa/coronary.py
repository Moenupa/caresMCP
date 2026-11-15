"""Coronary X-ray angiography stenosis and vessel segmentation U-Nets."""

from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

import cv2
import numpy as np
from pydantic import Field

from caresmcp.api.decorators import adapt_inputs, requires
from caresmcp.api.io import decode_as, encode_label_mask
from caresmcp.constants import DEVICE
from caresmcp.registry import DSA, TASK_LOC, TASK_SEG

if TYPE_CHECKING:
    from ._unet import UNet

CORONARY_MODEL_ID = "uzbtrust/stenoz-coronary-stenosis-unet"
_CHECKPOINTS = {
    "stenosis": "unet_stenosis.pt",
    "vessel": "unet_vessel.pt",
}
_CORONARY_DEPS = ["torch", "huggingface_hub"]

CoronaryAngiogramImage = Annotated[
    bytes,
    Field(
        description="Single-frame coronary X-ray angiography image with the ARCADE "
        "display polarity: dark vessels on a brighter background."
    ),
]
ScoreThreshold = Annotated[
    float,
    Field(ge=0, le=1, description="Minimum sigmoid probability for a pixel."),
]
MinAreaPixels = Annotated[
    int, Field(ge=1, description="Minimum connected-component area in pixels.")
]
InvertPolarity = Annotated[
    bool,
    Field(
        description="Invert grayscale values before normalization for frames whose "
        "display polarity is reversed relative to ARCADE."
    ),
]


def _decode_angiogram(image: bytes) -> np.ndarray:
    """Decode one frame into a finite two-dimensional float32 array."""
    source = decode_as(image)
    if source.mode == "I" or source.mode.startswith("I;16") or source.mode == "F":
        pixels = np.asarray(source)
    else:
        pixels = np.asarray(source.convert("L"))
    if pixels.ndim != 2 or min(pixels.shape) < 2:
        raise ValueError("image must contain one two-dimensional angiography frame")
    values = pixels.astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("angiography frame must contain only finite pixels")
    return values


def _normalize_frame(pixels: np.ndarray, invert_polarity: bool) -> np.ndarray:
    """Apply optional polarity inversion and the release's min-max scaling."""
    if invert_polarity:
        pixels = pixels.max() + pixels.min() - pixels
    span = float(pixels.max() - pixels.min())
    if span <= 0:
        raise ValueError("angiography frame must contain more than one intensity value")
    return (pixels - pixels.min()) / (span + 1e-6)


def _load_model(kind: Literal["stenosis", "vessel"]) -> "UNet":
    """Load one released checkpoint strictly onto the inference device."""
    from caresmcp.api.models import get_hf_checkpoint

    from ._unet import UNet

    checkpoint = get_hf_checkpoint(CORONARY_MODEL_ID, _CHECKPOINTS[kind])
    if not isinstance(checkpoint, dict) or "state" not in checkpoint:
        raise ValueError("coronary checkpoint must contain a state dictionary")
    model = UNet(base=int(checkpoint.get("base", 32)))
    model.load_state_dict(checkpoint["state"], strict=True)
    return model.to(DEVICE).eval()


def _predict(
    pixels: np.ndarray,
    kind: Literal["stenosis", "vessel"],
    invert_polarity: bool,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Return the cropped sigmoid probability map and original frame size."""
    import torch

    pixels = _normalize_frame(pixels, invert_polarity)
    height, width = pixels.shape
    padded = np.pad(
        pixels,
        ((0, (-height) % 16), (0, (-width) % 16)),
        mode="reflect",
    )
    tensor = torch.from_numpy(padded)[None, None].to(DEVICE)
    model = _load_model(kind)
    with torch.inference_mode():
        logits = model(tensor)
        if (
            logits.shape != (1, 1, padded.shape[0], padded.shape[1])
            or not torch.isfinite(logits).all()
        ):
            raise RuntimeError("coronary U-Net returned invalid logits")
        probabilities = logits[0, 0].sigmoid().float().cpu().numpy()
    return probabilities[:height, :width], (height, width)


def _label_components(
    probabilities: np.ndarray,
    score_threshold: float,
    min_area_pixels: int,
) -> tuple[np.ndarray, list[tuple[float, float, float, int]]]:
    """Threshold the probability map and keep components at or above the area."""
    mask = (probabilities > score_threshold).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=4
    )
    kept = np.zeros_like(labels)
    detections: list[tuple[float, float, float, int]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area_pixels:
            continue
        component = labels == label
        kept[component] = 1
        peak_score = float(probabilities[component].max())
        x, y = (float(value) for value in centroids[label])
        detections.append((x, y, peak_score, area))
    detections.sort(key=lambda item: -item[2])
    return kept, detections


@DSA.register("detect_coronary_stenosis")
@TASK_LOC.register("detect_coronary_stenosis")
@requires(_CORONARY_DEPS, extra="hf")
@adapt_inputs(image=_decode_angiogram)
def detect_coronary_stenosis(
    image: CoronaryAngiogramImage,
    score_threshold: ScoreThreshold = 0.5,
    min_area_pixels: MinAreaPixels = 40,
    invert_polarity: InvertPolarity = False,
) -> dict[str, Any]:
    """Locate candidate coronary artery stenosis regions in one angiography frame.

    Detections are connected components of the model's sigmoid lesion map.
    Scores are pixel confidences, not a diagnosis or a percent diameter
    stenosis. The release reports high lesion recall with lower precision on
    ARCADE, which provided essentially no image-level negatives; an empty
    result does not rule out disease.

    See Also:
        https://huggingface.co/uzbtrust/stenoz-coronary-stenosis-unet
        https://github.com/uzbtrust/stenoz
    """
    probabilities, (height, width) = _predict(
        cast(np.ndarray, image), "stenosis", invert_polarity
    )
    _, detections = _label_components(probabilities, score_threshold, min_area_pixels)
    return {
        "model_id": CORONARY_MODEL_ID,
        "checkpoint": _CHECKPOINTS["stenosis"],
        "task": "coronary_stenosis_detection",
        "score_mode": "sigmoid",
        "image_width": width,
        "image_height": height,
        "polarity_inverted": invert_polarity,
        "score_threshold": score_threshold,
        "min_area_pixels": min_area_pixels,
        "max_detection_score": round(
            max((item[2] for item in detections), default=0.0), 4
        ),
        "detection_count": len(detections),
        "detections": [
            {
                "centroid_xy": [round(x, 2), round(y, 2)],
                "peak_score": round(peak_score, 4),
                "area_pixels": area,
            }
            for x, y, peak_score, area in detections
        ],
    }


@DSA.register("segment_coronary_stenosis")
@TASK_SEG.register("segment_coronary_stenosis")
@requires(_CORONARY_DEPS, extra="hf")
@adapt_inputs(image=_decode_angiogram)
def segment_coronary_stenosis(
    image: CoronaryAngiogramImage,
    score_threshold: ScoreThreshold = 0.5,
    min_area_pixels: MinAreaPixels = 40,
    invert_polarity: InvertPolarity = False,
) -> bytes:
    """Segment candidate coronary artery stenosis regions in one angiography frame.

    Returns:
        Palette PNG at the input size: 0 is transparent background and 1
        (blue) is a candidate stenosis region.
    """
    probabilities, _ = _predict(cast(np.ndarray, image), "stenosis", invert_polarity)
    mask, _ = _label_components(probabilities, score_threshold, min_area_pixels)
    return encode_label_mask(mask)


@DSA.register("segment_coronary_vessels")
@TASK_SEG.register("segment_coronary_vessels")
@requires(_CORONARY_DEPS, extra="hf")
@adapt_inputs(image=_decode_angiogram)
def segment_coronary_vessels(
    image: CoronaryAngiogramImage,
    score_threshold: ScoreThreshold = 0.5,
    min_area_pixels: MinAreaPixels = 60,
    invert_polarity: InvertPolarity = False,
) -> bytes:
    """Segment the coronary artery tree in one angiography frame.

    Returns:
        Palette PNG at the input size: 0 is transparent background and 1
        (blue) is a predicted vessel pixel. The mask supports visualization;
        it does not identify arteries, lesions, or blood flow.
    """
    probabilities, _ = _predict(cast(np.ndarray, image), "vessel", invert_polarity)
    mask, _ = _label_components(probabilities, score_threshold, min_area_pixels)
    return encode_label_mask(mask)
