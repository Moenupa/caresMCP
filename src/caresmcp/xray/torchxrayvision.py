"""Chest radiograph pathology scores, anatomical masks, and saliency overlays."""

from typing import Annotated, Any, Literal, get_args

import numpy as np
from PIL import Image
from pydantic import Field

from caresmcp.api.decorators import requires
from caresmcp.api.imageio import decode_as, draw_legends, encode_as
from caresmcp.api.models import get_torchxrayvision_model
from caresmcp.api.saliency import blend_saliency, compute_gradcam
from caresmcp.api.tensorio import get_classification_scores
from caresmcp.constants import DEVICE
from caresmcp.registry import TASK_CLS, TASK_LOC, TASK_SEG, XRAY

CHEST_XRAY_IMAGE_FIELD = Field(description="Chest radiograph image")
TORCHXRAYVISION_WEIGHTS_FIELD = Field(
    description="Pretrained TorchXRayVision checkpoint to use"
)

ChestXrayWeights = Literal[
    "densenet121-res224-all",
    "densenet121-res224-rsna",
    "densenet121-res224-nih",
    "densenet121-res224-pc",
    "densenet121-res224-chex",
    "densenet121-res224-mimic_nb",
    "densenet121-res224-mimic_ch",
    "resnet50-res512-all",
]
ChestXrayLabels = Literal[
    "Atelectasis",
    "Consolidation",
    "Infiltration",
    "Pneumothorax",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Effusion",
    "Pneumonia",
    "Pleural_Thickening",
    "Cardiomegaly",
    "Nodule",
    "Mass",
    "Hernia",
    "Lung Lesion",
    "Fracture",
    "Lung Opacity",
    "Enlarged Cardiomediastinum",
]

AnatomicalTarget = Literal[
    "Left Clavicle",
    "Right Clavicle",
    "Left Scapula",
    "Right Scapula",
    "Left Lung",
    "Right Lung",
    "Left Hilus Pulmonis",
    "Right Hilus Pulmonis",
    "Heart",
    "Aorta",
    "Facies Diaphragmatica",
    "Mediastinum",
    "Weasand",
    "Spine",
]

ANATOMICAL_TARGETS = get_args(AnatomicalTarget)


def _preprocess_xray(
    image: bytes, xrv: Any, transforms: Any, weight_or_size: int | ChestXrayWeights
) -> tuple[Any, int]:
    """Return a model-ready radiograph array and its square side length."""

    if isinstance(weight_or_size, str):
        weight_or_size = 512 if weight_or_size.startswith("resnet") else 224
    pixels = np.asarray(decode_as(image, "L"))

    normalized = xrv.datasets.normalize(pixels, maxval=255, reshape=True)
    transform = transforms.Compose(
        [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(weight_or_size)]
    )
    return transform(normalized), weight_or_size


@XRAY.register("torchxrayvision_classify")
@TASK_CLS.register("torchxrayvision_classify")
@requires(["torch", "torchvision", "torchxrayvision"], extra="xray")
def torchxrayvision_classify(
    image: Annotated[bytes, CHEST_XRAY_IMAGE_FIELD],
    weights: Annotated[
        ChestXrayWeights, TORCHXRAYVISION_WEIGHTS_FIELD
    ] = "densenet121-res224-all",
) -> dict:
    """Return pathology multi-label scores for a chest radiograph.

    See Also:
        https://github.com/mlmed/torchxrayvision
    """
    import torch
    import torchxrayvision as xrv
    from torchvision import transforms

    image_array, _ = _preprocess_xray(image, xrv, transforms, weight_or_size=weights)
    tensor = torch.from_numpy(image_array).unsqueeze(0).to(DEVICE)

    model = get_torchxrayvision_model(weights, apply_sigmoid=True)
    with torch.no_grad():
        scores = model(tensor)

    return {
        "model_id": weights,
        "score_mode": "sigmoid",
        "scores": get_classification_scores(
            scores,
            dict(enumerate(model.pathologies)),
            "identity",
            skip_empty_labels=True,
        ),
    }


@XRAY.register("torchxrayvision_segment_anatomy")
@TASK_SEG.register("torchxrayvision_segment_anatomy")
@requires(["torch", "torchvision", "torchxrayvision"], extra="xray")
def torchxrayvision_segment_anatomy(
    image: Annotated[bytes, CHEST_XRAY_IMAGE_FIELD],
    probability_threshold: Annotated[
        float,
        Field(
            ge=0,
            le=1,
            description="Minimum sigmoid score for a pixel to receive a target label",
        ),
    ] = 0.5,
    output: Annotated[
        Literal["legend", "label_mask"],
        Field(
            description="PNG with a color legend, or a label mask with IDs 1–14 "
            "in anatomical target order (0 is background)"
        ),
    ] = "legend",
    region: Annotated[
        AnatomicalTarget | None,
        Field(
            description="Return a binary PNG for this structure, independently "
            "thresholded to preserve overlaps; overrides output"
        ),
    ] = None,
) -> bytes:
    """Return a PNG mask of 14 anatomical regions in a chest radiograph.

    Colors identify structures in the legend. Background is transparent, and
    overlapping structures use the highest-scoring label. Masks cover a
    512-pixel square center crop, with the legend panel added on the right.
    Binary region masks use white for the structure and black for background.

    See Also:
        https://github.com/mlmed/torchxrayvision
    """
    import seaborn as sns
    import torch
    import torchxrayvision as xrv
    from torchvision import transforms

    image_array, _ = _preprocess_xray(image, xrv, transforms, weight_or_size=512)
    tensor = torch.from_numpy(image_array).unsqueeze(0).to(DEVICE)
    model = get_torchxrayvision_model("anatomy")
    with torch.no_grad():
        probabilities = torch.sigmoid(model(tensor)[0])
        if region is not None:
            region_index = ANATOMICAL_TARGETS.index(region)
            mask = probabilities[region_index] >= probability_threshold
            return encode_as(
                Image.fromarray(mask.to(torch.uint8).cpu().numpy() * 255, mode="L")
            )
        confidence, labels = probabilities.max(dim=0)
        labels = torch.where(
            confidence >= probability_threshold,
            labels + 1,
            torch.zeros_like(labels),
        )
        mask = labels.to(torch.uint8).cpu().numpy()

    colors = sns.color_palette("tab20", n_colors=14)
    palette = [0, 0, 0]
    for color in colors:
        palette.extend(int(x * 255) for x in color)
    palette.extend([0] * (768 - len(palette)))

    mask_image = Image.fromarray(mask, mode="P")
    mask_image.putpalette(palette)
    if output == "label_mask":
        return encode_as(mask_image, transparency=0)
    canvas = draw_legends(mask_image, ANATOMICAL_TARGETS, title="Anatomical targets")

    return encode_as(canvas, transparency=0)


@XRAY.register("torchxrayvision_localize")
@TASK_LOC.register("torchxrayvision_localize")
@requires(["torch", "torchvision", "torchxrayvision"], extra="xray")
def torchxrayvision_localize(
    image: Annotated[bytes, CHEST_XRAY_IMAGE_FIELD],
    label: Annotated[
        ChestXrayLabels,
        Field(description="Pathology label exposed by the selected checkpoint"),
    ],
    weights: Annotated[
        ChestXrayWeights, TORCHXRAYVISION_WEIGHTS_FIELD
    ] = "densenet121-res224-all",
    output: Annotated[
        Literal["overlay", "saliency"],
        Field(
            description="Color saliency overlay or 8-bit grayscale saliency PNG (0–255)"
        ),
    ] = "overlay",
) -> bytes:
    """Return a PNG saliency overlay for the selected pathology label.

    Warmer colors indicate stronger positive contributions to the model score,
    not pathology boundaries; brighter grayscale pixels have the same meaning.
    Outputs cover a square center crop, 224 pixels for DenseNet or 512 for
    ResNet, without a legend.

    See Also:
        https://github.com/mlmed/torchxrayvision
        https://github.com/rajpurkarlab/cheXlocalize
    """
    import torch
    import torchxrayvision as xrv
    from torchvision import transforms

    image_array, input_size = _preprocess_xray(
        image, xrv, transforms, weight_or_size=weights
    )
    tensor = torch.from_numpy(image_array).unsqueeze(0).to(DEVICE)

    model = get_torchxrayvision_model(weights)
    target_layer = (
        model.model.layer4
        if weights.startswith("resnet")
        else model.features.denseblock4
    )
    class_index = list(model.pathologies).index(label)
    heatmap_array = compute_gradcam(
        model, tensor, target_layer, class_index, (input_size, input_size)
    )

    if output == "saliency":
        return encode_as(Image.fromarray(np.uint8(heatmap_array * 255), mode="L"))

    base = np.asarray(image_array[0], dtype=np.float32)
    base = np.clip((base + 1024.0) / 2048.0, 0, 1)
    base_image = Image.fromarray(np.uint8(base * 255), mode="L").convert("RGB")
    base_image = base_image.resize((input_size, input_size), Image.Resampling.BILINEAR)

    return encode_as(blend_saliency(base_image, heatmap_array))
