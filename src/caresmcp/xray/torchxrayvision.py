"""Chest radiograph pathology scores, anatomical masks, and Grad-CAM saliency.

Three TorchXRayVision tools share this module: multi-label pathology
classification with DenseNet-121 or ResNet-50 checkpoints, 14-structure
anatomical segmentation with a ChestX-Det PSPNet, and Grad-CAM saliency for
supported classifier labels. Classifier checkpoints were trained on public
chest radiograph datasets including NIH ChestX-ray14, RSNA Pneumonia, PadChest,
CheXpert, and MIMIC-CXR, so available labels and score behavior depend on the
selected checkpoint. Scores are model outputs, not calibrated probabilities or
diagnoses. Anatomical masks and saliency maps cover a 224- or 512-pixel square
center crop rather than the original image geometry, and saliency is not lesion
segmentation.

MONOCHROME1 DICOM files are inverted to MONOCHROME2, and stored pixel values
are scaled by the declared bit depth.

TorchXRayVision's core library and classifier checkpoints are Apache-2.0. Its
baseline-model checkpoints, including the PSPNet anatomy model, are separately
licensed upstream; the ChestX-Det source dataset is Apache-2.0. Verify
checkpoint and dataset terms before redistribution or clinical use.

References:
    Cohen et al., TorchXRayVision: A library of chest X-ray datasets and
    models, MIDL 2022. https://proceedings.mlr.press/v172/cohen22a

    Lian et al., A Structure-Aware Relation Network for Thoracic Diseases
    Detection and Segmentation, IEEE Transactions on Medical Imaging, 2021.
    https://doi.org/10.1109/TMI.2021.3070847

    Selvaraju et al., Grad-CAM: Visual Explanations from Deep Networks via
    Gradient-Based Localization, ICCV 2017.
    https://doi.org/10.1109/ICCV.2017.74

    Saporta et al., Benchmarking saliency methods for chest X-ray
    interpretation (CheXlocalize), Nature Machine Intelligence, 2022.
    https://doi.org/10.1038/s42256-022-00536-x

See Also:
    https://github.com/mlmed/torchxrayvision
    https://github.com/Deepwise-AILab/ChestX-Det-Dataset
"""

from io import BytesIO
from typing import TYPE_CHECKING, Annotated, Literal, get_args

import numpy as np
import seaborn as sns
from PIL import Image
from pydantic import Field

from caresmcp.api.decorators import requires
from caresmcp.api.io import (
    decode_as,
    draw_legends,
    encode_as,
    is_dicom,
)
from caresmcp.api.io.tensor import device_to, get_classification_scores
from caresmcp.api.models import get_torchxrayvision_model
from caresmcp.api.saliency import blend_saliency, compute_gradcam
from caresmcp.registry import TASK_CLS, TASK_LOC, TASK_SEG, XRAY

if TYPE_CHECKING:
    from torch import Tensor

CHEST_XRAY_IMAGE_FIELD = Field(
    description="Chest radiograph image or single-frame DICOM"
)
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


@requires(["pydicom"], extra="xray")
def _decode_dicom(image: bytes) -> np.ndarray:
    """Return a single-frame radiograph scaled to approximately [-1024, 1024]."""
    import torchxrayvision as xrv

    try:
        # pydicom accepts a file-like object; upstream annotates a filesystem path.
        pixels = xrv.utils.read_xray_dcm(BytesIO(image))  # ty: ignore[invalid-argument-type]
    except Exception as exc:
        raise ValueError("Invalid chest radiograph DICOM") from exc
    if pixels.ndim != 2 or not np.isfinite(pixels).all():
        raise ValueError(
            "Chest radiograph DICOM must contain one finite, two-dimensional frame"
        )
    return pixels[None, ...]


def _preprocess_xray(
    image_or_dicom: bytes, weight_or_size: int | ChestXrayWeights
) -> tuple["Tensor", int]:
    """Return a model-ready radiograph tensor and its square side length."""
    import torch
    import torchxrayvision as xrv
    from torchvision import transforms

    if isinstance(weight_or_size, str):
        weight_or_size = 512 if weight_or_size.startswith("resnet") else 224

    if is_dicom(image_or_dicom):
        pixels = _decode_dicom(image_or_dicom)
    else:
        pixels = xrv.datasets.normalize(
            np.asarray(decode_as(image_or_dicom, "L")), maxval=255, reshape=True
        )
    transform = transforms.Compose(
        [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(weight_or_size)]
    )

    tensor = device_to(torch.from_numpy(transform(pixels)).unsqueeze(0))
    return tensor, weight_or_size


@XRAY.register("torchxrayvision_classify")
@TASK_CLS.register("torchxrayvision_classify")
@requires(["torchxrayvision"], extra="xray")
def torchxrayvision_classify(
    image_or_dicom: Annotated[bytes, CHEST_XRAY_IMAGE_FIELD],
    weights: Annotated[
        ChestXrayWeights, TORCHXRAYVISION_WEIGHTS_FIELD
    ] = "densenet121-res224-all",
) -> dict:
    """Score chest radiograph findings using a TorchXRayVision multi-label classifier.

    Choose weights that support the findings of interest; each finding receives
    an independent, sigmoid-like score in [0, 1]. Labels the selected checkpoint
    cannot score are omitted or reported as null.

    References:
        Cohen et al., TorchXRayVision: A library of chest X-ray datasets and
        models, MIDL 2022. https://proceedings.mlr.press/v172/cohen22a

    See Also:
        https://github.com/mlmed/torchxrayvision
    """
    import torch

    image_tensor, _ = _preprocess_xray(image_or_dicom, weight_or_size=weights)

    model = get_torchxrayvision_model(weights, apply_sigmoid=True)
    with torch.inference_mode():
        scores = model(image_tensor)

    # xrv emits a constant 0.5 when a checkpoint has no operating threshold for
    # a label; report those labels as unavailable instead of a fake score.
    thresholds = getattr(model, "op_threshs", None)
    unavailable: set[str] = set()
    if thresholds is not None:
        unavailable = {
            name
            for index, name in enumerate(model.pathologies)
            if name and not bool(torch.isfinite(thresholds[index]))
        }
    pathology_scores = {
        name: None if name in unavailable else score
        for name, score in get_classification_scores(
            scores,
            dict(enumerate(model.pathologies)),
            # xrv's forward pass always applies sigmoid and then op_norm when
            # op_threshs is present (apply_sigmoid is ignored). op_norm is a
            # per-label piecewise-linear rescale mapping the checkpoint's
            # operating threshold to 0.5: sigmoid-like (bounded, monotone,
            # 0.5 decides positive) but not a probability. "identity" keeps it.
            "identity",
            skip_empty_labels=True,
        ).items()
    }

    return {
        "model_id": weights,
        "score_mode": "sigmoid",
        "scores": pathology_scores,
    }


@XRAY.register("torchxrayvision_segment_anatomy")
@TASK_SEG.register("torchxrayvision_segment_anatomy")
@requires(["torchxrayvision"], extra="xray")
def torchxrayvision_segment_anatomy(
    image_or_dicom: Annotated[bytes, CHEST_XRAY_IMAGE_FIELD],
    weights: Annotated[
        Literal["pspnet_chestxray"], TORCHXRAYVISION_WEIGHTS_FIELD
    ] = "pspnet_chestxray",
    probability_threshold: Annotated[
        float,
        Field(
            ge=0,
            le=1,
            description="Minimum score for a pixel to receive a target label",
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
    """Segment 14 anatomical structures in a chest radiograph.

    Use the legend output to inspect several structures, the label mask for
    anatomical target IDs, or select a region for a binary mask. Single-region
    masks preserve overlaps between structures; the combined mask assigns each
    pixel to the highest-scoring structure. Binary masks use white for the region
    and black for background. Masks cover a 512-pixel square center crop.

    References:
        Lian et al., A Structure-Aware Relation Network for Thoracic Diseases
        Detection and Segmentation, IEEE Transactions on Medical Imaging, 2021.
        https://doi.org/10.1109/TMI.2021.3070847

    See Also:
        - https://github.com/mlmed/torchxrayvision
        - https://github.com/Deepwise-AILab/ChestX-Det-Dataset
    """
    import torch

    image_tensor, _ = _preprocess_xray(image_or_dicom, weight_or_size=512)
    model = get_torchxrayvision_model("anatomy")
    with torch.inference_mode():
        probabilities = torch.sigmoid(model(image_tensor)[0])
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
@requires(["torchxrayvision"], extra="xray")
def torchxrayvision_localize(
    image_or_dicom: Annotated[bytes, CHEST_XRAY_IMAGE_FIELD],
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
    """Visualize Grad-CAM saliency for a chest radiograph finding.

    Select a finding supported by the weights to inspect which image regions
    contribute to its score. Choose an overlay for visual review or a grayscale
    map for further analysis. Warmer colors or brighter pixels indicate stronger
    positive contributions, not pathology boundaries. The map covers a square
    center crop: 224 pixels for DenseNet and 512 for ResNet.

    References:
        Selvaraju et al., Grad-CAM: Visual Explanations from Deep Networks via
        Gradient-Based Localization, ICCV 2017.
        https://doi.org/10.1109/ICCV.2017.74

        Saporta et al., Benchmarking saliency methods for chest X-ray
        interpretation (CheXlocalize), Nature Machine Intelligence, 2022.
        https://doi.org/10.1038/s42256-022-00536-x

    See Also:
        - https://github.com/mlmed/torchxrayvision
        - https://github.com/rajpurkarlab/cheXlocalize
    """
    import torch

    image_tensor, input_size = _preprocess_xray(image_or_dicom, weight_or_size=weights)

    model = get_torchxrayvision_model(weights)
    target_layer = (
        model.model.layer4
        if weights.startswith("resnet")
        else model.features.denseblock4
    )
    class_index = list(model.pathologies).index(label)
    thresholds = getattr(model, "op_threshs", None)
    if thresholds is not None and not bool(torch.isfinite(thresholds[class_index])):
        raise ValueError(
            f"{weights} has no operating threshold for {label!r}; "
            "saliency is unavailable."
        )
    heatmap_array = compute_gradcam(
        model, image_tensor, target_layer, class_index, (input_size, input_size)
    )

    if output == "saliency":
        return encode_as(Image.fromarray(np.uint8(heatmap_array * 255), mode="L"))

    base = image_tensor[0, 0].detach().cpu().numpy()
    base = np.clip((base + 1024.0) / 2048.0, 0, 1)
    base_image = Image.fromarray(np.uint8(base * 255), mode="L").convert("RGB")
    base_image = base_image.resize((input_size, input_size), Image.Resampling.BILINEAR)

    return encode_as(blend_saliency(base_image, heatmap_array))
