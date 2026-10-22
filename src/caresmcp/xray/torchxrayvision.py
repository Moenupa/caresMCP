"""Chest radiograph tools powered by TorchXRayVision.

Good test samples:
- Hernia: https://nih-chest-x-rays.s3.us-east-2.amazonaws.com/images_1024x1024/00000003_000.png
"""

from io import BytesIO
from typing import Annotated, Any, Literal

from pydantic import Field

from caresmcp.constants import DEVICE, MODEL_CACHE_DIR
from caresmcp.registry import TASK_CLS, TASK_SEG, XRAY
from caresmcp.util.io import draw_legends
from caresmcp.util.moduleloader import requires

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

ANATOMICAL_TARGETS = (
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
)


def _preprocess_xray(image: bytes, xrv: Any, transforms: Any, weight_or_size: int | ChestXrayWeights) -> tuple[Any, int]:
    """Decode and resize a radiograph using TorchXRayVision preprocessing."""
    import numpy as np
    from PIL import Image, UnidentifiedImageError

    if isinstance(weight_or_size, str):
        weight_or_size = 512 if weight_or_size.startswith("resnet") else 224
    assert isinstance(weight_or_size, int), f"Expected int, got {type(weight_or_size)}"
    try:
        with Image.open(BytesIO(image)) as source:
            pixels = np.asarray(source.convert("L"))
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError("image must contain a readable chest radiograph") from exc

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
) -> dict[str, float]:
    """Score by predicted pathology classes.

    Returns:
        A mapping from labels to sigmoid-scaled scores, including:
        Atelectasis, Consolidation, Infiltration, Pneumothorax, Edema, Emphysema,\
        Fibrosis, Effusion, Pneumonia, Pleural_Thickening, Cardiomegaly, Nodule,\
        Mass, Hernia, Lung Lesion, Fracture, Lung Opacity, Enlarged Cardiomediastinum

    See Also:
        https://github.com/mlmed/torchxrayvision
    """
    import torch
    import torchxrayvision as xrv
    from torchvision import transforms

    image_array, _ = _preprocess_xray(image, xrv, transforms, weight_or_size=weights)
    tensor = torch.from_numpy(image_array).unsqueeze(0).to(DEVICE)

    if weights.startswith("resnet"):
        model = xrv.models.ResNet(
            weights=weights, apply_sigmoid=True, cache_dir=MODEL_CACHE_DIR.as_posix()
        )
    else:
        model = xrv.models.DenseNet(
            weights=weights, apply_sigmoid=True, cache_dir=MODEL_CACHE_DIR.as_posix()
        )
    model = model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        scores = model(tensor)[0].detach().cpu().tolist()

    pathologies = list(model.pathologies)
    if len(pathologies) != len(scores):
        raise RuntimeError("Model output count does not match its pathology labels")
    # round decimal scores to 4 places for readability
    return {name: round(float(score), 4) for name, score in zip(pathologies, scores)}


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
) -> bytes:
    """Return a color-indexed PNG of 14 anatomical regions in a chest X-ray.

    The TorchXRayVision PSPNet predicts anatomical structures rather than
    disease regions. Each pixel is assigned to its highest-scoring target when that score meets
    ``probability_threshold``.

    See Also:
        https://github.com/mlmed/torchxrayvision
    """
    import seaborn as sns
    import torch
    import torchxrayvision as xrv
    from PIL import Image
    from torchvision import transforms

    image_array, _ = _preprocess_xray(image, xrv, transforms, weight_or_size=512)
    tensor = torch.from_numpy(image_array).unsqueeze(0).to(DEVICE)
    model = xrv.baseline_models.chestx_det.PSPNet(cache_dir=MODEL_CACHE_DIR.as_posix())
    model = model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        probabilities = torch.sigmoid(model(tensor)[0])
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
    canvas = draw_legends(
        mask_image,
        ANATOMICAL_TARGETS,
        title="Anatomical targets",
    )

    output = BytesIO()
    canvas.save(output, format="PNG", transparency=0)
    return output.getvalue()


@XRAY.register("torchxrayvision_localize")
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
) -> bytes:
    """Return a Grad-CAM overlay for one TorchXRayVision pathology score.

    The heatmap highlights image regions that contribute positively to the
    selected model output. It is a model explanation, not a pathology mask or
    a radiologist-validated localization.

    Returns:
        A color PNG overlay on the square, center-cropped model input.

    See Also:
        https://github.com/mlmed/torchxrayvision
        https://github.com/rajpurkarlab/cheXlocalize
    """
    import numpy as np
    import torch
    import torchxrayvision as xrv
    from PIL import Image
    from torchvision import transforms

    image_array, input_size = _preprocess_xray(image, xrv, transforms, weight_or_size=weights)
    tensor = torch.from_numpy(image_array).unsqueeze(0).to(DEVICE)

    if weights.startswith("resnet"):
        model = xrv.models.ResNet(weights=weights, cache_dir=MODEL_CACHE_DIR.as_posix())
        target_layer = model.model.layer4
    else:
        model = xrv.models.DenseNet(
            weights=weights, cache_dir=MODEL_CACHE_DIR.as_posix()
        )
        target_layer = model.features.denseblock4
    model = model.to(DEVICE)
    model.eval()

    pathologies = list(model.pathologies)
    if label not in pathologies:
        raise ValueError(
            "label must be one of the supported labels: " + ", ".join(pathologies)
        )
    class_index = pathologies.index(label)

    activations: list[Any] = []

    def capture_activation(
        _module: Any,
        _inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        activations.append(output)

    assert isinstance(target_layer, torch.nn.Module), (
        f"Expected torch.nn.Module, got {type(target_layer)}"
    )
    hook = target_layer.register_forward_hook(capture_activation)
    try:
        with torch.enable_grad():
            model.zero_grad(set_to_none=True)
            scores = model(tensor)
            if len(activations) != 1:
                raise RuntimeError("Could not capture the model's final feature map")
            feature_map = activations[0]
            gradients = torch.autograd.grad(
                scores[0, class_index], feature_map, retain_graph=False
            )[0]
            channel_weights = gradients.mean(dim=(2, 3), keepdim=True)
            heatmap = torch.relu((channel_weights * feature_map).sum(dim=1))[0]
    finally:
        hook.remove()

    heatmap = torch.nn.functional.interpolate(
        heatmap[None, None],
        size=(input_size, input_size),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    maximum = heatmap.max()
    if maximum > 0:
        heatmap = heatmap / maximum
    heatmap_array = heatmap.detach().cpu().numpy().clip(0, 1)

    base = np.asarray(image_array[0], dtype=np.float32)
    base = np.clip((base + 1024.0) / 2048.0, 0, 1)
    base_image = Image.fromarray(np.uint8(base * 255), mode="L").convert("RGB")
    base_image = base_image.resize((input_size, input_size), Image.Resampling.BILINEAR)

    red = np.clip(1.5 * heatmap_array, 0, 1)
    green = np.clip(1.5 - np.abs(2 * heatmap_array - 1.0) * 1.5, 0, 1)
    blue = np.clip(1.0 - 1.5 * heatmap_array, 0, 1)
    color_map = np.stack((red, green, blue), axis=-1)
    overlay = np.asarray(base_image, dtype=np.float32) / 255.0
    alpha = 0.45 * heatmap_array[..., None]
    result = np.uint8(np.clip((1 - alpha) * overlay + alpha * color_map, 0, 1) * 255)

    output_image = BytesIO()
    Image.fromarray(result, mode="RGB").save(output_image, format="PNG")
    return output_image.getvalue()
