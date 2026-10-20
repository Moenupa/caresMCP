"""Chest radiograph tools powered by TorchXRayVision."""

from io import BytesIO
from typing import Any, Literal

from caresmcp.constants import DEVICE, MODEL_CACHE_DIR
from caresmcp.registry import MEDAI, TAG_XRAY, TASK_CLS, TASK_SEG
from caresmcp.util.moduleloader import requires

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


def _preprocess_xray(image: bytes, xrv: Any, transforms: Any, size: int) -> Any:
    """Decode and resize a radiograph using TorchXRayVision preprocessing."""
    import numpy as np
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(BytesIO(image)) as source:
            pixels = np.asarray(source.convert("L"))
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError("image must contain a readable chest radiograph") from exc

    normalized = xrv.datasets.normalize(pixels, maxval=255, reshape=True)
    transform = transforms.Compose(
        [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(size)]
    )
    return transform(normalized)


@MEDAI.register("torchxrayvision_classify")
@TAG_XRAY.register("torchxrayvision_classify")
@TASK_CLS.register("torchxrayvision_classify")
@requires(["torch", "torchvision", "torchxrayvision"], extra="xray")
def torchxrayvision_classify(
    image: bytes,
    weights: ChestXrayWeights = "densenet121-res224-all",
) -> dict[str, float]:
    """Score by predicted pathology classes.

    Args:
        image: Chest radiograph image bytes.
        weights: Pretrained TorchXRayVision DenseNet or ResNet checkpoint.

    Returns:
        A mapping from labels to sigmoid-scaled scores, including:
        Atelectasis, Consolidation, Infiltration, Pneumothorax, Edema, Emphysema, Fibrosis, Effusion, Pneumonia, Pleural_Thickening, Cardiomegaly, Nodule, Mass, Hernia,
        Lung Lesion, Fracture, Lung Opacity, Enlarged Cardiomediastinum

    See Also:
        https://github.com/mlmed/torchxrayvision
    """
    import torch
    import torchxrayvision as xrv
    from torchvision import transforms

    input_size = 512 if weights.startswith("resnet") else 224
    image_array = _preprocess_xray(image, xrv, transforms, input_size)
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


@MEDAI.register("torchxrayvision_segment_anatomy")
@TAG_XRAY.register("torchxrayvision_segment_anatomy")
@TASK_SEG.register("torchxrayvision_segment_anatomy")
@requires(["torch", "torchvision", "torchxrayvision"], extra="xray")
def torchxrayvision_segment_anatomy(
    image: bytes,
    probability_threshold: float = 0.5,
) -> bytes:
    """Return a color-indexed PNG of 14 anatomical regions in a chest X-ray.

    The TorchXRayVision PSPNet predicts anatomical structures rather than
    disease regions. In the returned PNG, palette index 0 is background and
    indices 1 through 14 correspond in order to the targets in the legend.
    Each pixel is assigned to its highest-scoring target when that score meets
    ``probability_threshold``.

    Args:
        image: Chest radiograph image bytes.
        probability_threshold: Minimum sigmoid score for a pixel to receive a
            target label, from 0 to 1.

    Returns:
        A color-indexed PNG segmentation image with a transparent background
        and a legend mapping each color to its anatomical target.

    See Also:
        https://github.com/mlmed/torchxrayvision
    """
    import torch
    import torchxrayvision as xrv
    from PIL import Image, ImageDraw, ImageFont
    from torchvision import transforms

    if not 0 <= probability_threshold <= 1:
        raise ValueError("probability_threshold must be between 0 and 1")

    image_array = _preprocess_xray(image, xrv, transforms, 512)
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

    colors = (
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
        (127, 127, 127),
        (188, 189, 34),
        (23, 190, 207),
        (174, 199, 232),
        (255, 187, 120),
        (152, 223, 138),
        (255, 152, 150),
    )
    palette = [0, 0, 0]
    for color in colors:
        palette.extend(color)
    palette.extend((255, 255, 255, 0, 0, 0))
    palette.extend([0] * (768 - len(palette)))

    font = ImageFont.load_default()
    measure = ImageDraw.Draw(Image.new("P", (1, 1)))
    label_width = max(
        measure.textbbox((0, 0), target, font=font)[2] for target in ANATOMICAL_TARGETS
    )
    swatch_size = 14
    row_height = 20
    panel_padding = 12
    panel_width = int(panel_padding * 2 + swatch_size + 8 + label_width)
    legend_height = panel_padding * 2 + 20 + len(ANATOMICAL_TARGETS) * row_height

    mask_image = Image.fromarray(mask, mode="P")
    mask_image.putpalette(palette)
    canvas = Image.new(
        "P",
        (mask_image.width + panel_width, max(mask_image.height, legend_height)),
        15,
    )
    canvas.putpalette(palette)
    canvas.paste(mask_image, (0, 0))

    draw = ImageDraw.Draw(canvas)
    draw.line((mask_image.width, 0, mask_image.width, canvas.height), fill=16)
    legend_top = (canvas.height - legend_height) // 2
    panel_left = mask_image.width
    draw.text(
        (panel_left + panel_padding, legend_top + panel_padding),
        "Anatomical targets",
        fill=16,
        font=font,
    )

    rows_top = legend_top + panel_padding + 20
    for index, target in enumerate(ANATOMICAL_TARGETS, start=1):
        row_top = rows_top + (index - 1) * row_height
        swatch_top = row_top + (row_height - swatch_size) // 2
        draw.rectangle(
            (
                panel_left + panel_padding,
                swatch_top,
                panel_left + panel_padding + swatch_size - 1,
                swatch_top + swatch_size - 1,
            ),
            fill=index,
            outline=16,
        )
        draw.text(
            (panel_left + panel_padding + swatch_size + 8, row_top + 4),
            target,
            fill=16,
            font=font,
        )

    output = BytesIO()
    canvas.save(output, format="PNG", transparency=0)
    return output.getvalue()


@MEDAI.register("torchxrayvision_localize")
@TAG_XRAY.register("torchxrayvision_localize")
@requires(["torch", "torchvision", "torchxrayvision"], extra="xray")
def torchxrayvision_localize(
    image: bytes,
    pathology: str,
    weights: ChestXrayWeights = "densenet121-res224-all",
) -> bytes:
    """Return a Grad-CAM overlay for one TorchXRayVision pathology score.

    The heatmap highlights image regions that contribute positively to the
    selected model output. It is a model explanation, not a pathology mask or
    a radiologist-validated localization.

    Args:
        image: Chest radiograph image bytes.
        pathology: Pathology label exposed by the selected checkpoint.
        weights: Pretrained TorchXRayVision DenseNet or ResNet checkpoint.

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

    input_size = 512 if weights.startswith("resnet") else 224
    image_array = _preprocess_xray(image, xrv, transforms, input_size)
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
    if pathology not in pathologies:
        raise ValueError(
            "pathology must be one of the selected checkpoint labels: "
            + ", ".join(pathologies)
        )
    class_index = pathologies.index(pathology)

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
    base_image = Image.fromarray(np.uint8(base * 255), mode="L").convert("RGB")  # ty: ignore[invalid-argument-type]
    base_image = base_image.resize((input_size, input_size), Image.Resampling.BILINEAR)

    red = np.clip(1.5 * heatmap_array, 0, 1)
    green = np.clip(1.5 - np.abs(2 * heatmap_array - 1.0) * 1.5, 0, 1)
    blue = np.clip(1.0 - 1.5 * heatmap_array, 0, 1)
    color_map = np.stack((red, green, blue), axis=-1)
    overlay = np.asarray(base_image, dtype=np.float32) / 255.0
    alpha = 0.45 * heatmap_array[..., None]
    result = np.uint8(np.clip((1 - alpha) * overlay + alpha * color_map, 0, 1) * 255)

    output_image = BytesIO()
    Image.fromarray(result, mode="RGB").save(output_image, format="PNG")  # ty: ignore[invalid-argument-type]
    return output_image.getvalue()
