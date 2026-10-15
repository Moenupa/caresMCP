"""Research tools for chest X-ray classification, localization, and retrieval."""

from functools import lru_cache
from io import BytesIO
from typing import Any, Literal

from caresmcp.constants import DEVICE, MODEL_CACHE_DIR
from caresmcp.registry import MEDAI, TAG_XRAY
from caresmcp.util.moduleloader import requires

from ..image_processing import decode_rgb

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


def _transformer_load_kwargs(
    revision: str | None,
    local_files_only: bool,
) -> dict[str, Any]:
    """Build standard Transformers checkpoint loading arguments."""
    kwargs: dict[str, Any] = {"local_files_only": local_files_only}
    if revision is not None:
        kwargs["revision"] = revision
    return kwargs


@lru_cache(maxsize=2)
def _load_cxr_encoder(
    model_id: str,
    revision: str | None,
    local_files_only: bool,
) -> tuple[Any, Any]:
    """Load and cache an encoder and its image processor."""
    from transformers import AutoImageProcessor, AutoModel

    load_kwargs = _transformer_load_kwargs(revision, local_files_only)
    processor_kwargs = {
        **load_kwargs,
        "trust_remote_code": model_id.startswith("m42-health/"),
    }
    processor = AutoImageProcessor.from_pretrained(model_id, **processor_kwargs)
    model = AutoModel.from_pretrained(model_id, **load_kwargs).to(DEVICE)
    model.eval()
    return processor, model


@MEDAI.register("classify_chest_xray_pathologies")
@TAG_XRAY.register("classify_chest_xray_pathologies")
@requires(["torch", "torchvision", "torchxrayvision"], extra="xray")
def classify_chest_xray_pathologies(
    image: bytes,
    weights: ChestXrayWeights = "densenet121-res224-all",
) -> dict[str, float]:
    """Return TorchXRayVision scores for the pathologies supported by a model.

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


@MEDAI.register("segment_chest_xray_anatomy")
@TAG_XRAY.register("segment_chest_xray_anatomy")
@requires(["torch", "torchvision", "torchxrayvision"], extra="xray")
def segment_chest_xray_anatomy(
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


@MEDAI.register("extract_chest_xray_encoder_embedding")
@TAG_XRAY.register("extract_chest_xray_encoder_embedding")
@requires(["torch", "transformers"], extra="xray-hf")
def extract_chest_xray_encoder_embedding(
    image: bytes,
    model_id: Literal[
        "microsoft/rad-dino",
        "m42-health/CXformer-small",
        "m42-health/CXformer-base",
    ] = "microsoft/rad-dino",
    revision: str | None = None,
    local_files_only: bool = False,
    pooling: Literal["cls", "mean"] = "cls",
) -> dict[str, Any]:
    """Extract a feature vector from a published CXR image encoder.

    The supported encoders are RAD-DINO and CXFormer. The returned vector is
    a general visual representation and does not predict disease labels.
    Pooling is provided for downstream research and may differ from a
    checkpoint's preferred fine-tuning or evaluation recipe.

    Args:
        image: Chest radiograph image bytes.
        model_id: Hugging Face repository for RAD-DINO or CXFormer.
        revision: Optional model revision or commit identifier.
        local_files_only: Whether to use only locally cached checkpoint files.
        pooling: Use the first token or the mean of non-CLS tokens.

    Returns:
        Model provenance, pooling method, and a JSON-safe embedding vector.

    See Also:
        https://huggingface.co/microsoft/rad-dino
        https://huggingface.co/m42-health/CXformer-small
        https://huggingface.co/m42-health/CXformer-base
    """
    import torch

    if pooling not in ("cls", "mean"):
        raise ValueError("pooling must be 'cls' or 'mean'")

    processor, model = _load_cxr_encoder(model_id, revision, local_files_only)

    inputs = processor(images=decode_rgb(image), return_tensors="pt")
    inputs = {name: value.to(DEVICE) for name, value in inputs.items()}
    with torch.inference_mode():
        output = model(**inputs)

    tokens = getattr(output, "last_hidden_state", None)
    if tokens is None or tokens.ndim != 3 or tokens.shape[1] == 0:
        raise ValueError(
            "model must return last_hidden_state with shape [batch, tokens, features]"
        )
    if pooling == "cls" or tokens.shape[1] == 1:
        vector = tokens[:, 0, :]
    else:
        vector = tokens[:, 1:, :].mean(dim=1)

    embedding = vector[0].detach().float().cpu().tolist()
    return {
        "model_id": model_id,
        "revision": revision,
        "pooling": pooling,
        "embedding": [float(value) for value in embedding],
    }


@MEDAI.register("localize_chest_xray_pathology")
@TAG_XRAY.register("localize_chest_xray_pathology")
@requires(["torch", "torchvision", "torchxrayvision"], extra="xray")
def localize_chest_xray_pathology(
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


@MEDAI.register("evaluate_chest_xray_localization")
@TAG_XRAY.register("evaluate_chest_xray_localization")
def evaluate_chest_xray_localization(
    saliency_map: bytes,
    ground_truth_mask: bytes,
    saliency_threshold: float = 0.5,
) -> dict[str, float]:
    """Score one saliency map against a radiologist reference mask.

    This computes the per-image intersection-over-union and pointing-game hit
    used in saliency localization evaluation. The threshold converts the
    grayscale saliency map into a binary segmentation; the pointing-game
    location is the saliency map's maximum-valued pixel.

    Args:
        saliency_map: Grayscale PNG or JPEG bytes containing saliency values.
        ground_truth_mask: Grayscale reference mask bytes; nonzero pixels are
            treated as the localized finding.
        saliency_threshold: Saliency cutoff in the range [0, 1].

    Returns:
        Per-image intersection-over-union and pointing-game hit, each in [0, 1].

    See Also:
        https://github.com/rajpurkarlab/cheXlocalize
    """
    import numpy as np
    from PIL import Image, UnidentifiedImageError

    if not 0 <= saliency_threshold <= 1:
        raise ValueError("saliency_threshold must be between 0 and 1")

    decoded: list[Any] = []
    for label, image_bytes in (
        ("saliency_map", saliency_map),
        ("ground_truth_mask", ground_truth_mask),
    ):
        try:
            with Image.open(BytesIO(image_bytes)) as source:
                rgba = source.convert("RGBA")
                grayscale = np.asarray(rgba.convert("L"), dtype=np.uint8)
                alpha = np.asarray(rgba.getchannel("A"), dtype=np.uint8)
                decoded.append((grayscale * alpha // 255).astype(np.uint8))
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"{label} must contain a readable image") from exc

    saliency, reference = decoded
    if saliency.shape != reference.shape:
        raise ValueError("saliency_map and ground_truth_mask must have the same size")
    reference_mask = reference > 0
    reference_pixels = int(reference_mask.sum())
    if reference_pixels == 0:
        raise ValueError("ground_truth_mask must contain at least one positive pixel")

    predicted_mask = saliency >= round(saliency_threshold * 255)
    intersection = int(np.logical_and(predicted_mask, reference_mask).sum())
    union = int(np.logical_or(predicted_mask, reference_mask).sum())
    peak_y, peak_x = np.unravel_index(int(saliency.argmax()), saliency.shape)
    return {
        "intersection_over_union": float(intersection / union) if union else 0.0,
        "pointing_game_hit": float(reference_mask[peak_y, peak_x]),
    }


@MEDAI.register("segment_chest_xray_saliency_map")
@TAG_XRAY.register("segment_chest_xray_saliency_map")
def segment_chest_xray_saliency_map(
    saliency_map: bytes,
    threshold: float = 0.5,
    smoothing_kernel_size: int = 0,
) -> bytes:
    """Convert a grayscale chest X-ray saliency map to a binary mask.

    Args:
        saliency_map: Grayscale PNG or JPEG bytes containing saliency values.
        threshold: Saliency cutoff in the range [0, 1].
        smoothing_kernel_size: Optional square box-filter width in pixels;
            zero disables smoothing.

    Returns:
        A binary PNG mask with foreground value 255 and background value 0.

    See Also:
        https://github.com/rajpurkarlab/cheXlocalize
    """
    import cv2
    import numpy as np
    from PIL import Image, UnidentifiedImageError

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if smoothing_kernel_size < 0:
        raise ValueError("smoothing_kernel_size must not be negative")

    try:
        with Image.open(BytesIO(saliency_map)) as source:
            rgba = source.convert("RGBA")
            grayscale = np.asarray(rgba.convert("L"), dtype=np.uint8)
            alpha = np.asarray(rgba.getchannel("A"), dtype=np.uint8)
            values = (grayscale * alpha // 255).astype(np.float32) / 255.0
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError("saliency_map must contain a readable image") from exc

    if smoothing_kernel_size > 1:
        values = cv2.blur(
            values,
            (smoothing_kernel_size, smoothing_kernel_size),
        )
    mask = np.uint8(values >= threshold) * 255
    output = BytesIO()
    Image.fromarray(mask, mode="L").save(output, format="PNG")  # ty: ignore[invalid-argument-type]
    return output.getvalue()


@MEDAI.register("retrieve_chest_xray_reports")
@TAG_XRAY.register("retrieve_chest_xray_reports")
@requires(["torch", "transformers"], extra="xray-hf")
def retrieve_chest_xray_reports(
    image: bytes,
    reference_ids: list[str],
    reference_reports: list[str],
    reference_image_embeddings: list[list[float]],
    reference_model_id: Literal[
        "microsoft/rad-dino",
        "m42-health/CXformer-small",
        "m42-health/CXformer-base",
    ] = "microsoft/rad-dino",
    reference_revision: str | None = None,
    reference_pooling: Literal["cls", "mean"] = "cls",
    revision: str | None = None,
    local_files_only: bool = False,
    top_k: int = 5,
) -> dict[str, Any]:
    """Retrieve reports paired with the nearest reference CXR embeddings.

    Reference records must pair each report with a feature vector extracted
    from its associated image using the same encoder revision and pooling
    method. This tool performs image-embedding nearest-neighbor retrieval; it
    is not the separately trained CXR-RePaiR or X-REM pipeline.

    Args:
        image: Query chest radiograph image bytes.
        reference_ids: Identifiers for indexed studies or images.
        reference_reports: Report text paired with each reference image.
        reference_image_embeddings: Feature vectors from the reference images.
        reference_model_id: Encoder used to create the reference embeddings.
        reference_revision: Encoder revision used for the reference embeddings.
        reference_pooling: Pooling method used for the reference embeddings.
        revision: Optional query encoder revision; it must match the index.
        local_files_only: Whether to use only locally cached checkpoint files.
        top_k: Maximum number of reports to return.

    Returns:
        Model provenance and the highest cosine-similarity report matches.

    See Also:
        https://huggingface.co/microsoft/rad-dino
        https://github.com/rajpurkarlab/CXR-RePaiR
    """
    import torch
    from torch.nn import functional

    count = len(reference_ids)
    if count == 0:
        raise ValueError("at least one reference report is required")
    if len(reference_reports) != count or len(reference_image_embeddings) != count:
        raise ValueError(
            "reference IDs, reports, and embeddings must have equal lengths"
        )
    if len(set(reference_ids)) != count:
        raise ValueError("reference_ids must be unique")
    if reference_revision != revision:
        raise ValueError("query revision must match reference_revision")
    if not 1 <= top_k:
        raise ValueError("top_k must be positive")

    query_result = extract_chest_xray_encoder_embedding(
        image=image,
        model_id=reference_model_id,
        revision=revision,
        local_files_only=local_files_only,
        pooling=reference_pooling,
    )
    query = torch.tensor(query_result["embedding"], dtype=torch.float32)
    feature_size = query.numel()
    if any(len(embedding) != feature_size for embedding in reference_image_embeddings):
        raise ValueError("each reference embedding must match the query feature size")
    references = torch.tensor(reference_image_embeddings, dtype=torch.float32)
    if not torch.isfinite(references).all() or not torch.isfinite(query).all():
        raise ValueError("embeddings must contain only finite values")
    if (
        torch.any(torch.linalg.vector_norm(references, dim=1) == 0)
        or torch.linalg.vector_norm(query) == 0
    ):
        raise ValueError("embeddings must have non-zero norm")
    normalized_query = functional.normalize(query, dim=0)
    normalized_references = functional.normalize(references, dim=1)
    similarities = normalized_references @ normalized_query
    result_count = min(top_k, count)
    scores, indices = torch.topk(similarities, k=result_count)
    matches = [
        {
            "reference_id": reference_ids[index],
            "report": reference_reports[index],
            "cosine_similarity": float(score),
        }
        for index, score in zip(indices.tolist(), scores.tolist(), strict=True)
    ]
    return {
        "model_id": reference_model_id,
        "revision": revision,
        "pooling": reference_pooling,
        "matches": matches,
    }
