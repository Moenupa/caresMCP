"""BioViL image embeddings, phrase grounding, and image-text ranking."""

from typing import Annotated

import numpy as np
from PIL import Image
from pydantic import Field

from caresmcp.api.decorators import requires
from caresmcp.api.io import decode_as
from caresmcp.api.models._biovil import (
    BIOVIL_CHECKPOINTS,
    BioViLModel,
    get_biovil_image_model,
    get_biovil_text_model,
)
from caresmcp.constants import DEVICE
from caresmcp.registry import TASK_LOC, TASK_RET, XRAY


def _biovil_image_tensor(image: bytes, crop_size: int):
    """Apply the published min/max intensity remapping and image transforms."""
    from health_multimodal.image.data.io import remap_to_uint8
    from health_multimodal.image.data.transforms import (
        create_chest_xray_transform_for_inference,
    )

    source = decode_as(image)
    pixels = np.asarray(source.convert("RGB") if source.mode == "P" else source)
    grayscale = Image.fromarray(remap_to_uint8(pixels)).convert("L")
    transform = create_chest_xray_transform_for_inference(512, crop_size)
    return transform(grayscale).unsqueeze(0).to(DEVICE)


def _embed_biovil_image(
    image: Annotated[bytes, Field(description="Current chest radiograph image")],
    model_id: Annotated[
        BioViLModel,
        Field(description="Official BioViL or BioViL-T image checkpoint repository"),
    ] = "microsoft/BiomedVLP-BioViL-T",
    prior_image: Annotated[
        bytes | None,
        Field(
            description="Earlier chest radiograph from the same patient; BioViL-T only."
        ),
    ] = None,
) -> dict:
    """Embed a chest radiograph with BioViL, or a current/prior pair with BioViL-T.

    The normalized 128-dimensional vector is in the model's projected joint
    image-text space. It has no diagnosis head. Compare vectors only from the
    same model and temporal input configuration. BioViL-T uses its learned
    missing-prior representation when no prior image is supplied. Temporal
    embeddings do not provide trained improved/stable/worsened predictions.

    See Also:
        https://github.com/microsoft/hi-ml/tree/main/hi-ml-multimodal
        https://huggingface.co/microsoft/BiomedVLP-BioViL-T
    """
    import torch

    temporal = model_id == "microsoft/BiomedVLP-BioViL-T"
    if prior_image is not None and not temporal:
        raise ValueError("prior_image is supported only by BioViL-T")
    crop_size = 448 if temporal else 480
    current = _biovil_image_tensor(image, crop_size)
    model = get_biovil_image_model(model_id)
    with torch.inference_mode():
        if temporal:
            previous = (
                _biovil_image_tensor(prior_image, crop_size)
                if prior_image is not None
                else None
            )
            output = model(current_image=current, previous_image=previous)
        else:
            output = model(current)
        vector = output.projected_global_embedding[0].float()
        vector = torch.nn.functional.normalize(vector, dim=0)
    return {
        "model_id": model_id,
        "prior_image_used": prior_image is not None,
        "representation": "l2_normalized_projected_global_embedding",
        "embedding": vector.cpu().tolist(),
    }


@XRAY.register("biovil_ground_phrases")
@TASK_LOC.register("biovil_ground_phrases")
@TASK_RET.register("biovil_ground_phrases")
@requires(["torch", "health_multimodal", "transformers"], extra="xray")
def biovil_ground_phrases(
    image: Annotated[bytes, Field(description="Current chest radiograph image")],
    phrases: Annotated[
        list[Annotated[str, Field(min_length=1, max_length=2000)]],
        Field(
            min_length=1,
            max_length=16,
            description="Radiological phrases to locate and rank by image-text similarity",
        ),
    ],
    model_id: Annotated[
        BioViLModel, Field(description="Matching BioViL image and text checkpoint")
    ] = "microsoft/BiomedVLP-BioViL-T",
    prior_image: Annotated[
        bytes | None,
        Field(description="Earlier same-patient radiograph; BioViL-T only"),
    ] = None,
) -> dict:
    """Locate radiological phrases with BioViL patch-to-text similarity maps.

    Maps contain Gaussian-smoothed cosine similarities on the patch grid, ordered
    by rows then columns. Their coverage rectangle gives pixel-edge coordinates
    in the original image, with x rightward and y downward. Pixels outside this
    center crop are unobserved. The map grid spans the coverage rectangle evenly;
    patch receptive fields can overlap. Global image-text cosine scores also rank
    the supplied phrases. Scores are not probabilities or trained progression
    predictions, including when a prior image is supplied.

    See Also:
        https://github.com/microsoft/hi-ml/blob/main/hi-ml-multimodal/src/health_multimodal/vlp/inference_engine.py
    """
    import torch
    from scipy.ndimage import gaussian_filter

    temporal = model_id == "microsoft/BiomedVLP-BioViL-T"
    if prior_image is not None and not temporal:
        raise ValueError("prior_image is supported only by BioViL-T")
    crop_size = 448 if temporal else 480
    source = decode_as(image)
    width, height = source.size
    # torchvision Resize(int) floors the long side; CenterCrop rounds its offset.
    resized_width = 512 if width <= height else int(512 * width / height)
    resized_height = 512 if height <= width else int(512 * height / width)
    left = round((resized_width - crop_size) / 2)
    top = round((resized_height - crop_size) / 2)
    coverage = [
        left * width / resized_width,
        top * height / resized_height,
        (left + crop_size) * width / resized_width,
        (top + crop_size) * height / resized_height,
    ]
    current = _biovil_image_tensor(image, crop_size)
    previous = (
        _biovil_image_tensor(prior_image, crop_size)
        if prior_image is not None
        else None
    )
    image_model = get_biovil_image_model(model_id)
    text_model, tokenizer = get_biovil_text_model(model_id)
    tokens = tokenizer(phrases, padding=True, return_tensors="pt", truncation=False)
    if tokens["input_ids"].shape[1] > text_model.config.max_position_embeddings:
        raise ValueError("phrases exceed the text encoder token limit")
    with torch.inference_mode():
        output = (
            image_model(current_image=current, previous_image=previous)
            if temporal
            else image_model(current)
        )
        encoded = text_model(
            input_ids=tokens["input_ids"].to(DEVICE),
            attention_mask=tokens["attention_mask"].to(DEVICE),
            output_cls_projected_embedding=True,
            return_dict=True,
        )
        text = torch.nn.functional.normalize(
            encoded.cls_projected_embedding.float(), dim=1
        )
        patches = torch.nn.functional.normalize(
            output.projected_patch_embeddings[0].float(), dim=0
        )
        global_image = torch.nn.functional.normalize(
            output.projected_global_embedding[0].float(), dim=0
        )
        maps = torch.einsum("dhw,nd->nhw", patches, text).cpu().numpy()
        scores = (text @ global_image).cpu().tolist()
    maps = np.stack([gaussian_filter(values, sigma=1.5) for values in maps])
    if not np.isfinite(maps).all() or not np.isfinite(scores).all():
        raise RuntimeError("BioViL returned nonfinite image-text similarities")
    filename, revision = BIOVIL_CHECKPOINTS[model_id]
    return {
        "model_id": model_id,
        "checkpoint": filename,
        "checkpoint_revision": revision,
        "prior_image_used": prior_image is not None,
        "image_width": width,
        "image_height": height,
        "coverage_xyxy_original_pixels": coverage,
        "resized_image_wh": [resized_width, resized_height],
        "center_crop_xywh_resized_pixels": [left, top, crop_size, crop_size],
        "outside_coverage": "unobserved",
        "map_smoothing_sigma_grid_cells": 1.5,
        "phrases": [
            {
                "phrase": phrase,
                "global_cosine_similarity": score,
                "patch_similarity_map": values.tolist(),
            }
            for phrase, score, values in zip(phrases, scores, maps, strict=True)
        ],
        "ranked_phrase_indices": sorted(
            range(len(scores)), key=lambda index: scores[index], reverse=True
        ),
    }
