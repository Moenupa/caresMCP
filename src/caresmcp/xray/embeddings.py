"""One global embedding interface for retained chest radiograph encoders."""

from typing import Annotated, Literal

from pydantic import Field

from caresmcp.api.decorators import requires
from caresmcp.api.embeddings import cosine_similarity
from caresmcp.registry import TASK_EMB, TASK_RET, XRAY
from caresmcp.xray.biovil import _embed_biovil_image
from caresmcp.xray.cxr_encoder import _extract_cxr_embedding

XrayEncoder = Literal[
    "microsoft/rad-dino",
    "microsoft/BiomedVLP-CXR-BERT-specialized",
    "microsoft/BiomedVLP-BioViL-T",
]


@XRAY.register("xray_image_embed")
@TASK_EMB.register("xray_image_embed")
@requires(["torch", "transformers", "health_multimodal"], extra="xray")
def xray_image_embed(
    image: Annotated[bytes, Field(description="Current chest radiograph image")],
    model_id: Annotated[
        XrayEncoder, Field(description="Exact encoder repository")
    ] = "microsoft/rad-dino",
    pooling: Annotated[
        Literal["cls", "mean"] | None,
        Field(
            description="RAD-DINO token pooling, default CLS; omit for BioViL's fixed projector"
        ),
    ] = None,
    prior_image: Annotated[
        bytes | None,
        Field(
            description="Earlier same-patient chest radiograph, supported only by BioViL-T"
        ),
    ] = None,
) -> dict:
    """Generate an embedding of a chest radiograph.

    Use an embedding to compare radiographs or retrieve similar images. Choose
    RAD-DINO for visual features or BioViL for an image-text representation.
    Choose BioViL-T to include an earlier radiograph.

    See Also:
        https://huggingface.co/microsoft/rad-dino
        https://github.com/microsoft/hi-ml/tree/main/hi-ml-multimodal
    """
    if model_id == "microsoft/rad-dino":
        if prior_image is not None:
            raise ValueError("prior_image is supported only by BioViL-T")
        result = _extract_cxr_embedding(image, model_id, pooling=pooling or "cls")
    else:
        if pooling is not None:
            raise ValueError("Token pooling is supported only by RAD-DINO")
        result = _embed_biovil_image(image, model_id=model_id, prior_image=prior_image)
    return result


@XRAY.register("xray_image_similarity")
@TASK_RET.register("xray_image_similarity")
def xray_image_similarity(
    image: Annotated[bytes, Field(description="First chest radiograph image")],
    other_image: Annotated[bytes, Field(description="Second chest radiograph image")],
    model_id: Annotated[
        XrayEncoder, Field(description="Exact encoder repository")
    ] = "microsoft/rad-dino",
    pooling: Annotated[
        Literal["cls", "mean"] | None,
        Field(description="RAD-DINO token pooling; omit for BioViL"),
    ] = None,
) -> dict:
    """Compare two chest radiographs by the cosine similarity of their embeddings.

    Use this to compare images or rank candidates by visual similarity. Both
    images use the same encoder and pooling settings. Similarity reflects the
    model's representation, not diagnostic agreement.
    """
    first = xray_image_embed(image, model_id=model_id, pooling=pooling)
    second = xray_image_embed(other_image, model_id=model_id, pooling=pooling)
    return {
        "model_id": model_id,
        "representation": first["representation"],
        "pooling": pooling or ("cls" if model_id == "microsoft/rad-dino" else None),
        "cosine_similarity": cosine_similarity(first["embedding"], second["embedding"]),
    }
