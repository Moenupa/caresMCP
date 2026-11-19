"""Reproduce released BioViL embeddings, phrase grounding, and zero-shot labels.

Three released samples anchor the reproduction checks:

* the radiograph, phrase, and ground-truth box from the released
  ``hi-ml-multimodal`` phrase-grounding notebook;
* one positive case from the released zero-shot pneumonia test;
* one NIH ChestX-ray14 radiograph with its released finding box.

The embedding checks instead use the NIH radiographs downloaded in
``tests/xray/conftest.py``. Reference values come from the official
``ImageTextInferenceEngine`` or its released models evaluated on the same
inputs, never from a previous run of this tool. The paper's MS-CXR cohort
scores (BioViL CNR 1.027 +/- 0.02 and mIoU 0.266, arXiv:2204.09817 Table 5
and Table A.4) require credentialed PhysioNet access and are not reproduced
here.
"""

import csv
import gzip
import hashlib
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, get_args

import numpy as np
import pytest

from caresmcp.api.decorators import download_bytes, save_returns, skip_on_import_error
from caresmcp.api.decorators.bytes_io import _read_bytes
from caresmcp.api.io import decode_as
from caresmcp.constants import DEVICE
from caresmcp.xray.biovil import BioViLModel, biovil_ground_phrases
from caresmcp.xray.embeddings import xray_image_embed

if TYPE_CHECKING:
    from health_multimodal.vlp import ImageTextInferenceEngine

pytestmark = pytest.mark.slow

BIOVIL_RELEASE_ROOT = "https://github.com/microsoft/hi-ml/tree/main/hi-ml-multimodal"

# Released phrase-grounding notebook demo: an Open-i radiograph, the notebook's
# phrase, and its ground-truth box in original pixels as (x, y, width, height).
# https://github.com/microsoft/hi-ml/blob/main/hi-ml-multimodal/notebooks/phrase_grounding.ipynb
GROUNDING_DEMO_URL = (
    "https://openi.nlm.nih.gov/imgs/512/242/1445/CXR1445_IM-0287-4004.png"
)
GROUNDING_DEMO_SHA256 = (
    "73d0357ada8b8c845389bccb7e0dd7b5477acbdacb614981dc1102ef21d8dbcf"
)
GROUNDING_DEMO_PHRASE = "Left basilar consolidation seen"
GROUNDING_DEMO_BOX_XYWH = (306, 168, 124, 101)

# One positive case from the released zero-shot pneumonia test. The released
# test pins MD5 digests for twelve Open-i radiographs; the SHA-256 below is of
# the same bytes. One case exercises both prompt groups while keeping this
# suite at three image samples.
# https://github.com/microsoft/hi-ml/blob/main/hi-ml-multimodal/test_multimodal/vlp/test_zero_shot_classification.py
ZERO_SHOT_URL = "https://openi.nlm.nih.gov/imgs/512/173/1777/CXR1777_IM-0509-1001.png"
ZERO_SHOT_SHA256 = "cfd2ac60cc0d3927bd31a6d7323b48335b4c4d3c008746252a33cb2b81476506"
ZERO_SHOT_PNEUMONIA = True
ZERO_SHOT_POSITIVE = [
    "Findings consistent with pneumonia",
    "Findings suggesting pneumonia",
    "This opacity can represent pneumonia",
    "Findings are most compatible with pneumonia",
]
ZERO_SHOT_NEGATIVE = [
    "There is no pneumonia",
    "No evidence of pneumonia",
    "No evidence of acute pneumonia",
    "No signs of pneumonia",
]

# Released NIH ChestX-ray14 box bundled with TorchXRayVision. BioViL model
# cards list MIMIC-CXR as the image pretraining source, so this is a
# source-held-out check rather than a calibrated detection benchmark.
NIH_IMAGE_ROOT = "https://nih-chest-x-rays.s3.us-east-2.amazonaws.com/images_1024x1024/"
NIH_BOX_IMAGE = "00019892_003.png"
NIH_BOX_FINDING = "Pneumothorax"


def _download_pinned(url: str, checksum: str, attempts: int = 3) -> bytes:
    """Download a checksum-pinned asset, retrying transient host failures.

    Raises:
        ValueError: If every attempt fails, reported without the URL's query
            parameters, which may contain credentials or signed values.
    """
    errors: list[ValueError] = []
    for _ in range(attempts):
        try:
            return download_bytes(url, checksum=checksum)
        except ValueError as error:
            errors.append(error)
    # Drop query parameters, which may contain credentials or signed values.
    source = url.split("?", 1)[0]
    raise ValueError(f"Could not download the pinned asset from {source}") from errors[
        -1
    ]


@cache
def _official_engine(model_id: BioViLModel) -> "ImageTextInferenceEngine":
    """Load and cache the released inference engine for one checkpoint.

    Caching keeps a single engine per checkpoint, moved to ``DEVICE``, across
    all comparisons in the session.
    """
    import torch
    from health_multimodal.image import get_image_inference
    from health_multimodal.image.utils import ImageModelType
    from health_multimodal.text.utils import BertEncoderType, get_bert_inference
    from health_multimodal.vlp import ImageTextInferenceEngine

    temporal = model_id == "microsoft/BiomedVLP-BioViL-T"
    engine = ImageTextInferenceEngine(
        image_inference_engine=get_image_inference(
            ImageModelType.BIOVIL_T if temporal else ImageModelType.BIOVIL
        ),
        text_inference_engine=get_bert_inference(
            BertEncoderType.BIOVIL_T_BERT if temporal else BertEncoderType.CXR_BERT
        ),
    )
    engine.to(torch.device(DEVICE))
    return engine


def _official_global_embedding(
    image: bytes,
    model_id: BioViLModel,
    tmp_path: Path,
) -> np.ndarray:
    """Return the released engine's L2-normalized global image embedding.

    The engine reads images from disk, so the bytes are written under
    ``tmp_path`` first.
    """
    image_path = tmp_path / "reference.png"
    image_path.write_bytes(image)
    engine = _official_engine(model_id)
    embedding = engine.image_inference_engine.get_projected_global_embedding(image_path)
    return embedding.float().cpu().numpy()


def _official_temporal_embedding(
    current_image: bytes,
    prior_image: bytes,
    tmp_path: Path,
) -> np.ndarray:
    """Return the released BioViL-T model's paired global embedding.

    The released engine exposes no paired-image entry point, and
    ``ImageModel.forward`` accepts only a single image, so this calls the
    model's paired ``MultiImageEncoder`` and post-encoder exactly as the
    released ``MultiImageModel.forward`` does, then L2-normalizes the result
    as the engine's global-embedding path does.
    """
    import torch
    from health_multimodal.image.model.model import ImageModel

    current_path = tmp_path / "current.png"
    current_path.write_bytes(current_image)
    prior_path = tmp_path / "prior.png"
    prior_path.write_bytes(prior_image)

    engine = _official_engine("microsoft/BiomedVLP-BioViL-T")
    image_engine = engine.image_inference_engine
    current, _ = image_engine.load_and_transform_input_image(
        current_path, image_engine.transform
    )
    previous, _ = image_engine.load_and_transform_input_image(
        prior_path, image_engine.transform
    )
    image_model = cast(ImageModel, image_engine.model)
    with torch.inference_mode():
        patch_x, pooled_x = image_model.encoder(
            current_image=current,
            previous_image=previous,
            return_patch_embeddings=True,
        )
        output = image_model.forward_post_encoder(patch_x, pooled_x)
        embedding = torch.nn.functional.normalize(
            output.projected_global_embedding[0].float(), dim=0
        )
    return embedding.cpu().numpy()


def _official_global_cosines(
    image: bytes,
    phrases: list[str],
    model_id: BioViLModel,
    tmp_path: Path,
) -> np.ndarray:
    """Return the released engine's global image-text cosine per phrase.

    This mirrors ``get_similarity_score_from_raw_data`` called once per
    phrase: the engine's global image embedding and the text embeddings are
    both L2-normalized before the dot product, so a single image encode
    serves every phrase.
    """
    engine = _official_engine(model_id)
    image_embedding = _official_global_embedding(image, model_id, tmp_path)
    text_embeddings = engine.text_inference_engine.get_embeddings_from_prompt(
        phrases, verbose=False
    )
    return image_embedding @ text_embeddings.cpu().numpy().T


def _assert_phrase_matches_official(
    image: bytes,
    phrase: str,
    result: dict[str, Any],
    tmp_path: Path,
    model_id: BioViLModel,
) -> tuple[np.ndarray, np.ndarray, list[int], float]:
    """Assert one phrase's map and global cosine match the released engine.

    ``result`` must contain the requested phrase as its only phrase: every
    caller requests one phrase, and this reads ``result["phrases"][0]``.

    The official engine pads the area outside the center crop with NaN after
    resampling. Its observed extent must match the tool's reported coverage,
    and nearest-resampled patch centers must reproduce the tool's smoothed
    grid.

    Returns:
        Patch-center x and y coordinates in original pixels, the official
        observed extent as ``(x0, y0, x1, y1)``, and the official global
        cosine.
    """
    phrase_result = result["phrases"][0]
    scores = np.asarray(phrase_result["patch_similarity_map"])
    left, top, right, bottom = result["coverage_xyxy_original_pixels"]
    rows, columns = scores.shape
    xs = (left + (np.arange(columns) + 0.5) * (right - left) / columns).astype(int)
    ys = (top + (np.arange(rows) + 0.5) * (bottom - top) / rows).astype(int)

    image_path = tmp_path / "reference.png"
    image_path.write_bytes(image)
    engine = _official_engine(model_id)
    reference_map = engine.get_similarity_map_from_raw_data(
        image_path, phrase, interpolation="nearest"
    )
    reference_score = engine.get_similarity_score_from_raw_data(image_path, phrase)

    observed_y, observed_x = np.nonzero(~np.isnan(reference_map))
    observed_box = [
        int(observed_x.min()),
        int(observed_y.min()),
        int(observed_x.max()) + 1,
        int(observed_y.max()) + 1,
    ]
    assert observed_box == pytest.approx([left, top, right, bottom], abs=1)
    reference_grid = reference_map[ys[:, None], xs[None, :]]
    assert np.isfinite(reference_grid).all()
    assert np.abs(reference_grid - scores).max() <= 1e-4
    assert phrase_result["global_cosine_similarity"] == pytest.approx(
        reference_score, abs=1e-4
    )
    return xs, ys, observed_box, float(reference_score)


@pytest.fixture(scope="session")
def grounding_demo_image() -> bytes:
    """Download the pinned radiograph from the released grounding notebook."""
    return _download_pinned(GROUNDING_DEMO_URL, GROUNDING_DEMO_SHA256)


@pytest.fixture(scope="session")
def zero_shot_image() -> bytes:
    """Download the pinned radiograph from the released zero-shot test."""
    return _download_pinned(ZERO_SHOT_URL, ZERO_SHOT_SHA256)


@pytest.fixture(scope="module")
@skip_on_import_error
def nih_pneumothorax_box() -> tuple[bytes, list[list[float]], str, str, str]:
    """Return the released NIH pneumothorax radiograph, boxes, and provenance.

    The boxes come from the ``BBox_List_2017.csv.gz`` annotations bundled with
    TorchXRayVision; no region annotation is drawn by hand.

    Returns:
        ``(image, boxes_xyxy, finding, image_url, annotation_sha256)``, where
        the boxes are in original 1024-square pixels.
    """
    xrv = pytest.importorskip("torchxrayvision")
    annotation_path = Path(xrv.__file__).parent / "data" / "BBox_List_2017.csv.gz"
    with gzip.open(annotation_path, "rt") as stream:
        rows = list(csv.reader(stream))[1:]
    boxes = []
    for row in rows:
        if row[0] == NIH_BOX_IMAGE and row[1] == NIH_BOX_FINDING:
            left, top, width, height = map(float, row[2:6])
            boxes.append([left, top, left + width, top + height])
    assert boxes, f"No released {NIH_BOX_FINDING} annotation for {NIH_BOX_IMAGE}"
    url = NIH_IMAGE_ROOT + NIH_BOX_IMAGE
    image = _read_bytes(url)
    assert decode_as(image).size == (1024, 1024), (
        "NIH box geometry is defined at 1024-square resolution"
    )
    return (
        image,
        boxes,
        NIH_BOX_FINDING,
        url,
        hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
    )


@skip_on_import_error
@pytest.mark.parametrize("model_id", get_args(BioViLModel))
@save_returns()
def test_biovil_image_embedding_matches_released_encoder(
    cxr_image: bytes, tmp_path: Path, model_id: BioViLModel
):
    """Assert each checkpoint's normalized embedding matches its release.

    Both released checkpoints embed the same NIH radiograph, and the tool
    must return the same L2-normalized 128-dimensional vector as the
    checkpoint's official engine.
    """
    result = xray_image_embed(cxr_image, model_id=model_id)
    vector = np.asarray(result["embedding"])
    assert vector.shape == (128,)
    assert np.isfinite(vector).all()
    assert np.linalg.norm(vector) == pytest.approx(1, abs=1e-6)
    assert result["model_id"] == model_id
    assert result["prior_image_used"] is False
    reference = _official_global_embedding(cxr_image, model_id, tmp_path)
    assert np.abs(reference - vector).max() <= 1e-4
    return result


@skip_on_import_error
@save_returns(unpack=True)
def test_biovil_temporal_embedding_matches_released_encoder(
    cxr_image: bytes, cxr_image_followup: bytes, tmp_path: Path
):
    """Assert current-only and paired BioViL-T embeddings match the release.

    The paired reference comes from the released model's own paired-image
    path because the public engine API accepts a single image.
    """
    single = xray_image_embed(
        cxr_image_followup, model_id="microsoft/BiomedVLP-BioViL-T"
    )
    temporal = xray_image_embed(
        cxr_image_followup,
        model_id="microsoft/BiomedVLP-BioViL-T",
        prior_image=cxr_image,
    )
    for result in (single, temporal):
        vector = np.asarray(result["embedding"])
        assert vector.shape == (128,)
        assert np.isfinite(vector).all()
        assert np.linalg.norm(vector) == pytest.approx(1, abs=1e-6)
    assert single["prior_image_used"] is False
    assert temporal["prior_image_used"] is True
    assert not np.allclose(single["embedding"], temporal["embedding"])
    reference_single = _official_global_embedding(
        cxr_image_followup, "microsoft/BiomedVLP-BioViL-T", tmp_path
    )
    reference_temporal = _official_temporal_embedding(
        cxr_image_followup, cxr_image, tmp_path
    )
    assert np.abs(reference_single - np.asarray(single["embedding"])).max() <= 1e-4
    assert np.abs(reference_temporal - np.asarray(temporal["embedding"])).max() <= 1e-4
    return {"current_only": single, "current_with_prior": temporal}


@skip_on_import_error
@pytest.mark.parametrize("model_id", get_args(BioViLModel))
@save_returns(unpack=True)
def test_biovil_reproduces_released_grounding_demo(
    grounding_demo_image: bytes, tmp_path: Path, model_id: BioViLModel
):
    """Reproduce the released phrase-grounding demo and its ground-truth box.

    The official inference engine is the numeric reference for the pipeline,
    covering preprocessing, patch and text embeddings, the similarity map,
    and the global cosine. Both released checkpoints are compared; the
    notebook itself runs BioViL-T. The per-image CNR and grid AUC are
    reported, not asserted, because the paper's claimed cohort scores use the
    credentialed MS-CXR release.
    """
    temporal = model_id == "microsoft/BiomedVLP-BioViL-T"
    crop_size = 448 if temporal else 480

    result = biovil_ground_phrases(
        grounding_demo_image, [GROUNDING_DEMO_PHRASE], model_id=model_id
    )
    assert result["model_id"] == model_id
    assert result["prior_image_used"] is False
    assert result["outside_coverage"] == "unobserved"
    width, height = decode_as(grounding_demo_image).size
    assert (width, height) == (512, 420)
    assert (result["image_width"], result["image_height"]) == (width, height)
    phrase = result["phrases"][0]
    scores = np.asarray(phrase["patch_similarity_map"])
    # The official VLP test pins a 15x15 grid for the 480 crop; the BioViL-T
    # encoder's released grid shape is 14x14 for its 448 crop.
    assert scores.shape == (crop_size // 32, crop_size // 32)
    assert np.isfinite(scores).all()
    assert scores.min() >= -1.00001 and scores.max() <= 1.00001
    left, top, right, bottom = result["coverage_xyxy_original_pixels"]
    assert 0 < left < right < width
    assert 0 < top < bottom < height

    xs, ys, observed_box, reference_score = _assert_phrase_matches_official(
        grounding_demo_image,
        GROUNDING_DEMO_PHRASE,
        result,
        tmp_path,
        model_id,
    )

    # The notebook plots this released ground-truth box, so the phrase peak is
    # the executable form of the demo's localization claim.
    box_x, box_y, box_width, box_height = GROUNDING_DEMO_BOX_XYWH
    box = (
        (ys[:, None] >= box_y)
        & (ys[:, None] < box_y + box_height)
        & (xs[None, :] >= box_x)
        & (xs[None, :] < box_x + box_width)
    )
    assert box.any() and not box.all()
    peak = np.unravel_index(scores.argmax(), scores.shape)
    assert box[peak], "released demo ground-truth box must contain the peak"
    inside, outside = scores[box], scores[~box]
    # CNR is the paper's threshold-free localization metric (section 4.1).
    cnr = float(
        abs(inside.mean() - outside.mean()) / np.sqrt(inside.var() + outside.var())
    )
    auc = float(
        ((inside[:, None] > outside) + 0.5 * (inside[:, None] == outside)).mean()
    )
    return {
        "source": grounding_demo_image,
        "grounding": result,
        "metrics": {
            "source_image": GROUNDING_DEMO_URL,
            "source_image_sha256": GROUNDING_DEMO_SHA256,
            "source_notebook": (
                BIOVIL_RELEASE_ROOT + "/notebooks/phrase_grounding.ipynb"
            ),
            "phrase": GROUNDING_DEMO_PHRASE,
            "ground_truth_box_xywh_original_pixels": list(GROUNDING_DEMO_BOX_XYWH),
            "coverage_xyxy_original_pixels": list(
                result["coverage_xyxy_original_pixels"]
            ),
            "reference_observed_extent_xyxy": observed_box,
            "reference_global_cosine_similarity": reference_score,
            "global_cosine_similarity": phrase["global_cosine_similarity"],
            "patch_grid_shape": [int(scores.shape[0]), int(scores.shape[1])],
            "peak_xy_original_pixels": [
                float(xs[peak[1]]),
                float(ys[peak[0]]),
            ],
            "peak_inside_ground_truth_box": bool(box[peak]),
            "grid_auc_ground_truth_box": auc,
            "cnr_ground_truth_box": cnr,
            "cnr_definition": (
                "arXiv:2204.09817 section 4.1; a per-image value, not the "
                "MS-CXR cohort average"
            ),
        },
    }


@skip_on_import_error
@save_returns(unpack=True)
def test_biovil_reproduces_released_zero_shot_pneumonia(
    zero_shot_image: bytes, tmp_path: Path
):
    """Reproduce one released zero-shot pneumonia label.

    The released test averages each prompt group's unnormalized text
    embeddings and then normalizes the mean. The tool reports one cosine per
    phrase, so this check averages those cosines within each group and
    separately matches every phrase cosine against the released engine.

    The released test overrides the crop to 512 pixels; the tool uses the
    released ``get_image_inference`` default of 448 for BioViL-T, which the
    same release requires for its 14x14 temporal grid. The label direction is
    unchanged by that difference.
    """
    model_id = "microsoft/BiomedVLP-BioViL-T"
    phrases = ZERO_SHOT_POSITIVE + ZERO_SHOT_NEGATIVE
    result = biovil_ground_phrases(zero_shot_image, phrases, model_id=model_id)
    scores = np.asarray(
        [phrase["global_cosine_similarity"] for phrase in result["phrases"]]
    )
    assert scores.shape == (len(phrases),)
    ranked = [scores[index] for index in result["ranked_phrase_indices"]]
    assert ranked == sorted(ranked, reverse=True)

    reference_scores = _official_global_cosines(
        zero_shot_image, phrases, model_id, tmp_path
    )
    max_difference = float(np.abs(reference_scores - scores).max())
    assert max_difference <= 1e-4

    positive = float(scores[: len(ZERO_SHOT_POSITIVE)].mean())
    negative = float(scores[len(ZERO_SHOT_POSITIVE) :].mean())
    assert (positive > negative) == ZERO_SHOT_PNEUMONIA, (
        "released zero-shot pneumonia label not reproduced"
    )
    return {
        "metrics": {
            "source_image": ZERO_SHOT_URL,
            "source_image_sha256": ZERO_SHOT_SHA256,
            "source_test": (
                BIOVIL_RELEASE_ROOT
                + "/test_multimodal/vlp/test_zero_shot_classification.py"
            ),
            "model_id": model_id,
            "pneumonia": ZERO_SHOT_PNEUMONIA,
            "positive_prompts": ZERO_SHOT_POSITIVE,
            "negative_prompts": ZERO_SHOT_NEGATIVE,
            "positive_prompt_mean_similarity": positive,
            "negative_prompt_mean_similarity": negative,
            "max_official_cosine_difference": max_difference,
        }
    }


@pytest.mark.cuda
@pytest.mark.parametrize("model_id", get_args(BioViLModel))
@skip_on_import_error
@save_returns("biovil_nih_pneumothorax_grounding", unpack=True)
def test_biovil_nih_pneumothorax_grounding_matches_released_engine(
    nih_pneumothorax_box, tmp_path: Path, model_id: BioViLModel
):
    """Compare grounding with a released NIH pneumothorax annotation.

    The tool's similarity map and global cosine are asserted against the
    official engine on the same image and phrase. Localization metrics
    against the released boxes are reported, not asserted: the model cards do
    not claim per-image localization on NIH radiographs.
    """
    image, boxes, finding, source_url, annotation_sha = nih_pneumothorax_box
    result = biovil_ground_phrases(image, [finding], model_id=model_id)
    phrase = result["phrases"][0]
    scores = np.asarray(phrase["patch_similarity_map"])
    assert np.isfinite(scores).all()

    xs, ys, observed_box, reference_score = _assert_phrase_matches_official(
        image, finding, result, tmp_path, model_id
    )

    target = np.zeros(scores.shape, dtype=bool)
    for x0, y0, x1, y1 in boxes:
        target |= (
            (ys[:, None] >= y0)
            & (ys[:, None] < y1)
            & (xs[None, :] >= x0)
            & (xs[None, :] < x1)
        )
    assert target.any() and not target.all(), (
        "Expected both annotated and background grid cells"
    )
    foreground, background = scores[target], scores[~target]
    auc = float(
        (
            (foreground[:, None] > background)
            + 0.5 * (foreground[:, None] == background)
        ).mean()
    )
    peak_y, peak_x = np.unravel_index(scores.argmax(), scores.shape)
    return {
        "source": image,
        "grounding": result,
        "metrics": {
            "source_image": source_url,
            "source_annotation": "NIH BBox_List_2017.csv.gz bundled with TorchXRayVision",
            "annotation_sha256": annotation_sha,
            "source_patient_id": source_url.rsplit("/", 1)[1].split("_")[0],
            "model_training_image_source": "MIMIC-CXR according to model cards",
            "training_source_overlap": "NIH is not listed as an image pretraining source",
            "finding": finding,
            "boxes_xyxy_original_pixels": boxes,
            "reference_observed_extent_xyxy": observed_box,
            "reference_global_cosine_similarity": reference_score,
            "peak_xy_original_pixels": [float(xs[peak_x]), float(ys[peak_y])],
            "peak_inside_released_box": bool(target[peak_y, peak_x]),
            "grid_box_auc": auc,
            "annotated_grid_cells": int(target.sum()),
            "background_grid_cells": int((~target).sum()),
            "global_cosine_similarity": phrase["global_cosine_similarity"],
            "negative_note": "Outside-box grid cells are localization background, not verified healthy tissue",
        },
    }
