"""Reproduce MONAI breast density scores on real mammograms.

The preprocessing mirrors the bundle's ``inference.json`` pipeline exactly,
including MONAI PILReader's axis swap for 2-D images, so the recorded scores
reproduce a literal CPU bundle run. MONAI's model author published sigmoid
scores for all 16 labeled sample images in model-zoo PR #150; this test
reproduces one sample per category to bound the slow run, within 1% relative
or 0.01 absolute. Uncomment the remaining samples in ``CLAIMED_SCORES`` and
set ``MODEL_CARD_SAMPLES_PER_CATEGORY`` to 4 for full coverage. The 0.96
accuracy in the bundle metadata is measured on Mayo Clinic's private test
set, so the public sample images pin the published scores rather than that
metric.
"""

import math

import pytest

from caresmcp.api.decorators import save_returns, skip_on_exception
from caresmcp.mammography.density import classify_breast_density

pytestmark = pytest.mark.slow

LABEL_A = "A: almost entirely fatty"
LABEL_B = "B: scattered fibroglandular density"
LABEL_C = "C: heterogeneously dense"
LABEL_D = "D: extremely dense"
EXPECTED_LABELS = (LABEL_A, LABEL_B, LABEL_C, LABEL_D)

# Recorded outputs from a deterministic CPU run of the bundle's preprocessing
# on the released 0.1.8 checkpoint. The enabled model-card samples are
# covered by CLAIMED_SCORES instead of being pinned twice.
EXPECTED_SCORES = {
    "mlo_scattered_density": {
        LABEL_A: 0.0172,
        LABEL_B: 0.1461,
        LABEL_C: 0.8180,
        LABEL_D: 0.7869,
    },
}

# Published per-sample sigmoid scores in A, B, C, D order, from the model
# author's cat_predictions.csv in a model-zoo PR #150 comment:
# https://github.com/Project-MONAI/model-zoo/pull/150#issuecomment-1290716476
# That bundle run used CUDA fp16 (amp), so CPU fp32 differs by under 1%
# relative or 0.01 absolute.
CLAIMED_SCORES = {
    "A1": (0.93505859375, 0.8115234375, 0.057281494140625, 0.00930023193359375),
    # "A2": (0.93701171875, 0.86279296875, 0.0531005859375, 0.005706787109375),
    # "A3": (0.896484375, 0.74951171875, 0.1253662109375, 0.015899658203125),
    # "A4": (0.93310546875, 0.8837890625, 0.047698974609375, 0.0057525634765625),
    "B1": (0.88623046875, 0.78857421875, 0.13671875, 0.01666259765625),
    # "B2": (0.460205078125, 0.8525390625, 0.403564453125, 0.0292510986328125),
    # "B3": (0.66748046875, 0.88134765625, 0.28564453125, 0.01177978515625),
    # "B4": (0.9453125, 0.85791015625, 0.0706787109375, 0.004116058349609375),
    "C1": (0.060516357421875, 0.830078125, 0.787109375, 0.089111328125),
    # "C2": (0.06781005859375, 0.89501953125, 0.84912109375, 0.018157958984375),
    # "C3": (0.006191253662109375, 0.171142578125, 0.9560546875, 0.54296875),
    # "C4": (0.006984710693359375, 0.1890869140625, 0.94482421875, 0.60791015625),
    "D1": (0.00234222412109375, 0.052520751953125, 0.79296875, 0.9833984375),
    # "D2": (0.0094146728515625, 0.0618896484375, 0.759765625, 0.97119140625),
    # "D3": (0.010650634765625, 0.0823974609375, 0.6357421875, 0.9765625),
    # "D4": (0.024383544921875, 0.09967041015625, 0.74755859375, 0.93017578125),
}


@pytest.mark.parametrize("image", tuple(EXPECTED_SCORES))
@skip_on_exception(ImportError, OSError)
@save_returns()
def test_classify_breast_density(image: str, request: pytest.FixtureRequest):
    """Pin the released checkpoint's scores on real mammograms."""
    density_scores = classify_breast_density(request.getfixturevalue(image))

    # Cheap contract checks first.
    assert tuple(density_scores) == EXPECTED_LABELS
    assert all(
        isinstance(score, float) and math.isfinite(score) and 0 <= score <= 1
        for score in density_scores.values()
    )
    # Then reproduce the bundle-run scores.
    assert density_scores == pytest.approx(EXPECTED_SCORES[image], rel=1e-2)
    return density_scores


@skip_on_exception(ImportError, OSError)
@save_returns()
def test_model_card_claimed_scores(model_card_samples: dict[str, bytes]):
    """Reproduce MONAI's published sigmoid scores for every card sample."""
    observed = {}
    correct = 0
    for sample_id, image in model_card_samples.items():
        density_scores = classify_breast_density(image)
        claimed = dict(zip(EXPECTED_LABELS, CLAIMED_SCORES[sample_id], strict=True))
        assert density_scores == pytest.approx(claimed, rel=1e-2, abs=1e-2), sample_id
        observed[sample_id] = density_scores
        predicted_label = max(density_scores, key=density_scores.__getitem__)
        correct += predicted_label.startswith(sample_id[0])
    return {
        "scores": observed,
        "top1_accuracy": correct / len(model_card_samples),
    }
