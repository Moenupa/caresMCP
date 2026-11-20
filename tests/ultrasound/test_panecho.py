"""Check PanEcho against its released model and task semantics.

The JAMA 2025 report (doi:10.1001/jama.2025.8731) claims a median AUROC of
0.91 across diagnostic tasks and ejection fraction MAE of 4.2% internally and
4.5% externally. Those cohorts (Yale New Haven and external health systems)
are not publicly released, so the cohort scores are not reproduced here.
Instead, two official EchoNet-Dynamic examples are passed through
``panecho_predict_study`` while the released ``MultiTaskModel`` output is
captured, and the study predictions are derived from those outputs using the
authors' own activations (``src/models.py``), the GLS sign convention
(``content/tasks.md``), and the equal-weight study mean (``src/utils.py``).
"""

from statistics import fmean
from typing import Any, cast

import pytest

from caresmcp.api.decorators import save_returns, skip_on_exception
from caresmcp.api.models import get_torch_model
from caresmcp.ultrasound import panecho
from caresmcp.ultrasound.panecho import (
    _PANECHO_NAME,
    _PANECHO_REPOSITORY,
    panecho_predict_study,
)

# Binary heads that the released MultiTaskModel flips so their scalar denotes
# the positive class; copied from
# https://github.com/CarDS-Yale/PanEcho/blob/836c46e/src/models.py.
OFFICIAL_FLIPPED_BINARY_TASKS = frozenset(
    {
        "MVStenosis",
        "AVStructure",
        "RASize",
        "RVSystolicFunction",
        "LVWallThickness-increased-modsev",
        "LVWallThickness-increased-any",
        "pericardial-effusion",
    }
)
# Two of the three released EchoNet-Dynamic function groups keep the study
# aggregation test at two inferences.
STUDY_EXAMPLES = ("normal", "low_ef")


def _released_task_value(task: Any, output: Any) -> float | dict[str, float]:
    """Derive a task value from a released MultiTaskModel output."""
    values = output.detach().float().cpu().reshape(-1).tolist()
    if task.task_type == "regression":
        # tasks.md: the actual GLS value is -1 times the model output.
        return -values[0] if task.task_name == "GLS" else values[0]

    class_names = [str(label) for label in task.class_names]
    if task.task_type == "binary_classification":
        # The released model flips these heads so the scalar denotes the
        # positive class, which is class_names[0] for the listed tasks and
        # class_names[1] for every other binary head.
        flipped = task.task_name in OFFICIAL_FLIPPED_BINARY_TASKS
        first = values[0] if flipped else 1 - values[0]
        return {class_names[0]: first, class_names[1]: 1 - first}
    return dict(zip(class_names, values, strict=True))


@pytest.mark.slow
@skip_on_exception(ImportError, OSError)
@save_returns()
def test_study_matches_released_tasks(echonet_examples, monkeypatch):
    """Derive study predictions from the released model's own outputs."""
    model = get_torch_model(_PANECHO_REPOSITORY, _PANECHO_NAME, tasks="all")
    outputs: list[dict[str, Any]] = []
    hook = model.register_forward_hook(
        lambda _model, _inputs, output: outputs.append(output)
    )
    monkeypatch.setattr(panecho, "get_torch_model", lambda *args, **kwargs: model)
    videos = [echonet_examples[category] for category in STUDY_EXAMPLES]
    try:
        study = panecho_predict_study(videos)
    finally:
        hook.remove()

    assert len(outputs) == len(videos)
    assert study["video_count"] == len(videos)
    assert study["aggregation"] == "equal_weight_mean"

    tasks = cast(list[Any], model.tasks)
    task_by_name = {task.task_name: task for task in tasks}
    per_video: list[dict[str, Any]] = [
        {
            name: _released_task_value(task_by_name[name], output)
            for name, output in video_output.items()
        }
        for video_output in outputs
    ]
    expected: dict[str, Any] = {}
    for name, first_value in per_video[0].items():
        if isinstance(first_value, dict):
            expected[name] = {
                label: fmean(predictions[name][label] for predictions in per_video)
                for label in first_value
            }
        else:
            expected[name] = fmean(predictions[name] for predictions in per_video)

    assert study["predictions"].keys() == expected.keys()
    for name, expected_value in expected.items():
        if isinstance(expected_value, dict):
            assert study["predictions"][name].keys() == expected_value.keys()
            for label, value in expected_value.items():
                assert study["predictions"][name][label] == pytest.approx(
                    value, rel=1e-5, abs=1e-6
                )
        else:
            assert study["predictions"][name] == pytest.approx(
                expected_value, rel=1e-5, abs=1e-6
            )
    return study
