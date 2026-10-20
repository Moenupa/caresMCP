"""Run TotalSegmentator on real NIfTI volumes from data_subset10."""

import gzip
from pathlib import Path

import pytest

from caresmcp.constants import CACHE_ROOT
from caresmcp.medical_ai.totalsegmentator import totalsegmentator_segment_anatomy

SAMPLE_DIRECTORY = Path(__file__).resolve().parents[2] / "data_subset10"
SAMPLE_PATHS = tuple(
    sorted(
        path
        for path in SAMPLE_DIRECTORY.rglob("*")
        if path.is_file()
        and (path.name.endswith(".nii") or path.name.endswith(".nii.gz"))
    )
)


@pytest.mark.slow
@pytest.mark.cuda
@pytest.mark.parametrize(
    "sample_path",
    SAMPLE_PATHS,
    ids=[str(path.relative_to(SAMPLE_DIRECTORY)) for path in SAMPLE_PATHS],
)
def test_totalsegmentator(sample_path: Path) -> None:
    """Segment real volumes and verify the resulting spatial grid and labels."""
    nib = pytest.importorskip("nibabel")
    np = pytest.importorskip("numpy")
    pytest.importorskip("totalsegmentator")
    from totalsegmentator.map_to_binary import class_map

    source = nib.load(sample_path)
    task = (
        "total_mr"
        if "mr" in str(sample_path.relative_to(SAMPLE_DIRECTORY)).lower()
        else "total"
    )
    result = totalsegmentator_segment_anatomy(
        image=sample_path.read_bytes(), task=task, fast=True, device="gpu"
    )
    assert isinstance(result, bytes)
    segmentation = nib.Nifti1Image.from_bytes(gzip.decompress(result))
    assert segmentation.shape == source.shape
    np.testing.assert_allclose(segmentation.affine, source.affine)
    labels = np.unique(segmentation.get_fdata())
    assert np.any(labels > 0), "Expected anatomical structures in the real volume"
    assert set(labels).issubset({0, *class_map[task]})
    assert segmentation.header.extensions, "Expected anatomical label metadata"

    output_path = (
        CACHE_ROOT / "totalsegmentator" / sample_path.relative_to(SAMPLE_DIRECTORY)
    )
    output_path = (
        output_path.with_suffix(".nii.gz")
        if output_path.suffix == ".nii"
        else output_path
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(result)
