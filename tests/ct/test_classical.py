"""Check the classical CT tools on real acquisitions and save reviewable outputs."""

from dataclasses import dataclass
from io import BytesIO
from typing import Literal

import cv2
import numpy as np
import pytest
from PIL import Image

from caresmcp.api.decorators.file_io import save_results
from caresmcp.constants import CACHE_ROOT
from caresmcp.ct.classical import (
    CTMotionModel,
    denoise_ct_slice,
    evaluate_ct_segmentation,
    measure_ct_hu_regions,
    register_ct_slices,
    segment_ct_hu_range,
    window_ct_slice,
)

pytestmark = pytest.mark.slow
SOFT_TISSUE_HU = (-100, 200)


@dataclass
class CTSample:
    name: str
    dicom: bytes
    hu: np.ndarray
    spacing: tuple[float, float]

    def save(self, result: bytes | dict, tool: str, variant: str = "result") -> None:
        """Keep each acquisition's artifacts in its own directory."""
        save_results(result, CACHE_ROOT / "ct_classical" / self.name / tool / variant)


@pytest.fixture(scope="module", params=["ct_lung_cancer_dicom", "ct_lung_nodule_dicom"])
def sample(request: pytest.FixtureRequest) -> CTSample:
    import pydicom
    from pydicom.pixels import apply_modality_lut

    data = request.getfixturevalue(request.param)
    dataset = pydicom.dcmread(BytesIO(data))
    return CTSample(
        name=request.param,
        dicom=data,
        hu=np.asarray(
            apply_modality_lut(dataset.pixel_array, dataset), dtype=np.float32
        ),
        spacing=(float(dataset.PixelSpacing[0]), float(dataset.PixelSpacing[1])),
    )


def image_pixels(data: bytes) -> np.ndarray:
    with Image.open(BytesIO(data)) as image:
        return np.array(image)


def image_bytes(pixels: np.ndarray, format: str = "PNG") -> bytes:
    output = BytesIO()
    Image.fromarray(pixels).save(output, format=format)
    return output.getvalue()


@pytest.fixture
def tissue_mask(sample: CTSample) -> bytes:
    return segment_ct_hu_range(sample.dicom, *SOFT_TISSUE_HU)


@pytest.mark.parametrize(
    "center,width", [(40, 400), (-600, 1500)], ids=["soft_tissue", "lung"]
)
def test_window_ct_slice(sample: CTSample, center: float, width: float) -> None:
    """Window calibrated HU values using the DICOM LINEAR convention."""
    output = window_ct_slice(sample.dicom, window_center=center, window_width=width)
    sample.save(output, "window_ct_slice", f"center_{center}_width_{width}")

    lower = center - 0.5 - (width - 1) / 2
    upper = center - 0.5 + (width - 1) / 2
    expected = np.interp(sample.hu, [lower, upper], [0, 255]).round()
    np.testing.assert_array_equal(image_pixels(output), expected)


def test_segment_ct_hu_range(sample: CTSample, tissue_mask: bytes) -> None:
    """Select the requested HU interval and optionally retain its largest region."""
    sample.save(tissue_mask, "segment_ct_hu_range", "all_regions")
    minimum, maximum = SOFT_TISSUE_HU
    expected = (sample.hu >= minimum) & (sample.hu <= maximum)
    np.testing.assert_array_equal(
        image_pixels(tissue_mask), expected.astype(np.uint8) * 255
    )

    largest_mask = segment_ct_hu_range(
        sample.dicom, minimum, maximum, keep_largest_component=True
    )
    sample.save(largest_mask, "segment_ct_hu_range", "largest_region")
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        expected.astype(np.uint8), connectivity=8
    )
    largest_label = 1 + stats[1:, cv2.CC_STAT_AREA].argmax()
    np.testing.assert_array_equal(
        image_pixels(largest_mask) > 0, labels == largest_label
    )


def test_measure_ct_hu_regions(sample: CTSample, tissue_mask: bytes) -> None:
    """Region geometry, HU means, and physical areas describe the selected pixels."""
    result = measure_ct_hu_regions(
        sample.dicom, *SOFT_TISSUE_HU, maximum_regions=sample.hu.size
    )
    sample.save(result, "measure_ct_hu_regions", "all_regions")
    regions = result["regions"]
    _, labels = cv2.connectedComponents(image_pixels(tissue_mask), connectivity=8)
    assert sum(region["area_pixels"] for region in regions) == np.count_nonzero(labels)
    assert result["pixel_spacing_row_column_mm"] == pytest.approx(sample.spacing)

    for region in regions:
        rows, columns = np.nonzero(labels == region["label"])
        assert region["bounding_box_xywh"] == [
            columns.min(),
            rows.min(),
            np.ptp(columns) + 1,
            np.ptp(rows) + 1,
        ]
        assert region["centroid_xy"] == pytest.approx([columns.mean(), rows.mean()])
        assert region["mean_hu"] == pytest.approx(
            sample.hu[rows, columns].mean(), abs=1e-4
        )
        assert region["area_mm2"] == pytest.approx(len(rows) * np.prod(sample.spacing))

    # A limited response must identify omissions and keep the largest regions.
    limit = max(1, len(regions) // 2)
    limited = measure_ct_hu_regions(
        sample.dicom, *SOFT_TISSUE_HU, maximum_regions=limit
    )
    sample.save(limited, "measure_ct_hu_regions", "limited")
    assert (
        limited["regions"]
        == sorted(regions, key=lambda region: region["area_pixels"], reverse=True)[
            :limit
        ]
    )
    assert limited["qualifying_region_count"] == len(regions)
    assert limited["regions_truncated"] == (len(regions) > limit)


@pytest.mark.parametrize("method", ["bilateral", "gaussian", "median"])
def test_denoise_ct_slice(
    sample: CTSample, method: Literal["bilateral", "gaussian", "median"]
) -> None:
    """Smooth real CT data while preserving calibrated float values across formats."""
    output = denoise_ct_slice(sample.dicom, method=method)
    sample.save(output, "denoise_ct_slice", method)
    filtered = image_pixels(output)
    assert filtered.shape == sample.hu.shape
    assert filtered.dtype == np.float32

    # Local variation should decrease without quantizing HU to display intensities.
    def local_variation(values: np.ndarray) -> float:
        return float(
            np.mean(np.abs(np.diff(values, axis=0)))
            + np.mean(np.abs(np.diff(values, axis=1)))
        )

    assert local_variation(filtered) < local_variation(sample.hu)
    slope, intercept = 0.5, -1024.0
    stored_values = (sample.hu - intercept) / slope
    calibrated_tiff = image_bytes(stored_values, format="TIFF")
    raster_output = denoise_ct_slice(
        calibrated_tiff,
        method=method,
        raster_units="hu",
        rescale_slope=slope,
        rescale_intercept=intercept,
    )
    np.testing.assert_array_equal(filtered, image_pixels(raster_output))


@pytest.mark.parametrize("motion_model", ["translation", "rigid", "affine"])
def test_register_ct_slices(ct_heart_image: bytes, motion_model: CTMotionModel) -> None:
    """Recover a known motion of real anatomy in the documented moving-to-fixed direction."""
    fixed = image_pixels(ct_heart_image)
    height, width = fixed.shape
    forward = cv2.getRotationMatrix2D(
        (width / 2, height / 2), 0 if motion_model == "translation" else 2, 1
    )
    forward[:, 2] += [4, -3]
    if motion_model == "affine":
        forward[0, 1] += 0.01
    moving = cv2.warpAffine(fixed, forward, (width, height))
    result = register_ct_slices(
        ct_heart_image, image_bytes(moving), motion_model=motion_model
    )
    output_dir = (
        CACHE_ROOT
        / "ct_classical"
        / "ct_heart_image"
        / "register_ct_slices"
        / motion_model
    )
    save_results(result, output_dir / "transform")
    save_results(image_bytes(fixed), output_dir / "fixed")
    save_results(image_bytes(moving), output_dir / "moving")

    recovered = np.asarray(result["moving_to_fixed_matrix"])
    aligned = cv2.warpAffine(moving, recovered, (width, height))
    save_results(image_bytes(aligned), output_dir / "aligned")
    # Check displacement across the image, without relying on matrix coefficients
    # whose error depends on rotation center and image dimensions.
    landmarks = np.array([[0.2, 0.2], [0.8, 0.2], [0.5, 0.8]]) * [width, height]
    homogeneous = np.column_stack([landmarks, np.ones(len(landmarks))])
    moved = homogeneous @ forward.T
    restored = np.column_stack([moved, np.ones(len(moved))]) @ recovered.T
    np.testing.assert_allclose(restored, landmarks, atol=0.5)


def test_evaluate_ct_segmentation(sample: CTSample, tissue_mask: bytes) -> None:
    """Score an incomplete prediction against a real HU-selected reference."""
    reference = image_pixels(tissue_mask) > 0
    prediction = reference.copy()
    prediction[:, prediction.shape[1] // 2 :] = False
    prediction_bytes = image_bytes(prediction.astype(np.uint8) * 255)
    result = evaluate_ct_segmentation(prediction_bytes, tissue_mask)
    sample.save(prediction_bytes, "evaluate_ct_segmentation", "prediction")
    sample.save(tissue_mask, "evaluate_ct_segmentation", "reference")
    sample.save(result, "evaluate_ct_segmentation", "metrics")

    # Every predicted pixel is correct, but only part of the reference is covered.
    coverage = prediction.sum() / reference.sum()
    assert 0 < coverage < 1
    assert result == pytest.approx(
        {
            "precision": 1,
            "recall": coverage,
            "intersection_over_union": coverage,
            "dice": 2 * coverage / (1 + coverage),
        }
    )
