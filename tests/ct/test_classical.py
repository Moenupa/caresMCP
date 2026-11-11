"""Check the classical CT tools on real acquisitions and save reviewable outputs."""

from dataclasses import dataclass
from io import BytesIO

import cv2
import numpy as np
import pytest
from PIL import Image

from caresmcp.api.decorators import save_returns
from caresmcp.ct.classical import (
    CTMotionModel,
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
@save_returns("ct_classical")
def test_window_ct_slice(sample: CTSample, center: float, width: float):
    """Window calibrated HU values using the DICOM LINEAR convention."""
    output = window_ct_slice(
        image_or_dicom=sample.dicom, window_center=center, window_width=width
    )

    lower = center - 0.5 - (width - 1) / 2
    upper = center - 0.5 + (width - 1) / 2
    expected = np.interp(sample.hu, [lower, upper], [0, 255]).round()
    np.testing.assert_array_equal(image_pixels(output), expected)
    return output


@save_returns("ct_classical", unpack=True)
def test_segment_ct_hu_range(sample: CTSample, tissue_mask: bytes):
    """Select the requested HU interval and optionally retain its largest region."""
    minimum, maximum = SOFT_TISSUE_HU
    expected = (sample.hu >= minimum) & (sample.hu <= maximum)
    np.testing.assert_array_equal(
        image_pixels(tissue_mask), expected.astype(np.uint8) * 255
    )

    largest_mask = segment_ct_hu_range(
        sample.dicom, minimum, maximum, keep_largest_component=True
    )
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        expected.astype(np.uint8), connectivity=8
    )
    largest_label = 1 + stats[1:, cv2.CC_STAT_AREA].argmax()
    np.testing.assert_array_equal(
        image_pixels(largest_mask) > 0, labels == largest_label
    )
    return {"all_regions": tissue_mask, "largest_region": largest_mask}


@save_returns("ct_classical", unpack=True)
def test_measure_ct_hu_regions(sample: CTSample, tissue_mask: bytes):
    """Region geometry, HU means, and physical areas describe the selected pixels."""
    result = measure_ct_hu_regions(
        sample.dicom, *SOFT_TISSUE_HU, maximum_regions=sample.hu.size
    )
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
    assert (
        limited["regions"]
        == sorted(regions, key=lambda region: region["area_pixels"], reverse=True)[
            :limit
        ]
    )
    assert limited["qualifying_region_count"] == len(regions)
    assert limited["regions_truncated"] == (len(regions) > limit)
    return {"all_regions": result, "limited": limited}


@pytest.mark.parametrize("motion_model", ["translation", "rigid", "affine"])
@save_returns("ct_classical", unpack=True)
def test_register_ct_slices(ct_heart_image: bytes, motion_model: CTMotionModel):
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
        fixed_image_or_dicom=ct_heart_image,
        moving_image_or_dicom=image_bytes(moving),
        motion_model=motion_model,
    )

    recovered = np.asarray(result["moving_to_fixed_matrix"])
    aligned = cv2.warpAffine(moving, recovered, (width, height))
    # Check displacement across the image, without relying on matrix coefficients
    # whose error depends on rotation center and image dimensions.
    landmarks = np.array([[0.2, 0.2], [0.8, 0.2], [0.5, 0.8]]) * [width, height]
    homogeneous = np.column_stack([landmarks, np.ones(len(landmarks))])
    moved = homogeneous @ forward.T
    restored = np.column_stack([moved, np.ones(len(moved))]) @ recovered.T
    np.testing.assert_allclose(restored, landmarks, atol=0.5)
    return {
        "transform": result,
        "fixed": image_bytes(fixed),
        "moving": image_bytes(moving),
        "aligned": image_bytes(aligned),
    }
