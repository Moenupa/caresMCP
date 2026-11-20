"""Check classical CT tools against released DICOM semantics and reference code.

The window check compares with pydicom's implementation of the DICOM LINEAR
VOI LUT function, and segmentation and region geometry are checked against
independent SciPy connected-component labeling. The input is the CT slice
released by the pydicom project, so pixel data, HU calibration, and geometry
are all upstream artifacts.
"""

from io import BytesIO

import numpy as np
import pytest

from caresmcp.api.decorators import save_returns
from caresmcp.api.io import decode_as
from caresmcp.ct.classical import (
    measure_ct_hu_regions,
    segment_ct_hu_range,
    window_ct_slice,
)

pytestmark = pytest.mark.slow
SOFT_TISSUE_HU = (-100, 200)


@pytest.fixture(scope="module")
def sample(ct_dicom: bytes) -> tuple[np.ndarray, tuple[float, float]]:
    """Return calibrated HU values and row/column spacing from the release."""
    import pydicom
    from pydicom.pixels import apply_modality_lut

    dataset = pydicom.dcmread(BytesIO(ct_dicom))
    hu = np.asarray(apply_modality_lut(dataset.pixel_array, dataset), dtype=np.float32)
    spacing = (float(dataset.PixelSpacing[0]), float(dataset.PixelSpacing[1]))
    return hu, spacing


@pytest.fixture(scope="module")
def tissue_mask(ct_dicom: bytes) -> bytes:
    return segment_ct_hu_range(ct_dicom, *SOFT_TISSUE_HU)


def _dicom_linear_window(hu: np.ndarray, center: float, width: float) -> np.ndarray:
    """Apply the DICOM LINEAR function with pydicom's released implementation.

    The window is applied to already calibrated HU, so the reference dataset
    declares an unsigned 8-bit output range and no modality rescaling.
    """
    from pydicom.dataset import Dataset
    from pydicom.pixels import apply_windowing

    window = Dataset()
    window.PhotometricInterpretation = "MONOCHROME2"
    window.BitsStored = 8
    window.PixelRepresentation = 0
    window.WindowCenter = center
    window.WindowWidth = width
    return np.rint(apply_windowing(hu.astype(np.float64), window)).astype(np.uint8)


@pytest.mark.parametrize(
    "center,width", [(40, 400), (-600, 1500)], ids=["soft_tissue", "lung"]
)
@save_returns("ct_classical")
def test_window_ct_slice(
    ct_dicom: bytes,
    sample: tuple[np.ndarray, tuple[float, float]],
    center: float,
    width: float,
):
    """Window calibrated HU values using the DICOM LINEAR convention."""
    output = window_ct_slice(
        image_or_dicom=ct_dicom, window_center=center, window_width=width
    )

    hu, _ = sample
    np.testing.assert_array_equal(
        np.array(decode_as(output)), _dicom_linear_window(hu, center, width)
    )
    return output


@save_returns("ct_classical", unpack=True)
def test_segment_ct_hu_range(
    ct_dicom: bytes,
    sample: tuple[np.ndarray, tuple[float, float]],
    tissue_mask: bytes,
):
    """Select the requested HU interval and optionally retain its largest region."""
    from scipy import ndimage

    minimum, maximum = SOFT_TISSUE_HU
    hu, _ = sample
    expected = (hu >= minimum) & (hu <= maximum)
    np.testing.assert_array_equal(
        np.array(decode_as(tissue_mask)), expected.astype(np.uint8) * 255
    )

    largest_mask = segment_ct_hu_range(
        ct_dicom, minimum, maximum, keep_largest_component=True
    )
    labels, component_count = ndimage.label(expected, structure=np.ones((3, 3)))
    assert component_count > 0
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest_labels = np.flatnonzero(sizes == sizes.max())
    observed = np.array(decode_as(largest_mask)) > 0
    assert any(np.array_equal(observed, labels == label) for label in largest_labels)
    return {"all_regions": tissue_mask, "largest_region": largest_mask}


@save_returns("ct_classical", unpack=True)
def test_measure_ct_hu_regions(
    ct_dicom: bytes,
    sample: tuple[np.ndarray, tuple[float, float]],
):
    """Region geometry, HU means, and physical areas describe the selected pixels."""
    from scipy import ndimage

    minimum, maximum = SOFT_TISSUE_HU
    hu, spacing = sample
    expected_mask = (hu >= minimum) & (hu <= maximum)
    labels, component_count = ndimage.label(expected_mask, structure=np.ones((3, 3)))
    expected_regions = []
    for label in range(1, component_count + 1):
        rows, columns = np.nonzero(labels == label)
        area = int(rows.size)
        expected_regions.append(
            {
                "area_pixels": area,
                "bounding_box_xywh": [
                    int(columns.min()),
                    int(rows.min()),
                    int(np.ptp(columns) + 1),
                    int(np.ptp(rows) + 1),
                ],
                "centroid_xy": [float(columns.mean()), float(rows.mean())],
                "mean_hu": float(hu[rows, columns].astype(np.float64).mean()),
                "area_mm2": float(area * np.prod(spacing)),
            }
        )

    result = measure_ct_hu_regions(ct_dicom, minimum, maximum, maximum_regions=hu.size)
    regions = result["regions"]
    assert result["pixel_spacing_row_column_mm"] == pytest.approx(spacing)
    assert result["qualifying_region_count"] == len(expected_regions)
    assert result["regions_truncated"] is False
    assert len(regions) == len(expected_regions)

    def order(region: dict) -> tuple:
        return -region["area_pixels"], region["bounding_box_xywh"]

    for region, reference in zip(
        sorted(regions, key=order), sorted(expected_regions, key=order), strict=True
    ):
        assert region["area_pixels"] == reference["area_pixels"]
        assert region["bounding_box_xywh"] == reference["bounding_box_xywh"]
        assert region["centroid_xy"] == pytest.approx(
            reference["centroid_xy"], abs=1e-6
        )
        assert region["mean_hu"] == pytest.approx(reference["mean_hu"], abs=1e-4)
        assert region["area_mm2"] == pytest.approx(reference["area_mm2"])

    # A limited response keeps the largest regions and reports the omissions.
    limit = max(1, len(regions) // 2)
    limited = measure_ct_hu_regions(ct_dicom, minimum, maximum, maximum_regions=limit)
    assert limited["regions"] == regions[:limit]
    assert limited["qualifying_region_count"] == len(regions)
    assert limited["regions_truncated"] == (len(regions) > limit)
    return {"all_regions": result, "limited": limited}
