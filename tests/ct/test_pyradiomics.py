"""Exercise the public CT radiomics tool on real acquisitions."""

from io import BytesIO

import numpy as np
import pytest

from caresmcp.api.decorators import save_returns, skip_on_import_error
from caresmcp.api.io import decode_as
from caresmcp.ct.classical import segment_ct_hu_range
from caresmcp.ct.pyradiomics import radiomics_ct_slice

pytestmark = pytest.mark.slow
SOFT_TISSUE_HU = (-100, 200)
EXPECTED_CLASSES = {"firstorder", "glcm", "gldm", "glrlm", "glszm", "ngtdm"}


@pytest.fixture(params=["ct_lung_cancer_dicom", "ct_lung_nodule_dicom"])
def ct_sample(request: pytest.FixtureRequest) -> tuple[str, bytes]:
    return request.param, request.getfixturevalue(request.param)


@pytest.fixture
def roi_mask(ct_sample: tuple[str, bytes]) -> bytes:
    return segment_ct_hu_range(ct_sample[1], *SOFT_TISSUE_HU)


@skip_on_import_error
@save_returns("ct_pyradiomics")
def test_extract_dicom_slice_radiomics(
    ct_sample: tuple[str, bytes], roi_mask: bytes
) -> dict:
    """Radiomics summarize the mask ROI with calibrated DICOM spacing."""
    import pydicom
    from pydicom.pixels import apply_modality_lut

    _, dicom = ct_sample
    result = radiomics_ct_slice(dicom, roi_mask)
    features = result["features"]

    assert result["intensity_units"] == "hu"
    assert result["pixel_spacing_source"] == "dicom"
    assert result["bin_width"] == 25.0
    assert all(name.startswith("original_") for name in features)
    assert {name.split("_")[1] for name in features} == EXPECTED_CLASSES
    assert len(features) == 93
    assert "original_glrlm_RunVariance" in features
    assert "original_glszm_SizeZoneNonUniformity" in features
    assert "original_firstorder_StandardDeviation" not in features
    assert "original_gldm_GrayLevelNonUniformityNormalized" not in features
    assert np.isfinite(list(features.values())).all()

    dataset = pydicom.dcmread(BytesIO(dicom))
    hu = np.asarray(apply_modality_lut(dataset.pixel_array, dataset), dtype=np.float64)
    mask = np.asarray(decode_as(roi_mask)) > 0
    assert result["pixel_spacing_row_column_mm"] == pytest.approx(
        [float(dataset.PixelSpacing[0]), float(dataset.PixelSpacing[1])]
    )
    assert features["original_firstorder_Mean"] == pytest.approx(
        hu[mask].mean(), rel=1e-5
    )
    assert features["original_firstorder_Minimum"] == pytest.approx(hu[mask].min())
    assert features["original_firstorder_Maximum"] == pytest.approx(hu[mask].max())
    return result


@skip_on_import_error
@save_returns("ct_pyradiomics")
def test_extract_dicom_slice_radiomics_with_spacing(
    ct_sample: tuple[str, bytes], roi_mask: bytes
) -> dict:
    """Explicit spacing is reported as provided instead of acquisition spacing."""
    spacing = [2.5, 1.25]
    result = radiomics_ct_slice(
        ct_sample[1], roi_mask, pixel_spacing_row_column_mm=spacing
    )
    assert result["pixel_spacing_row_column_mm"] == spacing
    assert result["pixel_spacing_source"] == "provided"
    return result


@skip_on_import_error
@save_returns("ct_pyradiomics")
def test_extract_raster_slice_radiomics(ct_heart_image: bytes) -> dict:
    """A calibrated raster without spacing metadata is measured in assumed pixels."""
    slope, intercept = 2.0, -1024.0
    mask = segment_ct_hu_range(
        ct_heart_image,
        -1024,
        -500,
        rescale_slope=slope,
        rescale_intercept=intercept,
        raster_units="hu",
    )
    result = radiomics_ct_slice(
        ct_heart_image,
        mask,
        rescale_slope=slope,
        rescale_intercept=intercept,
        raster_units="hu",
    )
    assert result["intensity_units"] == "hu"
    assert result["pixel_spacing_row_column_mm"] == [1.0, 1.0]
    assert result["pixel_spacing_source"] == "assumed"
    assert result["features"]
    return result
