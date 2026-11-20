"""Check CT radiomics features against calibrated HU and released CT inputs."""

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from caresmcp.api.decorators import save_returns, skip_on_import_error
from caresmcp.api.io import decode_as, encode_as
from caresmcp.ct.pyradiomics import radiomics_ct_slice

pytestmark = pytest.mark.slow
SOFT_TISSUE_HU = (-100, 200)
EXPECTED_CLASSES = {"firstorder", "glcm", "gldm", "glrlm", "glszm", "ngtdm"}


@pytest.fixture(scope="module")
def dicom_sample(ct_dicom: bytes) -> tuple[np.ndarray, bytes]:
    """Return calibrated HU and an independently built soft-tissue ROI mask."""
    import pydicom
    from pydicom.pixels import apply_modality_lut

    dataset = pydicom.dcmread(BytesIO(ct_dicom))
    hu = np.asarray(apply_modality_lut(dataset.pixel_array, dataset), dtype=np.float64)
    mask = (hu >= SOFT_TISSUE_HU[0]) & (hu <= SOFT_TISSUE_HU[1])
    assert mask.any()
    return hu, encode_as(Image.fromarray(mask.astype(np.uint8) * 255))


@skip_on_import_error
@save_returns("ct_pyradiomics")
def test_extract_dicom_slice_radiomics(
    ct_dicom: bytes, dicom_sample: tuple[np.ndarray, bytes]
) -> dict:
    """Radiomics summarize the ROI using calibrated DICOM spacing."""
    import pydicom

    hu, roi_mask = dicom_sample
    result = radiomics_ct_slice(ct_dicom, roi_mask)
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

    dataset = pydicom.dcmread(BytesIO(ct_dicom))
    assert result["pixel_spacing_row_column_mm"] == pytest.approx(
        [float(value) for value in dataset.PixelSpacing]
    )
    mask = np.asarray(decode_as(roi_mask)) > 0
    assert features["original_firstorder_Mean"] == pytest.approx(
        hu[mask].mean(), rel=1e-5
    )
    assert features["original_firstorder_Minimum"] == pytest.approx(hu[mask].min())
    assert features["original_firstorder_Maximum"] == pytest.approx(hu[mask].max())
    return result


@skip_on_import_error
@save_returns("ct_pyradiomics")
def test_extract_dicom_slice_radiomics_with_spacing(
    ct_dicom: bytes, dicom_sample: tuple[np.ndarray, bytes]
) -> dict:
    """Explicit spacing is reported as provided instead of acquisition spacing."""
    _, roi_mask = dicom_sample
    spacing = [2.5, 1.25]
    result = radiomics_ct_slice(ct_dicom, roi_mask, pixel_spacing_row_column_mm=spacing)
    assert result["pixel_spacing_row_column_mm"] == spacing
    assert result["pixel_spacing_source"] == "provided"
    assert np.isfinite(list(result["features"].values())).all()
    return result


@skip_on_import_error
@save_returns("ct_pyradiomics")
def test_extract_raster_slice_radiomics(ct_raster: bytes) -> dict:
    """A calibrated raster without spacing metadata is measured in assumed pixels."""
    slope, intercept = 2.0, -1024.0
    hu = np.asarray(decode_as(ct_raster)).astype(np.float32) * slope + intercept
    mask = (hu >= -1024) & (hu <= -800)
    assert mask.any()
    roi_mask = encode_as(Image.fromarray(mask.astype(np.uint8) * 255))

    result = radiomics_ct_slice(
        ct_raster,
        roi_mask,
        rescale_slope=slope,
        rescale_intercept=intercept,
        raster_units="hu",
    )
    assert result["intensity_units"] == "hu"
    assert result["pixel_spacing_row_column_mm"] == [1.0, 1.0]
    assert result["pixel_spacing_source"] == "assumed"
    assert result["features"]["original_firstorder_Mean"] == pytest.approx(
        hu[mask].astype(np.float64).mean(), rel=1e-5
    )
    return result
