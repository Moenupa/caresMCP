"""Shared input annotations for CT tools."""

from typing import Annotated, Literal

from pydantic import Field

from .io import CTRasterUnits

CuriaCTPlane = Literal["axial", "coronal", "sagittal"]
CTClassifier = Literal[
    "oohtmeel/swin-tiny-patch4-finetuned-lung-cancer-ct-scans",
    "sukhmani1303/lung-cancer-vit-model",
]
CuriaCTTask = Literal[
    "ct_anatomy_classification",
    "covid_pneumonia_classification",
    "intracranial_hemorrhage_classification",
]

Mask = Annotated[
    bytes, Field(min_length=1, description="Single-frame binary mask image.")
]
MinHU = Annotated[
    float, Field(allow_inf_nan=False, description="Inclusive minimum HU.")
]
MaxHU = Annotated[
    float, Field(allow_inf_nan=False, description="Inclusive maximum HU.")
]
WindowWidth = Annotated[
    float, Field(ge=1, allow_inf_nan=False, description="Window width in HU.")
]
CTImage = Annotated[
    bytes,
    Field(
        min_length=1,
        description="Single CT DICOM slice or grayscale raster image.",
    ),
]
RasterUnits = Annotated[
    CTRasterUnits,
    Field(
        description="Use hu only for rasters with known HU calibration; display exports cannot recover HU values."
    ),
]
HURasterUnits = Annotated[
    CTRasterUnits,
    Field(
        description="Use hu only for rasters with known HU calibration; display rasters are rejected."
    ),
]
WindowCenter = Annotated[
    float, Field(allow_inf_nan=False, description="Window center in HU.")
]
PositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
PixelSpacing = Annotated[
    list[PositiveFloat],
    Field(
        min_length=2, max_length=2, description="Row and column spacing in millimeters."
    ),
]
RasterRescaleSlope = Annotated[
    float,
    Field(
        allow_inf_nan=False,
        description="Raster stored-value multiplier; ignored for DICOM. With HU units, HU = stored value * slope + intercept.",
    ),
]
RasterRescaleIntercept = Annotated[
    float,
    Field(
        allow_inf_nan=False,
        description="Raster stored-value offset; ignored for DICOM. With HU units, HU = stored value * slope + intercept.",
    ),
]
SegmentCtHuRangeClosingKernelSizeArg = Annotated[
    int,
    Field(ge=0, description="Square closing width in pixels; zero disables closing."),
]
SegmentCtHuRangeKeepLargestComponentArg = Annotated[
    bool,
    Field(description="Keep only the largest 8-connected foreground component."),
]
MeasureCtHuRegionsMinimumAreaPixelsArg = Annotated[
    int, Field(ge=1, description="Minimum connected component area in pixels.")
]
MeasureCtHuRegionsMaximumRegionsArg = Annotated[
    int,
    Field(ge=1, description="Maximum regions returned, ordered by decreasing area."),
]
ClassifyCtSliceWithCuriaAnatomicalPlaneArg = Annotated[
    CuriaCTPlane, Field(description="Anatomical plane of the supplied CT slice.")
]
ClassifyCtHeadHemorrhagePreviousImageArg = Annotated[
    bytes,
    Field(description="Immediately preceding CT slice; empty at the series boundary."),
]
ClassifyCtHeadHemorrhageNextImageArg = Annotated[
    bytes,
    Field(description="Immediately following CT slice; empty at the series boundary."),
]
ExtractCtSliceRadiomicsPixelSpacingRowColumnMmArg = PixelSpacing | None
