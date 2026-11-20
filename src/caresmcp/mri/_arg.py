"""Shared input annotations for MRI tools."""

from typing import Annotated

from pydantic import Field

MriSlice = Annotated[
    bytes,
    Field(min_length=1, description="Encoded grayscale or RGB 2D MRI slice image."),
]
MriNativeSlice = Annotated[
    bytes,
    Field(
        min_length=1,
        description="Floating-point TIFF slice preserving native MRI intensities.",
    ),
]
