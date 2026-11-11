"""Volumetric CT and MR anatomy segmentation with TotalSegmentator."""

import gzip
import logging
import struct
import zlib
from typing import Annotated, Literal

from pydantic import Field

from caresmcp.api.decorators import requires
from caresmcp.constants import DEVICE
from caresmcp.registry import MEDAI, TASK_SEG

logger = logging.getLogger(__name__)


@MEDAI.register("totalsegmentator_segment_anatomy")
@TASK_SEG.register("totalsegmentator_segment_anatomy")
@requires(["totalsegmentator", "nibabel"], extra="totalsegmentator")
def totalsegmentator_segment_anatomy(
    image: Annotated[
        bytes, Field(description="Three-dimensional NIfTI-1 volume (.nii or .nii.gz)")
    ],
    task: Annotated[
        Literal["total", "total_mr"], Field(description="CT or MR anatomy task")
    ] = "total",
    fast: Annotated[
        bool, Field(description="Use the faster, lower-resolution model")
    ] = False,
    roi_subset: Annotated[
        list[str] | None,
        Field(min_length=1, description="Anatomical names to segment, or null for all"),
    ] = None,
    force_split: Annotated[
        bool,
        Field(description="Split large volumes into three parts to reduce memory use"),
    ] = False,
) -> bytes:
    """Segment anatomical structures in a three-dimensional CT or MR volume.

    Returns a gzip-compressed NIfTI segmentation on the input spatial grid.
    Zero denotes background; the extended NIfTI header maps positive labels to
    anatomical names.

    References:
        Wasserthal et al., TotalSegmentator, Radiology: Artificial Intelligence,
        2023. https://doi.org/10.1148/ryai.230024
        https://github.com/wasserth/TotalSegmentator#python-api
    """
    import nibabel as nib
    from nibabel.filebasedimages import ImageFileError
    from nibabel.spatialimages import HeaderDataError
    from nibabel.wrapstruct import WrapStructError
    from totalsegmentator.map_to_binary import class_map
    from totalsegmentator.python_api import totalsegmentator

    if roi_subset is not None and any(
        name not in class_map[task].values() for name in roi_subset
    ):
        raise ValueError("roi_subset contains names unsupported by the selected task")

    try:
        payload = gzip.decompress(image) if image.startswith(b"\x1f\x8b") else image
        volume = nib.Nifti1Image.from_bytes(payload)
    except (
        OSError,
        EOFError,
        ValueError,
        struct.error,
        zlib.error,
        ImageFileError,
        HeaderDataError,
        WrapStructError,
    ):
        raise ValueError("image must contain a readable NIfTI-1 volume") from None
    if len(volume.shape) != 3 or any(size < 1 for size in volume.shape):
        raise ValueError("image must contain a non-empty three-dimensional volume")

    try:
        segmentation = totalsegmentator(
            volume,
            output=None,
            ml=True,
            task=task,
            fast=fast,
            roi_subset=roi_subset,
            device="gpu" if DEVICE == "cuda" else DEVICE,
            force_split=force_split,
            nr_thr_resamp=1,
            nr_thr_saving=1,
            quiet=True,
        )
        return gzip.compress(segmentation.to_bytes(), mtime=0)
    except Exception:
        logger.exception(
            "TotalSegmentator inference or output serialization failed for task %s",
            task,
        )
        raise RuntimeError("TotalSegmentator could not segment the volume") from None
