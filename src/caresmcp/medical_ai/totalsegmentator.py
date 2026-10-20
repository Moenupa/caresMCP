"""Volumetric CT and MR anatomy segmentation with TotalSegmentator."""

import gzip
import logging
import struct
import zlib
from typing import Literal

from caresmcp.constants import DEVICE
from caresmcp.registry import MEDAI, TAG_CT, TAG_MRI
from caresmcp.util.moduleloader import requires

logger = logging.getLogger(__name__)


@MEDAI.register("totalsegmentator_segment_anatomy")
@TAG_CT.register("totalsegmentator_segment_anatomy")
@TAG_MRI.register("totalsegmentator_segment_anatomy")
@requires(["totalsegmentator", "nibabel"], extra="totalsegmentator")
def totalsegmentator_segment_anatomy(
    image: bytes,
    task: Literal["total", "total_mr"] = "total",
    fast: bool = False,
    roi_subset: list[str] | None = None,
    device: Literal["auto", "cpu", "gpu", "mps"] = "auto",
    force_split: bool = False,
) -> bytes:
    """Segment anatomical structures in a three-dimensional CT or MR volume.

    Args:
        image: Complete NIfTI-1 volume bytes, uncompressed (.nii) or gzip
            compressed (.nii.gz). Two-dimensional pictures and DICOM are not
            supported.
        task: ``total`` for CT anatomy or ``total_mr`` for MR anatomy.
        fast: Use the lower-resolution model for faster inference.
        roi_subset: Anatomical names to segment, such as ``["liver", "spleen"]``,
            or None for all structures supported by the selected task.
        device: Inference device; auto uses the server's configured device.
        force_split: Process the volume in three parts to reduce memory use.
            Intended for large volumes; avoid using it with small volumes.

    Returns:
        Gzip-compressed NIfTI-1 segmentation bytes (.nii.gz) on the input volume's
        spatial grid. Zero denotes background; positive integer labels identify
        anatomy using TotalSegmentator's task-specific label map. The extended
        NIfTI header contains the anatomical names.

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

    if task not in ("total", "total_mr"):
        raise ValueError("task must be total or total_mr")
    if device not in ("auto", "cpu", "gpu", "mps"):
        raise ValueError("device must be auto, cpu, gpu, or mps")
    if roi_subset is not None:
        if (
            not isinstance(roi_subset, list)
            or not roi_subset
            or any(not isinstance(name, str) for name in roi_subset)
        ):
            raise ValueError("roi_subset must be a non-empty list of anatomical names")
        if any(name not in class_map[task].values() for name in roi_subset):
            raise ValueError(
                "roi_subset contains names unsupported by the selected task"
            )

    if not isinstance(image, bytes) or not image:
        raise ValueError("image must contain a non-empty NIfTI-1 volume")
    try:
        payload = gzip.decompress(image) if image.startswith(b"\x1f\x8b") else image
        volume = nib.Nifti1Image.from_bytes(payload)
        # Materialize the data before inference to reject truncated payloads.
        volume.get_fdata()
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

    resolved_device = (
        ("gpu" if DEVICE == "cuda" else DEVICE) if device == "auto" else device
    )
    try:
        segmentation = totalsegmentator(
            volume,
            output=None,
            ml=True,
            task=task,
            fast=fast,
            roi_subset=roi_subset,
            device=resolved_device,
            force_split=force_split,
            nr_thr_resamp=1,
            nr_thr_saving=1,
            quiet=True,
        )
        assert isinstance(segmentation, nib.Nifti1Image), (
            f"TotalSegmentator returned {type(segmentation).__name__}; expected Nifti1Image"
        )
        assert segmentation.shape == volume.shape, (
            f"TotalSegmentator output shape {segmentation.shape} differs from input {volume.shape}"
        )
        return gzip.compress(segmentation.to_bytes(), mtime=0)
    except Exception:
        logger.exception(
            "TotalSegmentator inference or output serialization failed for task %s",
            task,
        )
        raise RuntimeError("TotalSegmentator could not segment the volume") from None
