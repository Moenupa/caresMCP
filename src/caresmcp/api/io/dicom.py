"""Decode DICOM images into stored pixel frames and technical metadata.

Only image data and acquisition metadata leave this module; patient, study,
and series identifiers are discarded. Pixel values are returned as stored, so
modality LUT application and VOI windowing remain explicit caller decisions.

References:
    DICOM PS3.5, Pixel Data. https://dicom.nema.org/medical/dicom/current/output/html/part05.html
    DICOM PS3.3 C.11.1, Modality LUT. https://dicom.nema.org/medical/dicom/current/output/html/part03.html
"""

import math
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, Any

import filetype
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pydicom.dataset import Dataset


def is_dicom(image: bytes) -> bool:
    """Return True when the bytes carry the DICOM Part 10 file marker."""
    return filetype.guess_mime(image) == "application/dicom"


@dataclass(frozen=True)
class DicomImage:
    """Stored pixel frames and technical metadata decoded from one DICOM file.

    Fields are frozen; the underlying NumPy arrays remain mutable. Pixel
    layout is ``(frames, rows, columns)`` for monochrome images and
    ``(frames, rows, columns, samples)`` for color images, with color converted
    to RGB by pydicom. Values are stored values; ``modality_values`` holds
    ``float32`` values after the modality LUT when requested.
    """

    pixels: NDArray[Any]
    modality_values: NDArray[np.float32] | None
    photometric_interpretation: str
    modality: str | None
    sop_class_uid: str | None
    number_of_frames: int
    bits_stored: int | None
    pixel_representation: int | None
    rescale_slope: float | None
    rescale_intercept: float | None
    rescale_type: str | None
    has_modality_lut: bool
    modality_lut_type: str | None
    pixel_spacing_row_column_mm: tuple[float, float] | None
    image_orientation_patient: tuple[float, ...] | None
    image_position_patient: tuple[float, ...] | None
    slice_thickness_mm: float | None

    @property
    def is_monochrome(self) -> bool:
        """Whether the photometric interpretation is a monochrome format."""
        return self.photometric_interpretation.upper().startswith("MONOCHROME")

    def monochrome_frame(
        self, index: int = 0, *, invert_monochrome1: bool = True
    ) -> NDArray[Any]:
        """Return one stored monochrome frame.

        Args:
            index: Zero-based frame index.
            invert_monochrome1: Convert MONOCHROME1 stored values to the
                MONOCHROME2 convention using the declared ``BitsStored``.

        Returns:
            A two-dimensional frame of stored pixel values.

        Raises:
            ValueError: If the image is not monochrome, the index is out of
                range, or MONOCHROME1 inversion has no ``BitsStored``.
        """
        if not self.is_monochrome:
            raise ValueError("DICOM image must be monochrome")
        if not 0 <= index < self.number_of_frames:
            raise ValueError("DICOM frame index is out of range")
        frame = self.pixels[index]
        if (
            invert_monochrome1
            and self.photometric_interpretation.upper() == "MONOCHROME1"
        ):
            if self.bits_stored is None:
                raise ValueError("MONOCHROME1 DICOM image requires BitsStored")
            frame = ((1 << self.bits_stored) - 1) - frame.astype(np.int64)
        return frame


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _float(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return converted if math.isfinite(converted) else None


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _finite_sequence(value: Any, length: int) -> tuple[float, ...] | None:
    if value is None:
        return None
    try:
        converted = tuple(float(item) for item in value)
    except (TypeError, ValueError, OverflowError):
        return None
    if len(converted) != length or not all(math.isfinite(item) for item in converted):
        return None
    return converted


def _positive_sequence(value: Any, length: int) -> tuple[float, ...] | None:
    converted = _finite_sequence(value, length)
    if converted is None or any(item <= 0 for item in converted):
        return None
    return converted


def _positive_float(value: Any) -> float | None:
    converted = _float(value)
    return converted if converted is not None and converted > 0 else None


def _stack_frames(raw: NDArray[Any], photometric: str) -> NDArray[Any]:
    """Normalize pydicom pixel arrays to (frames, rows, columns[, samples])."""
    if photometric.upper().startswith("MONOCHROME"):
        if raw.ndim == 2:
            return raw[None, ...]
        if raw.ndim == 3:
            return raw
        raise ValueError("DICOM monochrome pixel data is not a supported frame layout")
    if raw.ndim == 3:
        return raw[None, ...]
    if raw.ndim == 4:
        return raw
    raise ValueError("DICOM color pixel data is not a supported frame layout")


def _frame_count(dataset: "Dataset", frames: NDArray[Any]) -> int:
    declared = _int(dataset.get("NumberOfFrames"))
    number_of_frames = 1 if declared is None else declared
    if number_of_frames < 1 or number_of_frames != frames.shape[0]:
        raise ValueError("DICOM image declares an inconsistent frame count")
    return number_of_frames


def _modality_lut_type(value: Any) -> str | None:
    if not value:
        return None
    try:
        return _text(value[0].get("ModalityLUTType"))
    except (AttributeError, IndexError, TypeError):
        return None


def decode_dicom(image: bytes, *, modality_lut: bool = False) -> DicomImage:
    """Decode one DICOM image into stored frames and technical metadata.

    Args:
        image: Complete DICOM Part 10 file bytes.
        modality_lut: Also populate ``modality_values`` by applying the
            modality LUT from ``RescaleSlope``/``RescaleIntercept`` or
            ``ModalityLUTSequence``. Color images cannot be rescaled.

    Returns:
        Stored pixel frames and the metadata needed by modality adapters.
        Metadata that the file does not declare is returned as None.

    Raises:
        ValueError: If the bytes are not a readable DICOM image with pixel
            data, or if the pixel data layout is inconsistent.
    """
    import pydicom

    try:
        dataset = pydicom.dcmread(BytesIO(image))
    except Exception:  # noqa: BLE001
        raise ValueError("input is not a readable DICOM image") from None

    if "PixelData" not in dataset:
        raise ValueError("DICOM image does not contain pixel data")
    try:
        raw = np.asarray(dataset.pixel_array)
    except Exception:  # noqa: BLE001
        raise ValueError("DICOM image does not contain readable pixel data") from None

    photometric = _text(dataset.get("PhotometricInterpretation"))
    if photometric is None:
        raise ValueError("DICOM image must declare its PhotometricInterpretation")
    frames = _stack_frames(raw, photometric)
    number_of_frames = _frame_count(dataset, frames)
    if frames.dtype.kind == "f" and not np.isfinite(frames).all():
        raise ValueError("DICOM float pixel data must be finite")

    modality_values = None
    if modality_lut:
        if not photometric.upper().startswith("MONOCHROME"):
            raise ValueError("DICOM modality LUT requires a monochrome image")
        from pydicom.pixels import apply_modality_lut

        try:
            rescaled = np.asarray(apply_modality_lut(raw, dataset), dtype=np.float32)
        except Exception:  # noqa: BLE001
            raise ValueError("DICOM modality LUT could not be applied") from None
        modality_values = _stack_frames(rescaled, photometric)

    spacing_values = _positive_sequence(dataset.get("PixelSpacing"), 2)
    spacing = (
        (spacing_values[0], spacing_values[1]) if spacing_values is not None else None
    )
    return DicomImage(
        pixels=frames,
        modality_values=modality_values,
        photometric_interpretation=photometric,
        modality=_text(dataset.get("Modality")),
        sop_class_uid=_text(dataset.get("SOPClassUID")),
        number_of_frames=number_of_frames,
        bits_stored=_int(dataset.get("BitsStored")),
        pixel_representation=_int(dataset.get("PixelRepresentation")),
        rescale_slope=_float(dataset.get("RescaleSlope")),
        rescale_intercept=_float(dataset.get("RescaleIntercept")),
        rescale_type=_text(dataset.get("RescaleType")),
        has_modality_lut=bool(dataset.get("ModalityLUTSequence")),
        modality_lut_type=_modality_lut_type(dataset.get("ModalityLUTSequence")),
        pixel_spacing_row_column_mm=spacing,
        image_orientation_patient=_finite_sequence(
            dataset.get("ImageOrientationPatient"), 6
        ),
        image_position_patient=_finite_sequence(dataset.get("ImagePositionPatient"), 3),
        slice_thickness_mm=_positive_float(dataset.get("SliceThickness")),
    )
