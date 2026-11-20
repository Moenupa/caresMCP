"""Classical pixel-grid registration for reconstructed MRI slices."""

from typing import Annotated, Literal

import numpy as np
from pydantic import Field

from caresmcp.api.decorators import requires
from caresmcp.api.io import encode_intensity_slice
from caresmcp.registry import MRI

from ._arg import MriSlice
from ._io import _decode_mri_slice


@MRI.register("register_mri_slices")
@requires("SimpleITK", extra="itk")
def register_mri_slices(
    fixed_image: MriSlice,
    moving_image: MriSlice,
    transform: Annotated[
        Literal["rigid", "affine"],
        Field(description="Rigid rotation/translation or affine 2D transform."),
    ] = "rigid",
    max_iterations: Annotated[
        int,
        Field(ge=1, le=1000, description="Maximum optimizer iterations."),
    ] = 100,
) -> bytes:
    """TODO: no demonstrated diagnostic or benchmark value; remove or replace.

    Align a moving 2D MRI slice to a fixed reference slice.

    Registration estimates a transform between two reconstructed slices; it
    cannot be inferred from one image alone. The moving slice is resampled on
    the fixed slice's pixel grid using Mattes mutual information and linear
    interpolation. Pixel spacing and anatomical orientation are unavailable
    in ordinary PNG/JPEG inputs, so this operation aligns image pixels only.

    Returns:
        Grayscale PNG (unsigned) or TIFF (signed/float) bytes of the
        resampled moving slice on the fixed grid.

    See Also:
        https://simpleitk.readthedocs.io/en/main/registrationOverview.html
    """
    import SimpleITK as sitk

    fixed_pixels, _, _, _ = _decode_mri_slice(fixed_image)
    moving_pixels, moving_minimum, moving_maximum, moving_dtype = _decode_mri_slice(
        moving_image
    )
    for name, pixels in (("fixed", fixed_pixels), ("moving", moving_pixels)):
        if min(pixels.shape) < 8:
            raise ValueError(f"{name} MRI slice must be at least 8 by 8 pixels")
        if float(pixels.max()) == float(pixels.min()):
            raise ValueError(f"{name} MRI slice must have varying pixel intensities")

    fixed = sitk.GetImageFromArray(fixed_pixels.astype(np.float32))
    moving = sitk.GetImageFromArray(moving_pixels.astype(np.float32))
    initial_transform = (
        sitk.Euler2DTransform() if transform == "rigid" else sitk.AffineTransform(2)
    )

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.2, seed=17)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=0.5,
        minStep=0.001,
        numberOfIterations=max_iterations,
        relaxationFactor=0.5,
        gradientMagnitudeTolerance=1e-6,
    )
    registration.SetOptimizerScalesFromPhysicalShift()

    levels = min(3, max(1, int(np.log2(min(fixed.GetSize() + moving.GetSize()) / 8))))
    shrink_factors = [2**exponent for exponent in range(levels - 1, -1, -1)]
    registration.SetShrinkFactorsPerLevel(shrinkFactors=shrink_factors)
    registration.SetSmoothingSigmasPerLevel(
        smoothingSigmas=[float(factor // 2) for factor in shrink_factors]
    )
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    try:
        initialized = sitk.CenteredTransformInitializer(
            fixed,
            moving,
            initial_transform,
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
        registration.SetInitialTransform(initialized, inPlace=False)
        estimated_transform = registration.Execute(fixed, moving)
        aligned = sitk.Resample(
            moving,
            fixed,
            estimated_transform,
            sitk.sitkLinear,
            0.0,
            sitk.sitkFloat32,
        )
        aligned_pixels = sitk.GetArrayFromImage(aligned)
    except RuntimeError as exc:
        raise ValueError("MRI slice registration failed") from exc

    return encode_intensity_slice(
        np.asarray(aligned_pixels, dtype=np.float32),
        moving_minimum,
        moving_maximum,
        moving_dtype,
    )
