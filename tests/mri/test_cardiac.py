"""Reproduce MONAI's published ventricular short-axis segmentation exactly.

The bundle ships the Sunnybrook SC-N-2 frame ``docs/SC-N-2-3-0.npy`` and
``docs/visualise.ipynb`` segments that frame with the released checkpoint.
This module feeds the same pixels to the tool and compares the result with a
reference run of the notebook's own normalization, softmax, and argmax path.
"""

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from caresmcp.api.decorators import (
    save_returns,
    skip_on_exception,
    skip_on_import_error,
)
from caresmcp.api.decorators.bytes_io import _read_bytes
from caresmcp.api.io import decode_as, encode_as, encode_label_mask
from caresmcp.api.models import get_hf_checkpoint
from caresmcp.constants import DEVICE
from caresmcp.mri.cardiac import (
    MRI_MONAI_CARDIAC_IMAGE_SIZE,
    MRI_MONAI_CARDIAC_LABELS,
    segment_cardiac_mri_slice,
)

pytestmark = pytest.mark.slow

MODEL_ID = "MONAI/ventricular_short_axis_3label"
MODEL_REVISION = "0.3.5"

# The example frame loaded by the bundle author's visualise.ipynb, from
# case SC-N-2 of the Sunnybrook Cardiac Dataset, pinned to the model revision.
SHORT_AXIS_URL = (
    "https://huggingface.co/MONAI/ventricular_short_axis_3label/"
    "resolve/0.3.5/docs/SC-N-2-3-0.npy"
)

# Class pixel counts recorded from the published notebook run (CPU, float32).
EXPECTED_LABEL_PIXELS = {0: 62381, 1: 1010, 2: 863, 3: 1282}


@pytest.fixture(scope="module")
def published_frame() -> np.ndarray:
    """Load the bundle's published 256-square short-axis frame."""
    frame = np.load(BytesIO(_read_bytes(SHORT_AXIS_URL)), allow_pickle=False)
    assert frame.shape == (MRI_MONAI_CARDIAC_IMAGE_SIZE,) * 2
    # The frame is integer-valued with a zero minimum, so a 16-bit PNG
    # preserves every pixel and, being min-max scaled, every model input.
    assert np.array_equal(frame, np.rint(frame))
    assert 0 <= frame.min() and frame.max() <= np.iinfo(np.uint16).max
    return frame


@pytest.fixture(scope="module")
def published_slice(published_frame: np.ndarray) -> bytes:
    """Encode the published frame losslessly as a 16-bit grayscale PNG."""
    return encode_as(Image.fromarray(published_frame.astype(np.uint16)))


def _notebook_segmentation(frame: np.ndarray) -> np.ndarray:
    """Run the published notebook's inference steps on the released model."""
    import torch
    from monai.networks.nets import UNet

    # The bundle's network_def, as instantiated by visualise.ipynb.
    network = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=4,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
    checkpoint = get_hf_checkpoint(MODEL_ID, "models/model.pt", revision=MODEL_REVISION)
    network.load_state_dict(checkpoint)
    network = network.to(DEVICE).eval()
    inputd = torch.from_numpy(frame / frame.max()).float().to(DEVICE)
    with torch.inference_mode():
        # Softmax preserves the argmax but mirrors the notebook exactly.
        probabilities = torch.softmax(network(inputd[None, None])[0], dim=0)
        return torch.argmax(probabilities, dim=0).cpu().numpy()


@skip_on_exception(ImportError, OSError)
@save_returns("mri_cardiac", unpack=True)
def test_segment_cardiac_mri_slice_reproduces_published_example(
    published_frame: np.ndarray, published_slice: bytes
) -> dict[str, object]:
    """Reproduce the bundle's published segmentation at the pixel level."""
    mask_bytes = segment_cardiac_mri_slice(published_slice)
    mask = decode_as(mask_bytes)
    assert mask.mode == "P"
    assert mask.size == (MRI_MONAI_CARDIAC_IMAGE_SIZE,) * 2
    assert mask.info["transparency"] == 0

    labels = np.asarray(mask)
    assert {int(label) for label in np.unique(labels)} == set(MRI_MONAI_CARDIAC_LABELS)

    # Pin the published result, then require an exact match with the reference.
    counts = {
        int(label): int(count)
        for label, count in zip(*np.unique(labels, return_counts=True), strict=True)
    }
    assert counts == EXPECTED_LABEL_PIXELS

    published_mask = _notebook_segmentation(published_frame)
    assert np.array_equal(labels, published_mask)

    return {
        "source": published_slice,
        "mask": mask_bytes,
        "published_mask": encode_label_mask(published_mask),
    }


@skip_on_import_error
def test_segment_cardiac_mri_slice_rejects_wrong_size(published_slice: bytes) -> None:
    """Reject the published frame when it is not the pinned input size."""
    source = decode_as(published_slice)
    half = MRI_MONAI_CARDIAC_IMAGE_SIZE // 2
    wrong_size = encode_as(source.resize((half, half), Image.Resampling.BILINEAR))
    with pytest.raises(ValueError, match="256 by 256"):
        segment_cardiac_mri_slice(wrong_size)
