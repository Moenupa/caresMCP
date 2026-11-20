"""Compare DALE-CT-2S embeddings with the released upstream implementation.

The model card reports linear-probing MIL AUROC on gated CT-RATE and
RAD-ChestCT scans (0.8247 in-domain; 0.6252/0.7389 external), which cannot be
reproduced from the released checkpoint alone. This test instead reproduces
the preprocessing and checkpoint load in the release's ``load_hf/lejepa_2s.py``
(https://github.com/Kentucky-Open-Science/DALE-CT) on the pydicom project's
CQ500 slice and requires the global feature to match the upstream
implementation.
"""

import functools
from io import BytesIO

import numpy as np
import pytest

from caresmcp.api.decorators import save_returns, skip_on_exception
from caresmcp.constants import DEVICE
from caresmcp.ct.dale_ct_2s import dale_ct_2s_image_embed

pytestmark = pytest.mark.slow
MODEL_ID = "hf-hub:Kentucky-Open-Science/DALE-CT-2S"
REPO_ID = "Kentucky-Open-Science/DALE-CT-2S"
CLIP_MIN_HU = -997.0
CLIP_MAX_HU = 888.0
MEAN_HU = -142.39
STD_HU = 360.97


@functools.cache
def _released_model():
    """Load the checkpoint exactly as the release's ``lejepa_2s.py`` does."""
    import timm
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    model = timm.create_model(
        "vit_large_patch14_dinov2",
        pretrained=False,
        num_classes=0,
        in_chans=1,
        patch_size=16,
        img_size=512,
        dynamic_img_size=True,
    )
    state = load_file(hf_hub_download(REPO_ID, "model.safetensors"))
    model.load_state_dict(state, strict=False)
    return model.eval().to(DEVICE)


def _released_preprocess(hu: np.ndarray):
    """Apply the release's HU clipping, scaling, and training normalization."""
    import torch

    values = torch.as_tensor(hu, dtype=torch.float32)
    if values.ndim == 2:
        values = values.unsqueeze(0)
    value_range = CLIP_MAX_HU - CLIP_MIN_HU
    normalized_mean = (MEAN_HU - CLIP_MIN_HU) / value_range
    normalized_std = STD_HU / value_range
    values = torch.clamp(values, CLIP_MIN_HU, CLIP_MAX_HU)
    values = (values - CLIP_MIN_HU) / value_range
    values = (values - normalized_mean) / normalized_std
    return values.unsqueeze(0).to(DEVICE)


@skip_on_exception(ImportError, OSError)
@save_returns("ct_dale_ct_2s")
def test_dicom_embedding_matches_release(ct_dicom: bytes):
    """Reproduce the released global feature for the same calibrated slice."""
    import pydicom
    import torch
    from pydicom.pixels import apply_modality_lut

    dataset = pydicom.dcmread(BytesIO(ct_dicom))
    hu = np.asarray(apply_modality_lut(dataset.pixel_array, dataset), dtype=np.float32)

    with torch.inference_mode():
        released = _released_model()(_released_preprocess(hu))[0].float().cpu()
    result = dale_ct_2s_image_embed(ct_dicom)
    observed = torch.tensor(result["embedding"])

    assert result["model_id"] == MODEL_ID
    assert result["embedding_type"] == "global"
    assert result["image_shape_hw"] == [512, 512]
    torch.testing.assert_close(observed, released, rtol=1e-5, atol=1e-6)
    return result
