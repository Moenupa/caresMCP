"""Exercise the released FleXray bundle on a real radiograph."""

from io import BytesIO
from typing import get_args

import numpy as np
import pytest
from PIL import Image

from caresmcp.api.decorators import save_returns, skip_on_import_error
from caresmcp.api.models.flexray import get_flexray_model
from caresmcp.xray.flexray import AnatomicalRegion, flexray_segment_anatomy

pytestmark = [pytest.mark.slow, pytest.mark.cuda]


@pytest.mark.parametrize("region", ["lungs", "heart", "clavicles"])
@skip_on_import_error
@save_returns("flexray")
def test_flexray_segment_anatomy(cxr_image: bytes, region: AnatomicalRegion):
    """Verify independent masks, released label order, and input-sized geometry."""
    model = get_flexray_model()
    assert model.label_names == ("background", *get_args(AnatomicalRegion))
    assert model.subfolders == ("members/flux0375",)
    result = flexray_segment_anatomy(cxr_image, region)
    with Image.open(BytesIO(cxr_image)) as source, Image.open(BytesIO(result)) as mask:
        assert mask.size == source.size
        assert mask.mode == "L"
        pixels = np.asarray(mask)
        assert set(np.unique(pixels)) == {0, 255}
    return result


@skip_on_import_error
@save_returns("flexray", unpack=True)
def test_flexray_wrist_anatomy(wrist_example):
    """Exercise a non-square pediatric wrist with independent radius/ulna masks."""
    image, _ = wrist_example
    results = {
        name: flexray_segment_anatomy(image, name) for name in ("radii", "ulnae")
    }
    with Image.open(BytesIO(image)) as source:
        for result in results.values():
            with Image.open(BytesIO(result)) as mask:
                assert mask.size == source.size
                assert set(np.unique(np.asarray(mask))) == {0, 255}
    return {"source": image, **results}
