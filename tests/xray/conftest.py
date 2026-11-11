import json
from pathlib import Path

import pytest

from caresmcp.api.decorators.bytes_io import _read_bytes
from caresmcp.constants import CACHE_ROOT


@pytest.fixture(scope="module")
def cxr_image() -> bytes:
    # a hernia case
    return _read_bytes(
        "https://nih-chest-x-rays.s3.us-east-2.amazonaws.com/images_1024x1024/00000003_001.png"
    )


@pytest.fixture(scope="module")
def cxr_image_followup() -> bytes:
    # a hernia case
    return _read_bytes(
        "https://nih-chest-x-rays.s3.us-east-2.amazonaws.com/images_1024x1024/00000003_002.png"
    )


def _read_wrist_source(offset: int, image_id: str) -> tuple[bytes, dict]:
    """Cache genuine source data so signed asset URLs need not stay valid."""
    from tempfile import NamedTemporaryFile

    from curl_cffi import requests

    root = CACHE_ROOT / "wrist_reference"
    image_path, metadata_path = root / (image_id + ".png"), root / (image_id + ".json")
    if image_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        assert metadata["image_id"] == image_id, (
            "Cached reference image identity changed"
        )
        return image_path.read_bytes(), metadata
    url = (
        "https://datasets-server.huggingface.co/rows?"
        "dataset=tirandazdylan/open-radiograph-findings&config=default&split=test"
        f"&offset={offset}&length=1"
    )
    # Cold dataset-server rows take longer than ordinary byte downloads.
    response = requests.get(url, impersonate="chrome", timeout=60)
    response.raise_for_status()
    row = response.json()["rows"][0]["row"]
    assert row["image_id"] == image_id, (
        "Dataset row order changed; review source identity"
    )
    assert row["source"] == "graz"
    image = _read_bytes(row["image"]["src"])
    metadata = {key: value for key, value in row.items() if key != "image"}
    metadata["source_row_url"] = url
    root.mkdir(parents=True, exist_ok=True)
    # Multiple slow-test workers may fetch the same reference concurrently.
    for destination, content in (
        (image_path, image),
        (metadata_path, json.dumps(metadata).encode()),
    ):
        with NamedTemporaryFile(dir=root, delete=False) as temporary:
            temporary.write(content)
        Path(temporary.name).replace(destination)
    return image, metadata


@pytest.fixture(scope="module")
def wrist_source_reader():
    """Supply cached, identity-checked real GRAZPEDWRI-DX examples."""
    return _read_wrist_source


@pytest.fixture(scope="module", params=[0, 1])
def wrist_example(request, wrist_source_reader):
    """Fetch two original smoke-test projections and source annotations."""
    image_ids = (
        "0032_1146492991_01_WRI-R1_M007",
        "0032_1146493016_02_WRI-R2_M007",
    )
    return wrist_source_reader(request.param, image_ids[request.param])
