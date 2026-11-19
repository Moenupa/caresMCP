import json
from pathlib import Path

import pytest

from caresmcp.api.decorators import download_bytes
from caresmcp.constants import CACHE_ROOT


@pytest.fixture(scope="session")
def hernia_image() -> bytes:
    """First study of NIH patient 00000003, whose record includes a hernia."""
    return download_bytes(
        "https://nih-chest-x-rays.s3.us-east-2.amazonaws.com/images_1024x1024/00000003_000.png"
    )


@pytest.fixture(scope="session")
def cxr_image() -> bytes:
    """Second study of NIH patient 00000003."""
    return download_bytes(
        "https://nih-chest-x-rays.s3.us-east-2.amazonaws.com/images_1024x1024/00000003_001.png"
    )


@pytest.fixture(scope="session")
def cxr_image_followup() -> bytes:
    """Third study of the same patient, used as the later temporal image."""
    return download_bytes(
        "https://nih-chest-x-rays.s3.us-east-2.amazonaws.com/images_1024x1024/00000003_002.png"
    )


@pytest.fixture(scope="session")
def chest_dicom() -> bytes:
    """A real 1024x1024 PA chest radiograph DICOM from TorchXRayVision's tests.

    jsDelivr mirrors the file at the pinned tag; raw.githubusercontent.com
    intermittently returns 503 for this path.
    """
    return download_bytes(
        "https://cdn.jsdelivr.net/gh/mlmed/torchxrayvision@1.5.7/tests/1.2.276.0.7230010.3.1.4.8323329.6904.1517875201.850819.dcm"
    )


# A revision-pinned mirror of the released 16-bit PNGs and annotations. Tests
# fetch only the rows below, never the full dataset parquet files.
WRIST_DATASET_REVISION = "20f26bf9436b23af8b6f9ae171f96d86f837bb17"
WRIST_ROW_OFFSETS = {
    "0032_1146492991_01_WRI-R1_M007": 0,
    "0032_1146493016_02_WRI-R2_M007": 1,
    "0755_0509327622_01_WRI-R1_F012": 18,
    "2022_1185533267_01_WRI-L1_F012": 79,
}
WRIST_IMAGE_SHA256 = {
    "0032_1146492991_01_WRI-R1_M007": (
        "25628bd54621bd74085aacbad6fe75a059a433cf412b6602854b881292eaefc3"
    ),
    "0032_1146493016_02_WRI-R2_M007": (
        "a972b3096d05cc8917d5bbced93f450fa68f3a5944419166adf8d31a4e41ca70"
    ),
    "0755_0509327622_01_WRI-R1_F012": (
        "27923597306fbb6edf937ee22b5cc196501b9fec42595cd536b3eb3dbd505413"
    ),
    "2022_1185533267_01_WRI-L1_F012": (
        "8b3ac772282b0baf77fccbdd4dbcdfdbbd259373c4821142d27ac3a3ae7d9813"
    ),
}

# Two projections of one annotated patient, shared with other wrist tests.
WRIST_EXAMPLE_IDS = (
    "0032_1146492991_01_WRI-R1_M007",
    "0032_1146493016_02_WRI-R2_M007",
)
# One fracture-positive and one fracture-negative projection from other patients.
WRIST_CASE_IDS = (
    "0755_0509327622_01_WRI-R1_F012",
    "2022_1185533267_01_WRI-L1_F012",
)


def _read_wrist_source(image_id: str) -> tuple[bytes, dict]:
    """Cache genuine source data so signed asset URLs need not stay valid."""
    from tempfile import NamedTemporaryFile

    from curl_cffi import requests

    root = CACHE_ROOT / "wrist_reference"
    image_path, metadata_path = root / (image_id + ".png"), root / (image_id + ".json")
    checksum = WRIST_IMAGE_SHA256[image_id]
    if image_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        assert metadata["image_id"] == image_id, (
            "Cached reference image identity changed"
        )
        return download_bytes(image_path, checksum=checksum), metadata

    url = (
        "https://datasets-server.huggingface.co/rows?"
        "dataset=tirandazdylan/open-radiograph-findings&config=default&split=test"
        f"&offset={WRIST_ROW_OFFSETS[image_id]}&length=1"
        f"&revision={WRIST_DATASET_REVISION}"
    )
    # Cold dataset-server rows take longer than ordinary byte downloads.
    response = requests.get(url, impersonate="chrome", timeout=60)
    response.raise_for_status()
    row = response.json()["rows"][0]["row"]
    assert row["image_id"] == image_id, (
        "Dataset row order changed; review source identity"
    )
    assert row["source"] == "graz"
    image = download_bytes(row["image"]["src"], checksum=checksum)
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


@pytest.fixture(scope="session")
def wrist_source_reader():
    """Supply cached, identity-checked real GRAZPEDWRI-DX examples."""
    return _read_wrist_source


@pytest.fixture(scope="session", params=WRIST_EXAMPLE_IDS)
def wrist_example(request, wrist_source_reader):
    """Fetch two original projections of one annotated wrist patient."""
    return wrist_source_reader(request.param)


@pytest.fixture(scope="session", params=WRIST_CASE_IDS)
def wrist_case(request, wrist_source_reader):
    """Fetch one fracture-positive and one fracture-negative projection."""
    return wrist_source_reader(request.param)
