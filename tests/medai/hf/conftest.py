"""Real media shared by the Hugging Face pipeline tests."""

import pytest

from caresmcp.api.decorators.bytes_io import _read_bytes


@pytest.fixture(scope="session")
def sample() -> tuple[bytes, set[str]]:
    """Download one real cat photo and provide its expected ImageNet labels."""
    return (
        _read_bytes(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"
        ),
        {"tabby, tabby cat", "tiger cat", "Egyptian cat"},
    )


@pytest.fixture(scope="module")
def hernia_image() -> bytes:
    """A real chest radiograph from the NIH chest X-ray sample set."""
    return _read_bytes(
        "https://nih-chest-x-rays.s3.us-east-2.amazonaws.com/images_1024x1024/00000003_000.png"
    )
