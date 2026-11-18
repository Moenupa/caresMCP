"""
* |x| Hugging Face checkpoints through the shared loaders in
  :mod:`hf.py <caresmcp.medai.hf>`: one function per pipeline, and each
  accepts compatible Hub checkpoints through ``model_id``.

  * |x| :func:`hf_classifier <caresmcp.medai.hf.hf_classifier>`: scores from
    trained classification heads, with labels read from the checkpoint's
    ``id2label``; see the `Transformers image-classification guide
    <https://huggingface.co/docs/transformers/tasks/image_classification>`_.
  * |x| :func:`hf_image_embed <caresmcp.medai.hf.hf_image_embed>`: CLS or
    mean-pooled features from Hub image encoders, defaulting to
    `facebook/dinov2-base <https://huggingface.co/facebook/dinov2-base>`_
    :footcite:`oquab2023dinov2`.
  * |x| :func:`hf_image_text_retrieval
    <caresmcp.medai.hf.hf_image_text_retrieval>`: image-text cosine ranking
    with `google/medsiglip-448
    <https://huggingface.co/google/medsiglip-448>`_
    :footcite:`tschannen2025siglip2`, `openai/clip-vit-base-patch32
    <https://huggingface.co/openai/clip-vit-base-patch32>`_
    :footcite:`radford2021clip`, or `google/siglip-base-patch16-224
    <https://huggingface.co/google/siglip-base-patch16-224>`_
    :footcite:`zhai2023siglip`.
  * |x| :func:`hf_segment_image <caresmcp.medai.hf.hf_segment_image>`:
    box-prompted segmentation with `wanglab/medsam-vit-base
    <https://huggingface.co/wanglab/medsam-vit-base>`_
    :footcite:`ma2024medsam`, which keeps its original square-canvas
    preprocessing, or `facebook/sam2.1-hiera-large
    <https://huggingface.co/facebook/sam2.1-hiera-large>`_
    :footcite:`ravi2024sam2`.
* |x| `TotalSegmentator <https://github.com/wasserth/TotalSegmentator>`_:
  volumetric CT and MR anatomy segmentation for the released task set in
  :mod:`totalsegmentator.py <caresmcp.medai.totalsegmentator>`.
"""

import importlib
from pathlib import Path

_pkg_path = Path(__file__).parent

for _p in _pkg_path.glob("*.py"):
    if _p.name == "__init__.py" or _p.name.startswith(".") or _p.name.startswith("_"):
        continue

    importlib.import_module(f"{__name__}.{_p.stem}")
