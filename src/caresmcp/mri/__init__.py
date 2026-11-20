"""
Each tool accepts one reconstructed 2D MRI slice; sequence identity,
acquisition geometry, and anatomical correspondence are not inferred from
pixels. Multi-slice tools require explicitly ordered inputs from co-registered
data.

* |x| `MONAI/ventricular_short_axis_3label
  <https://huggingface.co/MONAI/ventricular_short_axis_3label>`_
  :footcite:`kerfoot2019lvquant`: LV cavity, myocardium, and RV cavity masks
  for 256-square short-axis slices in
  :mod:`cardiac.py <caresmcp.mri.cardiac>`; the checkpoint and revision are
  pinned, and most training data are end-diastolic frames. MIT.
* |x| `A4-Unet <https://github.com/RoxanneWAAANG/A4-Unet>`_: binary whole-tumor
  masks from four aligned BraTS contrasts in
  :mod:`a4unet.py <caresmcp.mri.a4unet>`; the original source requires CUDA
  device 0, and equal input sizes do not establish registration.
* |x| SimpleITK rigid/affine pixel-grid registration in :mod:`classical.py
  <caresmcp.mri.classical>`; no transform, spacing, or orientation can be
  recovered from one PNG/JPEG.

Denoising, box- and point-prompted MedSAM/SAM2 segmentation, generic semantic
segmentation, and image-text retrieval are provided by
:func:`denoise <caresmcp.tools.skimage.denoise>`,
:func:`hf_segment_image <caresmcp.medai.hf.hf_segment_image>`,
:func:`hf_segment_semantic <caresmcp.medai.hf.hf_segment_semantic>`, and
:func:`hf_image_text_retrieval <caresmcp.medai.hf.hf_image_text_retrieval>`
rather than MRI-specific tools; no validated MRI-only image-text checkpoint is
bundled.

Deferred priority candidates:

* |o| `SPiN
  <https://github.com/alexklwong/subpixel-embedding-segmentation>`_
  :footcite:`wong2021small`: ischemic-stroke lesion masks from five adjacent
  ATLAS T1 slices in :mod:`spin.py <caresmcp.mri.spin>`; the official source,
  ordering, and checkpoint are the caller's responsibility.
* |o| MONAI cardiac valve landmarks: released as a bundle, deferred until its
  long-axis view and named-point contract are validated.
* |o| Encoder-only foundation models (`MRI-CORE
  <https://github.com/mazurowski-lab/mri_foundation>`_, `RadImageNet
  <https://github.com/BMEII-AI/RadImageNet>`_): deferred because they provide
  features rather than a task-specific head.
* |o| Acquisition-parameter prediction, brain/fetal quality scoring, learned
  denoising, contrast synthesis, and super-resolution: deferred until each
  task-specific checkpoint and its evaluation data are verified.
* |o| Volume models (HD-BET, SynthSeg, MRAnnotator, 3D BraTS networks):
  deferred because they require volume geometry and spacing contracts rather
  than one reconstructed slice.

Out of scope:

* Generative diagnosis narratives and report writing: provided by
  well-established LLM/VLM APIs and excluded by the library's serving model.
* k-space reconstruction and coil-sensitivity estimation: they require raw
  acquisition data rather than reconstructed slices.
"""

import importlib
from pathlib import Path

_pkg_path = Path(__file__).parent

for _p in _pkg_path.glob("*.py"):
    if _p.name == "__init__.py" or _p.name.startswith(".") or _p.name.startswith("_"):
        continue

    importlib.import_module(f"{__name__}.{_p.stem}")
