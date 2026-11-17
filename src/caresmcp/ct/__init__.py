"""
Each tool accepts one CT slice as DICOM bytes or a single-frame grayscale
raster; head-hemorrhage scoring also accepts physically adjacent slices.
HU-based tools require CT DICOM or a raster with known HU calibration;
display exports cannot recover HU. Series- and volume-level processing is
not supported.

* |x| DICOM LINEAR windowing, inclusive-HU segmentation, and 8-connected
  region measurement in :mod:`classical.py <caresmcp.ct.classical>`:
  windowing returns an 8-bit grayscale PNG, segmentation a binary PNG, and
  measurement reports bounding boxes, centroids, areas, and mean HU, with
  physical areas when DICOM pixel spacing is available. Threshold masks do
  not identify an organ or lesion.
* |x| `PyRadiomics <https://pyradiomics.readthedocs.io/>`_
  :footcite:`vangriethuysen2017pyradiomics`: two-dimensional first-order,
  co-occurrence, run-length, size-zone, gray-level dependence, and
  neighborhood features from a supplied ROI in
  :mod:`pyradiomics.py <caresmcp.ct.pyradiomics>`; bin width and row/column
  spacing are explicit, and spacing provenance is reported as ``provided``,
  ``dicom``, or ``assumed`` at 1 mm.
* |x| `oohtmeel/swin-tiny-patch4-finetuned-lung-cancer-ct-scans
  <https://huggingface.co/oohtmeel/swin-tiny-patch4-finetuned-lung-cancer-ct-scans>`_:
  binary lung-cancer softmax scores for one CT slice in
  :mod:`huggingface.py <caresmcp.ct.huggingface>`.
* |x| `sukhmani1303/lung-cancer-vit-model
  <https://huggingface.co/sukhmani1303/lung-cancer-vit-model>`_: binary
  lung-cancer softmax scores for one CT slice through the same
  ``classify_ct_slice`` tool; the checkpoint's index-to-label order is
  ambiguous at the source.
* |x| `raidium/curia <https://huggingface.co/raidium/curia>`_
  :footcite:`dancette2025curia`: anatomy, COVID/pneumonia, and intracranial
  hemorrhage slice heads in :mod:`curia.py <caresmcp.ct.curia>`; access is
  gated and the weights are Research-only RAIL-M. Disease heads require
  axial slices, and orthogonal DICOM axes are reoriented from orientation
  metadata.
* |x| `ianpan/ct-head-hemorrhage-detection
  <https://huggingface.co/ianpan/ct-head-hemorrhage-detection>`_: six
  sigmoid hemorrhage scores for a center noncontrast axial head CT slice
  with optional physically adjacent slices in
  :mod:`hemorrhage.py <caresmcp.ct.hemorrhage>`; only the baseline slice
  classifier is served, not the upstream contextual GRU or segmentation
  ensemble. Research software, not a medical device.
* |x| `Kentucky-Open-Science/DALE-CT-2S
  <https://huggingface.co/Kentucky-Open-Science/DALE-CT-2S>`_
  :footcite:`damron2026dale`: one global chest-CT slice embedding in
  :mod:`dale_ct_2s.py <caresmcp.ct.dale_ct_2s>`; inputs must be HU with
  height and width divisible by 16. CC-BY-NC-SA-4.0 weights for
  non-commercial research.

Deferred priority candidates:

* |o| Contextual hemorrhage inference and localization from the same
  `ianpan/ct-head-hemorrhage-detection
  <https://huggingface.co/ianpan/ct-head-hemorrhage-detection>`_ release:
  the upstream two-layer bidirectional GRU and five DeepLabV3+ decoders
  require series-level input and a segmentation output contract.
* |o| `mlinslab/neurovfm-dx-ct
  <https://huggingface.co/mlinslab/neurovfm-dx-ct>`_
  :footcite:`kondepudi2026neurovfm`: multilabel study-level CT diagnosis
  over an 82-diagnosis ontology; blocked by gated CC-BY-NC-SA-4.0 weights
  and a series-level multiple-instance aggregation pipeline.
* |o| Inception-v4 checkpoint leads and other previously surveyed CT
  classifiers: no adapter or current investigation notes; task fit, access,
  and licensing need reassessment before integration.
"""

import importlib
from pathlib import Path

_pkg_path = Path(__file__).parent

for _p in _pkg_path.glob("*.py"):
    if _p.name == "__init__.py" or _p.name.startswith(".") or _p.name.startswith("_"):
        continue

    importlib.import_module(f"{__name__}.{_p.stem}")
