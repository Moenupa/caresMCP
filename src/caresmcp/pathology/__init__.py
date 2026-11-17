"""
No pathology tool reads a whole-slide image.

* |x| Tile encoders in
  :mod:`pathology_encoder.py <caresmcp.pathology.pathology_encoder>`: one
  encoded tile or ROI becomes a feature vector, not a diagnosis.

  * |x| `bioptimus/H0-mini <https://huggingface.co/bioptimus/H0-mini>`_ and
    `bioptimus/H-optimus-0 <https://huggingface.co/bioptimus/H-optimus-0>`_:
    compact and large histology encoders; H-optimus-0 accepts only 224×224
    tiles. H0-mini is CC-BY-NC-ND-4.0 with manually granted access;
    H-optimus-0 is Apache-2.0, gated.
  * |x| `MahmoodLab/UNI <https://huggingface.co/MahmoodLab/UNI>`_
    :footcite:`chen2024uni` and
    `MahmoodLab/UNI2-h <https://huggingface.co/MahmoodLab/UNI2-h>`_:
    general-purpose and larger 1536-value histology features;
    CC-BY-NC-ND-4.0, gated.
  * |x| `owkin/phikon <https://huggingface.co/owkin/phikon>`_
    :footcite:`filiot2023phikon` and
    `owkin/phikon-v2 <https://huggingface.co/owkin/phikon-v2>`_
    :footcite:`filiot2024phikonv2`: masked-image-modeling features with 768 and
    1024 values; Owkin non-commercial terms.
  * |x| `histai/hibou-b <https://huggingface.co/histai/hibou-b>`_
    :footcite:`nechaev2024hibou`: ViT-B histology features with 768 values;
    Apache-2.0, gated.
  * |x| `paige-ai/Virchow <https://huggingface.co/paige-ai/Virchow>`_
    :footcite:`vorontsov2024virchow` and
    `paige-ai/Virchow2 <https://huggingface.co/paige-ai/Virchow2>`_
    :footcite:`zimmermann2024virchow2`: CLS or patch-mean 1280-value features,
    or their 2560-value concatenation; Virchow is Apache-2.0 and Virchow2 is
    CC-BY-NC-ND-4.0, both gated.
* |x| `MahmoodLab/CONCH <https://huggingface.co/MahmoodLab/CONCH>`_
  :footcite:`lu2024conch`: cosine similarities between one tile and candidate
  pathology descriptions in :mod:`conch.py <caresmcp.pathology.conch>`;
  CC-BY-NC-ND-4.0, gated.
* |x| `paige-ai/Prism <https://huggingface.co/paige-ai/Prism>`_
  :footcite:`shaikovski2024prism` and
  `paige-ai/Prism2 <https://huggingface.co/paige-ai/Prism2>`_
  :footcite:`vorontsov2026prism2`: slide-level analysis on one tile per call.
  PRISM turns Virchow CLS+mean features into a 1280-value embedding or
  negative/positive prompt-group probabilities in
  :mod:`prism.py <caresmcp.pathology.prism>` and does not generate reports.
  PRISM2 provides base (2560) and diagnostic (3072) embeddings, yes/no question
  scores, and prompt-conditioned caption generation in
  :mod:`prism2.py <caresmcp.pathology.prism2>`; it requires CUDA with bfloat16,
  and its gated non-commercial academic access is not provided here. Both are
  CC-BY-NC-ND-4.0 and gated.
* |x| `MONAI/pathology_tumor_detection
  <https://huggingface.co/MONAI/pathology_tumor_detection>`_: patch-wise tumor
  scores and pixel boxes for H&E lymph-node regions in
  :mod:`monai.py <caresmcp.pathology.monai>`; scores are not calibrated
  probabilities.
* |x| `MONAI/pathology_nuclei_classification
  <https://huggingface.co/MONAI/pathology_nuclei_classification>`_:
  nucleus-type scores for a 128×128 colorectal H&E patch and target-nucleus
  mask in :mod:`monai.py <caresmcp.pathology.monai>`; scores are not calibrated
  probabilities.
* |x| `MONAI/pathology_nuclei_segmentation_classification
  <https://huggingface.co/MONAI/pathology_nuclei_segmentation_classification>`_
  :footcite:`graham2019hovernet`: sliding-window HoVerNet instance and type
  maps for colorectal H&E regions at CoNSeP's 40x scale in
  :mod:`monai.py <caresmcp.pathology.monai>`.
* |x| `Cellpose <https://github.com/MouseLand/cellpose>`_
  :footcite:`stringer2020cellpose,pachitariu2025cellposesam`: cell-boundary
  overlays and a rendered instance count for pathology images in
  :mod:`cellpose.py <caresmcp.pathology.cellpose>`; it does not type cells and
  returns a rendered PNG rather than structured masks.

Deferred priority candidates:

* |o| WSI input and region handling: read slides with OpenSlide, TIAToolbox, or
  TRIDENT, preserve pyramid level and MPP, and extract coordinate-preserving
  tiles; the current tools accept one encoded image only.
* |o| WSI-level classification and localization: add a fixed-feature MIL
  baseline such as CLAM or ABMIL, then compare DSMIL, TransMIL, and ACMIL;
  multi-tile PRISM and TITAN/GigaPath aggregation need coordinates and a tile
  encoder.
* |o| Additional encoders: CHIEF, Prov-GigaPath, HIPT, MUSK, Google Path
  Foundation/REMEDIS, GPFM, PLIP, CTransPath, Kaiko, Midnight-12k, and other
  surveyed checkpoints; select by task, access terms, hardware, and benchmark
  evidence rather than wrapping every release.
* |o| Additional nuclei and tissue-QC models: CellViT/CellViT++, StarDist,
  NuClick, DeepLIIF, HistoQC, WSISegQC, and SlideSegmenter; these need
  scale-aware tiling and model-specific label mappings.
* |o| Disease, biomarker, and prognosis heads: PANDA/AGGC2022 grading,
  IDaRS/HistoBistro biomarkers, DeepHistology/DeepPATH, CAMIL, and
  SurvPath/PORPOISE/MCAT; each needs its organ, endpoint, cohort, labels,
  patient-level split, and evaluation contract.
* |o| Spatial-expression models: ST-Net, HE2RNA, DeepPT, GHIST, STPath,
  Path2Space, and related methods; they require aligned image/spot/cell assays
  and return imputed measurements, not measured results.
* |o| Per-tool provenance and uncertainty: record checkpoint revision,
  preprocessing, stain, MPP/magnification, label schema, and access/license
  state; support calibration, OOD evaluation, and abstention for task
  predictions.
* |o| TITAN: its released package omits the caption decoder; only an
  encoder/retrieval path would be in scope if added.

Out of scope:

* PathChat and other open-ended pathology chat, ROI description, or VQA
  models: served through the LLM/VLM API layer, not as per-model tools here.
* HistGen, R2Gen/R2GenCMN, and free-text pathology report generation:
  narrative report workflows belong to the LLM/VLM serving layer.
* Generic models already exposed by an LLM API: reuse that serving route
  instead of duplicating model-specific wrappers here; bounded visual tasks
  such as embedding, candidate matching, segmentation, and scoring remain in
  scope.
* A generic "pathology diagnosis" wrapper: disease and biomarker heads must
  name their task, labels, cohort, and evaluation contract.
"""

import importlib
from pathlib import Path

_pkg_path = Path(__file__).parent

for _p in _pkg_path.glob("*.py"):
    if _p.name == "__init__.py" or _p.name.startswith(".") or _p.name.startswith("_"):
        continue

    importlib.import_module(f"{__name__}.{_p.stem}")
