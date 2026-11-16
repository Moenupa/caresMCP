"""
* |x| `TorchXRayVision <https://github.com/mlmed/torchxrayvision>`_
  :footcite:`cohen2022torchxrayvision`: chest pathology scores, 14-structure
  anatomy masks, and Grad-CAM saliency in
  :mod:`torchxrayvision.py <caresmcp.xray.torchxrayvision>`; anatomy and
  saliency maps cover center crops rather than original-image coordinates.
* |x| `microsoft/rad-dino <https://huggingface.co/microsoft/rad-dino>`_
  :footcite:`perezgarcia2025rad` and
  `microsoft/BiomedVLP-CXR-BERT-specialized
  <https://huggingface.co/microsoft/BiomedVLP-CXR-BERT-specialized>`_ /
  `microsoft/BiomedVLP-BioViL-T
  <https://huggingface.co/microsoft/BiomedVLP-BioViL-T>`_
  :footcite:`boecking2022biovil,bannur2023biovilt`: one global-embedding
  interface and similarity ranking in
  :mod:`embeddings.py <caresmcp.xray.embeddings>`; phrase grounding and
  image-text ranking, optionally with a prior image, in
  :mod:`biovil.py <caresmcp.xray.biovil>`. BioViL-T has no trained progression
  classifier.
* |x| `VictorButoi/flexray <https://huggingface.co/VictorButoi/flexray>`_
  :footcite:`butoi2026flexray`: 60 selectable anatomical masks across
  radiographic regions in :mod:`flexray.py <caresmcp.xray.flexray>`; paired
  structures have no laterality split. CC-BY-NC-4.0.
* |x| `mdciri/YOLOv7-Bone-Fracture-Detection
  <https://github.com/mdciri/YOLOv7-Bone-Fracture-Detection>`_
  :footcite:`nagy2022grazpedwri`: pediatric wrist findings trained on
  GRAZPEDWRI-DX, with nine upstream classes and original-image boxes in
  :mod:`wrist.py <caresmcp.xray.wrist>`. GPL-3.0.
* |x| `skhoha/SKELEX <https://huggingface.co/skhoha/SKELEX>`_
  :footcite:`kim2026skelex`: seeded, multi-mask reconstruction-error maps for
  musculoskeletal radiographs in :mod:`skelex.py <caresmcp.xray.skelex>`; not a
  calibrated disease score. CC-BY-NC-ND-4.0.

Out of scope:

* `DeepWrist <https://github.com/imedslab/DeepWrist>`_ and
  `AIML-MED DRF <https://github.com/AIML-MED/DRF_Classification_Public>`_:
  distal-radius fracture models requiring legacy PyTorch/Lightning stacks or a
  manual Box download and Python 3.8 Detectron2 environment.
* `Urinary-stone segmentation <https://github.com/job-preedanan/urinary-stones-segmentation>`_:
  excluded because its two-stage cascade requires TensorFlow/Keras; the
  `KUB YOLOv8 study <https://doi.org/10.1007/s44163-026-01210-6>`_ is an
  unevaluated detection alternative.
* Bone-lesion and osteochondroma subtyping: deferred after the released YOLOv8
  segmenter underdetected release examples and no subtype checkpoint was found;
  see `BTXRD <https://github.com/SHUNHANYAO/BTXRD>`_ and
  `Bone-Tumor-Classification-XAI
  <https://github.com/psxdj6/Bone-Tumor-Classification-XAI>`_.
* Chest support-device paths/tips and carina: `ianpan/cxr-lines-and-tubes
  <https://huggingface.co/ianpan/cxr-lines-and-tubes>`_ and
  `ranzcr_clip <https://github.com/asnorkin/ranzcr_clip>`_: deferred because the
  former's loader downloads a multi-fold snapshot, mutates ``sys.path``, and
  loads checkpoints without a safe packaged loader; the latter relies on
  DVC/Google Drive retrieval that was not verified.
* `Ankle fractures <https://github.com/FaresAlMohamad/ankle-fractures-llm>`_ and
  pediatric Salter–Harris typing: skipped because access is restricted and no
  verified pediatric ankle checkpoint exists; binary labels cannot yield Weber
  or Salter–Harris classes.
* `BoneView <https://www.gleamer.ai/copilot/boneview>`_ and
  `Rayvolve <https://www.azmed.co/terms-of-use>`_: excluded because vendor
  deployment agreements and credentials are required.
* Cervical foreign bodies, endobronchial blockers, and neonatal umbilical
  catheters: skipped because no task-specific localization or tip-placement
  checkpoint exists, and 2D masks cannot establish depth or position.
  `Cervical-Spine-Segmentation-App
  <https://github.com/nievsky/Cervical-Spine-Segmentation-App>`_ covers C2–C7
  anatomy only; `Neonatal ResNet-50
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC8455735/>`_ and `NeoCLIP
  <https://doi.org/10.1038/s41746-025-01922-6>`_ provide presence scores
  without UVC tip placement; ETT models cannot be relabeled for endobronchial
  blockers.
* TMJ and first-MTP prostheses: skipped because no implant-specific head or foot
  landmarks exist; OrthoNet covers hip/knee implants only, and the
  `panoramic dental Fast R-CNN <https://doi.org/10.1093/dmfr/twaf079>`_ addresses
  dental restorations rather than TMJ replacement.
* `ChEX <https://github.com/philip-mueller/chex>`_ and
  `CheXseg <https://github.com/stanfordmlgroup/CheXseg>`_: skipped because
  checkpoints require manual Google Drive/Box placement and both target legacy
  environments (CheXseg Python 3.7.6; ChEX a custom DETR/CLIP/generator stack).
* ReMed, Ark/Ark+, CXR-RePaiR/X-REM, and HybridGNet/XLSor: deferred until they
  add a capability beyond the integrated models and their access is rechecked;
  XLSor also needs a legacy CUDA/model port.
* LLM/VLM tools, free-text report generation, and text-to-X-ray synthesis are
  out of scope.
"""

import importlib
from pathlib import Path

_pkg_path = Path(__file__).parent

for _p in _pkg_path.glob("*.py"):
    if _p.name == "__init__.py" or _p.name.startswith(".") or _p.name.startswith("_"):
        continue

    importlib.import_module(f"{__name__}.{_p.stem}")
