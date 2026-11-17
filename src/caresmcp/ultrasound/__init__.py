"""
* |x| `CarDS-Yale/PanEcho <https://huggingface.co/CarDS-Yale/PanEcho>`_
  :footcite:`holste2025panecho`: 39 measurement and classification heads for
  transthoracic echocardiography, with equal-weight study aggregation and
  768-dimensional video and frame embeddings in
  :mod:`panecho.py <caresmcp.ultrasound.panecho>`. Inference samples 16 frames
  uniformly and resizes them to 224 pixels, so it is not established as
  upstream evaluation-pipeline parity, and embeddings are not disease scores.
* |x| `EchoNet-Dynamic <https://github.com/echonet/dynamic>`_
  :footcite:`ouyang2020echonet`: apical four-chamber ejection fraction and
  left-ventricle segmentation in
  :mod:`echonet_dynamic.py <caresmcp.ultrasound.echonet_dynamic>`. EF averages
  up to 32 evenly spaced clips with per-video channel statistics because
  training statistics were unpublished; masks are single-frame and returned at
  the input size. MIT code.
* |x| `CAIR-HKISI/EchoCare <https://github.com/CAIR-HKISI/EchoCare>`_
  :footcite:`zhang2025echocare`: 2048-dimensional image embeddings from a
  self-supervised ultrasound foundation model in
  :mod:`echocare.py <caresmcp.ultrasound.echocare>`; the package loads the
  `Moenupa/EchoCare <https://huggingface.co/Moenupa/EchoCare>`_ checkpoint.
  The upstream release specifies no canonical embedding inference transform,
  and embeddings alone do not identify disease.

Deferred priority candidates:

* |o| `NCLS
  <https://github.com/Je1zzz/Neonatal_cerebral_lesions_screening_NCLS>`_
  :footcite:`lin2025ncls`: severe-versus-non-severe neonatal cerebral-lesion
  screening from cranial ultrasound videos; its detector, five diagnostic
  folds, view extraction, and multi-view study pipeline exceed a thin wrapper,
  and the weight license and preprocessing remain unverified. Grade III/IV IVH
  is grouped with other severe lesions, so its score is not an IVH grade.
* |o| `SonoBase <https://github.com/AlfredQin/sonobase>`_: point- or
  box-prompted ultrasound segmentation and prompt-once video propagation;
  blocked by its custom SAM2-derived architecture and a 706 MB checkpoint whose
  weights are CC BY-NC 4.0. A mask delineates a user-selected structure and
  does not identify disease.
* |o| `UltraSam <https://github.com/CAMMA-public/UltraSam>`_: prompted
  ultrasound segmentation; blocked by its MMDetection stack; its released
  weights, license, and coordinate conventions need verification.
* |o| `Cond-UNet <https://huggingface.co/AImageLab-Zip/US_Cond-UNet>`_:
  testicular organ segmentation; its foreground-mask labels and weight terms
  remain unverified, its aspect-ratio padding is not removed from predicted
  masks, and its Transformers custom code does not import with the current
  dependency set. A testis mask is not intratesticular cyst detection.
* |o| `EchoNet-Measurements <https://github.com/echonet/measurements>`_:
  cardiac Doppler segmentation and measurement; file-oriented scripts with
  DICOM calibration do not form a callable byte interface, and transfer to
  uterine-artery waveforms is unvalidated.
* |o| `EchoNet-AS <https://github.com/echonet/AS>`_: aortic-stenosis workflow
  combining Doppler inference, six view-specific models, DICOM-to-video
  conversion, and file-based study aggregation; adapting it to a bytes-based
  study API exceeds a thin wrapper.
* |o| `Retuve <https://github.com/radoss-org/retuve>`_: hip DICOM and image
  segmentation with a bundled TinyUNet checkpoint for ilium, acetabulum,
  femoral head, and ischium; blocked by its ``stem-cv`` dependency pinning
  ``timm==0.4.12``, which conflicts with this project. Weights are
  CC BY-NC-SA 3.0.
* |o| `FUSEP <https://github.com/ahuteam/FUSEP>`_: fetal-anatomy detection
  checkpoints for 11–14-week screening; blocked by its custom Relation-DETR
  stack, and its targets are fetal structures rather than gestational-sac or
  cervical findings.
* |o| `SonoNet <https://github.com/bkainz/SonoNet-weights>`_: fetal
  standard-plane detection; its Theano/Lasagne model needs a runtime port.
* |o| `FetalNet <https://github.com/SanoScience/FetalNet>`_: fetal plane
  detection and biometric measurement; only a model definition and a Drive
  checkpoint are released, without inference or measurement postprocessing.
* |o| `FUVAI <https://github.com/SanoScience/FUVAI>`_: fetal ultrasound video
  biometric measurement; model definitions and a Drive checkpoint are
  released, but no packaged inference or preprocessing workflow is.
* |o| `EchoTracker <https://github.com/riponazad/echotracker>`_: myocardial
  point tracking in echocardiography; blocked because the released code is not
  packaged, imports top-level modules, and initializes a CUDA-only custom
  architecture with a separate Drive checkpoint. CC BY-NC 4.0.
* |o| `CVA-Net <https://github.com/jhl-Det/CVA-Net>`_: ultrasound video
  analysis; blocked by custom CUDA operators and video processing, with dataset
  labels that needed correction; weights are linked separately.
* |o| `BINDS <https://github.com/lyhkevin/BINDS>`_: multimodal breast-cancer
  diagnosis from ultrasound, mammography, and MRI; blocked by a separate
  lesion-ROI preprocessing workflow, unbundled weights, and a Python
  3.8/PyTorch 1.11 environment.
* |o| `OpenUS <https://github.com/XZheng0427/OpenUS>`_: ultrasound
  foundation-model embeddings; blocked by its ``timm==0.4.12`` pin and custom
  VMamba/CUDA stack.
* |o| `EchoFM <https://github.com/SekeunKim/EchoFM>`_: echocardiography
  foundation-model embeddings; blocked by its unpackaged custom video-ViT
  architecture and preprocessing.
* |o| `USFM <https://github.com/openmedlab/USFM>`_: ultrasound foundation-model
  embeddings; blocked by its modified MMSegmentation runtime.
* |o| `UltraFedFM <https://github.com/yuncheng97/UltraFedFM>`_: federated
  ultrasound foundation-model embeddings; pretrained checkpoints and training
  scripts are linked, but no packaged embedding inference interface is.
* |o| `QingboKang/URFM <https://huggingface.co/QingboKang/URFM>`_: ultrasound
  representation foundation model; its Hub checkpoint is gated, so anonymous
  automatic loading cannot be assumed.
* |o| `USF-MAE <https://github.com/Yusufii9/USF-MAE>`_: ultrasound
  masked-autoencoder embeddings; Drive checkpoints and pretraining notebooks
  are released without a ready-use encoder inference API.
* |o| `MUST-Sub <https://github.com/takimailto/MUST-Sub>`_: breast-cancer
  molecular subtyping from multimodal ultrasound; task weights remain
  unverified.
* |o| `EchoNet-LVH <https://github.com/echonet/lvh>`_: left-ventricular
  hypertrophy classification and etiology prediction; task weights remain
  unverified.
* |o| `DSMA-Net <https://github.com/guangxin1990/DSMA-Net>`_: thyroid-nodule
  classification; task weights remain unverified.

Out of scope:

* Deep pelvic endometriosis (`AI-DE study
  <https://www.umbjournal.org/article/S0301-5629%2825%2900091-2/fulltext>`_):
  YOLOv8 nodule detection on ultrasound images and videos; no accessible
  trained weights or code were verified.
* `Multiclass TVUS segmentation
  <https://github.com/cumcrad/MulticlassSegmentationTVUS>`_: bladder, cervical
  region, and canal masks in pregnancy; the repository has evaluation code but
  no released trained ensemble checkpoints.
* `Gestational-sac segmentation
  <https://www.frontiersin.org/journals/pediatrics/articles/10.3389/fped.2024.1453302/full>`_:
  miscarriage cases are included, but no usable checkpoint was verified.
* Uterine-artery spectral Doppler: no released envelope, notch-detection, or
  PI/RI model validated on uterine waveforms was verified; the published
  `waveform simulator <https://pmc.ncbi.nlm.nih.gov/articles/PMC6511649/>`_ is
  not a screenshot inference tool.
* Thumb MCP calcific periarthritis: no verified hand or joint calcification
  detector was found; shoulder calcification models are not established
  transfer.
* Pyramidalis muscle: no verified abdominal-wall muscle model was found;
  prompted segmentation would be user-directed delineation only.
* `TRUSFormer <https://github.com/med-i-lab/TRUSFormer>`_: experimental
  training framework with server-dependent data and backbone pretraining, and
  no directly usable task inference release.
* `URI-CADS <https://github.com/miguel55/URI-CADS>`_: weights and data require
  a request, and inference depends on a bundled modified torchvision 0.5 and
  PyTorch 1.3.1.
* `Open Kidney Ultrasound <https://github.com/rsingla92/kidneyUS>`_: evaluation
  code and weight links are present, but no inference architecture or API is
  released, and its image data require registration.
* `BiometryNet
  <https://github.com/surgical-vision/Multicentre-Fetal-Biometry>`_: training
  and evaluation scripts target Python 3.6 and PyTorch 1.0, and no trained
  task weights or usable inference wrapper were found.
* `OBUS-GHL <https://github.com/Global-Health-Labs/OBUS-GHL>`_: gestational
  age, fetal presentation, and weight framework requiring a separately trained
  checkpoint; no downloadable task weights were found.
"""

import importlib
from pathlib import Path

_pkg_path = Path(__file__).parent

for _p in _pkg_path.glob("*.py"):
    if _p.name == "__init__.py" or _p.name.startswith(".") or _p.name.startswith("_"):
        continue

    importlib.import_module(f"{__name__}.{_p.stem}")
