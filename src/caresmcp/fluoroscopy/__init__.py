"""
No fluoroscopy-specific task-trained checkpoint with a verified license was
found. Generic box-prompted segmentation is available as
:func:`caresmcp.medical_ai.hf.hf_segment_image`; it is not fluoroscopy-specific
and is not claimed as coverage for any examination below.

Out of scope:

* Retrograde urethrogram stricture classification (`Kim et al.
  <https://doi.org/10.1007/s00345-022-04199-6>`_): no code, trained weights, or
  license were verified. An image-level binary label cannot localize the
  narrowing or measure its length.
* Hysterosalpingography tubal patency (`YOLO11s study
  <https://doi.org/10.1038/s41598-026-62943-z>`_) and `HSG-Assistant
  <https://doi.org/10.3233/SHTI250604>`_: no verified dataset, task-trained
  weights, or license. The reported categories do not identify the blocked
  side, hydrosalpinx, or a T-shaped uterus.
* `VCUG-DAM <https://github.com/wangzheyuan-666/VCUG-DAM>`_: official code only;
  no trained checkpoint was identified and the repository declares no license.
  Its pooled abnormal-urethra score is not a dedicated stricture diagnosis.
* Intraoperative cholangiogram segmentation (`DeeplabV3+ and U-Net study
  <https://doi.org/10.1007/s00464-024-10768-0>`_): no code, weights, or license
  were verified; an intraoperative cholangiogram is an adjacent examination.
* Barium esophagram cancer detection (`two-stage detector
  <https://doi.org/10.3389/fonc.2022.766243>`_): no code, checkpoint, or license
  were verified. It targets esophageal cancer, not an epiphrenic diverticulum.
* Pediatric vesicoureteral reflux grading (`Li-zhanchi/VCUG
  <https://github.com/Li-zhanchi/VCUG>`_): code only, with no trained weights or
  license. Reflux grading is not a fistula or urinary-diversion assessment.
* Pediatric posterior urethral valves (`Kabir et al.
  <https://doi.org/10.1016/j.compbiomed.2024.109509>`_): no code, weights, or
  license were verified. Congenital valve detection is not adult urethral
  stricture detection.
* Videofluoroscopic swallowing analysis: the penetration/aspiration study
  (`<https://doi.org/10.1038/s41598-025-10397-0>`_) did not identify a
  task-trained artifact. `Cubero et al.
  <https://doi.org/10.5281/zenodo.17191973>`_ released CC-BY-4.0 nnU-Net v1
  weights, but the accompanying code has no license and its video preprocessing
  plus 21-parameter pipeline is not a thin integration. Bolus segmentation
  (`<https://doi.org/10.1088/2057-1976/ad0bb3>`_) reports unavailable data and
  no verified weights.
* Catheter and guidewire segmentation: `pambros
  <https://github.com/pambros/CNN-2D-X-Ray-Catheter-Detection>`_ requires
  TensorFlow and trains on synthetic noise; `CathMTL
  <https://github.com/xilin-x/CathMTL>`_ is MIT but releases no weights;
  `FluoroSAM <https://github.com/arcadelab/fluorosam>`_ is CC-BY-NC-SA with
  2.5 GB manual Google Drive weights and a large bundled architecture;
  `Elakiya17/fluoroscopy-sam2
  <https://huggingface.co/Elakiya17/fluoroscopy-sam2>`_ and
  `Sim2real-Guidewire-Seg
  <https://github.com/Yuxuan-Wen/Sim2real-Guidewire-Seg>`_ declare no license.
* No verified task-trained model was found for epiphrenic diverticulum,
  microcolon due to meconium ileus, T-shaped uterus, hydrosalpinx,
  rectovesical or vesicocutaneous fistula, or Mitrofanoff and ileocystoplasty
  assessment. Absence of a verified model is not proof that none exists.
"""

import importlib
from pathlib import Path

_pkg_path = Path(__file__).parent

for _p in _pkg_path.glob("*.py"):
    if _p.name == "__init__.py" or _p.name.startswith(".") or _p.name.startswith("_"):
        continue

    importlib.import_module(f"{__name__}.{_p.stem}")
