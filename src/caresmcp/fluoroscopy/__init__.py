"""
No fluoroscopy-specific task-trained checkpoint has been integrated. Generic
box-prompted segmentation is available as
:func:`caresmcp.medai.hf.hf_segment_image`; it is not fluoroscopy-specific.

Leads with released code and trained weights are deferred; code-only or
paper-only leads are out of scope because no servable checkpoint exists.

Deferred priority candidates:

* |o| `VFSS_analysis <https://github.com/BSEL-UC3M/VFSS_analysis>`_ with
  `CC-BY-4.0 nnU-Net weights <https://doi.org/10.5281/zenodo.17191973>`_:
  videofluoroscopic swallowing segmentation and parameters; blocked by
  unlicensed code and a 6.1 GB nnU-Net v1 video pipeline.

Out of scope:

* Retrograde urethrogram stricture (`Kim et al.
  <https://doi.org/10.1007/s00345-022-04199-6>`_): no code, weights, or data.
* HSG tubal patency (`YOLO11s study
  <https://doi.org/10.1038/s41598-026-62943-z>`_): no code, weights, or license.
* `HSG-Assistant <https://doi.org/10.3233/SHTI250604>`_: no code or weights.
* `VCUG-DAM <https://github.com/wangzheyuan-666/VCUG-DAM>`_: code only; no
  trained checkpoint, private training data, no license.
* Intraoperative cholangiogram segmentation (`DeeplabV3+/U-Net study
  <https://doi.org/10.1007/s00464-024-10768-0>`_): paper only; no code,
  weights, or public data.
* Barium esophagram cancer detection (`two-stage detector
  <https://doi.org/10.3389/fonc.2022.766243>`_): paper only; no code,
  weights, or public data.
* Pediatric VUR grading (`Deep-VCUG <https://github.com/Li-zhanchi/VCUG>`_):
  code only; no weights or license, data on request.
* Pediatric posterior urethral valves (`Kabir et al.
  <https://doi.org/10.1016/j.compbiomed.2024.109509>`_): no code or weights.
* VFSS penetration/aspiration (`Kwak et al.
  <https://doi.org/10.1038/s41598-025-10397-0>`_): no code or weights.
* VFSS bolus segmentation (`<https://doi.org/10.1088/2057-1976/ad0bb3>`_): no
  code or weights; data unavailable.
* No verified model for epiphrenic diverticulum, microcolon, T-shaped uterus,
  hydrosalpinx, rectovesical or vesicocutaneous fistula, or Mitrofanoff and
  ileocystoplasty assessment.
"""

import importlib
from pathlib import Path

_pkg_path = Path(__file__).parent

for _p in _pkg_path.glob("*.py"):
    if _p.name == "__init__.py" or _p.name.startswith(".") or _p.name.startswith("_"):
        continue

    importlib.import_module(f"{__name__}.{_p.stem}")
