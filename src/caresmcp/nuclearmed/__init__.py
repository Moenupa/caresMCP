"""
No nuclear-medicine checkpoint has been downloaded, run, or integrated; the
leads below are tracked until their artifacts, licenses, and PET/CT
preprocessing are verified.

* |o| `MedMLX/nnunet-autopet-mlx <https://huggingface.co/MedMLX/nnunet-autopet-mlx>`_
  :footcite:`isensee2023autopet`: five-fold AutoPET II nnU-Net converted from
  the PyTorch original; input is aligned CT and SUV FDG PET NIfTI volumes and
  output is a tumor mask. Integration is blocked because the conversion
  targets Apple Silicon MLX and Python 3.13, so the original PyTorch
  checkpoint and a Python 3.12-compatible loader need to be located and
  verified. The card states CC BY 4.0, research and education only; audit
  checkpoint provenance and attribution.
* |o| `ENHANCE-PET/MOOSE <https://github.com/ENHANCE-PET/MOOSE>`_: CT organ,
  lung, and bone anatomy masks for PET/CT localization and uptake summaries,
  not lesion, node, metastasis, or VIPoma detectors. Stated licenses are
  Apache 2.0 for code and CC BY 4.0 for models; verify each artifact and
  dependency. Overlaps existing CT anatomy tools, so integrate only for a
  clear PET/CT benefit.

Out of scope:

* Carlsen et al. Cu-64 DOTATATE nnU-Net :footcite:`carlsen2022dotatate`: no
  open code, weights, input channels, or license were verified.
* Haque et al. Ga-68 PPGL nnU-Net :footcite:`haque2024ppgl`: no open code,
  weights, or license were verified.
* Yang et al. PET-only hepatic network :footcite:`yang2026dotatate`: no open
  code, checkpoint, or license was verified.
* mSegResRF-SPECT :footcite:`ji2024msegresrf`: no open implementation,
  weights, or license was verified.
* Sentinel-node localization on lymphoscintigraphy: no node-localization
  checkpoint was verified; FDG-avid node models are not lymphatic mapping.
* `Zohaib3810/BS-80K-Bone-Scan-Metastasis-Detection-Regional-Analysis
  <https://github.com/Zohaib3810/BS-80K-Bone-Scan-Metastasis-Detection-Regional-Analysis>`_:
  prediction head requires TensorFlow/Keras.

Every candidate additionally needs licensed images with expert annotations, a
checkpoint and license audit, lesion-level evaluation by site and size across
scanners and tracers, and preservation of PET/CT alignment, DICOM scaling,
attenuation and decay correction, activity units, and SUV conversion.
"""

import importlib
from pathlib import Path

_pkg_path = Path(__file__).parent

for _p in _pkg_path.glob("*.py"):
    if _p.name == "__init__.py" or _p.name.startswith(".") or _p.name.startswith("_"):
        continue

    importlib.import_module(f"{__name__}.{_p.stem}")
