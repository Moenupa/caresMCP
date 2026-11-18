"""
* |x| Hugging Face checkpoints through the shared loaders in
  :mod:`hf.py <caresmcp.medai.hf>`: image classification, image embedding, and
  box-prompted segmentation for any compatible Hub checkpoint.
* |x| `TotalSegmentator <https://github.com/wasserth/TotalSegmentator>`_: volumetric CT
  and MR anatomy segmentation for the released task set in
  :mod:`totalsegmentator.py <caresmcp.medai.totalsegmentator>`.
"""

import importlib
from pathlib import Path

_pkg_path = Path(__file__).parent

for _p in _pkg_path.glob("*.py"):
    if _p.name == "__init__.py" or _p.name.startswith(".") or _p.name.startswith("_"):
        continue

    importlib.import_module(f"{__name__}.{_p.stem}")
