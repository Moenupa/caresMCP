"""
* |x| Image download, inspection, and pixel-exact preparation in
  :mod:`pil.py <caresmcp.tools.pil>`: download with optional mode conversion,
  inspect format, dimensions, and mode, and crop to explicit or centered pixel
  bounds.
* |x| `scikit-image <https://scikit-image.org/>`_ classical denoising in
  :mod:`skimage.py <caresmcp.tools.skimage>`: bilateral, Gaussian, median,
  non-local means, total-variation (Bregman and Chambolle), and wavelet filters,
  with optional J-invariant self-supervised denoising. Grayscale output is a
  floating-point TIFF preserving the input value scale; RGB output is an 8-bit
  PNG.
"""

import importlib
from pathlib import Path

_pkg_path = Path(__file__).parent

for _p in _pkg_path.glob("*.py"):
    if _p.name == "__init__.py" or _p.name.startswith(".") or _p.name.startswith("_"):
        continue

    importlib.import_module(f"{__name__}.{_p.stem}")
