"""
* |x| `MONAI breast density <https://huggingface.co/MONAI/breast_density_classification>`_
  :footcite:`gupta2022density`: BI-RADS A–D scores in
  :mod:`density.py <caresmcp.mammography.density>`.
* |x| `umairinayat/medical-models <https://huggingface.co/umairinayat/medical-models>`_:
  one release providing two mammography checkpoints.

  * |x| `Patch classification <https://huggingface.co/umairinayat/medical-models/blob/main/disease_models/mammography/classification/best_classifier.pt>`_: benign, malignant, and normal scores in
    :mod:`umair.py <caresmcp.mammography.umair>`; trained on
    CLAHE-preprocessed images.
  * |o| `Lesion segmentation <https://huggingface.co/umairinayat/medical-models/blob/main/disease_models/mammography/segmentation/best_segmenter.pt>`_: pending integration.

Out of scope:

* `NYU breast density classifier <https://github.com/nyukat/breast_density_classifier>`_
  :footcite:`wu2018density`: excluded due to input sensitivity.
* `GLAM <https://github.com/nyukat/GLAM>`_ :footcite:`liu2021glam`: deferred due
  to its AGPL3 license restrictions.
* `abdullahtahir/resnet50-mammography-birads <https://huggingface.co/abdullahtahir/resnet50-mammography-birads>`_:
  deferred because it requires TensorFlow.
* `batmanLab/Mammo-FM <https://huggingface.co/batmanLab/Mammo-FM>`_
  :footcite:`ghosh2025mammofm`: its weight license prohibits API serving.
* `batmanLab/Mammo-CLIP <https://github.com/batmanLab/Mammo-CLIP>`_
  :footcite:`ghosh2024mammoclip`: not inference-ready checkpoints.
"""

import importlib
from pathlib import Path

_pkg_path = Path(__file__).parent

for _p in _pkg_path.glob("*.py"):
    if _p.name == "__init__.py" or _p.name.startswith(".") or _p.name.startswith("_"):
        continue

    importlib.import_module(f"{__name__}.{_p.stem}")
