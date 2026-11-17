"""
* |x| `uzbtrust/stenoz-coronary-stenosis-unet
  <https://huggingface.co/uzbtrust/stenoz-coronary-stenosis-unet>`_
  (`code <https://github.com/uzbtrust/stenoz>`_) :footcite:`popov2024arcade`:
  MIT-licensed X-ray coronary angiography U-Nets for candidate stenosis regions
  and vessel masks in :mod:`coronary.py <caresmcp.dsa.coronary>`, trained on the
  `ARCADE dataset <https://zenodo.org/records/10390295>`_.

Deferred priority candidates:

* |o| `DSA sequence classifier <https://osf.io/n8k4r/>`_
  :footcite:`mittmann2022dsa`: MCA occlusion and reperfusion classification;
  the `code
  <https://github.com/BMittmannTest/DeepLearningBasedDsaClassification>`_ and
  OSF weights are public but declare no license. The NAMI demonstrator
  :footcite:`baumgaertner2023translation` wraps the same model with `MIT code
  <https://github.com/NAMI-THU/DSA-Thrombus-Classifier>`_ and the same
  unlicensed weights; later validation :footcite:`rosskopf2026validation`.
* |o| `Coronary stenosis YOLO checkpoint
  <https://github.com/Miguel-De-Sousa/live-coronary-angiography/blob/main/models/best_stenosis.pt>`_
  (`code <https://github.com/Miguel-De-Sousa/live-coronary-angiography>`_):
  coronary stenosis localization; the checkpoint embeds an AGPL-3.0 Ultralytics
  license despite the MIT repository, so its licensing must be resolved.
* |o| `FPD-SegDSA <https://github.com/kkk123lll/FPD-SegDSA>`_
  :footcite:`liu2026fpdsegdsa`: cerebral artery masks from temporal DSA; code
  and a `checkpoint
  <https://drive.google.com/drive/folders/1hFghqzO3E4eeEjH8v-giJsi9qwZjgrdi>`_
  are released, but the project declares no license and requires a custom Mamba
  stack.

Out of scope:

* `Angio-DualDx <https://github.com/SarvinChitsaz/Angio-DualDx>`_ (`checkpoint
  instructions
  <https://github.com/SarvinChitsaz/Angio-DualDx/blob/main/models/checkpoints/README.md>`_):
  coronary branch labeling with stenosis masks on ARCADE
  :footcite:`popov2024arcade`; the code is released but trained weights are
  not.
* Temporal multisource coronary stenosis detection
  :footcite:`sun2026coronary`: no implementation, checkpoint, or weight license
  was verified.
* `DIAS <https://github.com/lseventeen/DIAS>`_ :footcite:`liu2024dias`:
  intracranial artery segmentation dataset and benchmark; code and a
  `CC-BY-4.0 dataset <https://zenodo.org/records/11637181>`_ are released, but
  no trained checkpoint is.
* `DSCA dataset <https://zenodo.org/records/11255024>`_: cerebral artery
  segmentation dataset only; no accompanying model or checkpoint was verified.
* `dsa-vessel-seg <https://github.com/arvindmvepa/dsa-vessel-seg>`_
  :footcite:`vepa2022vessel`: single-frame cerebral vessel masks; code and
  test-annotated images are released, but no trained checkpoint or license.
* `Graph-Mask2former <https://github.com/hermition/Graph-Mask2former>`_
  :footcite:`ye2025graphmask2former`: intracranial aneurysm lesion
  segmentation; the repository is a GPL-3.0 model-only overlay without
  checkpoints or dataset configs.
* Intracranial aneurysm segmentation multicenter study
  :footcite:`ruiboliu2025aneurysm`: no open code or weights were verified.
* Endoleak detection :footcite:`smorenburg2025endoleak`: paper-only lead with no
  verified code or weights.
* Uterine-ovarian artery anastomosis detection :footcite:`lei2025anastomosis`:
  paper-only lead with no verified code or weights.
"""

import importlib
from pathlib import Path

_pkg_path = Path(__file__).parent

for _p in _pkg_path.glob("*.py"):
    if _p.name == "__init__.py" or _p.name.startswith(".") or _p.name.startswith("_"):
        continue

    importlib.import_module(f"{__name__}.{_p.stem}")
