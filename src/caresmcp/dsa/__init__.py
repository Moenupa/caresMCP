"""
* |x| `uzbtrust/stenoz-coronary-stenosis-unet <https://huggingface.co/uzbtrust/stenoz-coronary-stenosis-unet>`_
  (`code <https://github.com/uzbtrust/stenoz>`_):
  MIT-licensed X-ray
  coronary angiography U-Nets for candidate stenosis regions and vessel masks
  in :mod:`coronary.py <caresmcp.dsa.coronary>`. Trained on the
  `ARCADE dataset <https://zenodo.org/records/10390295>`_
  :footcite:`popov2024arcade`.

Deferred priority candidates:

* |o| **MCA occlusion / reperfusion: DSA sequence classifier.**
  `DSA sequence classifier <https://osf.io/n8k4r/>`_
  :footcite:`mittmann2022dsa`, with `code
  <https://github.com/BMittmannTest/DeepLearningBasedDsaClassification>`_;
  later validation :footcite:`rosskopf2026validation`. Excluded because the
  repository and OSF weights declare no license. The NAMI demonstrator
  :footcite:`baumgaertner2023translation` wraps the same model; its
  `code <https://github.com/NAMI-THU/DSA-Thrombus-Classifier>`_ is MIT but its
  `DSA sequence classifier weights <https://osf.io/n8k4r/>`_ have no declared
  license.
* |o| **Coronary stenosis localization: YOLO checkpoint.** `Coronary
  stenosis YOLO checkpoint
  <https://github.com/Miguel-De-Sousa/live-coronary-angiography/blob/main/models/best_stenosis.pt>`_
  with `code <https://github.com/Miguel-De-Sousa/live-coronary-angiography>`_.
  Excluded because the checkpoint embeds an AGPL-3.0 Ultralytics license
  despite the MIT repository.
* |o| **Coronary branch labeling plus stenosis masks: Angio-DualDx.**
  `code <https://github.com/SarvinChitsaz/Angio-DualDx>`_ and
  `Angio-DualDx checkpoint instructions
  <https://github.com/SarvinChitsaz/Angio-DualDx/blob/main/models/checkpoints/README.md>`_;
  the `official ARCADE record <https://zenodo.org/records/10390295>`_
  :footcite:`popov2024arcade`. Deferred because trained weights are not released.
* |o| **Coronary stenosis localization: temporal multisource model.** The
  paper :footcite:`sun2026coronary` reports a temporal multisource model.
  Deferred because no implementation, checkpoint, or weight license was
  verified.

Additional DSA tasks:

* |o| **Cerebral artery masks from temporal DSA.** FPD-SegDSA
  :footcite:`liu2026fpdsegdsa`, with `code
  <https://github.com/kkk123lll/FPD-SegDSA>`_ and an `FPD-SegDSA checkpoint
  <https://drive.google.com/drive/folders/1hFghqzO3E4eeEjH8v-giJsi9qwZjgrdi>`_;
  DIAS :footcite:`liu2024dias` with `code
  <https://github.com/lseventeen/DIAS>`_ and a `CC-BY-4.0 DIAS dataset
  <https://zenodo.org/records/11637181>`_; the `DSCA dataset
  <https://zenodo.org/records/11255024>`_. Deferred because FPD-SegDSA declares
  no license and uses a custom Mamba stack; the benchmark code has no released
  checkpoint.
* |o| **Cerebral vessel masks, single-frame alternative.** The single-frame
  approach :footcite:`vepa2022vessel`, with `code
  <https://github.com/arvindmvepa/dsa-vessel-seg>`_ that also provides
  cerebral DSA images with test annotations. Deferred because no trained
  checkpoint or license was released.
* |o| **Intracranial aneurysm lesion segmentation.** Graph-Mask2former
  :footcite:`ye2025graphmask2former`, with
  `code <https://github.com/hermition/Graph-Mask2former>`_, and the multicenter
  DSA aneurysm study :footcite:`ruiboliu2025aneurysm`. Deferred because the
  repository is a GPL-3.0 model-only overlay without checkpoints or dataset
  configs.
* |o| **Other catheter DSA diagnoses.** Endoleak detection
  :footcite:`smorenburg2025endoleak` and uterine-ovarian artery anastomosis
  detection :footcite:`lei2025anastomosis`. Deferred as paper-only leads with
  no verified code or weights.
"""

import importlib
from pathlib import Path

_pkg_path = Path(__file__).parent

for _p in _pkg_path.glob("*.py"):
    if _p.name == "__init__.py" or _p.name.startswith(".") or _p.name.startswith("_"):
        continue

    importlib.import_module(f"{__name__}.{_p.stem}")
