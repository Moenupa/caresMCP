🧠 MONAI
========

Models are built from the architectures released with the MONAI model zoo and loaded
from pinned Hub revisions, so an upstream update cannot silently change the weights
behind a tool.

📦 Supported checkpoints:

- ``MONAI/breast_density_classification``
- ``MONAI/pathology_tumor_detection``
- ``MONAI/pathology_nuclei_classification``
- ``MONAI/pathology_nuclei_segmentation_classification``

🧩 Requires the ``monai`` extra.

.. autofunction:: caresmcp.api.models.monai.get_monai_model
