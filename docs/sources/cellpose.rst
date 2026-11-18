🔬 Cellpose
===========

Cell segmentation models are created and loaded by Cellpose from its `pretrained model
zoo <https://cellpose.readthedocs.io/en/latest/models.html>`_. Weights are downloaded on
first use to the location set by :envvar:`CELLPOSE_LOCAL_MODELS_PATH` and reused from
disk afterwards. Each call builds a new model on ``caresmcp.constants.DEVICE``; CaresMCP
does not retain the instance, so callers that segment repeatedly should keep and reuse
the returned model.

📦 Supported pretrained models:

- ``cpsam`` — Cellpose-SAM.
- ``cpsam_v2`` — updated Cellpose-SAM with fewer spurious masks in low-contrast regions.
- ``cpdino`` — Cellpose-DINO with a DINOv3 ViT-L backbone.
- ``cpdino-vitb`` — Cellpose-DINO with a DINOv3 ViT-B backbone.

🧩 Requires the ``pathology`` extra.

.. autofunction:: caresmcp.api.models.cellpose.get_cellpose_model
