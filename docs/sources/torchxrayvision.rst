🫁 TorchXRayVision
==================

Chest X-ray classifiers, anatomical segmenters, and pathology localizers are loaded from
the TorchXRayVision model zoo. The ``anatomy`` weights select the PSPNet chest-anatomy
segmenter; other names select a DenseNet, or a ResNet when the name starts with
``resnet``.

Weights are cached under ``$CARESMCP_CACHE/models`` (see :ref:`environment-variables`).

🧩 Requires the ``xray`` extra.

.. autofunction:: caresmcp.api.models.torchxrayvision.get_torchxrayvision_model
