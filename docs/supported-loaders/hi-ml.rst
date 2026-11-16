🧬 BioViL (hi-ml)
=================

BioViL image and text encoders are loaded from the ``hi-ml-multimodal`` package, with
checkpoints taken from pinned Hub revisions. The image encoder matches the projected
text encoder, so image and phrase embeddings share one joint space.

📦 Supported checkpoints:

- ``microsoft/BiomedVLP-CXR-BERT-specialized``
- ``microsoft/BiomedVLP-BioViL-T``

🧩 Requires the ``xray`` extra.

🤖 Image and text encoders
--------------------------

.. autofunction:: caresmcp.api.models.hi_ml.get_biovil_image_model

.. autofunction:: caresmcp.api.models.hi_ml.get_biovil_text_model
