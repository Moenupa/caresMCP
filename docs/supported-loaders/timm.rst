🌱 timm
========

Models are created from the timm registry. With ``pretrained=True``, weights are
downloaded through the Hugging Face Hub; ``hf-hub:<model_id>`` names select
checkpoints hosted on the Hub directly.

🧩 Requires the ``timm`` extra.

🧱 Models
---------

.. autofunction:: caresmcp.api.models.timm.get_timm_model

🛠️ Custom checkpoints
----------------------

.. autofunction:: caresmcp.api.models.timm.build_timm_model
