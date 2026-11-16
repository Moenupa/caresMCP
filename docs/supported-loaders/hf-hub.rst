🤗 Hugging Face Hub
====================

Checkpoints are loaded from the Hugging Face Hub through ``transformers``. Models and
their processors are paired by :class:`~caresmcp.api.models.hf_hub.HFModelType`, while
individual repository files (JSON metadata and tensor checkpoints) can be fetched
directly.

🧩 Requires the ``hf`` extra.

🤖 Models and processors
------------------------

.. autofunction:: caresmcp.api.models.hf_hub.get_hf_model

.. autofunction:: caresmcp.api.models.hf_hub.get_hf_model_and_processor

.. autoclass:: caresmcp.api.models.hf_hub.HFModelType
   :members:

💾 Checkpoint files
-------------------

.. autofunction:: caresmcp.api.models.hf_hub.get_hf_checkpoint

.. autofunction:: caresmcp.api.models.hf_hub.get_hf_json
