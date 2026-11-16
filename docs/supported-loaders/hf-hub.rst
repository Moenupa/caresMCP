🤗 Hugging Face Hub
===================

Models are loaded through ``transformers``; individual repository files (JSON metadata
and tensor checkpoints) are fetched with ``huggingface_hub``. Models and their
processors are paired by :class:`~caresmcp.api.models.hf_hub.HFModelType`.

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
