🔥 PyTorch Hub
===============

Models are loaded from GitHub repositories or local directories with
``torch.hub.load``. The loader trusts the requested repository and skips
torch.hub's validation step, so callers should pin repositories they control.
Checkpoint state dicts can also be downloaded directly from a URL.

🧩 Requires the ``torch`` extra.

🧱 Models
---------

.. autofunction:: caresmcp.api.models.torch_hub.get_torch_model

💾 Checkpoint files
-------------------

.. autofunction:: caresmcp.api.models.torch_hub.load_torch_checkpoint
