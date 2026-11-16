🧩 Supported Sources
====================

CaresMCP accesses every checkpoint through the loaders in ``caresmcp.api.models``. Each
loader covers one model source and standardizes how its weights are fetched, cached, and
returned, so tool modules only contain model-specific preprocessing, inference, and
postprocessing.

⚙️ Shared Behavior
------------------

- 🔌 **Lazy imports** — loader modules import their backend lazily, so the package can be
  installed and documented without every optional dependency.
- 💾 **Bounded caching** — in-memory caches are bounded by
  ``caresmcp.constants.LRU_MAX_SIZE``. Hub checkpoints and TorchXRayVision weights are
  stored under ``CACHE_ROOT/models`` (``CACHE_ROOT`` defaults to ``.cache``); PyTorch Hub
  keeps its downloads in torch's own cache.
- 🧪 **Evaluation ready** — model-returning loaders hand out models in evaluation mode,
  moved to ``caresmcp.constants.DEVICE`` when that is not the CPU.
- ♻️ **Device copies** — the transformers, timm, PyTorch Hub, and TorchXRayVision loaders
  cache a master model on CPU and return a device copy; MONAI caches the checkpoint
  through the Hub loader and rebuilds its architecture on every call.

.. list-table::
   :header-rows: 1

   - - Loader
     - Backend
     - Entry points
   - - :doc:`hf-hub`
     - transformers
     - ``get_hf_model``, ``get_hf_model_and_processor``, ``get_hf_checkpoint``,
       ``get_hf_json``
   - - :doc:`timm`
     - timm
     - ``get_timm_model``, ``build_timm_model``
   - - :doc:`torch-hub`
     - torch.hub
     - ``get_torch_model``, ``load_torch_checkpoint``
   - - :doc:`torchxrayvision`
     - torchxrayvision
     - ``get_torchxrayvision_model``
   - - :doc:`monai`
     - MONAI
     - ``get_monai_model``

.. toctree::
   :maxdepth: 1

   hf-hub
   timm
   torch-hub
   torchxrayvision
   monai
