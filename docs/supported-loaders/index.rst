🧩 Supported Sources
====================

CaresMCP accesses every checkpoint through the loaders in ``caresmcp.api.models``. Each
loader covers one model source and standardizes how its weights are fetched, cached, and
returned, so tool modules only contain model-specific preprocessing, inference, and
postprocessing.

⚙️ Shared Behavior
------------------

- 🔌 **Lazy imports** — loader modules import their backend lazily, so the package can
  be installed and documented without every optional dependency.
- 💾 **Bounded caching** — in-memory caches are bounded by :envvar:`CARESMCP_LRU_SIZE`;
  downloaded weights and checkpoints follow the :ref:`environment-variables` cache
  layout.
- 🧪 **Evaluation ready** — model-returning loaders hand out models in evaluation mode,
  moved to ``caresmcp.constants.DEVICE`` when that is not the CPU.
- 🔬 **Fresh Cellpose models** — Cellpose builds a new model on ``DEVICE`` for every
  call, while reusing weights already downloaded to the model cache.

.. list-table::
   :header-rows: 1

   - - Loader
     - Backend
   - - :doc:`hf-hub`
     - :external+transformers:doc:`transformers <index>`
   - - :doc:`timm`
     - :external+timm:doc:`timm <index>`
   - - :doc:`torch-hub`
     - :external+torch:doc:`torch.hub <hub>`
   - - :doc:`torchxrayvision`
     - :external+torchxrayvision:doc:`torchxrayvision <index>`
   - - :doc:`hi-ml`
     - :external+hi-ml:doc:`hi-ml-multimodal <multimodal>`
   - - :doc:`flexray`
     - `flexray <https://github.com/VictorButoi/FleXray/tree/main/docs>`_
   - - :doc:`wrist`
     - :external+onnxruntime:doc:`ONNX Runtime <index>`
   - - :doc:`cellpose`
     - :external+cellpose:doc:`Cellpose <index>`
   - - :doc:`monai`
     - :external+monai:doc:`MONAI <index>`

.. toctree::
   :hidden:

   hf-hub
   timm
   torch-hub
   torchxrayvision
   hi-ml
   flexray
   wrist
   cellpose
   monai
