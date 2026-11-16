.. _environment-variables:

⚙️ Environment Variables
========================

Set these variables before starting the CLI or MCP server; CaresMCP reads them at import
time.

.. envvar:: CARESMCP_CACHE

   Root directory for downloaded model files and for artifacts written by the test
   helpers. Default: ``~/.cache/caresmcp``.

   Model files are organized as follows, and the source-specific variables below are
   defaulted into this tree unless they are already set:

   - ``models/hf`` — Hugging Face Hub models and processors.
   - ``models/cellpose`` — Cellpose weights.
   - ``models/hub`` — PyTorch Hub clones and checkpoints.
   - ``models`` — pinned downloads and TorchXRayVision weights.

.. envvar:: CARESMCP_LRU_SIZE

   Maximum number of entries retained by each in-memory model cache. Default: ``8``.

.. envvar:: CARESMCP_TIMEOUT

   Timeout in seconds for HTTP requests made when reading byte inputs. Default: ``15``.

.. envvar:: HF_HUB_CACHE

   Hugging Face Hub cache directory. CaresMCP defaults it to
   ``$CARESMCP_CACHE/models/hf`` when it is not already set.

.. envvar:: CELLPOSE_LOCAL_MODELS_PATH

   Directory for Cellpose weights. CaresMCP defaults it to
   ``$CARESMCP_CACHE/models/cellpose`` when it is not already set.

.. envvar:: TORCH_HOME

   Base directory for PyTorch Hub downloads; PyTorch Hub stores clones and checkpoints
   under ``$TORCH_HOME/hub``. CaresMCP defaults it to ``$CARESMCP_CACHE/models`` when it
   is not already set.
