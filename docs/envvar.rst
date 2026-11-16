.. _environment-variables:

⚙️ Environment Variables
========================

Set these variables before starting the CLI or MCP server; CaresMCP reads them at import
time. Values already present in the environment always win — CaresMCP only fills in
variables that are unset.

.. envvar:: CARESMCP_CACHE

   Root directory for downloaded model files and test-helper artifacts. Default:
   ``~/.cache/caresmcp`` (``~`` is expanded; the model tree is created at import).

   Unless already set, each source-specific variable is defaulted into this tree, so one
   value relocates all model storage:

   .. list-table::
      :header-rows: 1

      * - Variable
        - Default
        - Used by
      * - :envvar:`HF_HUB_CACHE`
        - ``$CARESMCP_CACHE/models/hf``
        - :doc:`hf-hub </sources/hf-hub>`, :doc:`timm </sources/timm>`,
          :doc:`hi-ml </sources/hi-ml>`, :doc:`flexray </sources/flexray>`,
          :doc:`monai </sources/monai>`
      * - :envvar:`HF_MODULES_CACHE`
        - ``$CARESMCP_CACHE/models/modules``
        - :doc:`hf-hub </sources/hf-hub>`
      * - :envvar:`CELLPOSE_LOCAL_MODELS_PATH`
        - ``$CARESMCP_CACHE/models/cellpose``
        - :doc:`cellpose </sources/cellpose>`
      * - :envvar:`TORCH_HOME`
        - ``$CARESMCP_CACHE/models``
        - :doc:`torch-hub </sources/torch-hub>`
      * - :envvar:`TOTALSEG_HOME_DIR`
        - ``$CARESMCP_CACHE/models/totalseg``
        - :doc:`medai </tools/medai>`
      * - —
        - ``$CARESMCP_CACHE/models``
        - :doc:`onnx </sources/onnx>`, :doc:`torchxrayvision
          </sources/torchxrayvision>`

.. envvar:: CARESMCP_LRU_SIZE

   Maximum number of entries retained by each in-memory model cache. Default: ``2``.

.. envvar:: CARESMCP_TIMEOUT

   Timeout in seconds for HTTP requests made when reading byte inputs. Default: ``15``.

.. envvar:: HF_HUB_CACHE

   Hugging Face Hub cache directory. CaresMCP defaults it to
   ``$CARESMCP_CACHE/models/hf`` when it is not already set.

.. envvar:: HF_MODULES_CACHE

   Directory for remote code cached when loading models with ``trust_remote_code=True``.
   CaresMCP defaults it to ``$CARESMCP_CACHE/models/modules`` when it is not already set.

.. envvar:: CELLPOSE_LOCAL_MODELS_PATH

   Directory for Cellpose weights. CaresMCP defaults it to
   ``$CARESMCP_CACHE/models/cellpose`` when it is not already set.

.. envvar:: TORCH_HOME

   Base directory for PyTorch Hub downloads; PyTorch Hub stores clones and checkpoints
   under ``$TORCH_HOME/hub``. CaresMCP defaults it to ``$CARESMCP_CACHE/models`` when it
   is not already set.

.. envvar:: TOTALSEG_HOME_DIR

   Directory for TotalSegmentator's config file and nnU-Net weights. CaresMCP defaults
   it to ``$CARESMCP_CACHE/models/totalseg`` when it is not already set.

The following Hugging Face variables are not managed by CaresMCP:

.. envvar:: HF_TOKEN

   Access token for gated and private repositories. See the `Hugging Face documentation
   <https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hftoken>`_.

.. envvar:: HF_XET_CACHE

   Chunk cache for Xet-backed Hub downloads; defaults to ``$HF_HOME/xet``. See the
   `Hugging Face documentation
   <https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfxetcache>`_.
