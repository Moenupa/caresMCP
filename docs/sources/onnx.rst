🖥️ ONNX Runtime
===============

ONNX checkpoints are downloaded from a pinned URL, verified against a SHA-256 digest,
and cached under :envvar:`CARESMCP_CACHE` (see :ref:`environment-variables`). The loader
returns an ONNX Runtime inference session that is not retained in memory, because a
session may hold CUDA resources; callers that run inference repeatedly should keep and
reuse the returned instance.

Execution providers follow the globally managed ``caresmcp.constants.DEVICE``: CUDA with
CPU fallback when both the device and the installed ONNX Runtime support it, otherwise
the CPU execution provider. Using CUDA requires an ONNX Runtime build that provides
``CUDAExecutionProvider``, such as ``onnxruntime-gpu``.

🧩 Requires the ``xray`` extra.

🧱 Models
---------

.. autofunction:: caresmcp.api.models.onnx.get_onnx_session
