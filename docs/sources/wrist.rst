🦴 Pediatric wrist detector
===========================

The pediatric wrist fracture detector is a fixed ONNX release downloaded from GitHub,
verified against a pinned SHA-256 digest, and cached under :envvar:`CARESMCP_CACHE` (see
:ref:`environment-variables`). The returned inference session uses the CPU execution
provider.

🧩 Requires the ``xray`` extra.

🧱 Models
---------

.. autofunction:: caresmcp.api.models.wrist.get_wrist_onnx_model
