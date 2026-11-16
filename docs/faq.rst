❓ FAQ
======

Do I need to install every model backend? 📦
--------------------------------------------

No. Backends are optional extras grouped by modality, such as ``xray``, ``mri``, ``ct``,
``ultrasound``, and ``pathology``. The base install provides the CLI and the MCP server;
loader modules import their backend lazily, so missing extras only affect the tools that
need them. See :doc:`installation`.

How do I connect an MCP client? 🔌
----------------------------------

Run ``uv run caresmcp mcp`` and point the client at ``http://127.0.0.1:8000/mcp``. Use
``--host``, ``--port``, and ``--path`` to change the address. Client examples are in
:doc:`quick-start`.

Where are model weights cached? 💾
----------------------------------

Checkpoints are stored under ``CACHE_ROOT/models``, and ``CACHE_ROOT`` defaults to
``.cache`` in the working directory. Set the ``CACHE_ROOT`` environment variable to
relocate the cache. The Hugging Face Hub and Cellpose caches follow this directory,
while PyTorch Hub keeps its downloads in torch's own cache.

Can I run CaresMCP without a GPU? 🖥️
------------------------------------

Yes. The active device is CUDA when torch reports one and CPU otherwise. Tools still run
on CPU, though large models take noticeably longer. Some loaders keep a master model on
CPU and return a device copy on every call.

How do I try a single tool without an MCP client? 🧪
----------------------------------------------------

Use ``uv run caresmcp debug <category> <tool> kwargs.json``. The file may contain one
JSON object or one JSON object per line; byte-valued inputs such as ``image`` accept a
local path, a base64 data URI, or an HTTP URL. Outputs are written under
``.cache/<category>/``.

Does CaresMCP run LLMs or VLMs? 🤖
----------------------------------

No. Inference for language and vision-language models belongs to their dedicated APIs.
CaresMCP wraps medical imaging models and utilities — including CLIP-style image and
text encoders — so an agent can pair them with any LLM it already uses.
