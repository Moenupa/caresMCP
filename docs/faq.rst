❓ FAQ
======

📦 Do I need to install every model backend?
--------------------------------------------

No. Backends are optional extras grouped by modality, such as ``xray``, ``mri``, ``ct``,
``ultrasound``, and ``pathology``. The base install provides the CLI and the MCP server;
loader modules import their backend lazily, so missing extras only affect the tools that
need them. See :doc:`installation`.

🖥️ Can I run CaresMCP without a GPU?
------------------------------------

Yes. The active device is CUDA when torch reports one and CPU otherwise. Tools still run
on CPU, though large models take noticeably longer. On CUDA, some loaders keep a master
model on CPU and return an independent eval-mode copy; on CPU they return the cached
instance itself.

🧪 How do I try a single tool without an MCP client?
----------------------------------------------------

Run ``caresmcp debug <category> <tool> <kwargs>``; pass ``help`` as the last argument to
print the tool's signature. See :doc:`quick-start` for accepted input formats and how
outputs are saved.

🤖 Does CaresMCP run LLMs or VLMs?
----------------------------------

No. Inference for language and vision-language models belongs to their dedicated APIs.
CaresMCP wraps medical imaging models and utilities — including CLIP-style image and
text encoders — so an agent can pair them with any LLM it already uses.
