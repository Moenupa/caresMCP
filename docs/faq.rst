❓ FAQ
======

📦 Do I need to install every model backend?
--------------------------------------------

No. Backends are optional extras grouped by modality (``xray``, ``ct``, ``mri``,
``dsa``, ``ultrasound``, ``pathology``) or by model backend (``hf``, ``monai``,
``timm``). The base install provides the CLI and the MCP server; loader modules import
their backend lazily, so missing extras only affect the tools that need them, and the
server logs a warning and skips those tools. See :doc:`installation`.

🖥️ Can I run CaresMCP without a GPU?
------------------------------------

Yes. The active device is CUDA when torch reports one and CPU otherwise. Tools still run
on CPU, though large models take noticeably longer. Loaders that cache a CPU master
model — such as the Hugging Face Hub, timm, torch.hub, and TorchXRayVision loaders —
return an independent evaluation-mode copy on the active device; uncached loaders such
as FleXray, MONAI, UMAIR, and the BioViL image encoder build a fresh model per call. See
:doc:`sources/index`.

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

🌑 Why are some rows dimmed in ``caresmcp list``?
-------------------------------------------------

A dimmed row marks a tool that is registered but not currently servable: its docstring
is still a TODO placeholder, or an optional package required by the tool is missing. The
MCP server logs a warning and skips these tools. Installing the matching extra (or
``caresmcp[all]``) clears dependency-related dimming. See :doc:`installation` and
:doc:`quick-start`.

🔑 How do I access gated Hugging Face checkpoints?
--------------------------------------------------

Set ``HF_TOKEN`` to a Hugging Face access token with access to the repository before
starting the CLI or MCP server. CaresMCP does not manage this variable; see
:doc:`envvar`.
