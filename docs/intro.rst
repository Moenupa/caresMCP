📖 Introduction
===============

CaresMCP brings dedicated medical ML models and tools to agents through the Model
Context Protocol (MCP). It targets the bottlenecks that hold medical vision-language
models back — medical reasoning, data sourcing, and AI explainability — so that
diagnostic conclusions stay traceable to evidence and even to individual images.

The same tool interface doubles as a foundation for test-time training and self-evolving
agent workflows.

🧰 What's Inside
----------------

- 🧠 **Workflow APIs** — 50+ end-to-end tools for classification, segmentation,
  localization, retrieval, embeddings, denoising, and image utilities.
- 🔌 **Framework adapters** — unified loaders for ``transformers``, MONAI, timm,
  ``torch.hub``, etc., so new models only need thin wrappers for inference logic.
- 🩻 **Modality coverage** — MRI, X-ray and CT, echocardiography and ultrasound,
  mammography, DSA and catheter angiography, pathology, and musculoskeletal imaging.
- 📤 **Agent-friendly outputs** — JSON-serializable results, with masks, heatmaps, and
  other image outputs accompanied by legends or descriptions that explain their colors.

🔍 Why Traceable Evidence?
--------------------------

A VLM alone can be unsure or distracted by spurious findings. With CaresMCP, an agent
can cross-check its reading against dedicated models:

1. 🎯 **Classify** the image with an expert classifier to obtain label probabilities.
2. 🗺️ **Localize** the finding to inspect the visual evidence as a heatmap.
3. 🧮 **Calibrate** the final answer by combining the model output with that evidence.

Every conclusion then points back to the tool result and the image region supporting it,
which keeps the diagnostic trail auditable.

🗺️ Explore the Docs
-------------------

- :doc:`installation` — set up CaresMCP and its optional model backends.
- :doc:`quick-start` — list tools, start the MCP server, and run a tool.
- :doc:`tools/index` — browse tools for each clinical category.
- :doc:`sources/index` — learn how checkpoints are fetched and cached.
