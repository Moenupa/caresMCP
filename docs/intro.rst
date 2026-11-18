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
- 🔌 **Framework adapters** — unified loaders for
  :external+transformers:doc:`transformers <index>`, :external+monai:doc:`MONAI
  <index>`, :external+timm:doc:`timm <index>`, :external+torch:doc:`torch.hub <hub>`,
  and more (see :doc:`sources/index`), so new models only need thin wrappers for
  inference logic.
- 🩻 **Modality coverage** — DSA and catheter angiography, echocardiography and
  ultrasound, mammography, MRI, musculoskeletal imaging, pathology, and X-ray and CT.
- 📤 **Agent-friendly outputs** — JSON-serializable results, with masks, heatmaps, and
  other image outputs accompanied by legends or descriptions that explain their colors.

🔍 Why Traceable Evidence?
--------------------------

A VLM alone can be unsure or distracted by spurious findings. With CaresMCP, an agent
can cross-check its reading against dedicated models. Tools declare the tasks they
perform, and each contributes a different kind of inspectable evidence:

- 🎯 **Classify** and **regress** — label probabilities and quantitative measurements.
- 🗺️ **Localize** and **segment** — heatmaps, detections, or masks showing *where* the
  evidence is.
- 🔎 **Retrieve** and **embed** — similarity scores against other images, tiles, or
  text.
- 🧮 **Evaluate** — reconstruction-error maps as an unsupervised signal.
- 🧹 **Denoise** — artifact and noise removal before evidence is gathered.
- 📝 **Caption** — textual descriptions citable alongside the VLM's reading.

These tasks compose into an evidence trail: an agent can denoise, score, inspect where
the score comes from, compare against references, and calibrate — so its conclusion
stays tied to the tool result and image region supporting it.

🗺️ Explore the Docs
-------------------

- :doc:`installation` — set up CaresMCP and its optional model backends.
- :doc:`quick-start` — list tools, start the MCP server, and run a tool.
- :doc:`tools/index` — browse tools for each clinical category.
- :doc:`sources/index` — learn how checkpoints are fetched and cached.
- :doc:`api/index` — look up tool signatures and implementation details.
