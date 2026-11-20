⚠️ Limitations and Safety
=========================

CaresMCP is research software for building and evaluating medical imaging agents. It is
not a medical device, and its outputs are not clinical diagnoses. Do not use it for
patient care without independent validation and the appropriate regulatory review.

🧪 Intended Use
---------------

CaresMCP exposes upstream research checkpoints and classical image tools to an agent. It
does not create a validated clinical workflow, does not replace a radiologist or
pathologist, and does not establish that a tool is appropriate for a particular
population, scanner, or acquisition protocol.

🧠 Model Weights and Access Terms
---------------------------------

Each checkpoint carries its own license and access conditions. Some are gated on Hugging
Face, some are limited to non-commercial research, and some are released under copyleft
or no-derivatives terms. The :doc:`coverage/index` pages record the terms known at
integration time. Check the upstream model card before using a checkpoint in a new
setting.

📥 Input Scope
--------------

Many tools accept one 2D image, slice, or frame. Volumetric or series-level processing
is available only where a tool documents it, such as TotalSegmentator anatomy
segmentation and the video-level echocardiography tools. Passing a volume to a slice
tool, or a series to a single-frame tool, is outside the supported input contract.

📤 Output Interpretation
------------------------

- Scores are model outputs, not calibrated clinical probabilities. Their scale and
  operating point depend on the upstream checkpoint and training data.
- Masks and heatmaps can cover a center crop or a resampled image rather than the
  original acquisition grid. Read the tool docstring for coordinate and crop notes.
- Image outputs document their colors or provide a legend. Do not infer a diagnosis from
  a color without reading that documentation.
- The :doc:`caresagentbench` page records a work-in-progress pilot study, not a clinical
  validation.

💾 Downloads, Hardware, and Time
--------------------------------

The first model-backed call downloads its checkpoint into the CaresMCP cache (default
``~/.cache/caresmcp``). Later calls reuse the cache. CPU execution is supported, but
large models take noticeably longer; CUDA is used when torch reports an available
device. See :doc:`envvar` for cache relocation and HTTP timeouts.

🔐 Privacy and Network Behavior
-------------------------------

- Local files are read by the local CaresMCP process. CaresMCP does not upload them to a
  remote service.
- HTTP(S) inputs are fetched from the URL you provide. Use a URL only for data you are
  permitted to retrieve.
- Model weights are downloaded from their upstream sources, typically the Hugging Face
  Hub or a project release.
- ``caresmcp mcp`` binds to ``127.0.0.1`` by default. If you bind it to another
  interface, add your own authentication and access controls; the development server is
  not a hardened multi-tenant service.

🚫 Not Included
---------------

CaresMCP does not run LLM or VLM inference, does not train models, and does not provide
PACS, EHR, or DICOM series-management integration. Pair it with the agent and data
infrastructure you already use.

See :doc:`faq` for common setup questions and :doc:`coverage/index` for model-specific
caveats.
