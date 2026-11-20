🧠 Concepts
===========

This page defines the vocabulary used throughout the docs and explains what a call
accepts and returns.

🛠️ Tools, Categories, and Names
-------------------------------

A **tool** is one callable function registered with CaresMCP. Each tool has a stable
name, such as ``torchxrayvision_classify``. The MCP server exposes that name directly;
the CLI also requires the tool's **category**:

.. code-block:: console

   $ caresmcp debug xray torchxrayvision_classify '{"image_or_dicom": "..."}'

A **category** groups tools for listing and CLI invocation, such as ``xray``, ``ct``, or
``pathology``. Categories are a navigation aid, not part of the MCP tool name. Use
``caresmcp list`` to see the category, name, models, task tags, and summary for every
registered tool.

🏷️ Tasks
--------

A **task** describes the kind of evidence a tool returns. The registry uses these tags:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   - - Task
     - Typical evidence
   - - Classify
     - Label probabilities or scores.
   - - Regress
     - Quantitative measurements.
   - - Segment
     - Masks, usually returned as image bytes.
   - - Localize
     - Heatmaps, boxes, or ranked regions.
   - - Retrieve
     - Similarity scores or rankings between images and text or other images.
   - - Embed
     - Feature vectors for downstream comparison.
   - - Caption
     - Text descriptions.
   - - Denoise
     - A filtered image.
   - - Evaluate
     - Unsupervised signals such as reconstruction-error maps.

A tool can carry more than one task. Tools without a task tag are analysis helpers or
image utilities. The :doc:`capabilities/catalog` groups tools by task and the
:doc:`capabilities/by-modality` page groups them by category.

🧠 Models, Checkpoints, and Loaders
-----------------------------------

A **model** or **checkpoint** is the upstream network behind a tool. Some tools accept a
model choice through a ``weights``, ``model_id``, or similar parameter; generic tools
such as ``hf_classifier`` accept any compatible checkpoint. The :doc:`coverage/index`
pages record which models are integrated, which are deferred, and their access terms.

A **loader** standardizes how checkpoints are downloaded, cached, and placed on the
active device. CaresMCP uses one loader per model source, so tool modules only contain
model-specific preprocessing, inference, and postprocessing. See :doc:`sources/index`.

📥 Inputs
---------

Tools that accept image bytes declare a parameter such as ``image`` or
``image_or_dicom``. Accepted sources differ by surface:

- **CLI** — a local path, a base64 data URI, or an HTTP(S) URL.
- **MCP** — a base64 data URI or an HTTP(S) URL as a string. A server-side local path is
  not accepted over MCP; a client that has a local file should send its bytes as a data
  URI.

The exact input contract for a tool is documented in its API entry; the parameter
description also appears in ``caresmcp debug <category> <tool> help``.

📤 Outputs
----------

Tools return JSON-serializable values or image bytes:

- **Structured results** — dictionaries and lists with scores, labels, masks metadata,
  measurements, or embeddings. They are returned as JSON over MCP.
- **Image results** — masks, heatmaps, and other rasters. Over MCP they are returned as
  image content; with ``caresmcp debug`` they are written to the current directory as
  ``<category>-<tool>.<ext>``, with a numeric suffix when one call runs several inputs.
- **Files** — non-image bytes are returned as file content over MCP and saved by
  ``caresmcp debug``.

Image-producing tools document their colors or provide a legend output. For medical
interpretation, read the tool's docstring or the corresponding :doc:`api/index` entry
rather than assuming that a mask color encodes a diagnosis.

🧩 Availability and Extras
--------------------------

Model backends are optional and grouped into extras. A tool is **dimmed** in ``caresmcp
list`` and skipped by the MCP server when its docstring is still a TODO placeholder or
when a package it declares is missing. Installing the matching extra clears
dependency-related dimming; a TODO placeholder stays unavailable until the tool is
implemented. See :doc:`installation` and :doc:`quick-start`.

.. note::

   Missing extras never change a tool's name or input contract. They only determine
   whether the tool is currently servable in your environment.
