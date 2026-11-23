🚀 Getting Started
==================

Get from a clean environment to a tool call and a running MCP server. This page uses a
chest X-ray example because the ``xray`` extra is small; the same steps work for any
category.

📋 Prerequisites
----------------

- Python 3.12 and `uv <https://docs.astral.sh/uv/>`_.
- Network access on the first run: model checkpoints are downloaded into the CaresMCP
  cache (default ``~/.cache/caresmcp``).
- A GPU is optional. CUDA is used when torch reports an available device; otherwise
  tools run on CPU and large models take longer.

📦 Install
----------

Install CaresMCP from `PyPI <https://pypi.org/project/caresmcp/>`_ as a `uv tool
<https://docs.astral.sh/uv/concepts/tools/>`_ with the extra for the category you want
to try:

.. code-block:: console

   $ uv tool install "caresmcp[xray]"

Use ``caresmcp[all]`` to install every backend, or pass several extras. See
:doc:`installation` for the extra list and the source-install path.

📋 List Available Tools
-----------------------

``caresmcp list`` prints every registered tool grouped by category, with the models it
accepts, its task tags, and a short description. The **Name** column is the MCP tool
name used by agents.

.. code-block:: console

   $ caresmcp list xray

   Categ │ Name                     │ Models                         │ Tags     │ Description
   xray  │ biovil_ground_phrases    │ microsoft/BiomedVLP-CXR-BERT-… │ LOC, RET │ Locate and rank …
   xray  │ torchxrayvision_classify │ densenet121-res224-all, …      │ CLS      │ Score chest radiograph …
   xray  │ xray_image_embed         │ microsoft/rad-dino, …          │ EMB      │ Generate an embedding …
   …

Rows are dimmed when a tool's docstring is still a TODO placeholder or when an optional
package is missing; dimmed tools are not served over MCP. Installing the matching extra
(or ``caresmcp[all]``) clears dependency-related dimming. Pass ``--output path.tsv`` to
also write the listing as tab-separated values.

.. _run-one-tool:

🧪 Run One Tool Directly
------------------------

``caresmcp debug`` runs a registered tool without an MCP client. It takes the category,
the tool name, and the tool's keyword arguments as inline JSON, a JSON or JSON Lines
file, or ``help`` to print the tool's signature. Start with an image utility that needs
no model download:

.. code-block:: console

   $ caresmcp debug image inspect \
       '{"image": "https://nih-chest-x-rays.s3.us-east-2.amazonaws.com/images_1024x1024/00000003_001.png"}'
   Debugging 'image' 'inspect'
   01. {'image': 'https://nih-chest-x-rays.s3.us-east-2.amazonaws.com/…'}
       Results: PosixPath('image-inspect.json')

The result is written to the current directory and printed after the run. The saved JSON
contains the image metadata:

.. code-block:: json

   {
     "format": "PNG",
     "width": 1024,
     "height": 1024,
     "mode": "L"
   }

Now run a model-backed tool. The first call downloads the checkpoint:

.. code-block:: console

   $ caresmcp debug xray torchxrayvision_classify \
       '{"image_or_dicom": "https://nih-chest-x-rays.s3.us-east-2.amazonaws.com/images_1024x1024/00000003_001.png"}'
   Debugging 'xray' 'torchxrayvision_classify'
   01. {'image_or_dicom': 'https://nih-chest-x-rays.s3.us-east-2.amazonaws.com/…'}
       Results: PosixPath('xray-torchxrayvision_classify.json')

The saved JSON maps each finding to an independent score:

.. code-block:: json

   {
     "model_id": "densenet121-res224-all",
     "score_mode": "sigmoid",
     "scores": {
       "Atelectasis": 0.2836,
       "Consolidation": 0.1119,
       "Emphysema": 0.5067,
       "Pneumothorax": 0.1086,
       "Hernia": 0.06
     }
   }

A full response includes every label the checkpoint can score. Labels the checkpoint
cannot score are omitted or reported as ``null``.

.. tip::

   Byte-valued inputs (for example ``image``) accept a local path, a base64 data URI, or
   an HTTP URL in the CLI. Each output is written as ``<category>-<tool>.<ext>``; use
   ``--save-dir DIR`` to change the directory. When one call runs several
   keyword-argument entries, outputs gain ``_1``, ``_2``, ... suffixes.

🔌 Serve with MCP
-----------------

Start the FastMCP server over streamable HTTP:

.. code-block:: console

   $ caresmcp mcp  # pass --help for host, port, and transport options

The server listens on ``http://127.0.0.1:8000/mcp``. Point an MCP client at that URL,
for example:

.. code-block:: console

   $ claude mcp add --transport http caresmcp http://127.0.0.1:8000/mcp

Or in a JSON client configuration:

.. code-block:: json

   {
     "mcpServers": {
       "caresmcp": {
         "type": "http",
         "url": "http://127.0.0.1:8000/mcp"
       }
     }
   }

The server exposes the **Name** values from ``caresmcp list`` directly as MCP tools.
Tools whose optional packages are missing are skipped at startup with a warning.

To inspect and call tools interactively in a browser instead, run ``caresmcp mcpdev``
(requires Node.js and ``npx``); see :doc:`installation` for more options.

🩺 Troubleshooting
------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   - - Symptom
     - Fix
   - - A tool is dimmed in ``caresmcp list``.
     - Install the extra shown by the tool's requirement, or ``caresmcp[all]``.
   - - A tool name is rejected by ``caresmcp debug``.
     - Pass the category first, for example ``caresmcp debug xray <tool> help``.
   - - The first model-backed call is slow.
     - The checkpoint is downloading. Later calls reuse the cache; set
       :envvar:`CARESMCP_CACHE` to relocate it.
   - - A gated Hugging Face checkpoint fails to load.
     - Set ``HF_TOKEN`` before starting the CLI or server; see :doc:`envvar`.
   - - ``caresmcp mcp`` cannot bind its port.
     - Pass ``--host`` and ``--port`` to select an available address.

Next, browse :doc:`capabilities/index` by task or modality, or read :doc:`concepts` for
the input and output contract.
