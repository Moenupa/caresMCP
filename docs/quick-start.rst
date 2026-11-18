🚀 Quick Start
==============

🏁 Get from an installed CaresMCP to a running MCP server in three steps.

📋 List Available Tools
-----------------------

``caresmcp list`` prints every registered tool grouped by category, with its supported
models, tags, and a short description:

.. code-block:: console

   $ caresmcp list

Tools are dimmed when their docstring is still a TODO placeholder or when an optional
dependency is missing; dimmed tools are not served over MCP. Installing the matching
extra (or ``caresmcp[all]``) supplies the dependencies, but TODO placeholders stay
dimmed until implemented.

Pass a category to narrow the listing, and an output path to also write it as
tab-separated values:

.. code-block:: console

   $ caresmcp list xray --output xray-tools.tsv

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

To inspect and call tools interactively in a browser instead, run ``caresmcp mcpdev``
(requires Node.js and ``npx``); see :doc:`installation` for more options.

🧪 Call a Tool Directly
-----------------------

``caresmcp debug`` runs a single registered tool without an MCP client. It takes the
category, the tool name, and the tool's keyword arguments as inline JSON, a JSON or JSON
Lines file, or ``help`` to print the tool's signature:

.. code-block:: console

   $ caresmcp debug xray torchxrayvision_classify \
       '{"image_or_dicom": "https://nih-chest-x-rays.s3.us-east-2.amazonaws.com/images_1024x1024/00000003_001.png"}'
   $ caresmcp debug xray torchxrayvision_classify kwargs.json
   $ caresmcp debug xray torchxrayvision_classify help

Inline text may hold a single JSON object, a list of objects, or one object per line. A
``.json`` file holds one object and a ``.jsonl`` file one object per line.

.. tip::

   Byte-valued inputs (for example ``image``) accept a local path, a base64 data URI, or
   an HTTP URL. Each output is written to the current directory as
   ``<category>-<tool>.<ext>`` (use ``--save-dir DIR`` to change the directory) and
   printed after the run. When one call runs several keyword-argument entries, outputs
   gain ``_1``, ``_2``, ... suffixes.
