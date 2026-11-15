Usage
=====

List tools
----------

``caresmcp list`` prints every registered tool grouped by category, with the supported
models, tags, and a short description:

.. code-block:: console

   $ uv run caresmcp list

Pass a category to narrow the listing, and an output path to also write it as
tab-separated values:

.. code-block:: console

   $ uv run caresmcp list xray --output xray-tools.tsv

Run the MCP server
------------------

Start the FastMCP server over streamable HTTP:

.. code-block:: console

   $ uv run caresmcp mcp

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

Call a tool directly
--------------------

``caresmcp debug`` runs a single registered tool without an MCP client. It takes the
category, the tool name, and a JSON or JSON Lines file of keyword arguments:

.. code-block:: console

   $ uv run caresmcp debug xray torchxrayvision_classify kwargs.json

Byte-valued inputs (for example ``image``) accept a local path, a base64 data URI, or an
HTTP URL. Results are written under ``.cache/<category>/``.
