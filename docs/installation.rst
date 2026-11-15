Installation
============

.. note::

   Installation from `PyPI <https://pypi.org/>`_ is coming soon.

CaresMCP requires Python 3.12 and uses `uv <https://docs.astral.sh/uv/>`_ for dependency
management.

Clone the repository and install the base dependencies:

.. code-block:: console

   $ git clone https://github.com/Moenupa/caresMCP.git
   $ cd caresMCP
   $ uv sync

Model backends are optional and grouped into extras by modality, so install only the
ones you need:

.. code-block:: console

   $ uv sync --extra xray --extra mri
   $ uv sync --all-extras

See ``pyproject.toml`` for the full list of extras. Verify the installation by listing
the registered tools:

.. code-block:: console

   $ uv run caresmcp list

Verify Installation
-------------------

You may run MCP inspector to examine the interface and test the functionality of each
tool by:

.. code-block:: console

   $ uv run fastmcp dev inspector mcpserver.py
