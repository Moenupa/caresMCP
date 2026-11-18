📦 Installation
===============

.. note::

   🚧 Installation from `PyPI <https://pypi.org/>`_ is coming soon.

CaresMCP requires **Python 3.12** and uses `uv <https://docs.astral.sh/uv/>`_ for
dependency management.

🧰 Install globally
-------------------

Install CaresMCP as a `uv tool <https://docs.astral.sh/uv/concepts/tools/>`_. This adds
the ``caresmcp`` executable to your ``PATH`` in an isolated environment, without cloning
the repository:

.. code-block:: console

   $ uv tool install git+https://github.com/Moenupa/caresMCP.git

Model backends are optional and grouped into extras by modality or backend. Select them with
`caresmcp[extra1,extra2]`.

.. code-block:: console

   $ uv tool install "caresmcp[xray,ct] @ git+https://github.com/Moenupa/caresMCP.git"
   $ uv tool install "caresmcp[all] @ git+https://github.com/Moenupa/caresMCP.git"

Reinstall to update the tool to the latest commit:

.. code-block:: console

   $ uv tool upgrade caresmcp --reinstall

⏬ Install from Source
----------------------

1. Clone the repository and install the base dependencies:

   .. code-block:: console

      $ git clone https://github.com/Moenupa/caresMCP.git
      $ cd caresMCP
      $ uv sync

2. 🧩 Install the model backends you need. Extras are optional and grouped by modality
   or backend; pass one ``--extra`` flag per group, or install them all:

   .. code-block:: console

      $ uv sync --extra xray --extra ct
      $ uv sync --all-extras

See ``pyproject.toml`` for the full list of extras.

✅ Verify Installation
----------------------

.. note::

   If you installed from source, activate the environment first (e.g., ``source
   .venv/bin/activate``), or prefix commands with ``uv run`` (e.g., ``uv run caresmcp
   list``).

List the registered tools to confirm the package is installed correctly:

.. code-block:: console

   $ caresmcp list

Launch MCP Inspector to examine the interface and call each tool interactively (requires
Node.js and ``npx``):

.. code-block:: console

   $ caresmcp mcpdev

The Inspector prints the UI URL in the terminal. Use ``--ui-port`` or ``--server-port``
to change its ports, ``--inspector-version`` to pin the Inspector release, and
``--no-reload`` to disable auto-reload on file changes.
