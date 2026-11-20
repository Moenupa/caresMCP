📦 Installation
===============

CaresMCP requires **Python 3.12** and uses `uv <https://docs.astral.sh/uv/>`_ for
dependency management. The package is installed from Git while the PyPI release is
pending.

🧰 Install as a uv tool
-----------------------

Install CaresMCP as a `uv tool <https://docs.astral.sh/uv/concepts/tools/>`_. This adds
the ``caresmcp`` executable to your ``PATH`` in an isolated environment, without cloning
the repository:

.. code-block:: console

   $ uv tool install git+https://github.com/Moenupa/caresMCP.git

The base install provides the CLI, the MCP server, and the PIL-based image utilities.
Model backends are optional and grouped into extras:

.. code-block:: console

   $ uv tool install "caresmcp[xray] @ git+https://github.com/Moenupa/caresMCP.git"
   $ uv tool install "caresmcp[xray,ct] @ git+https://github.com/Moenupa/caresMCP.git"
   $ uv tool install "caresmcp[all] @ git+https://github.com/Moenupa/caresMCP.git"

Reinstall to update the tool to the latest commit:

.. code-block:: console

   $ uv tool upgrade caresmcp --reinstall

🧩 Choose an extra
------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   - - Extra
     - What it enables
   - - ``xray``
     - Chest X-ray and musculoskeletal radiographs: classification, localization,
       segmentation, embeddings, phrase grounding, and reconstruction-error maps.
   - - ``ct``
     - Head and chest CT: hemorrhage scoring, slice classification, embeddings,
       radiomics, and HU-based image utilities.
   - - ``dsa``
     - Coronary stenosis detection and vessel segmentation in catheter angiography.
   - - ``mri``
     - Cardiac, stroke, and brain-tumor segmentation and slice registration.
   - - ``ultrasound``
     - Echocardiography measurements, ejection fraction, left-ventricle segmentation,
       and frame/video embeddings.
   - - ``pathology``
     - Nuclei and tumor analysis, pathology embeddings, PRISM/PRISM2, CONCH, and
       Cellpose.
   - - ``all``
     - Every modality extra above.

Generic medical-AI and mammography tools use the backend extras ``hf``, ``monai``,
``timm``, ``sam``, ``skimage``, ``totalseg``, ``itk``, and ``clip``. Install a backend
extra directly when you want a smaller environment:

.. code-block:: console

   $ uv tool install "caresmcp[hf] @ git+https://github.com/Moenupa/caresMCP.git"

See ``pyproject.toml`` for the full dependency tree behind each extra.

⏬ Install from Source
----------------------

1. Clone the repository and install the base dependencies:

   .. code-block:: console

      $ git clone https://github.com/Moenupa/caresMCP.git
      $ cd caresMCP
      $ uv sync

2. 🧩 Install the model backends you need. Pass one ``--extra`` flag per group, or
   install them all:

   .. code-block:: console

      $ uv sync --extra xray --extra ct
      $ uv sync --all-extras

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

💾 First Run and Cache
----------------------

The first model-backed call downloads its checkpoint into the CaresMCP cache (default
``~/.cache/caresmcp``). Later calls reuse the cache. Set :envvar:`CARESMCP_CACHE` to
relocate the model tree, :envvar:`CARESMCP_LRU_SIZE` to bound in-memory model caches,
and ``HF_TOKEN`` for gated Hugging Face repositories; see :doc:`envvar`.

.. note::

   🚧 Installation from `PyPI <https://pypi.org/>`_ is coming soon. Until then, use the
   Git URL shown above.
