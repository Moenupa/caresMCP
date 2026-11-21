CaresMCP
========

Medical imaging tools for AI agents, served over the Model Context Protocol (MCP).
CaresMCP wraps dedicated medical models behind 50+ callable tools and returns evidence
an agent can cite: scores, masks, heatmaps, measurements, and embeddings.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 🤖 Connect an agent
      :link: quick-start
      :link-type: doc

      Install CaresMCP, start the MCP server, and add it to a client.

   .. grid-item-card:: 🧰 Browse capabilities
      :link: capabilities/index
      :link-type: doc

      Find a tool by task or modality. The catalog is generated from the
      registry.

   .. grid-item-card:: 🧪 Run one tool
      :link: run-one-tool
      :link-type: ref

      Call a registered tool directly with JSON arguments and inspect its
      output.

   .. grid-item-card:: 🧩 Add a tool or model
      :link: contributing
      :link-type: doc

      Register a new tool, add a model survey, and rebuild the docs.

⚡ 60-Second Example
--------------------

.. code-block:: console

   $ uv tool install "caresmcp[xray] @ git+https://github.com/Moenupa/caresMCP.git"
   $ caresmcp list xray
   $ caresmcp debug image inspect \
       '{"image": "https://nih-chest-x-rays.s3.us-east-2.amazonaws.com/images_1024x1024/00000003_001.png"}'
   {
     "format": "PNG",
     "width": 1024,
     "height": 1024,
     "mode": "L"
   }

The tool name is the same one the MCP server exposes. See :doc:`quick-start` for the
full install, serve, and client-configuration steps.

🧰 What's Inside
----------------

- **Clinical categories** — CT, DSA and catheter angiography, mammography, MRI,
  pathology, ultrasound, X-ray, and generic medical-AI and image tools.
- ️**Task-oriented tools** — classification, regression, segmentation, localization,
  retrieval, embeddings, captioning, denoising, and evaluation.
- **Agent-friendly evidence** — JSON scores and measurements, plus masks and heatmaps
  with documented colors.
- **Shared model loaders** — checkpoint download, caching, and device placement are
  handled once per model source; see :doc:`sources/index`.

🧭 Where to Go Next
-------------------

- **New here?** Read :doc:`intro`, then :doc:`concepts`, then :doc:`quick-start`.
- **Choosing a tool?** Start with :doc:`capabilities/index`.
- **Checking provenance?** Use :doc:`coverage/index` and :doc:`sources/index`.
- **Looking up a signature?** Use :doc:`api/index`.
- **Tuning caches or timeouts?** See :doc:`envvar`.
- **Running in production?** Read :doc:`limitations`.

.. toctree::
   :hidden:
   :caption: 🚀 Start Here

   intro
   concepts
   installation
   quick-start
   envvar
   limitations
   faq

.. toctree::
   :hidden:
   :caption: 🧰 Capabilities

   capabilities/index
   coverage/index

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: 🛠 Reference

   contributing
   sources/index
   api/index
