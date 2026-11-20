🌐 Browse
=========

CaresMCP exposes one interface for many medical imaging tasks. Use the task-first
:doc:`catalog` to find a tool by what you want to do, or the :doc:`by-modality` view to
browse by clinical category. Both pages are generated from the tool registry, so they
reflect the registered tool set.

🎯 Browse by task
-----------------

- :ref:`capabilities-classify` — label probabilities or scores.
- :ref:`capabilities-regress` — quantitative measurements.
- :ref:`capabilities-segment` — masks as PNG bytes.
- :ref:`capabilities-localize` — heatmaps, boxes, or ranked regions.
- :ref:`capabilities-retrieve` — similarity scores or rankings.
- :ref:`capabilities-embed` — feature vectors.
- :ref:`capabilities-caption` — text descriptions.
- :ref:`capabilities-denoise` — filtered image bytes.
- :ref:`capabilities-evaluate` — reconstruction-error maps.
- :ref:`capabilities-other` — analysis and image utilities.

🩻 Browse by modality
---------------------

The :doc:`by-modality` page groups tools under CT, DSA, mammography, MRI, pathology,
ultrasound, X-ray, and the generic medical-AI and image categories.

🔬 Browse by model
------------------

The :doc:`../coverage/index` pages record which models are integrated, which are
deferred, and why. Use them when you need provenance or licensing details; use the
catalog when you need a callable tool.

.. note::

   Each tool's optional backend is installed through an extra. The catalog omits tools
   whose docstring is still a TODO placeholder; ``caresmcp list`` additionally dims
   tools whose extras are missing in your environment. See :doc:`../installation` and
   :doc:`../quick-start`.

.. toctree::
   :hidden:

   catalog
   by-modality
