🔬 Pathology
============

Registry category: ``pathology`` — use ``caresmcp debug pathology <tool> ...`` to run a
tool directly.

Overview
--------

Pathology tools for MONAI tumor and nuclei analysis, pathology foundation-model
embeddings, PRISM and PRISM2 tile scoring and captioning, CONCH image-text similarity,
and Cellpose cell segmentation.

Implementation
--------------

.. automodule:: caresmcp.pathology.monai
   :no-members:

.. autofunction:: caresmcp.pathology.monai.monai_pathology_tumor_detection

.. autofunction:: caresmcp.pathology.monai.monai_pathology_nuclei_classification

.. autofunction:: caresmcp.pathology.monai.monai_pathology_nuclei_segmentation_classification

.. automodule:: caresmcp.pathology.pathology_encoder
   :no-members:

.. autofunction:: caresmcp.pathology.pathology_encoder.pathology_image_embed

.. automodule:: caresmcp.pathology.prism
   :no-members:

.. autofunction:: caresmcp.pathology.prism.prism_slide_embed

.. autofunction:: caresmcp.pathology.prism.prism_zero_shot_classify

.. automodule:: caresmcp.pathology.prism2
   :no-members:

.. autofunction:: caresmcp.pathology.prism2.prism2_image_embed

.. autofunction:: caresmcp.pathology.prism2.prism2_image_caption

.. autofunction:: caresmcp.pathology.prism2.prism2_score_yes_no_question

.. automodule:: caresmcp.pathology.conch
   :no-members:

.. autofunction:: caresmcp.pathology.conch.conch_score_candidate_texts

.. automodule:: caresmcp.pathology.cellpose
   :no-members:

.. autofunction:: caresmcp.pathology.cellpose.cellpose_segment_pathology_cells
