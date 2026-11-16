🤖 Medical AI
=============

Registry category: ``medai`` — use ``caresmcp debug medai <tool> ...`` to run a tool
directly.

Overview
--------

Generic medical-imaging tools backed by Hugging Face transformers and TotalSegmentator:
classification, image embedding, box-prompted segmentation, and volumetric anatomy
segmentation.

Implementation
--------------

.. automodule:: caresmcp.medical_ai.hf
   :no-members:

.. autofunction:: caresmcp.medical_ai.hf.hf_classifier

.. autofunction:: caresmcp.medical_ai.hf.hf_image_embed

.. autofunction:: caresmcp.medical_ai.hf.hf_segment_image

.. automodule:: caresmcp.medical_ai.totalsegmentator
   :no-members:

.. autofunction:: caresmcp.medical_ai.totalsegmentator.totalsegmentator_segment_anatomy
