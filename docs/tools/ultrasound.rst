🫀 Ultrasound
=============

Registry category: ``ultrasound`` — use ``caresmcp debug ultrasound <tool> ...`` to run
a tool directly.

Overview
--------

Echocardiography tools for ejection-fraction estimation, left-ventricle segmentation,
study- and video-level measurements, and frame, video, and still-image embeddings.

Implementation
--------------

.. automodule:: caresmcp.ultrasound.echonet_dynamic
   :no-members:

.. autofunction:: caresmcp.ultrasound.echonet_dynamic.echonet_dynamic_predict_ef

.. autofunction:: caresmcp.ultrasound.echonet_dynamic.echonet_dynamic_segment_lv

.. automodule:: caresmcp.ultrasound.panecho
   :no-members:

.. autofunction:: caresmcp.ultrasound.panecho.panecho_predict_study

.. autofunction:: caresmcp.ultrasound.panecho.panecho_predict_video

.. autofunction:: caresmcp.ultrasound.panecho.panecho_image_embed

.. autofunction:: caresmcp.ultrasound.panecho.panecho_video_embed

.. automodule:: caresmcp.ultrasound.echocare
   :no-members:

.. autofunction:: caresmcp.ultrasound.echocare.echocare_image_embed
