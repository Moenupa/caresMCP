📁 Tools by Modality
====================

CaresMCP registers **52 interfaces** across **9 categories**. Categories with no
servable interfaces are documented in :doc:`../coverage/index`.

.. list-table::
   :class: capability-table tool-summary-table
   :header-rows: 1
   :widths: 25 10 25 40

   - - Category
     - Interfaces
     - Tasks
     - Representative interfaces
   - - :ref:`capabilities-category-medai`
     - 5
     - Classify, Segment, Retrieve, Embed
     - :func:`hf_classifier <caresmcp.medai.hf.hf_classifier>`, :func:`hf_image_embed
       <caresmcp.medai.hf.hf_image_embed>`, :func:`hf_image_text_retrieval
       <caresmcp.medai.hf.hf_image_text_retrieval>`
   - - :ref:`capabilities-category-image`
     - 5
     - Denoise
     - :func:`crop <caresmcp.tools.pil.crop>`, :func:`cropcenter
       <caresmcp.tools.pil.cropcenter>`, :func:`denoise
       <caresmcp.tools.skimage.denoise>`
   - - :ref:`capabilities-category-ct`
     - 8
     - Classify, Segment, Embed
     - :func:`classify_ct_head_hemorrhage
       <caresmcp.ct.hemorrhage.classify_ct_head_hemorrhage>`, :func:`classify_ct_slice
       <caresmcp.ct.huggingface.classify_ct_slice>`, :func:`classify_ct_slice_with_curia
       <caresmcp.ct.curia.classify_ct_slice_with_curia>`
   - - :ref:`capabilities-category-dsa`
     - 3
     - Segment, Localize
     - :func:`detect_coronary_stenosis
       <caresmcp.dsa.coronary.detect_coronary_stenosis>`,
       :func:`segment_coronary_stenosis
       <caresmcp.dsa.coronary.segment_coronary_stenosis>`,
       :func:`segment_coronary_vessels <caresmcp.dsa.coronary.segment_coronary_vessels>`
   - - :ref:`capabilities-category-mammography`
     - 2
     - Classify
     - :func:`classify_breast_density
       <caresmcp.mammography.density.classify_breast_density>`,
       :func:`classify_mammogram_patch
       <caresmcp.mammography.umair.classify_mammogram_patch>`
   - - :ref:`capabilities-category-mri`
     - 2
     - Segment
     - :func:`segment_cardiac_mri_slice
       <caresmcp.mri.cardiac.segment_cardiac_mri_slice>`, :func:`segment_mri_a4unet
       <caresmcp.mri.a4unet.segment_mri_a4unet>`
   - - :ref:`capabilities-category-pathology`
     - 11
     - Classify, Segment, Retrieve, Embed, Caption
     - :func:`cellpose_segment_pathology_cells
       <caresmcp.pathology.cellpose.cellpose_segment_pathology_cells>`,
       :func:`conch_score_candidate_texts
       <caresmcp.pathology.conch.conch_score_candidate_texts>`,
       :func:`monai_pathology_nuclei_classification
       <caresmcp.pathology.monai.monai_pathology_nuclei_classification>`
   - - :ref:`capabilities-category-ultrasound`
     - 7
     - Classify, Regress, Segment, Embed
     - :func:`echocare_image_embed <caresmcp.ultrasound.echocare.echocare_image_embed>`,
       :func:`echonet_dynamic_predict_ef
       <caresmcp.ultrasound.echonet_dynamic.echonet_dynamic_predict_ef>`,
       :func:`echonet_dynamic_segment_lv
       <caresmcp.ultrasound.echonet_dynamic.echonet_dynamic_segment_lv>`
   - - :ref:`capabilities-category-xray`
     - 9
     - Classify, Segment, Localize, Retrieve, Embed, Evaluate
     - :func:`biovil_ground_phrases <caresmcp.xray.biovil.biovil_ground_phrases>`,
       :func:`detect_pediatric_wrist_findings
       <caresmcp.xray.wrist.detect_pediatric_wrist_findings>`,
       :func:`flexray_segment_anatomy <caresmcp.xray.flexray.flexray_segment_anatomy>`

.. _capabilities-category-medai:

🤖 Medical AI (5 interfaces)
----------------------------

.. list-table::
   :class: capability-table
   :header-rows: 1
   :widths: 30 20 50

   - - Interface
     - Tasks
     - Description
   - - :func:`hf_classifier <caresmcp.medai.hf.hf_classifier>`
     - Classify
     - Run a trained Transformers image-classification head.
   - - :func:`hf_image_embed <caresmcp.medai.hf.hf_image_embed>`
     - Embed
     - Extract image features with a Hugging Face image encoder.
   - - :func:`hf_image_text_retrieval <caresmcp.medai.hf.hf_image_text_retrieval>`
     - Retrieve
     - Rank candidate texts against one image with a dual-encoder model.
   - - :func:`hf_segment_image <caresmcp.medai.hf.hf_segment_image>`
     - Segment
     - Segment a box-prompted region in one medical image (white foreground PNG).
   - - :func:`totalsegmentator_segment_anatomy
       <caresmcp.medai.totalsegmentator.totalsegmentator_segment_anatomy>`
     - Segment
     - Segment anatomical structures in a three-dimensional CT or MR volume.

.. _capabilities-category-image:

📐 Generic Tools (5 interfaces)
-------------------------------

.. list-table::
   :class: capability-table
   :header-rows: 1
   :widths: 30 20 50

   - - Interface
     - Tasks
     - Description
   - - :func:`crop <caresmcp.tools.pil.crop>`
     - —
     - Crop a 2D image to a pixel box.
   - - :func:`cropcenter <caresmcp.tools.pil.cropcenter>`
     - —
     - Crop the center of a 2D image to the given pixel size.
   - - :func:`denoise <caresmcp.tools.skimage.denoise>`
     - Denoise
     - Denoise a grayscale or RGB image using a classical filter.
   - - :func:`download <caresmcp.tools.pil.download>`
     - —
     - Download and return an image.
   - - :func:`inspect <caresmcp.tools.pil.inspect>`
     - —
     - Get metadata for a 2D image.

.. _capabilities-category-ct:

🖥️ CT (8 interfaces)
--------------------

.. list-table::
   :class: capability-table
   :header-rows: 1
   :widths: 30 20 50

   - - Interface
     - Tasks
     - Description
   - - :func:`classify_ct_head_hemorrhage
       <caresmcp.ct.hemorrhage.classify_ct_head_hemorrhage>`
     - Classify
     - Score six overlapping hemorrhage labels for one noncontrast axial head CT slice.
   - - :func:`classify_ct_slice <caresmcp.ct.huggingface.classify_ct_slice>`
     - Classify
     - Return classification scores for one CT slice.
   - - :func:`classify_ct_slice_with_curia
       <caresmcp.ct.curia.classify_ct_slice_with_curia>`
     - Classify
     - Classify CT anatomy, COVID/pneumonia, or hemorrhage with a Curia slice head.
   - - :func:`dale_ct_2s_image_embed <caresmcp.ct.dale_ct_2s.dale_ct_2s_image_embed>`
     - Embed
     - Extract a global DALE-CT-2S feature vector from one chest CT slice.
   - - :func:`extract_ct_slice_radiomics <caresmcp.ct.pyradiomics.radiomics_ct_slice>`
     - —
     - Extract two-dimensional PyRadiomics features from one CT slice ROI.
   - - :func:`measure_ct_hu_regions <caresmcp.ct.classical.measure_ct_hu_regions>`
     - —
     - Measure connected components selected by a CT HU interval.
   - - :func:`segment_ct_hu_range <caresmcp.ct.classical.segment_ct_hu_range>`
     - Segment
     - Create a binary mask (in 8-bit PNG) for pixels within a supplied HU interval.
   - - :func:`window_ct_slice <caresmcp.ct.classical.window_ct_slice>`
     - —
     - Render a single CT slice (in 8-bit PNG) with DICOM LINEAR HU window convention.

.. _capabilities-category-dsa:

🫀 DSA and Catheter Angiography (3 interfaces)
----------------------------------------------

.. list-table::
   :class: capability-table
   :header-rows: 1
   :widths: 30 20 50

   - - Interface
     - Tasks
     - Description
   - - :func:`detect_coronary_stenosis <caresmcp.dsa.coronary.detect_coronary_stenosis>`
     - Localize
     - Locate candidate coronary artery stenosis regions in one angiography frame.
   - - :func:`segment_coronary_stenosis
       <caresmcp.dsa.coronary.segment_coronary_stenosis>`
     - Segment
     - Segment candidate coronary artery stenosis regions in one angiography frame.
   - - :func:`segment_coronary_vessels <caresmcp.dsa.coronary.segment_coronary_vessels>`
     - Segment
     - Segment the coronary artery tree in one angiography frame.

.. _capabilities-category-mammography:

🩺 Mammography (2 interfaces)
-----------------------------

.. list-table::
   :class: capability-table
   :header-rows: 1
   :widths: 30 20 50

   - - Interface
     - Tasks
     - Description
   - - :func:`classify_breast_density
       <caresmcp.mammography.density.classify_breast_density>`
     - Classify
     - Score the four BI-RADS breast density categories on one mammogram (in sigmoid).
   - - :func:`classify_mammogram_patch
       <caresmcp.mammography.umair.classify_mammogram_patch>`
     - Classify
     - Score benign, malignant, and normal classes for a mammogram patch.

.. _capabilities-category-mri:

🧲 MRI (2 interfaces)
---------------------

.. list-table::
   :class: capability-table
   :header-rows: 1
   :widths: 30 20 50

   - - Interface
     - Tasks
     - Description
   - - :func:`segment_cardiac_mri_slice
       <caresmcp.mri.cardiac.segment_cardiac_mri_slice>`
     - Segment
     - Segment the LV cavity, myocardium, and RV cavity in a cardiac MR slice.
   - - :func:`segment_mri_a4unet <caresmcp.mri.a4unet.segment_mri_a4unet>`
     - Segment
     - Segment the whole-tumor region from four aligned BraTS contrasts.

.. _capabilities-category-pathology:

🔬 Pathology (11 interfaces)
----------------------------

.. list-table::
   :class: capability-table
   :header-rows: 1
   :widths: 30 20 50

   - - Interface
     - Tasks
     - Description
   - - :func:`cellpose_segment_pathology_cells
       <caresmcp.pathology.cellpose.cellpose_segment_pathology_cells>`
     - Segment
     - Segment cells in a pathology image and show their boundaries.
   - - :func:`conch_score_candidate_texts
       <caresmcp.pathology.conch.conch_score_candidate_texts>`
     - Retrieve
     - Score how closely each candidate description matches a pathology image.
   - - :func:`monai_pathology_nuclei_classification
       <caresmcp.pathology.monai.monai_pathology_nuclei_classification>`
     - Classify
     - Classify the nucleus marked in a colorectal tissue patch.
   - - :func:`monai_pathology_nuclei_segmentation_classification
       <caresmcp.pathology.monai.monai_pathology_nuclei_segmentation_classification>`
     - Classify, Segment
     - Segment nuclei and assign a type to each in colorectal tissue.
   - - :func:`monai_pathology_tumor_detection
       <caresmcp.pathology.monai.monai_pathology_tumor_detection>`
     - Classify
     - Score tumor-like content in an H&E lymph-node tile or ROI.
   - - :func:`pathology_image_embed
       <caresmcp.pathology.pathology_encoder.pathology_image_embed>`
     - Embed
     - Encode a pathology tile or ROI as a feature vector.
   - - :func:`prism2_image_caption <caresmcp.pathology.prism2.prism2_image_caption>`
     - Caption
     - Generate text about a pathology tile or ROI from a prompt.
   - - :func:`prism2_image_embed <caresmcp.pathology.prism2.prism2_image_embed>`
     - Embed
     - Encode a pathology tile or ROI as a PRISM2 embedding.
   - - :func:`prism2_score_yes_no_question
       <caresmcp.pathology.prism2.prism2_score_yes_no_question>`
     - Classify
     - Score how likely a yes/no answer is for a pathology tile or ROI.
   - - :func:`prism_slide_embed <caresmcp.pathology.prism.prism_slide_embed>`
     - Embed
     - Encode a pathology tile or ROI as a 1280-dimensional PRISM vector.
   - - :func:`prism_zero_shot_classify
       <caresmcp.pathology.prism.prism_zero_shot_classify>`
     - Retrieve
     - Compare two groups of pathology descriptions with a tile or ROI using PRISM.

.. _capabilities-category-ultrasound:

💓 Ultrasound (7 interfaces)
----------------------------

.. list-table::
   :class: capability-table
   :header-rows: 1
   :widths: 30 20 50

   - - Interface
     - Tasks
     - Description
   - - :func:`echocare_image_embed <caresmcp.ultrasound.echocare.echocare_image_embed>`
     - Embed
     - Extract a 2048-dimensional EchoCare embedding from an ultrasound image.
   - - :func:`echonet_dynamic_predict_ef
       <caresmcp.ultrasound.echonet_dynamic.echonet_dynamic_predict_ef>`
     - Regress
     - Estimate left ventricular ejection fraction from an apical four-chamber video.
   - - :func:`echonet_dynamic_segment_lv
       <caresmcp.ultrasound.echonet_dynamic.echonet_dynamic_segment_lv>`
     - Segment
     - Segment the left ventricle in an apical four-chamber echocardiogram frame.
   - - :func:`panecho_image_embed <caresmcp.ultrasound.panecho.panecho_image_embed>`
     - Embed
     - Extract a 768-dimensional PanEcho feature vector from one echo frame.
   - - :func:`panecho_predict_study <caresmcp.ultrasound.panecho.panecho_predict_study>`
     - Classify, Regress
     - Predict echocardiography measurements and findings for one study.
   - - :func:`panecho_predict_video <caresmcp.ultrasound.panecho.panecho_predict_video>`
     - Classify, Regress
     - Predict echocardiography measurements and findings from one video.
   - - :func:`panecho_video_embed <caresmcp.ultrasound.panecho.panecho_video_embed>`
     - Embed
     - Extract a 768-dimensional PanEcho feature vector from an echo video.

.. _capabilities-category-xray:

🩻 X-ray (9 interfaces)
-----------------------

.. list-table::
   :class: capability-table
   :header-rows: 1
   :widths: 30 20 50

   - - Interface
     - Tasks
     - Description
   - - :func:`biovil_ground_phrases <caresmcp.xray.biovil.biovil_ground_phrases>`
     - Localize, Retrieve
     - Locate and rank radiological phrases in a chest radiograph.
   - - :func:`detect_pediatric_wrist_findings
       <caresmcp.xray.wrist.detect_pediatric_wrist_findings>`
     - Localize
     - Detect fractures and associated findings on pediatric wrist radiographs.
   - - :func:`flexray_segment_anatomy <caresmcp.xray.flexray.flexray_segment_anatomy>`
     - Segment
     - Segment an anatomical structure in a radiograph.
   - - :func:`skelex_reconstruction_error
       <caresmcp.xray.skelex.skelex_reconstruction_error>`
     - Evaluate
     - Map reconstruction error in a musculoskeletal radiograph.
   - - :func:`torchxrayvision_classify
       <caresmcp.xray.torchxrayvision.torchxrayvision_classify>`
     - Classify
     - Score chest radiograph findings using a TorchXRayVision multi-label classifier.
   - - :func:`torchxrayvision_localize
       <caresmcp.xray.torchxrayvision.torchxrayvision_localize>`
     - Localize
     - Visualize Grad-CAM saliency for a chest radiograph finding.
   - - :func:`torchxrayvision_segment_anatomy
       <caresmcp.xray.torchxrayvision.torchxrayvision_segment_anatomy>`
     - Segment
     - Segment 14 anatomical structures in a chest radiograph.
   - - :func:`xray_image_embed <caresmcp.xray.embeddings.xray_image_embed>`
     - Embed
     - Generate an embedding of a chest radiograph.
   - - :func:`xray_image_similarity <caresmcp.xray.embeddings.xray_image_similarity>`
     - Retrieve
     - Compare two chest radiographs by the cosine similarity of their embeddings.

.. note::

   See :doc:`../installation` for extras and :doc:`../quick-start` for the first call.

.. rst-class:: generated-page-note

Auto-generated.
