🗃 Tool Catalog
==============

CaresMCP registers **52 interfaces** across **9 categories**. An interface appears under
each task it declares, so task counts overlap. Interfaces whose docstring is still a
TODO placeholder are omitted; install the optional extra for an interface's backend
before serving it over MCP.

.. list-table::
   :class: capability-table
   :header-rows: 1
   :widths: 25 10 65

   - - Task
     - Interfaces
     - Typical output
   - - :ref:`capabilities-classify`
     - 13
     - Label probabilities or scores.
   - - :ref:`capabilities-regress`
     - 3
     - Quantitative measurements.
   - - :ref:`capabilities-segment`
     - 12
     - Masks as PNG bytes.
   - - :ref:`capabilities-localize`
     - 4
     - Heatmaps, boxes, or ranked regions.
   - - :ref:`capabilities-retrieve`
     - 5
     - Similarity scores or rankings.
   - - :ref:`capabilities-embed`
     - 9
     - Feature vectors.
   - - :ref:`capabilities-caption`
     - 1
     - Text descriptions.
   - - :ref:`capabilities-denoise`
     - 1
     - Filtered image bytes.
   - - :ref:`capabilities-evaluate`
     - 1
     - Reconstruction-error maps.
   - - :ref:`capabilities-other`
     - 7
     - Analysis and image utilities without a task tag.

.. _capabilities-classify:

🎯 Classify (13 interfaces)
---------------------------

.. list-table::
   :class: capability-table tool-catalog-table
   :header-rows: 1
   :widths: 20 20 25 35

   - - Interface
     - Category
     - Models
     - Description
   - - :func:`classify_breast_density
       <caresmcp.mammography.density.classify_breast_density>`
     - :doc:`mammography <../api/mammography>`
     - —
     - Score the four BI-RADS breast density categories on one mammogram (in sigmoid).
   - - :func:`classify_ct_head_hemorrhage
       <caresmcp.ct.hemorrhage.classify_ct_head_hemorrhage>`
     - :doc:`ct <../api/ct>`
     - —
     - Score six overlapping hemorrhage labels for one noncontrast axial head CT slice.
   - - :func:`classify_ct_slice <caresmcp.ct.huggingface.classify_ct_slice>`
     - :doc:`ct <../api/ct>`
     - `oohtmeel/swin-tiny-patch4-finetuned-lung-cancer-ct-scans
       <https://huggingface.co/oohtmeel/swin-tiny-patch4-finetuned-lung-cancer-ct-scans>`_,
       `sukhmani1303/lung-cancer-vit-model
       <https://huggingface.co/sukhmani1303/lung-cancer-vit-model>`_
     - Return classification scores for one CT slice.
   - - :func:`classify_ct_slice_with_curia
       <caresmcp.ct.curia.classify_ct_slice_with_curia>`
     - :doc:`ct <../api/ct>`
     - —
     - Classify CT anatomy, COVID/pneumonia, or hemorrhage with a Curia slice head.
   - - :func:`classify_mammogram_patch
       <caresmcp.mammography.umair.classify_mammogram_patch>`
     - :doc:`mammography <../api/mammography>`
     - —
     - Score benign, malignant, and normal classes for a mammogram patch.
   - - :func:`hf_classifier <caresmcp.medai.hf.hf_classifier>`
     - :doc:`medai <../api/medai>`
     - —
     - Run a trained Transformers image-classification head.
   - - :func:`monai_pathology_nuclei_classification
       <caresmcp.pathology.monai.monai_pathology_nuclei_classification>`
     - :doc:`pathology <../api/pathology>`
     - —
     - Classify the nucleus marked in a colorectal tissue patch.
   - - :func:`monai_pathology_nuclei_segmentation_classification
       <caresmcp.pathology.monai.monai_pathology_nuclei_segmentation_classification>`
     - :doc:`pathology <../api/pathology>`
     - —
     - Segment nuclei and assign a type to each in colorectal tissue.
   - - :func:`monai_pathology_tumor_detection
       <caresmcp.pathology.monai.monai_pathology_tumor_detection>`
     - :doc:`pathology <../api/pathology>`
     - —
     - Score tumor-like content in an H&E lymph-node tile or ROI.
   - - :func:`panecho_predict_study <caresmcp.ultrasound.panecho.panecho_predict_study>`
     - :doc:`ultrasound <../api/ultrasound>`
     - —
     - Predict echocardiography measurements and findings for one study.
   - - :func:`panecho_predict_video <caresmcp.ultrasound.panecho.panecho_predict_video>`
     - :doc:`ultrasound <../api/ultrasound>`
     - —
     - Predict echocardiography measurements and findings from one video.
   - - :func:`prism2_score_yes_no_question
       <caresmcp.pathology.prism2.prism2_score_yes_no_question>`
     - :doc:`pathology <../api/pathology>`
     - —
     - Score how likely a yes/no answer is for a pathology tile or ROI.
   - - :func:`torchxrayvision_classify
       <caresmcp.xray.torchxrayvision.torchxrayvision_classify>`
     - :doc:`xray <../api/xray>`
     - ``densenet121-res224-all``, ``densenet121-res224-rsna``,
       ``densenet121-res224-nih`` … (+5)
     - Score chest radiograph findings using a TorchXRayVision multi-label classifier.

.. _capabilities-regress:

📏 Regress (3 interfaces)
-------------------------

.. list-table::
   :class: capability-table tool-catalog-table
   :header-rows: 1
   :widths: 20 20 25 35

   - - Interface
     - Category
     - Models
     - Description
   - - :func:`echonet_dynamic_predict_ef
       <caresmcp.ultrasound.echonet_dynamic.echonet_dynamic_predict_ef>`
     - :doc:`ultrasound <../api/ultrasound>`
     - —
     - Estimate left ventricular ejection fraction from an apical four-chamber video.
   - - :func:`panecho_predict_study <caresmcp.ultrasound.panecho.panecho_predict_study>`
     - :doc:`ultrasound <../api/ultrasound>`
     - —
     - Predict echocardiography measurements and findings for one study.
   - - :func:`panecho_predict_video <caresmcp.ultrasound.panecho.panecho_predict_video>`
     - :doc:`ultrasound <../api/ultrasound>`
     - —
     - Predict echocardiography measurements and findings from one video.

.. _capabilities-segment:

🗺️ Segment (12 interfaces)
--------------------------

.. list-table::
   :class: capability-table tool-catalog-table
   :header-rows: 1
   :widths: 20 20 25 35

   - - Interface
     - Category
     - Models
     - Description
   - - :func:`cellpose_segment_pathology_cells
       <caresmcp.pathology.cellpose.cellpose_segment_pathology_cells>`
     - :doc:`pathology <../api/pathology>`
     - ``cpsam``, ``cpsam_v2``, ``cpdino`` … (+1)
     - Segment cells in a pathology image and show their boundaries.
   - - :func:`echonet_dynamic_segment_lv
       <caresmcp.ultrasound.echonet_dynamic.echonet_dynamic_segment_lv>`
     - :doc:`ultrasound <../api/ultrasound>`
     - —
     - Segment the left ventricle in an apical four-chamber echocardiogram frame.
   - - :func:`flexray_segment_anatomy <caresmcp.xray.flexray.flexray_segment_anatomy>`
     - :doc:`xray <../api/xray>`
     - —
     - Segment an anatomical structure in a radiograph.
   - - :func:`hf_segment_image <caresmcp.medai.hf.hf_segment_image>`
     - :doc:`medai <../api/medai>`
     - `flaviagiammarino/medsam-vit-base
       <https://huggingface.co/flaviagiammarino/medsam-vit-base>`_,
       `facebook/sam2.1-hiera-tiny
       <https://huggingface.co/facebook/sam2.1-hiera-tiny>`_,
       `facebook/sam2.1-hiera-small
       <https://huggingface.co/facebook/sam2.1-hiera-small>`_ … (+2)
     - Segment a box-prompted region in one medical image (white foreground PNG).
   - - :func:`monai_pathology_nuclei_segmentation_classification
       <caresmcp.pathology.monai.monai_pathology_nuclei_segmentation_classification>`
     - :doc:`pathology <../api/pathology>`
     - —
     - Segment nuclei and assign a type to each in colorectal tissue.
   - - :func:`segment_cardiac_mri_slice
       <caresmcp.mri.cardiac.segment_cardiac_mri_slice>`
     - :doc:`mri <../api/mri>`
     - `MONAI/ventricular_short_axis_3label
       <https://huggingface.co/MONAI/ventricular_short_axis_3label>`_
     - Segment the LV cavity, myocardium, and RV cavity in a cardiac MR slice.
   - - :func:`segment_coronary_stenosis
       <caresmcp.dsa.coronary.segment_coronary_stenosis>`
     - :doc:`dsa <../api/dsa>`
     - —
     - Segment candidate coronary artery stenosis regions in one angiography frame.
   - - :func:`segment_coronary_vessels <caresmcp.dsa.coronary.segment_coronary_vessels>`
     - :doc:`dsa <../api/dsa>`
     - —
     - Segment the coronary artery tree in one angiography frame.
   - - :func:`segment_ct_hu_range <caresmcp.ct.classical.segment_ct_hu_range>`
     - :doc:`ct <../api/ct>`
     - —
     - Create a binary mask (in 8-bit PNG) for pixels within a supplied HU interval.
   - - :func:`segment_mri_a4unet <caresmcp.mri.a4unet.segment_mri_a4unet>`
     - :doc:`mri <../api/mri>`
     - `Roxanne-WANG/A4-Unet <https://huggingface.co/Roxanne-WANG/A4-Unet>`_
     - Segment the whole-tumor region from four aligned BraTS contrasts.
   - - :func:`torchxrayvision_segment_anatomy
       <caresmcp.xray.torchxrayvision.torchxrayvision_segment_anatomy>`
     - :doc:`xray <../api/xray>`
     - ``pspnet_chestxray``
     - Segment 14 anatomical structures in a chest radiograph.
   - - :func:`totalsegmentator_segment_anatomy
       <caresmcp.medai.totalsegmentator.totalsegmentator_segment_anatomy>`
     - :doc:`medai <../api/medai>`
     - —
     - Segment anatomical structures in a three-dimensional CT or MR volume.

.. _capabilities-localize:

🔎 Localize (4 interfaces)
--------------------------

.. list-table::
   :class: capability-table tool-catalog-table
   :header-rows: 1
   :widths: 20 20 25 35

   - - Interface
     - Category
     - Models
     - Description
   - - :func:`biovil_ground_phrases <caresmcp.xray.biovil.biovil_ground_phrases>`
     - :doc:`xray <../api/xray>`
     - `microsoft/BiomedVLP-CXR-BERT-specialized
       <https://huggingface.co/microsoft/BiomedVLP-CXR-BERT-specialized>`_,
       `microsoft/BiomedVLP-BioViL-T
       <https://huggingface.co/microsoft/BiomedVLP-BioViL-T>`_
     - Locate and rank radiological phrases in a chest radiograph.
   - - :func:`detect_coronary_stenosis <caresmcp.dsa.coronary.detect_coronary_stenosis>`
     - :doc:`dsa <../api/dsa>`
     - —
     - Locate candidate coronary artery stenosis regions in one angiography frame.
   - - :func:`detect_pediatric_wrist_findings
       <caresmcp.xray.wrist.detect_pediatric_wrist_findings>`
     - :doc:`xray <../api/xray>`
     - —
     - Detect fractures and associated findings on pediatric wrist radiographs.
   - - :func:`torchxrayvision_localize
       <caresmcp.xray.torchxrayvision.torchxrayvision_localize>`
     - :doc:`xray <../api/xray>`
     - ``densenet121-res224-all``, ``densenet121-res224-rsna``,
       ``densenet121-res224-nih`` … (+5)
     - Visualize Grad-CAM saliency for a chest radiograph finding.

.. _capabilities-retrieve:

🔁 Retrieve (5 interfaces)
--------------------------

.. list-table::
   :class: capability-table tool-catalog-table
   :header-rows: 1
   :widths: 20 20 25 35

   - - Interface
     - Category
     - Models
     - Description
   - - :func:`biovil_ground_phrases <caresmcp.xray.biovil.biovil_ground_phrases>`
     - :doc:`xray <../api/xray>`
     - `microsoft/BiomedVLP-CXR-BERT-specialized
       <https://huggingface.co/microsoft/BiomedVLP-CXR-BERT-specialized>`_,
       `microsoft/BiomedVLP-BioViL-T
       <https://huggingface.co/microsoft/BiomedVLP-BioViL-T>`_
     - Locate and rank radiological phrases in a chest radiograph.
   - - :func:`conch_score_candidate_texts
       <caresmcp.pathology.conch.conch_score_candidate_texts>`
     - :doc:`pathology <../api/pathology>`
     - —
     - Score how closely each candidate description matches a pathology image.
   - - :func:`hf_image_text_retrieval <caresmcp.medai.hf.hf_image_text_retrieval>`
     - :doc:`medai <../api/medai>`
     - `google/medsiglip-448 <https://huggingface.co/google/medsiglip-448>`_,
       `openai/clip-vit-base-patch32
       <https://huggingface.co/openai/clip-vit-base-patch32>`_,
       `google/siglip-base-patch16-224
       <https://huggingface.co/google/siglip-base-patch16-224>`_
     - Rank candidate texts against one image with a dual-encoder model.
   - - :func:`prism_zero_shot_classify
       <caresmcp.pathology.prism.prism_zero_shot_classify>`
     - :doc:`pathology <../api/pathology>`
     - `paige-ai/Prism <https://huggingface.co/paige-ai/Prism>`_
     - Compare two groups of pathology descriptions with a tile or ROI using PRISM.
   - - :func:`xray_image_similarity <caresmcp.xray.embeddings.xray_image_similarity>`
     - :doc:`xray <../api/xray>`
     - `microsoft/rad-dino <https://huggingface.co/microsoft/rad-dino>`_,
       `microsoft/BiomedVLP-CXR-BERT-specialized
       <https://huggingface.co/microsoft/BiomedVLP-CXR-BERT-specialized>`_,
       `microsoft/BiomedVLP-BioViL-T
       <https://huggingface.co/microsoft/BiomedVLP-BioViL-T>`_
     - Compare two chest radiographs by the cosine similarity of their embeddings.

.. _capabilities-embed:

🧬 Embed (9 interfaces)
-----------------------

.. list-table::
   :class: capability-table tool-catalog-table
   :header-rows: 1
   :widths: 20 20 25 35

   - - Interface
     - Category
     - Models
     - Description
   - - :func:`dale_ct_2s_image_embed <caresmcp.ct.dale_ct_2s.dale_ct_2s_image_embed>`
     - :doc:`ct <../api/ct>`
     - —
     - Extract a global DALE-CT-2S feature vector from one chest CT slice.
   - - :func:`echocare_image_embed <caresmcp.ultrasound.echocare.echocare_image_embed>`
     - :doc:`ultrasound <../api/ultrasound>`
     - —
     - Extract a 2048-dimensional EchoCare embedding from an ultrasound image.
   - - :func:`hf_image_embed <caresmcp.medai.hf.hf_image_embed>`
     - :doc:`medai <../api/medai>`
     - —
     - Extract image features with a Hugging Face image encoder.
   - - :func:`panecho_image_embed <caresmcp.ultrasound.panecho.panecho_image_embed>`
     - :doc:`ultrasound <../api/ultrasound>`
     - —
     - Extract a 768-dimensional PanEcho feature vector from one echo frame.
   - - :func:`panecho_video_embed <caresmcp.ultrasound.panecho.panecho_video_embed>`
     - :doc:`ultrasound <../api/ultrasound>`
     - —
     - Extract a 768-dimensional PanEcho feature vector from an echo video.
   - - :func:`pathology_image_embed
       <caresmcp.pathology.pathology_encoder.pathology_image_embed>`
     - :doc:`pathology <../api/pathology>`
     - `bioptimus/H0-mini <https://huggingface.co/bioptimus/H0-mini>`_, `MahmoodLab/UNI
       <https://huggingface.co/MahmoodLab/UNI>`_, `MahmoodLab/UNI2-h
       <https://huggingface.co/MahmoodLab/UNI2-h>`_ … (+6)
     - Encode a pathology tile or ROI as a feature vector.
   - - :func:`prism2_image_embed <caresmcp.pathology.prism2.prism2_image_embed>`
     - :doc:`pathology <../api/pathology>`
     - —
     - Encode a pathology tile or ROI as a PRISM2 embedding.
   - - :func:`prism_slide_embed <caresmcp.pathology.prism.prism_slide_embed>`
     - :doc:`pathology <../api/pathology>`
     - `paige-ai/Prism <https://huggingface.co/paige-ai/Prism>`_
     - Encode a pathology tile or ROI as a 1280-dimensional PRISM vector.
   - - :func:`xray_image_embed <caresmcp.xray.embeddings.xray_image_embed>`
     - :doc:`xray <../api/xray>`
     - `microsoft/rad-dino <https://huggingface.co/microsoft/rad-dino>`_,
       `microsoft/BiomedVLP-CXR-BERT-specialized
       <https://huggingface.co/microsoft/BiomedVLP-CXR-BERT-specialized>`_,
       `microsoft/BiomedVLP-BioViL-T
       <https://huggingface.co/microsoft/BiomedVLP-BioViL-T>`_
     - Generate an embedding of a chest radiograph.

.. _capabilities-caption:

📝 Caption (1 interface)
------------------------

.. list-table::
   :class: capability-table tool-catalog-table
   :header-rows: 1
   :widths: 20 20 25 35

   - - Interface
     - Category
     - Models
     - Description
   - - :func:`prism2_image_caption <caresmcp.pathology.prism2.prism2_image_caption>`
     - :doc:`pathology <../api/pathology>`
     - —
     - Generate text about a pathology tile or ROI from a prompt.

.. _capabilities-denoise:

🧹 Denoise (1 interface)
------------------------

.. list-table::
   :class: capability-table tool-catalog-table
   :header-rows: 1
   :widths: 20 20 25 35

   - - Interface
     - Category
     - Models
     - Description
   - - :func:`denoise <caresmcp.tools.skimage.denoise>`
     - :doc:`image <../api/tools>`
     - —
     - Denoise a grayscale or RGB image using a classical filter.

.. _capabilities-evaluate:

🧮 Evaluate (1 interface)
-------------------------

.. list-table::
   :class: capability-table tool-catalog-table
   :header-rows: 1
   :widths: 20 20 25 35

   - - Interface
     - Category
     - Models
     - Description
   - - :func:`skelex_reconstruction_error
       <caresmcp.xray.skelex.skelex_reconstruction_error>`
     - :doc:`xray <../api/xray>`
     - `skhoha/SKELEX <https://huggingface.co/skhoha/SKELEX>`_
     - Map reconstruction error in a musculoskeletal radiograph.

.. _capabilities-other:

🧰 Other interfaces (7 interfaces)
----------------------------------

.. list-table::
   :class: capability-table tool-catalog-table
   :header-rows: 1
   :widths: 20 20 25 35

   - - Interface
     - Category
     - Models
     - Description
   - - :func:`crop <caresmcp.tools.pil.crop>`
     - :doc:`image <../api/tools>`
     - —
     - Crop a 2D image to a pixel box.
   - - :func:`cropcenter <caresmcp.tools.pil.cropcenter>`
     - :doc:`image <../api/tools>`
     - —
     - Crop the center of a 2D image to the given pixel size.
   - - :func:`download <caresmcp.tools.pil.download>`
     - :doc:`image <../api/tools>`
     - —
     - Download and return an image.
   - - :func:`extract_ct_slice_radiomics <caresmcp.ct.pyradiomics.radiomics_ct_slice>`
     - :doc:`ct <../api/ct>`
     - —
     - Extract two-dimensional PyRadiomics features from one CT slice ROI.
   - - :func:`inspect <caresmcp.tools.pil.inspect>`
     - :doc:`image <../api/tools>`
     - —
     - Get metadata for a 2D image.
   - - :func:`measure_ct_hu_regions <caresmcp.ct.classical.measure_ct_hu_regions>`
     - :doc:`ct <../api/ct>`
     - —
     - Measure connected components selected by a CT HU interval.
   - - :func:`window_ct_slice <caresmcp.ct.classical.window_ct_slice>`
     - :doc:`ct <../api/ct>`
     - —
     - Render a single CT slice (in 8-bit PNG) with DICOM LINEAR HU window convention.

.. note::

   See :doc:`../installation` for extras and :doc:`../quick-start` for the first call.

.. rst-class:: generated-page-note

Auto-generated.
