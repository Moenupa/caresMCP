🧰 Tools by Modality
====================

All tools are grouped by the clinical or functional category they serve. Each page
summarizes the models behind the registered tools and their input and output formats;
see :doc:`../api/index` for function-level reference. Pages without integrated tools
record the candidate model survey instead.

- :doc:`medai` — generic Hugging Face classification, embedding, segmentation, and
  TotalSegmentator anatomy segmentation.
- :doc:`tools` — generic image download, inspection, cropping, and denoising.
- :doc:`ct` — head-CT hemorrhage scoring, slice classification, chest-CT embeddings and
  radiomics, and HU-based image utilities.
- :doc:`dsa` — coronary stenosis detection and vessel segmentation in catheter
  angiography.
- :doc:`fluoroscopy` — model survey for fluoroscopic examinations; no licensed, servable
  model is integrated.
- :doc:`mammography` — breast-density and mammogram-patch classification.
- :doc:`mri` — task-specific cardiac, stroke, and brain-tumor segmentation and slice
  registration.
- :doc:`nuclearmed` — FDG PET/CT tumor segmentation and PET/CT anatomical-context
  candidates; no checkpoint integrated yet.
- :doc:`pathology` — nuclei and tumor analysis, pathology embeddings, PRISM/PRISM2
  scoring, CONCH similarity, and Cellpose segmentation.
- :doc:`ultrasound` — echocardiography measurements, ejection fraction, segmentation,
  and embeddings.
- :doc:`xray` — chest X-ray classification, segmentation, localization, embeddings, and
  phrase grounding.

.. toctree::
   :hidden:

   medai
   tools
   ct
   dsa
   fluoroscopy
   mammography
   mri
   nuclearmed
   pathology
   ultrasound
   xray
