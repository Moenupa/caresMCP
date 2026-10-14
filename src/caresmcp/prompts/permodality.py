from enum import StrEnum


class INSTRUCTIONS(StrEnum):
    CT = """\
Check body region and coverage; supplied acquisition phase and contrast status; slice
thickness/reconstruction and windowing if known; motion, metal, or other artifacts; and whether
comparison studies are available. Do not infer a contrast phase or dose from an exported image.""".replace(
        "\n", " "
    )

    MRI = """\
Check anatomy, planes, and named sequences actually supplied; contrast status and timing
if stated; motion, susceptibility, distortion, or incomplete coverage; and whether sequences are
comparable across time. Do not infer a sequence from appearance or make MRI safety determinations.""".replace(
        "\n", " "
    )

    ULTRASOUND = """\
Check the documented organ/site, approach, planes, cine coverage, acoustic window,
and Doppler mode/settings only when supplied. State operator and window limitations when described;
do not infer that an unshown region is normal.""".replace("\n", " ")

    X_RAY = """\
Check body part, laterality, number and type of views, positioning/rotation, coverage,
exposure adequacy, and portable or weight-bearing context when documented. Do not treat a single
view as a complete multi-view examination.""".replace("\n", " ")

    PATHOLOGY = """\
Identify whether the input is a whole-slide image, field of view, gross photograph,
or supplied text. Check specimen/site, stain, scale and magnification if documented; tissue
adequacy, focus, artifacts, and annotation provenance. For whole-slide images, note scanner and
local validation status if supplied. Do not infer unsampled tissue or substitute image review for
the pathologist's validated diagnostic workflow.""".replace("\n", " ")

    NUCLEAR_MEDICINE = """\
Check radiopharmaceutical/tracer, administered activity and uptake time only if
provided; acquisition type and timing; attenuation correction/reconstruction if documented; and
alignment/registration in hybrid studies. Distinguish expected physiologic distribution from
reported abnormalities only when supported by supplied context; do not infer tracer or protocol.""".replace(
        "\n", " "
    )

    FLUOROSCOPY = """\
Check the stated procedure/question, anatomic coverage, views or recorded runs,
contrast opacification and timing when applicable, motion/artifacts, and any supplied dose indices
or exposure time. Do not estimate radiation dose from images or judge procedural appropriateness.""".replace(
        "\n", " "
    )

    MAMMOGRAPHY = """\
Check laterality, views and positioning/coverage, image quality, prior comparison
availability, and supplied breast density. Use BI-RADS terminology only when the source provides
enough information and the task calls for it; do not invent a category or management recommendation
from incomplete findings.""".replace("\n", " ")

    DSA_ANGIOGRAPHY = """\
Check vascular territory, projections, arterial/venous timing, opacification,
subtraction/motion artifacts, and the completeness of the supplied runs. State when only selected
frames are available; do not infer absence of disease in unshown vessels or phases. Separate
diagnostic observations from supplied procedural/intervention details.""".replace(
        "\n", " "
    )

    ILLUSTRATION = """\
Establish whether the image is a schematic, educational drawing, reconstruction, or
patient-specific rendering from its supplied caption and provenance. Check labels, orientation,
scale, legend, and whether elements are schematic or measured. Do not treat a conceptual illustration
as direct evidence of patient anatomy or findings.""".replace("\n", " ")


class GUIDELINES(StrEnum):
    CT = """\
- [ACR Practice Parameters and Technical Standards](https://www.acr.org/clinical-resources/clinical-tools-and-reference/practice-parameters-and-technical-standards) — examination performance and interpretation; the library contains region-specific CT parameters.
- [ACR Manual on Contrast Media](https://www.acr.org/Clinical-Resources/Clinical-Tools-and-Reference/Contrast-Manual) — contrast use and safety topics; not a substitute for patient-specific decisions.
- [UKHSA National Diagnostic Reference Levels](https://www.gov.uk/government/publications/diagnostic-radiology-national-diagnostic-reference-levels-ndrls) — UK dose optimisation benchmarks for common examinations, including CT.
- [EU eHealth Network guidelines on medical imaging studies and reports](https://health.ec.europa.eu/publications/ehn-guidelines-medical-imaging-studies-and-reports_en) — cross-modal study and report information.
- [RCR interpretation and reporting standards, third edition](https://www.rcr.ac.uk/our-services/all-our-publications/clinical-radiology-publications/standards-for-interpretation-and-reporting-of-imaging-investigations-third-edition/) — general UK reporting baseline."""

    MRI = """\
- [ACR Practice Parameters and Technical Standards](https://www.acr.org/clinical-resources/clinical-tools-and-reference/practice-parameters-and-technical-standards) — region- and examination-specific performance and interpretation parameters.
- [ACR Manual on MR Safety (2024)](https://cs.acr.org/-/media/ACR/Files/Radiology-Safety/MR-Safety/Manual-on-MR-Safety.pdf) — MR environment and safety program guidance; it is not a prompt for making implant clearance decisions.
- [UK MHRA MRI safety guidelines](https://www.gov.uk/government/publications/safety-guidelines-for-magnetic-resonance-imaging-equipment-in-clinical-use) — UK equipment, installation, and operational safety guidance; updated in 2026.
- [ESUR Contrast Media Safety Committee guidelines (2025)](https://www.esur.org/esur-guidelines-2025/) — gadolinium and other contrast safety topics.
- [EU eHealth Network guidelines on medical imaging studies and reports](https://health.ec.europa.eu/publications/ehn-guidelines-medical-imaging-studies-and-reports_en) — cross-modal study and report information.
- [RCR interpretation and reporting standards, third edition](https://www.rcr.ac.uk/our-services/all-our-publications/clinical-radiology-publications/standards-for-interpretation-and-reporting-of-imaging-investigations-third-edition/) — general UK reporting baseline."""

    ULTRASOUND = """\
- [ACR Practice Parameters and Technical Standards](https://www.acr.org/clinical-resources/clinical-tools-and-reference/practice-parameters-and-technical-standards) — includes general and anatomy-specific performance parameters.
- [AIUM ultrasound practice parameters](https://www.aium.org/resources/practice-parameters) — performance and recording criteria for high-quality examinations.
- [ACR-AAPM technical standard for diagnostic ultrasound equipment](https://gravitas.acr.org/PPTS/GetDocumentView?docId=118) — equipment performance monitoring and quality control.
- [NHS England AAA screening ultrasound image-quality guidance](https://www.gov.uk/government/publications/aaa-screening-ultrasound-image-quality-guidance) — image quality factors, capture, and QA review in the specific UK screening context.
- [EFSUMB guidelines and recommendations](https://efsumb.org/guidelines-recommendations-english-versions/) — European ultrasound recommendations, including contrast-enhanced ultrasound.
- [ACR BI-RADS v2025](https://www.acr.org/Clinical-Resources/Clinical-Tools-and-Reference/Reporting-and-Data-Systems/BI-RADS) — breast ultrasound lexicon and structured assessment only; not a general ultrasound standard."""

    X_RAY = """\
- [ACR-AAPM-SIIM practice parameter for digital radiography](https://www.acr.org/-/media/ACR/Files/Practice-Parameters/Rad-Digital.pdf) — digital radiography, data handling, image quality, dose, and quality improvement.
- [ACR Practice Parameters and Technical Standards](https://www.acr.org/clinical-resources/clinical-tools-and-reference/practice-parameters-and-technical-standards) — includes chest, extremity, paediatric, and other examination-specific parameters.
- [UKHSA National Diagnostic Reference Levels](https://www.gov.uk/government/publications/diagnostic-radiology-national-diagnostic-reference-levels-ndrls) — UK dose benchmarks for radiography, updated in 2025.
- [FDA Medical X-ray Imaging](https://www.fda.gov/radiation-emitting-products/medical-imaging/medical-x-ray-imaging) — U.S. information on justification, optimisation, and device oversight.
- [Euratom Basic Safety Standards Directive 2013/59](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=celex%3A32013L0059) — EU principles for justification, optimisation, protocols, dose information, and diagnostic reference levels.
- [EU eHealth Network guidelines on medical imaging studies and reports](https://health.ec.europa.eu/publications/ehn-guidelines-medical-imaging-studies-and-reports_en) — cross-modal study and report information."""

    PATHOLOGY = """\
- [CAP guideline on validating whole-slide imaging for diagnostic purposes](https://www.cap.org/protocols-and-guidelines/cap-guidelines/current-cap-guidelines/validating-whole-slide-imaging-for-diagnostic-purposes-in-pathology) — validation of WSI for its intended diagnostic use.
- [FDA technical performance assessment of digital pathology WSI devices](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/technical-performance-assessment-digital-pathology-whole-slide-imaging-devices) — technical WSI performance for regulatory evaluation, not clinical validation by itself.
- [RCPath cancer datasets and tissue pathways](https://www.rcpath.org/profession/guidelines/cancer-datasets-and-tissue-pathways.html) — structured histopathology reporting datasets and tissue pathways; select the dataset for the actual specimen/diagnosis.
- [RCPath best-practice recommendations for implementing digital pathology](https://www.rcpath.org/static/f465d1b3-797b-4297-b7fedc00b4d77e51/Best-practice-recommendations-for-implementing-digital-pathology.pdf) — UK digital pathology implementation and validation examples.
- [FDA whole-slide imaging system classification](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpcd/classification.cfm?ID=PSY) — intended-use and scope limitations of the U.S. WSI device category."""

    NUCLEAR_MEDICINE = """\
- [EANM/UEMS European Nuclear Medicine Guide](https://eanm.org/publications/useful-resources/nuclear-medicine-guide/) — broad European reference covering procedures, hybrid imaging, dosimetry, and radiation protection.
- [EANM clinical guidelines](https://eanm.org/publications/guidelines/overview/) — current procedure guidance by organ system, tracer, and application.
- [SNMMI procedure standards](https://sites.snmmi.org/Web/Web/Clinical-Practice/Procedure-Standards/Default.aspx) — U.S. procedure and interpretation standards, many jointly produced with EANM and other societies.
- [UK ARSAC Notes for Guidance (updated July 2026)](https://www.gov.uk/government/publications/arsac-notes-for-guidance) — UK good clinical practice for administration of radiopharmaceuticals in diagnosis, treatment, or research.
- [UKHSA National Diagnostic Reference Levels](https://www.gov.uk/government/publications/diagnostic-radiology-national-diagnostic-reference-levels-ndrls) — includes nuclear medicine administered-activity references via ARSAC guidance.
- [Euratom Basic Safety Standards Directive 2013/59](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=celex%3A32013L0059) — EU medical-exposure optimisation and administered-activity verification principles."""

    FLUOROSCOPY = """\
- [ACR-AAPM technical standard for management of fluoroscopic procedures](https://www.acr.org/-/media/ACR/Files/Practice-Parameters/MgmtFluoroProc.pdf) — dose recording, image acquisition, safety, and QA topics.
- [ACR-SIR-SNIS practice parameters](https://www.acr.org/clinical-resources/clinical-tools-and-reference/practice-parameters-and-technical-standards) — procedure-specific interventional radiology guidance and documentation.
- [UKHSA National Diagnostic Reference Levels](https://www.gov.uk/government/publications/diagnostic-radiology-national-diagnostic-reference-levels-ndrls) — UK fluoroscopy and interventional radiology dose benchmarks, updated in 2025.
- [RCR IR(ME)R guidance, second edition (2026)](https://www.rcr.ac.uk/our-services/all-our-publications/clinical-radiology-publications/ir-me-r-implications-for-clinical-practice-in-diagnostic-imaging-interventional-radiology-and-diagnostic-nuclear-medicine-second-edition/) — UK medical-exposure regulation implementation for imaging, interventional radiology, and diagnostic nuclear medicine.
- [Euratom Basic Safety Standards Directive 2013/59](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=celex%3A32013L0059) — EU justification, optimisation, QA, and dose recording requirements.
- [ACR-SIR practice parameter for reporting and archiving interventional radiology procedures](https://gravitas.acr.org/PPTS/GetDocumentView?docId=39) — final reports, structured reporting, image and dose-data archiving."""

    MAMMOGRAPHY = """\
- [ACR BI-RADS v2025](https://www.acr.org/Clinical-Resources/Clinical-Tools-and-Reference/Reporting-and-Data-Systems/BI-RADS) — standardized breast imaging terminology, report organization, assessment, and outcome monitoring for mammography, ultrasound, and breast MRI.
- [FDA Mammography Quality Standards Act program](https://www.fda.gov/radiation-emitting-products/mammography-quality-standards-act-mqsa-and-mqsa-program) — U.S. facility, personnel, equipment, dose, QA, recordkeeping, reporting, and clinical-image review requirements.
- [ACR Digital Mammography Quality Control Manual resources](https://www.acr.org/Accreditation/Resources/Digital-Mammography-QC-Manual) — QC for digital mammography and tomosynthesis.
- [NHS breast screening guidance for image reading](https://www.gov.uk/government/publications/breast-screening-guidance-for-image-reading/breast-screening-guidance-for-image-reading) — UK reading conditions, information checks, and image-quality requirements for screening.
- [NHS breast screening mammographer guidance](https://www.gov.uk/government/publications/breast-screening-quality-assurance-for-mammography-and-radiography/guidance-for-breast-screening-mammographers) — image quality, dose, repeat rates, audit, and peer review.
- [European Commission breast cancer guidelines](https://cancer-screening-and-care.jrc.ec.europa.eu/en/ecibc/european-breast-cancer-guidelines) and [European breast services QA scheme](https://cancer-screening-and-care.jrc.ec.europa.eu/en/ecibc/breast-quality-assurance-scheme) — EU evidence-based recommendations and service quality indicators."""

    DSA_ANGIOGRAPHY = """\
- [ACR-ASNR-SIR-SNIS practice parameter for diagnostic cervicocerebral catheter angiography](https://gravitas.acr.org/PPTS/GetDocumentView?docId=167) — examination performance, documentation, radiation safety, and quality improvement for adult cervicocerebral angiography.
- [ACR practice parameters for arteriography and angiographic procedures](https://www.acr.org/clinical-resources/clinical-tools-and-reference/practice-parameters-and-technical-standards) — select the relevant vascular territory and procedure from the current library.
- [ACR-SIR practice parameter for reporting and archiving interventional radiology procedures](https://gravitas.acr.org/PPTS/GetDocumentView?docId=39) — structured procedural reporting, image archiving, and dose-data archiving.
- [ACR-AAPM technical standard for management of fluoroscopic procedures](https://www.acr.org/-/media/ACR/Files/Practice-Parameters/MgmtFluoroProc.pdf) — fluoroscopy acquisition, dose documentation, and QA.
- [RCR IR(ME)R guidance, second edition (2026)](https://www.rcr.ac.uk/our-services/all-our-publications/clinical-radiology-publications/ir-me-r-implications-for-clinical-practice-in-diagnostic-imaging-interventional-radiology-and-diagnostic-nuclear-medicine-second-edition/) — UK medical-exposure governance for interventional radiology.
- [Euratom Basic Safety Standards Directive 2013/59](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=celex%3A32013L0059) — EU justification, optimisation, QA, and patient-dose principles."""

    ILLUSTRATION = """\
No modality-specific clinical interpretation standard was identified for generic illustrations. Use these sources for the narrower tasks of accessibility and communication, not for diagnostic truth:
- [NIH guidance on tables, charts, and figures](https://grants.nih.gov/grants-process/write-application/general-grant-writing-tips/tips-for-tables-charts-and-figures) — figure clarity, captions, and meaningful alternative text.
- [NIH Style Guide: alt text](https://www.nih.gov/nih-style-guide/structure-formatting) — concise image descriptions for accessible communication.
- [NHS England Accessible Information Standard](https://www.england.nhs.uk/accessible-information-standard/) — accessible health and care information.
- [GOV.UK guidance on publishing accessible documents](https://www.gov.uk/guidance/publishing-accessible-documents) — text alternatives for informative images, charts, and diagrams.
- [EU eHealth Network guidelines on medical imaging studies and reports](https://health.ec.europa.eu/publications/ehn-guidelines-medical-imaging-studies-and-reports_en) — general medical imaging study/report content; not an illustration interpretation standard."""
