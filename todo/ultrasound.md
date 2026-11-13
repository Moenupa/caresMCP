# Ultrasound tool plan

Survey: 2026-11-12. Implementation pass: 2026-11-13. Scope: the cases in
[ultrasound_cases.md](ultrasound_cases.md).

A linked checkpoint is a lead, not proof that it can be downloaded, loaded, or
used under this project's license. This pass verified the actual files,
dependency versions, and terms only for the implemented tool below. A thin
wrapper reproduces released inference; it does not adapt custom architectures,
transcode multi-script pipelines, or establish clinical accuracy.

## Completed in this pass

- [x] Prompted ultrasound segmentation: [sam2.py](../src/caresmcp/ultrasound/sam2.py)
  `segment_ultrasound_sam2`. Positive/negative point prompts and/or a box prompt
  select one user-nominated structure, and the tool returns an input-sized
  palette PNG: 0 is transparent background and 1 (blue) is the prompted
  structure. This reproduces the image usage documented for the released
  `facebook/sam2.1-hiera-{tiny,small,base-plus,large}` checkpoints with
  Transformers `Sam2Model` and `Sam2Processor`. Code and weights are Apache-2.0;
  the `ultrasound` extra already provides Transformers, so no new dependency or
  extra was added. Verified with Transformers 4.57.6 on Python 3.12 and CPU.
- [x] Shared loader support: [hf_hub.py](../src/caresmcp/api/models/hf_hub.py).
  Added `HFModelType.SAM2`, which loads the image-mode `Sam2Model` and
  `Sam2Processor` through the existing cached loader. The released checkpoints
  are stored as `sam2_video`; Transformers warns about that mismatch when
  loading the documented image interface, and inference was verified to produce
  plausible, non-empty masks despite the warning. Video propagation is not
  exposed.
- [x] Real-sample verification: [test_sam2.py](../tests/ultrasound/test_sam2.py).
  Point and box prompts on the Radiopaedia hand-ultrasound fixture produce
  non-empty, input-sized binary masks; prompt validation is covered. The file
  passes 9 tests with `RUN_SLOW=1`; inference artifacts are saved with
  `@save_returns("ultrasound_sam2")`. The 630 x 408 fixture and fixed prompts
  are illustrative, not a reference calcification or anatomy label.
- [x] Post-change quick suite: `pytest tests/ultrasound` reports 10 passed and
  16 slow-skipped. The existing EchoNet-Dynamic, PanEcho and EchoCare tools and
  their preprocessing caveats are unchanged from the 2026-11-12 pass.
- [x] Scope limits documented in the tool docstring: this is general-domain
  SAM 2, not an ultrasound-adapted segmenter. The mask follows user prompts; it
  does not identify, grade, or measure pathology. Accuracy for any target case
  below is unverified, and the mask must not be relabeled as a diagnosis.

## Current coverage

The eight tools in [ultrasound](../src/caresmcp/ultrasound/) now include cardiac
PanEcho video/study prediction and video/image embeddings, EchoNet-Dynamic EF
and LV segmentation, general ultrasound EchoCare image embeddings, and SAM 2
prompted segmentation. Only the EchoNet-Dynamic tools are task-specific
(cardiac); SAM 2 is assistive and prompt-driven. None is a validated
task-specific solution for the target cases. Embeddings alone do not identify a
disease. Existing inference tests are slow-gated; the larger models are also
CUDA-gated.

## Candidate assessment for the target cases

All remaining leads below were rechecked against their current releases and
skipped under the thin-wrapper constraint.

- [x] **Neonatal brain: NCLS** — skipped. NCLS is the strongest direct match for
  `germinal-matrix-hemorrhage-grade-iii-2`, but its upstream inference combines
  an RT-DETR standard-view detector, five diagnostic folds, and study-level
  processing through separate file-oriented scripts. The repository ships code
  only; example videos and all detector/diagnostic weights are Google Drive
  downloads, and CUDA is recommended. Adapting that pipeline exceeds a thin
  wrapper. Grade III/IV IVH is included in a severe-lesion group alongside
  other conditions; the published task does not specifically diagnose or grade
  IVH, so a score must not be converted into a grade or inferred from one
  image. [code](https://github.com/Je1zzz/Neonatal_cerebral_lesions_screening_NCLS),
  [paper](https://www.nature.com/articles/s41467-025-63096-9).
- [x] **Cross-case interactive segmentation: SonoBase / UltraSam** — skipped as
  checkpoints; the assistive use case is covered by `segment_ultrasound_sam2`.
  SonoBase requires its `nemo_cv` training/evaluation stack with hydra configs
  and a custom SAM2-derived TriBranch encoder; code is MIT but the 706 MB
  inference checkpoint is CC BY-NC 4.0.
  [SonoBase](https://github.com/AlfredQin/sonobase),
  [weights](https://huggingface.co/AlfredQin/sonobase). UltraSam requires its
  own MMDetection-based stack. [UltraSam](https://github.com/CAMMA-public/UltraSam).
  The SAM 2 wrapper is Apache-2.0 and loadable through the shared loader, but it
  is not ultrasound-adapted; SonoBase/UltraSam-specific accuracy claims do not
  transfer to it. Prompted masks are user-directed delineation, not automatic
  disease identification.
- [x] **Scrotum: Cond-UNet** — skipped. The Hub checkpoint exposes a binary
  foreground mask with an optional organ token (`7` is Testicle); it does not
  detect cysts. As verified in the 2026-11-12 pass, its custom code failed the
  Hub dependency scan on an unpublished `utils.paths` module even after
  installing `ptwt`, `wandb`, and `peft`; its weight license is unspecified,
  and the released pipeline resizes the padded prediction back without removing
  padding. The associated TesticulUS dataset
  describes testicular organ segmentation, not intratesticular cyst detection.
  Do not expose an ambiguous foreground mask as a testis or cyst result.
  [code](https://github.com/AImageLab-zip/US_Cond-UNet),
  [weights](https://huggingface.co/AImageLab-Zip/US_Cond-UNet),
  [TesticulUS](https://ditto.ing.unimore.it/testiculus/).
- [x] **Pelvic deep endometriosis** — skipped. The AI-DE study describes YOLOv8
  detection of pelvic deep-endometriosis nodules on ultrasound images and
  videos, but no accessible trained weights were verified. Check with the
  authors for weights, code, license, class definitions, anatomical sites, and
  video preprocessing before planning a detector. The linked rectosigmoid
  segmentation work uses MRI, so it is not an ultrasound candidate. Prompted
  masks do not establish endometriosis or bowel invasion.
  [AI-DE](https://www.umbjournal.org/article/S0301-5629%2825%2900091-2/fulltext),
  [rectosigmoid MRI](https://www.sciencedirect.com/science/article/pii/S0262885624003664).
- [x] **Pregnant uterus/cervix** — skipped. The multiclass TVUS study links
  training/evaluation code but no trained ensemble checkpoints; access would
  require the authors. Its out-of-distribution images were from 22–25 weeks, so
  applicability to `inevitable-abortion-4` is unknown. A gestational-sac
  segmentation study includes miscarriage cases, but no usable checkpoint was
  verified. Neither anatomy segmentation nor a pregnancy-loss score diagnoses
  inevitable abortion or establishes cervical dilation.
  [multiclass TVUS](https://www.nature.com/articles/s44294-025-00075-x),
  [code](https://github.com/cumcrad/MulticlassSegmentationTVUS),
  [gestational sac](https://www.frontiersin.org/journals/pediatrics/articles/10.3389/fped.2024.1453302/full).
- [x] **Uterine artery spectral Doppler** — skipped. No uterine-artery-specific
  spectral-envelope or notch-detection weights were verified.
  [EchoNet-Measurements](https://github.com/echonet/measurements) provides
  cardiac Doppler segmentation/measurement scripts; its cardiac task and
  file-oriented preprocessing do not establish transfer to uterine arteries.
  The published [waveform simulator](https://pmc.ncbi.nlm.nih.gov/articles/PMC6511649/)
  models physiology and is not a screenshot inference tool. A future tool must
  specify spectral ROI, baseline, axis calibration, velocity direction, beat
  selection, and units, and must distinguish a notch or PI/RI measurement from
  a pre-eclampsia diagnosis.
- [x] **Thumb MCP calcific periarthritis and pyramidalis muscle** — skipped. No
  verified task-specific pretrained checkpoint was found for either case. The
  SAM 2 tool can delineate a user-selected echogenic calcification or muscle
  boundary, especially when prompted with points and a box, but it does not
  detect, classify, or label calcific periarthritis or pyramidalis. Do not
  transfer shoulder calcification or limb-muscle models without evidence, and
  do not label the prompted mask as either finding.

## Validation and integration requirements

- [x] Released weights, architecture, runtime and licenses: verified for SAM 2
  (Apache-2.0 code and weights, Transformers 4.57.6, Python 3.12, shared loader,
  CPU inference). Skipped candidates retain the per-candidate reasons above.
  No related anatomical or clinical task was silently substituted for a target
  finding.
- [x] Real samples and test conventions: [test_sam2.py](../tests/ultrasound/test_sam2.py)
  uses a real public ultrasound image, is slow-gated, saves artifacts with
  `@save_returns`, and does not need `@pytest.mark.cuda` (the default tiny model
  is about 39M parameters). The SAM 2 checkpoints are not clinically validated
  for these cases; patient/study splits and reference labels remain absent.
- [x] Input/output conversion and loader use: the public wrapper decodes bytes,
  validates prompt geometry on the original pixel grid, uses the shared cached
  HF loader, and returns a self-explanatory palette mask. No runtime assertions
  were added, and Pydantic `Field` annotations carry the input descriptions.

## Existing tool maintenance

The following review and verification record from 2026-11-12 still applies; only
the test-suite line was rechecked after adding SAM 2.

- [x] EchoNet-Dynamic image and video decoding and normalization run at the
  public interface. The model loaders remain local: the shared loader targets
  hub models or checkpoints and does not construct these modified torchvision
  architectures. The real-image/video test file passed after that change.
- [x] Explicit ultrasound input and output errors use fixed messages;
  dependency and upstream loader exceptions remain subject to the shared
  framework's error handling. No user image or video contents are interpolated
  into those messages.
- [x] Preprocessing and checkpoint behavior were reviewed against upstream;
  the differences and limits are below. The existing real-sample tests passed
  on CPU, including both CUDA-marked EchoCare tests. Study aggregation has a
  passing two-view test using patient-matched clips from the CC BY 4.0
  [EchoNSTE-MI dataset](https://data.mendeley.com/datasets/y6z64ntstg/1),
  fetched by HTTP range requests without downloading the full archive.
- [x] PanEcho's upstream `EchoDataset` samples contiguous random clips and
  center crops frames; this wrapper uses uniform samples across the full video
  and resizes RGB frames to 224 pixels. Both apply ImageNet normalization, but
  the wrapper's results have not been established as evaluation-pipeline
  parity.
- [x] EchoNet-Dynamic's upstream converts OpenCV frames to RGB, normalizes with
  training-set RGB statistics, pads short videos after normalization, and uses
  random clips or averages all clips at evaluation. The wrapper instead uses a
  single centered clip and resizes frames to 112 pixels, so its smoke test is
  not an all-clips evaluation or a measure of clinical performance.
- [x] EchoCare's upstream pretraining converts to RGB, uses random resized
  crops and converts pixels to [0, 1]; the wrapper uses a deterministic 256×256
  resize and the same pixel scale. The upstream release does not specify a
  canonical embedding inference transform, so this does not establish
  downstream parity.
- [x] Channel-statistic validation uses the public LV tool in real-sample
  tests; EF registration is checked and EchoNet/PanEcho inference tests save
  artifacts. EchoNet-Dynamic's two models have fewer than 40M parameters, so
  their tests can run on CPU. Its full test file passed on CPU with public real
  samples and release weights (10 passed). The EchoNet smoke-test statistics
  from a sample image are not training statistics and cannot establish
  inference parity.
- [x] Retuve/TinyUNet was assessed and skipped: its plugin requires `stem-cv`,
  whose `timm==0.4.12` conflicts with this project's `timm>=1.0.27`. Its weights
  are CC BY-NC-SA 3.0. Bypassing the dependency or porting the model falls
  outside a thin wrapper.

## Other ultrasound backlog (lower priority for these cases)

These are retained for future work, not proposed as solutions to the target
cases. They were not re-run in this pass; their release-level reasons for
skipping remain as recorded in the 2026-11-12 survey.

- [x] Cardiac: skip [EchoNet-Measurements](https://github.com/echonet/measurements)
  for this pass: its release is a collection of file-oriented scripts, some
  parse arguments at import, with DICOM calibration and task-specific inference
  and output workflows. Adapting those to callable, calibrated byte interfaces
  is more than a thin wrapper. Its cardiac Doppler models do not establish
  uterine-artery applicability. Skip [EchoNet-AS](https://github.com/echonet/AS):
  its published workflow runs Doppler inference, six view-specific models,
  DICOM-to-video conversion, and file-based study aggregation as separate
  scripts; adapting it to a bytes-based study API would exceed a thin wrapper.
- [x] Video/other organs: skip [CVA-Net](https://github.com/jhl-Det/CVA-Net),
  whose unpublished-as-a-package inference workflow requires compiling its
  custom CUDA operators and adapting video processing; its dataset labels also
  needed correction. Skip [TRUSFormer](https://github.com/med-i-lab/TRUSFormer):
  its repository is an experimental training framework with server-dependent
  data and backbone pretraining, without a directly usable task inference
  release. Skip [URI-CADS](https://github.com/miguel55/URI-CADS): weights and
  data require a request and its inference depends on a bundled modified
  torchvision 0.5 implementation and PyTorch 1.3.1; adapting that is beyond a
  thin wrapper. Skip [BINDS](https://github.com/lyhkevin/BINDS): ultrasound
  inference depends on a separate lesion-ROI detection/selection workflow and
  unbundled weights, with an upstream Python 3.8/PyTorch 1.11 environment. Skip
  [Open Kidney Ultrasound](https://github.com/rsingla92/kidneyUS): the
  repository supplies evaluation code and links weights, but no inference
  architecture or API; its actual image data also require registration.
  [EchoTracker](https://github.com/riponazad/echotracker) is skipped: its
  released code is not packaged, imports top-level `model` and `utils` modules,
  initializes a CUDA-only custom architecture, and relies on a separate Drive
  checkpoint. An integration would require packaging and adapting the upstream
  project rather than a thin wrapper. The code and weights are CC BY-NC 4.0.
- [x] Obstetric and hip anatomy: skip [SonoNet](https://github.com/bkainz/SonoNet-weights)
  because its Theano/Lasagne model requires a runtime port. Skip
  [FetalNet](https://github.com/SanoScience/FetalNet): its repository contains
  only a model definition and a Drive checkpoint link, with no inference
  interface or preprocessing and measurement postprocessing. Skip
  [BiometryNet](https://github.com/surgical-vision/Multicentre-Fetal-Biometry):
  the cited repository supplies training/evaluation scripts but its documented
  task checkpoints are produced by training, and the setup targets Python 3.6
  and PyTorch 1.0; a usable inference wrapper and trained task weights were not
  found. Skip [FUVAI](https://github.com/SanoScience/FUVAI): its release
  contains model definitions and a Drive checkpoint, but no packaged inference
  API or usable preprocessing and result decoding, so integration would require
  implementing a complete video workflow. [OBUS-GHL](https://github.com/Global-Health-Labs/OBUS-GHL)
  provides an obstetric inference framework for gestational age, fetal
  presentation and weight, but its documented inference needs a separately
  trained checkpoint; no downloadable task weights were found in the
  repository. Its tasks also do not identify inevitable abortion.
- [x] Ultrasound image encoders URFM and USF-MAE were considered for embedding
  use cases; pretrained embeddings are not disease-specific heads. EchoPrime,
  EchoCLIP and FetalCLIP are VLMs and are outside this repository's scope.
  Skip [OpenUS](https://github.com/XZheng0427/OpenUS), whose released workflow
  specifies an incompatible `timm==0.4.12` and custom VMamba/CUDA stack, and
  [EchoFM](https://github.com/SekeunKim/EchoFM), whose encoder requires its
  unpackaged custom video-ViT architecture and preprocessing pipeline; adapting
  either exceeds a thin wrapper. Skip [USFM](https://github.com/openmedlab/USFM),
  whose published runtime requires a modified MMSegmentation stack, and
  [UltraFedFM](https://github.com/yuncheng97/UltraFedFM), whose release provides
  training and dataset-oriented evaluation scripts rather than a packaged
  embedding inference interface. The official
  [URFM checkpoint repository](https://huggingface.co/QingboKang/URFM) is gated
  (`gated: auto` in its Hub metadata), so anonymous automatic loading cannot be
  assumed. [USF-MAE](https://github.com/Yusufii9/USF-MAE) publishes Google Drive
  checkpoints and pretraining notebooks but no ready-to-use encoder inference
  API in its repository. Skip both for this pass: URFM needs model access, and
  adapting USF-MAE's checkpoint and preprocessing into a supported inference
  path exceeds a thin wrapper.
- [x] FUSEP's [current release](https://github.com/ahuteam/FUSEP) now links
  fetal-anatomy detection checkpoints, correcting the previous no-weights note.
  They target fetal structures at 11–14 weeks, not gestational-sac or cervical
  findings in the target pregnancy-loss case; the release also requires its
  custom Relation-DETR inference stack. Skip it for this pass.
- [x] Previously unverified task weights were not rechecked because no relevant
  task remained: [MUST-Sub](https://github.com/takimailto/MUST-Sub),
  [EchoNet-LVH](https://github.com/echonet/lvh),
  [DSMA-Net](https://github.com/guangxin1990/DSMA-Net).

`data_subset10/ultrasound/` was absent in the prior checkout review. Existing
fixtures use remote Radiopaedia examples; other datasets linked by the projects
above may require registration or research-only agreements. Check access and
suitability independently before using them for tests.
