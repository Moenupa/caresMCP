:orphan:

🚧 CaresAgentBench (Work in Progress)
=====================================

.. note::

   🚧 CaresAgentBench is under development. The pilot below is the motivating study; the
   full protocol, dataset, prompts, scoring definitions, and per-case results are still
   being finalized.

CaresAgentBench measures whether tool access makes a medical agent better at answering
imaging questions. It is not a model leaderboard: it evaluates the agent's final answer
together with the evidence it used.

🎯 The Point
------------

A general VLM can be unsure, overconfident, or distracted by spurious findings. A
dedicated classifier can be sharper on its narrow labels but knows nothing outside them.
CaresAgentBench is built around the gap between those two: the agent should call the
right tools, inspect the returned evidence, and calibrate its answer against that
evidence.

The benchmark therefore scores more than accuracy:

- **Outcome** — final answer quality on the task.
- **Evidence use** — whether the answer cites tool scores, masks, heatmaps, or
  measurements that support it.
- **Calibration** — whether tool evidence changes the agent's confidence in the right
  direction.
- **Tool selection** — whether the agent calls an appropriate tool instead of guessing.

🩻 Pilot: Chest X-ray Calibration
---------------------------------

The pilot compared three configurations on a chest X-ray task. These are the motivating
numbers, not a clinical validation.

.. list-table::
   :header-rows: 1
   :widths: 35 20 20

   - - Configuration
     - Soft accuracy (%)
     - Hard accuracy (%)
   - - VLM alone
     - 56.81
     - 57.14
   - - Dedicated classifier
     - 51.14
     - 61.90
   - - VLM + CaresMCP tool
     - 63.57
     - 57.14

The agent configuration gave the best soft accuracy, while the standalone classifier was
best on hard accuracy. That split is the point: the agent is not a replacement for the
specialist model. It is an orchestration layer that can use the specialist as evidence
and still reason about the whole case.

One pilot case shows the calibration loop:

1. The VLM initially scored **Hernia 0.62** and **Emphysema 0.13**, with several weaker
   findings.
2. The pilot's expert classifier returned **Hernia 0.9897**, plus Mass 0.5351,
   Infiltration 0.5139, and Fibrosis 0.503.
3. The pilot's localization tool returned a Grad-CAM heatmap over the lower abdomen and
   diaphragm region.
4. The agent combined the classifier score and the heatmap, raised Hernia to **0.96**,
   and removed three false-positive labels.

The final answer is grounded in a score and a region, not just a stronger guess. That is
the behavior CaresAgentBench is designed to measure.

🧭 Status
---------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   - - Item
     - Status
   - - Pilot result
     - Available above; motivating only.
   - - Full protocol and scoring definitions
     - In progress.
   - - Dataset and prompts
     - In progress.
   - - Exact checkpoints and tool versions
     - In progress.
   - - Per-case results and tool traces
     - In progress.
   - - Reproducibility package
     - Planned.

The pilot reported two accuracy variants; the full benchmark will publish their
definitions, dataset, prompts, and per-case results. The numbers do not establish
performance on a new dataset, scanner, population, or clinical task. See
:doc:`limitations` for the broader safety and validation caveats.
