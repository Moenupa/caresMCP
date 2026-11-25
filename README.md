# CaresMCP

[![DOI](https://zenodo.org/badge/1407386890.svg)](https://doi.org/10.5281/zenodo.24521705)
[![PyPI](https://img.shields.io/pypi/v/caresmcp)](https://pypi.org/project/caresmcp/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Docs](https://img.shields.io/readthedocs/caresmcp)](https://caresmcp.readthedocs.io/)

Medical imaging tools for AI agents, served over the Model Context Protocol
(MCP). CaresMCP wraps dedicated medical models behind 50+ callable tools and
returns evidence an agent can cite: scores, masks, heatmaps, measurements, and
embeddings.

**Package:** <https://pypi.org/project/caresmcp/> ·
**Documentation:** <https://caresmcp.readthedocs.io/>

Starting from the patient's presentation, an agent proactively orders
examinations, calls medical imaging tools on the studies, and grounds its
diagnosis in the evidence:

```mermaid
---
config:
  markdownAutoWrap: false
---
flowchart TD
    patient(["`<b>Patient</b><br/>demographics · symptoms`"])
    agent["`<b>Agent</b> · <i>orchestrator</i><br/>perception · reasoning`"]
    his[("`<b>HIS</b> <i>(simulated)</i><br/>records · reports · images`")]

    subgraph caresmcp["<b>CaresMCP</b> · <i>diagnostic toolkit</i>"]
        direction TB
        evidence(["`<b>Evidence</b><br/>scores · masks · measurements`"])
    end

    diagnosis(["`<b>Diagnosis</b><br/>grounded in evidence`"])
    human["<b>Human<br/>Clinician</b>"]

    patient -->|"① presents"| agent
    his <-->|"② orders exam → images"| agent
    agent <-.->|"④ if uncertain"| human
    agent <==>|"③ calls tools → evidence"| caresmcp
    caresmcp -.-|"<i>traceability</i>"| diagnosis
    agent -->|"④ concludes"| diagnosis
    human <-.->|"⑤ reviews"| diagnosis

    %% Semantic color classes
    classDef patient fill:#EFF6FF,stroke:#3B82F6,color:#1E40AF,stroke-width:1.5px
    classDef agent fill:#ECFDF5,stroke:#059669,color:#065F46,stroke-width:2px
    classDef his fill:#F1F5F9,stroke:#64748B,color:#334155,stroke-width:1.5px
    classDef evidence fill:#FFFBEB,stroke:#D97706,color:#92400E,stroke-width:1.5px
    classDef human fill:#F5F3FF,stroke:#8B5CF6,color:#5B21B6,stroke-width:1.5px
    classDef result fill:#EEF2FF,stroke:#4F46E5,color:#312E81,stroke-width:2.5px

    class patient patient
    class agent agent
    class his his
    class evidence evidence
    class human human
    class diagnosis result

    style caresmcp fill:#FFFCF5,stroke:#D97706,stroke-width:1.5px,color:#92400E

    linkStyle default stroke:#64748B,stroke-width:1.5px
```

## Quick start

```console
uv tool install "caresmcp[xray]"
caresmcp list xray
caresmcp debug xray torchxrayvision_classify \
  '{"image_or_dicom": "https://nih-chest-x-rays.s3.us-east-2.amazonaws.com/images_1024x1024/00000003_001.png"}'
```

The first model-backed call downloads its checkpoint. To serve the tools over
MCP:

```console
caresmcp mcp
claude mcp add --transport http caresmcp http://127.0.0.1:8000/mcp
```

Full walkthrough, output examples, and troubleshooting:
[Getting started](https://caresmcp.readthedocs.io/en/latest/quick-start.html).

## Documentation

- [Capabilities](https://caresmcp.readthedocs.io/en/latest/capabilities/) —
  browse tools by task or modality; the catalog is generated from the registry.
- [Concepts](https://caresmcp.readthedocs.io/en/latest/concepts.html) — tools,
  categories, tasks, models, and the input/output contract.
- [Model coverage](https://caresmcp.readthedocs.io/en/latest/coverage/) —
  integrated, deferred, and out-of-scope models with access terms.
- [Installation](https://caresmcp.readthedocs.io/en/latest/installation.html) —
  PyPI, extras, and source installs.
- [Limitations and safety](https://caresmcp.readthedocs.io/en/latest/limitations.html) —
  research use, checkpoint licenses, and privacy.
- [CaresAgentBench (WIP)](https://caresmcp.readthedocs.io/en/latest/caresagentbench.html) —
  the agent benchmark pilot and its work-in-progress status.
- [Contributing](https://caresmcp.readthedocs.io/en/latest/contributing.html) —
  add a tool or update a model survey.

CaresMCP is research software, not a medical device. Check each upstream
checkpoint's license and access terms before use.
