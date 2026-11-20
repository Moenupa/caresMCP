# CaresMCP

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Docs](https://img.shields.io/readthedocs/caresmcp)](https://caresmcp.readthedocs.io/)

Medical imaging tools for AI agents, served over the Model Context Protocol
(MCP). CaresMCP wraps dedicated medical models behind 50+ callable tools and
returns evidence an agent can cite: scores, masks, heatmaps, measurements, and
embeddings.

**Documentation:** <https://caresmcp.readthedocs.io/>

Starting from the patient's presentation, an agent proactively orders
examinations, calls medical imaging tools on the studies, and grounds its
diagnosis in the evidence:

```mermaid
flowchart TD
    patient(["<b>Patient</b><br/><i>demographics · symptoms</i>"])

    subgraph agent["<b>Agent</b> · <i>orchestrator</i>"]
        direction TB
        vlm[["<b>VLM</b><br/><i>perception · reasoning</i>"]]
    end

    his[("<b>HIS</b> <i>(simulated)</i><br/>records · lab tests · images")]

    subgraph caresmcp["<b>CaresMCP</b> · <i>diagnostic toolkit</i>"]
        direction TB
        evidence[/"<b>Evidence</b><br/><i>scores · masks · measurements</i>"/]
    end

    diagnosis(["<b>Diagnosis</b><br/><i>grounded in evidence</i>"])

    patient -->|"① presents"| agent
    agent <-->|"② orders exam → images"| his
    agent ==>|"③ calls tools"| caresmcp
    caresmcp ==>|"④ returns evidence"| agent
    caresmcp -.->|"<i>traceability</i>"| diagnosis
    agent -->|"⑤ concludes"| diagnosis

    classDef io fill:#EEF2FF,stroke:#6366F1,color:#1E1B4B,stroke-width:1.5px
    classDef model fill:#DCFCE7,stroke:#16A34A,color:#14532D,stroke-width:1.5px
    classDef db fill:#CCFBF1,stroke:#0D9488,color:#042F2E,stroke-width:1.5px
    classDef evid fill:#FEF3C7,stroke:#D97706,color:#451A03,stroke-width:1.5px
    classDef result fill:#FFE4E6,stroke:#E11D48,color:#4C0519,stroke-width:2.5px

    class patient io
    class vlm model
    class his db
    class evidence evid
    class diagnosis result

    style agent fill:#F0FDF4,stroke:#16A34A,stroke-width:2px,color:#14532D
    style caresmcp fill:#FFFBEB,stroke:#D97706,stroke-width:2px,color:#451A03
```

## Quick start

```console
uv tool install "caresmcp[xray] @ git+https://github.com/Moenupa/caresMCP.git"
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
  extras and source installs.
- [Limitations and safety](https://caresmcp.readthedocs.io/en/latest/limitations.html) —
  research use, checkpoint licenses, and privacy.
- [CaresAgentBench (WIP)](https://caresmcp.readthedocs.io/en/latest/caresagentbench.html) —
  the agent benchmark pilot and its work-in-progress status.
- [Contributing](https://caresmcp.readthedocs.io/en/latest/contributing.html) —
  add a tool or update a model survey.

CaresMCP is research software, not a medical device. Check each upstream
checkpoint's license and access terms before use.
