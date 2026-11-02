# CaresMCP


CaresMCP brings dedicated medical ML models & tools to agents through
the Model Context Protocol (MCP). Our goal is to help medical vision-language
models (VLMs) overcome bottlenecks in medical reasoning, data sourcing, and
interpretability: diagnostic conclusions should be traceable to evidence and
even to individual images. These tools also provide a foundation for test-time training and
self-evolving agent workflows.

```mermaid
flowchart TD
    input(["<b>Input</b><br/>medical image, clinical question"])
    vlm[["<b>VLM</b><br/>Model API"]]
    agent{{"<b>Agent</b><br/>plans, calls tools, checks evidence"}}

    framework["<b>CaresMCP</b><br/>Medical AI, Image tools"]

    evidence("<b>Evidence</b><br/>findings, groundings, measurements")
    diagnosis(["<b>Diagnosis</b><br/>grounded in evidence"])

    subgraph adaptation["<b>Self-Evolution</b> · our vision"]
        direction LR
        training["Test-time training"]
        evaluation["Agent Evaluation"]
        evolution["Self-evolving workflows"]
        training -.-> evaluation -.-> evolution
    end

    input -->|"① question + images"| agent
    agent <-->|"② reason / perceive"| vlm
    agent ==>|"③ select and execute"| framework
    framework ==>|"④ return results"| evidence
    evidence ==>|"⑤ ground reasoning,<br/>guide next step"| agent
    agent -->|"⑥ conclude"| diagnosis
    evidence -->|"traceability"| diagnosis

    evidence -.->|"inform adaptation"| adaptation
    adaptation -.->|"refine future tool use"| agent

    classDef io fill:#EEF2FF,stroke:#6366F1,color:#1E1B4B,stroke-width:1.5px
    classDef tool fill:#FEF3C7,stroke:#D97706,color:#451A03,stroke-width:3px
    classDef model fill:#F3E8FF,stroke:#9333EA,color:#3B0764,stroke-width:1.5px
    classDef agentNode fill:#F0FDF4,stroke:#16A34A,stroke-width:2px,color:#14532D
    classDef evid fill:#E0F2FE,stroke:#0284C7,color:#082F49,stroke-width:1.5px
    classDef result fill:#FFE4E6,stroke:#E11D48,color:#4C0519,stroke-width:2.5px
    classDef vision fill:#F8FAFC,stroke:#64748B,color:#0F172A,stroke-width:1.5px,stroke-dasharray:5 4

    class input io
    class agent agentNode
    class vlm model
    class framework tool
    class evidence evid
    class diagnosis result
    class training,evaluation,evolution vision

    style adaptation fill:#F8FAFC,stroke:#64748B,stroke-width:2px,stroke-dasharray:6 4,color:#334155
```

![framework overview](CaresMCP.png)

## Architecture

We support over 100 models or tools with dedicated end-to-end workflow function APIs,
and generic adapters to popular libraries.

```mermaid
flowchart LR
    S["CaresMCP server<br/>(MedToolLib)"]
    S --> W[("Workflow APIs<br/>(50+ tools)")]
    S --> G[("Framework adapters")]
    W --> M1["Segmentation"]
    W --> M2["Classification"]
    W --> M3["Localization"]
    W --> M4["..."]
    G --> F1["transformers"]
    G --> F2["monai"]
    G --> F3["..."]
    style S fill:#FEF3C7,stroke:#D97706,color:#451A03,stroke-width:3px
```

## Coverage by Body-Part and Modality

```mermaid
mindmap
  root((CaresMCP))
    Brain: 2
      MRI tumor, BraTS
      Stroke lesion
    Chest and lungs: 9
      CXR classify
      CXR localize
      CXR segment
      CXR embed
      Chest CT embed
    Heart: 9
      Echo EF
      LV segmentation
      PanEcho
      Cardiac MRI seg
    Any anatomy: 21
      CT
      MRI
      TotalSegmentator
      Image utilities
    Microscopic tissue: 11
      Pathology encoders
      Zero-shot classifiers
      Cellpose
    Gap: 0
      Abdomen
      Bone
      Breast
```

```mermaid
pie showData title Tools by task type
    "Segmentation" : 13
    "Embedding" : 11
    "Classification" : 3
    "Image-text (CLIP)" : 6
    "Analysis and utilities" : 19
```

## Example workflow: chest X-ray with expert-tool calibration

A VLM alone can be unsure or distracted by spurious findings. CaresMCP lets the agent call an expert classifier, then verify visually, and use the result to calibrate its own probabilities (test-time calibration).

```mermaid
sequenceDiagram
    autonumber
    participant V as VLM agent
    participant M as CaresMCP
    participant C as vit_classify()
    participant L as vit_localize()

    V->>V: Initial read of CXR (VLM probs)
    Note over V: Hernia 0.62, Emphysema 0.13, others low
    V->>M: Call expert classifier
    M->>C: Run ViT on image
    C-->>V: ViT probs (14+ labels)
    Note over C,V: Hernia 0.9897, Mass 0.5351, Infiltration 0.5139, Fibrosis 0.503 ...
    V->>M: Visual evidence check
    M->>L: Localize top finding
    L-->>V: Attention heatmap (lower abdomen / diaphragm region)
    V->>V: Combine VLM + ViT + heatmap
    Note over V: Agent probs: Hernia 0.96, others suppressed (3 FP labels removed)
```

- The expert classifier raises confidence in the true label (0.62 to 0.96 after calibration).
- The visual-evidence step filters out distractors, removing 3 false-positive labels and improving accuracy.
- The agent output acts as test-time calibration of the VLM using tool evidence.

## CXR Benchmark Pilot Study

```mermaid
xychart-beta
    title "Chest X-ray accuracy (%)"
    x-axis ["GPT-6-Luna", "torchxrayvision", "GPT-6-Luna + tool"]
    y-axis "Accuracy" 40 --> 70
    bar [56.81, 51.14, 63.57]
    bar [57.14, 61.90, 57.14]
```

Agent gives the best **soft** accuracy (63.57), while the standalone classifier is best on **hard** accuracy (61.90).
