# ForensiXplain

## An Evidence-Grounded Explainable Framework for Temporal Reconstruction and Anomaly Detection in Digital Forensics

ForensiXplain is a research-oriented digital forensics framework designed to transform heterogeneous forensic artifacts into a structured temporal representation, detect anomalous event patterns, connect anomalies back to forensic evidence, and generate investigator-readable explanations.

The project combines **digital forensics, temporal event reconstruction, graph-based representation, unsupervised anomaly detection, and explainable AI**.

---

## Research Question

> **Can heterogeneous digital forensic artifacts be transformed into an explainable temporal knowledge graph that automatically detects anomalous event sequences and provides evidence-grounded explanations for investigators?**

---

## Project Objectives

ForensiXplain aims to:

1. Extract structured observations from heterogeneous forensic artifacts.
2. Normalize forensic observations into a unified event schema.
3. Reconstruct chronological timelines while preserving timestamp confidence.
4. Represent relationships between processes and forensic artifacts as a temporal graph.
5. Detect anomalous process and event patterns using unsupervised machine learning.
6. Explain anomaly scores using SHAP.
7. Link anomalous events back to their supporting forensic evidence.
8. Generate investigator-readable explanations without treating anomaly scores as proof of malicious activity.
9. Provide a reproducible research pipeline suitable for academic evaluation.

---

# Architecture

```text
                 Digital Forensic Sources
                          |
        +-----------------+-----------------+
        |                 |                 |
       Disk             Memory            Logs
        |                 |                 |
        +-----------------+-----------------+
                          |
                   Artifact Extraction
                          |
                   Event Normalization
                          |
                 Unified Event Schema
                          |
              +-----------+-----------+
              |                       |
          Timeline               Evidence
          Reconstruction          Provenance
              |                       |
              +-----------+-----------+
                          |
                Temporal Knowledge
                     Graph
                          |
             +------------+------------+
             |                         |
       Temporal Features         Graph Features
             |                         |
             +------------+------------+
                          |
                Unsupervised Anomaly
                     Detection
                          |
                 +--------+--------+
                 |                 |
                SHAP          Evidence
                 |            Attribution
                 |                 |
                 +--------+--------+
                          |
                Investigator Explanation
                          |
                  Human-in-the-Loop
```

---

# Core Pipeline

ForensiXplain follows a structured research pipeline:

```text
Forensic Artifacts
        ↓
Artifact Extraction
        ↓
Event Normalization
        ↓
Temporal Reconstruction
        ↓
Temporal Knowledge Graph
        ↓
Feature Engineering
        ↓
Unsupervised Anomaly Detection
        ↓
SHAP-Based Explanation
        ↓
Evidence Attribution
        ↓
Investigator-Readable Explanation
```

---

# Key Components

## 1. Artifact Extraction

The framework processes heterogeneous digital forensic artifacts and extracts structured observations from available forensic sources.

Potential sources include:

* Disk artifacts
* Memory artifacts
* System logs
* Process information
* File-system activity
* Network-related artifacts
* Other forensic evidence sources

---

## 2. Event Normalization

Forensic observations from different sources are converted into a common event representation.

A normalized event can contain information such as:

```text
Event ID
Timestamp
Timestamp Confidence
Event Type
Process
Artifact
Source
Action
Evidence Reference
```

This allows evidence from different forensic sources to be analyzed within a unified representation.

---

## 3. Temporal Reconstruction

The framework reconstructs the sequence of forensic events chronologically.

The reconstruction process considers:

* Event timestamps
* Timestamp confidence
* Event ordering
* Process relationships
* Artifact relationships
* Cross-source temporal relationships

The goal is to produce a timeline that preserves the uncertainty inherent in forensic timestamps.

---

## 4. Temporal Knowledge Graph

Normalized events are represented as a temporal knowledge graph.

The graph captures relationships between entities such as:

```text
Process
   |
   +----> Event
   |
   +----> File
   |
   +----> Network Artifact
   |
   +----> System Artifact
```

Temporal relationships allow the framework to represent not only **what happened**, but also **when it happened and how events are related**.

---

## 5. Feature Engineering

Features are extracted from reconstructed events and graph structures.

These may include:

* Temporal features
* Event frequency
* Process behavior
* Event sequences
* Graph connectivity
* Process-artifact relationships
* Structural graph characteristics

The resulting feature representation is used for anomaly detection.

---

## 6. Unsupervised Anomaly Detection

ForensiXplain uses unsupervised machine learning to identify unusual event and process patterns.

The anomaly detection component is intended to identify observations that differ from established behavioral patterns.

An anomaly score represents **unusualness**, not proof of malicious activity.

---

## 7. Explainable AI with SHAP

SHAP is used to analyze the contribution of individual features to model outputs.

This helps answer questions such as:

* Why was an event considered anomalous?
* Which features contributed most to the anomaly score?
* Which behavioral characteristics distinguish the event from other observations?

The objective is to make machine-learning outputs more interpretable to forensic investigators.

---

## 8. Evidence Attribution

Detected anomalies are connected back to the forensic evidence that supports the corresponding events.

This creates an evidence chain:

```text
Anomaly
   ↓
Anomalous Event
   ↓
Process / Artifact Relationship
   ↓
Source Evidence
   ↓
Forensic Artifact
```

This evidence-grounded design helps investigators inspect the underlying evidence rather than relying solely on a machine-learning score.

---

## 9. Investigator-Readable Explanation

The final stage converts the analytical results into explanations that can be interpreted by investigators.

An explanation may contain:

```text
Observed Event
        ↓
Why It Is Unusual
        ↓
Important Contributing Features
        ↓
Supporting Evidence
        ↓
Temporal Context
```

The framework is designed to support investigators rather than replace human forensic judgment.

---

# Research Principles

ForensiXplain follows several important principles:

### Evidence Grounding

Explanations should be connected to the underlying forensic evidence.

### Temporal Awareness

Events should be interpreted in their chronological and contextual sequence.

### Explainability

Anomaly detection results should provide interpretable reasons rather than unexplained scores.

### Human-in-the-Loop Analysis

The framework supports investigators in evaluating evidence and analytical results.

### Reproducibility

The research pipeline is intended to be reproducible for academic experimentation and evaluation.

### Conservative Interpretation

An anomalous event should not automatically be interpreted as malicious activity. Additional forensic evidence and investigator assessment are required.

---

# Project Structure

The repository is organized around the research pipeline and supporting resources.

A typical structure is:

```text
ForensiXplain-Project/
│
├── configs/
│   └── Configuration files
│
├── data/
│   └── Datasets and forensic artifacts
│
├── results/
│   └── Experimental results
│
├── src/
│   └── Source code
│
├── .gitignore
├── README.md
└── requirements.txt
```

The exact structure may evolve as the research implementation develops.

---

# Environment Setup

## 1. Clone the Repository

```bash
git clone https://github.com/iamanshulhere/ForensiXplain-Project.git
cd ForensiXplain-Project
```

## 2. Create a Virtual Environment

### Windows

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Git Contribution Workflow

For contributors working on the project:

```bash
git checkout -b feature/your-feature-name
```

Make the required changes, then:

```bash
git status
git add .
git commit -m "Describe your change"
git push -u origin feature/your-feature-name
```

After pushing the branch, create a Pull Request for review.

---

# Current Research Direction

The project focuses on developing an integrated pipeline that connects:

```text
Digital Forensics
       +
Temporal Reconstruction
       +
Knowledge Graphs
       +
Anomaly Detection
       +
Explainable AI
       +
Evidence Attribution
       =
ForensiXplain
```

The intended outcome is a research framework capable of connecting **raw forensic observations → temporal events → graph relationships → anomaly detection → explanations → supporting evidence**.

---

# Disclaimer

ForensiXplain is a research-oriented framework.

Anomaly detection results should be treated as analytical indicators rather than definitive conclusions about malicious activity. Final interpretation should consider the underlying forensic evidence, temporal context, model limitations, and investigator judgment.

---

# Project Status

🚧 **Research and Development**

The framework is under active development. Components, datasets, experiments, and implementation details may change as the research progresses.

---

# Contributors

Contributions to the project are tracked through Git history and GitHub.

To contribute:

1. Clone the repository.
2. Create a feature branch.
3. Implement your changes.
4. Commit the changes using your GitHub-associated identity.
5. Push the branch.
6. Submit a Pull Request.

---

# License

License information will be added as the project is finalized.
