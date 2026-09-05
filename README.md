# ForensiXplain

## An Evidence-Grounded Explainable Framework for Temporal Reconstruction and Anomaly Detection in Digital Forensics

ForensiXplain is a research-oriented digital forensics framework designed to transform heterogeneous forensic artifacts into a structured temporal representation, detect anomalous event patterns, connect anomalies back to forensic evidence, and generate investigator-readable explanations.

The project combines digital forensics, temporal event reconstruction, graph-based representation, unsupervised anomaly detection, and explainable AI.

---

## Research Question

> Can heterogeneous digital forensic artifacts be transformed into an explainable temporal knowledge graph that automatically detects anomalous event sequences and provides evidence-grounded explanations for investigators?

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