# ForensiXplain

## An Evidence-Grounded Explainable Framework for Temporal Reconstruction and Anomaly Detection in Digital Forensics

ForensiXplain is a research-oriented digital forensics framework designed to transform heterogeneous forensic artifacts into a structured temporal representation, detect anomalous event and process patterns, connect anomalies back to forensic evidence, and generate investigator-readable explanations.

<<<<<<< HEAD
The project combines:

* Digital forensics
* Temporal event reconstruction
* Graph-based representation
* Unsupervised anomaly detection
* Explainable AI
* Evidence attribution
* Investigator-readable reporting
=======
The project combines **digital forensics, temporal event reconstruction, graph-based representation, unsupervised anomaly detection, and explainable AI**.
>>>>>>> origin/utkarsh-readme-update

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
9. Compare temporal and graph-based anomaly detection perspectives.
10. Provide a reproducible research pipeline suitable for academic evaluation.

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
             |            |             |
        Temporal      Graph-aware   Graph-only
        Features       Features      Features
             |            |             |
             +------------+-------------+
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
<<<<<<< HEAD
                   Model Comparison
                          |
=======
>>>>>>> origin/utkarsh-readme-update
                  Human-in-the-Loop
```

---

<<<<<<< HEAD
# Implemented Analytical Pipeline

The current implementation contains three complementary anomaly-detection perspectives.

```text
                         M57-Jean
                            |
              +-------------+-------------+
              |             |             |
              v             v             v
          Temporal      Graph-aware   Graph-only
          Features       Features      Features
              |             |             |
              v             v             v
        Isolation       Isolation     Isolation
         Forest          Forest        Forest
              |             |             |
              v             v             v
          Temporal      Graph SHAP   Graph-only
            SHAP                       SHAP
              |             |             |
              v             v             v
          Evidence      Evidence      Evidence
         Attribution   Attribution   Attribution
              |             |             |
              v             v             v
        Investigator   Investigator  Investigator
         Explanation    Explanation   Explanation
              \             |             /
               \            |            /
                +-----------+-----------+
                            |
                     Model Comparison
```

The three paths are intended to provide different analytical views of unusual process/event behavior rather than to establish malicious activity.

---

# Core Pipeline

ForensiXplain follows the following research workflow:

```text
Forensic Artifacts
        |
Artifact Extraction
        |
Event Normalization
        |
Temporal Reconstruction
        |
Temporal Knowledge Graph
        |
Feature Engineering
        |
+-------------------------------+
|                               |
Temporal Features          Graph Features
|                               |
v                               v
Temporal Model             Graph-aware Model
|                               |
v                               v
Temporal SHAP               Graph SHAP
|                               |
v                               v
Temporal Evidence           Graph Evidence
Attribution                 Attribution
|                               |
+---------------+---------------+
                |
       Investigator Explanation
                |
         Model Comparison
                |
        Human-in-the-Loop
=======
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
>>>>>>> origin/utkarsh-readme-update
```

---

<<<<<<< HEAD
# Dataset / Experimental Case

The current implemented experiment uses the **M57-Jean** forensic case.

The case is represented through normalized forensic observations and a reconstructed logical timeline.

Key data products include:

```text
data/normalized/M57-Jean/events.csv
data/normalized/M57-Jean/logical_timeline.csv
data/features/M57-Jean/graph_features.csv
```

The forensic evidence represented in the current experiment includes Windows memory-forensics observations such as:

* `pslist`
* `pstree`
* `cmdline`
* `dlllist`
* `malfind`

The current M57-Jean experiment contains:

**47 logical timeline events.**

These events are used as the common observation set for the temporal, graph-aware, and graph-only anomaly-detection experiments.

---

# 1. Artifact Extraction

The framework processes heterogeneous digital forensic artifacts and converts available forensic observations into structured records.

For the current M57-Jean experiment, the evidence includes Windows memory-forensics observations produced through Volatility-based analysis.

Examples include:

```text
Process listing
Process tree
Command-line observations
Loaded-module observations
Memory-region observations
Malfind observations
```

---

# 2. Event Normalization

Forensic observations from different sources are converted into a common event representation.

A normalized event can contain:
=======
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
>>>>>>> origin/utkarsh-readme-update

```text
Event ID
Timestamp
Timestamp Confidence
Event Type
<<<<<<< HEAD
Process ID
=======
>>>>>>> origin/utkarsh-readme-update
Process
Artifact
Source
Action
Evidence Reference
```

<<<<<<< HEAD
This allows observations from different forensic artifacts to be analyzed within a unified representation.

---

# 3. Temporal Reconstruction

The framework reconstructs the sequence of forensic events chronologically.

Temporal processing considers:
=======
This allows evidence from different forensic sources to be analyzed within a unified representation.

---

## 3. Temporal Reconstruction

The framework reconstructs the sequence of forensic events chronologically.

The reconstruction process considers:
>>>>>>> origin/utkarsh-readme-update

* Event timestamps
* Timestamp confidence
* Event ordering
<<<<<<< HEAD
* Previous and next events
* Process transitions
* Time gaps
* Local event density
* Temporal relationships

The resulting logical timeline provides the temporal foundation for subsequent analysis.

---

# 4. Temporal Features

Temporal features capture characteristics of event timing and process transitions.

Implemented examples include:

* `gap_log_seconds`
* `local_density_10s`
* `local_density_30s`
* `local_density_60s`
* `process_changed`
* `rapid_event`
* `short_event_gap`
* `medium_event_gap`
* `long_event_gap`

These features are used by the temporal anomaly-detection path.

---

# 5. Temporal Anomaly Detection

The temporal anomaly-detection path uses an unsupervised Isolation Forest model to identify unusual temporal/event profiles.

The current M57-Jean experiment produced:

```text
Logical events:       47
Temporal anomalies:    5
```

The temporal model identified the following process IDs as anomalous:

```text
1204
2004
2840
3992
4028
```

An anomaly represents an unusual feature profile under the corresponding model. It does not independently establish malicious activity.

---

# 6. Temporal SHAP Explainability

SHAP is used to examine the contribution of temporal features to anomaly-model outputs.

This provides feature-level explanations for why an event received its model output.

The temporal explanation pipeline connects:

```text
Temporal anomaly
       |
Temporal SHAP
       |
Important temporal features
       |
Temporal evidence
       |
Investigator explanation
```

---

# 7. Graph Representation

The framework represents process and forensic relationships using graph-derived features.

Graph characteristics include relationships such as:
=======
* Process relationships
* Artifact relationships
* Cross-source temporal relationships

The goal is to produce a timeline that preserves the uncertainty inherent in forensic timestamps.

---

## 4. Temporal Knowledge Graph

Normalized events are represented as a temporal knowledge graph.

The graph captures relationships between entities such as:
>>>>>>> origin/utkarsh-readme-update

```text
Process
   |
<<<<<<< HEAD
   +---- Parent Process
   |
   +---- Child Process
   |
   +---- Command-line Evidence
   |
   +---- Loaded Modules
   |
   +---- Memory Regions
   |
   +---- Other Relationships
```

The graph representation allows process behavior to be examined from a structural perspective in addition to temporal behavior.

---

# 8. Graph-aware Anomaly Detection

The graph-aware anomaly-detection path uses graph-derived features to identify unusual process profiles.

The current M57-Jean experiment produced:

```text
Logical events:       47
Graph-aware anomalies: 5
```

The graph-aware model identified:

```text
1372
1944
2892
3560
3992
```

as anomalous process IDs.

---

# 9. Graph SHAP Explainability

Graph SHAP explanations identify graph-derived features contributing to graph anomaly-model outputs.

Examples of graph-derived features include:

* `parent_count`
* `child_count`
* `graph_degree`
* `in_degree`
* `out_degree`
* `command_line_count`
* `module_count`
* `memory_region_count`
* `relationship_type_count`

The graph explanation pipeline connects:

```text
Graph anomaly
       |
Graph SHAP
       |
Important graph features
       |
Graph relationships
       |
Forensic evidence
       |
Investigator explanation
```

---

# 10. Graph-only Anomaly Detection

A separate graph-only analytical path was implemented to evaluate graph-derived behavior independently.

The graph-only model uses graph-derived features without the temporal feature set.

The current M57-Jean experiment produced:

```text
Logical events:       47
Graph-only anomalies: 5
```

The graph-only model identified:

```text
1372
1836
1944
2892
3560
```

as anomalous process IDs.

---

# 11. Graph-only SHAP

Graph-only SHAP explanations provide feature-level attribution for the graph-only anomaly model.

The current implementation generates:

```text
results/M57-Jean/graph_only_shap_explanations.csv
```

The explanation records contain SHAP values for graph-derived features including:

```text
parent_count
child_count
graph_degree
in_degree
out_degree
command_line_count
module_count
memory_region_count
relationship_type_count
```

For example, the graph-only analysis can identify a feature such as `module_count`, `child_count`, or `memory_region_count` as an important contributor to an individual model output.

---

# 12. Evidence Attribution

Detected anomalies are connected back to the underlying forensic evidence.

The evidence-attribution stage creates an evidence chain:

```text
Anomaly
   |
Anomalous Event
   |
Process
   |
Graph Relationship
   |
Forensic Evidence
   |
Artifact Observation
```

For the graph-only pipeline, the generated output is:

```text
results/M57-Jean/graph_only_evidence_attribution.csv
```

The attribution records connect graph-only anomaly candidates with information such as:

* Parent processes
* Child processes
* Graph features
* Raw forensic event counts
* Artifact types
* Event types
* Evidence IDs
* Command-line observations
* Timeline evidence
* Provenance

---

# 13. Investigator-Readable Explanations

The framework converts analytical results into investigator-readable explanations.

Current outputs include:

```text
Temporal:
results/M57-Jean/temporal_investigator_explanations.csv
results/M57-Jean/temporal_investigator_report.txt

Graph-aware:
results/M57-Jean/graph_investigator_explanations.csv
results/M57-Jean/graph_investigator_report.txt

Graph-only:
results/M57-Jean/graph_only_evidence_attribution.csv
results/M57-Jean/graph_only_shap_explanations.csv
```

The explanations provide context such as:

```text
Process
Anomaly score
Important contributing features
Process relationships
Temporal context
Raw evidence
Artifact information
Evidence references
Model limitations
```

The purpose is to help investigators inspect the evidence supporting an analytical observation.

---

# 14. Model Comparison

The project includes a model-comparison stage that compares anomaly flags from:

1. Temporal model
2. Graph-aware model
3. Graph-only model

Output files:

```text
results/M57-Jean/model_comparison.csv
results/M57-Jean/model_comparison_report.txt
```

The comparison records whether a process was identified by each model.

---

## Current M57-Jean Comparison

The current experiment contains:

| Model       | Logical Events | Anomalies |
| ----------- | -------------: | --------: |
| Temporal    |             47 |         5 |
| Graph-aware |             47 |         5 |
| Graph-only  |             47 |         5 |

Observed overlap:

```text
Temporal ∩ Graph-aware:
[3992]

Temporal ∩ Graph-only:
[]

Graph-aware ∩ Graph-only:
[1372, 1944, 2892, 3560]

All three:
[]
```

The corresponding comparison categories include:

```text
Temporal-only:
1204
2004
2840
4028

Graph-aware and graph-only:
1372
1944
2892
3560

Graph-only model only:
1836

Temporal and graph-aware:
3992
```

These results describe agreement and disagreement between analytical models. They should not be interpreted as independent proof that any listed process is malicious.

---

# Results Directory

The current M57-Jean experiment produces the following major analytical outputs:

```text
results/M57-Jean/
|
+-- isolation_forest_results.csv
|
+-- temporal_anomalies.csv
+-- temporal_shap_explanations.csv
+-- temporal_evidence_attribution.csv
+-- temporal_investigator_explanations.csv
+-- temporal_investigator_report.txt
|
+-- graph_anomalies.csv
+-- graph_shap_explanations.csv
+-- graph_evidence_attribution.csv
+-- graph_investigator_explanations.csv
+-- graph_investigator_report.txt
|
+-- graph_only_anomalies.csv
+-- graph_only_shap_explanations.csv
+-- graph_only_evidence_attribution.csv
|
+-- model_comparison.csv
+-- model_comparison_report.txt
```
=======
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
>>>>>>> origin/utkarsh-readme-update

---

# Project Structure

<<<<<<< HEAD
```text
ForensiXplain-Project/
|
+-- data/
|   +-- normalized/
|   |   +-- M57-Jean/
|   |
|   +-- features/
|       +-- M57-Jean/
|
+-- results/
|   +-- M57-Jean/
|
+-- src/
|   +-- anomaly/
|   |   +-- evidence_analysis.py
|   |   +-- features.py
|   |   +-- graph_features.py
|   |   +-- graph_isolation_forest.py
|   |   +-- graph_only_isolation_forest.py
|   |   +-- isolation_forest.py
|   |   +-- temporal_features.py
|   |   +-- temporal_isolation_forest.py
|   |
|   +-- explainability/
|   |   +-- evidence_attribution.py
|   |   +-- explanation_generator.py
|   |   +-- graph_evidence_attribution.py
|   |   +-- graph_explanation_generator.py
|   |   +-- graph_only_evidence_attribution.py
|   |   +-- graph_only_shap.py
|   |   +-- graph_shap.py
|   |   +-- isolation_forest_shap.py
|   |   +-- temporal_evidence_attribution.py
|   |   +-- temporal_explanation_generator.py
|   |   +-- temporal_shap.py
|   |
|   +-- reporting/
|   |   +-- model_comparison.py
|   |
|   +-- graph/
|   +-- ingestion/
|   +-- normalization/
|   +-- temporal/
|   +-- timeline/
|
+-- README.md
+-- requirements.txt
+-- .gitignore
```

=======
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

>>>>>>> origin/utkarsh-readme-update
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

<<<<<<< HEAD
## Run the Active M57-Jean Pipeline

From the repository root, after `data/normalized/M57-Jean/events.csv` is available:

```bash
python run_pipeline.py
```

The runner validates each active stage's required inputs and expected outputs,
and stops immediately if a stage fails.

=======
>>>>>>> origin/utkarsh-readme-update
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

<<<<<<< HEAD
# Research Principles

## Evidence Grounding

Explanations should be connected to the underlying forensic evidence.

## Temporal Awareness

Events should be interpreted in their chronological and contextual sequence.

## Graph Awareness

Process and artifact relationships provide structural context for anomaly analysis.

## Explainability

Anomaly detection results should provide interpretable feature-level information rather than unexplained scores.

## Model Comparison

Different analytical perspectives should be compared to identify agreement and disagreement between temporal and graph-derived anomaly signals.

# 15. Final Evaluation

The final evaluation stage consolidates the outputs of the temporal, graph-aware, and graph-only anomaly-detection paths.

It compares model agreement and disagreement and correlates graph-involved anomaly candidates with `malfind` forensic evidence.

The evaluation produces:

```text
results/M57-Jean/evaluation_summary.csv
results/M57-Jean/evaluation_candidates.csv
results/M57-Jean/evaluation_report.txt

---

# 16. Experimental Results

## M57-Jean

Three complementary anomaly-detection models were evaluated:

| Model | Rows | Anomalies |
|---|---:|---:|
| Temporal | 47 | 5 |
| Graph-aware | 47 | 5 |
| Graph-only | 47 | 5 |

### Detection overlap

- Temporal-only: 4 candidates
- Temporal + Graph-aware: 1 candidate
- Graph-aware + Graph-only: 4 candidates
- Graph-only model only: 1 candidate
- All three models: 0 candidates

### Evidence correlation

- Graph-involved candidates with malfind: 5/6
- Temporal-only candidates with malfind: 0/4

The results indicate that graph-involved anomaly detections showed greater overlap
with the available `malfind` observations in this experiment. This should not be
interpreted as proof of malicious activity or as a definitive comparison of model
accuracy. Further validation across additional forensic cases and evidence types
is required.

## Human-in-the-Loop Analysis

The framework supports investigators in evaluating evidence and analytical results rather than replacing forensic judgment.

## Reproducibility

The research pipeline is intended to be reproducible for academic experimentation and evaluation.

## Conservative Interpretation

An anomalous event should not automatically be interpreted as malicious activity. Additional forensic evidence, contextual analysis, model limitations, and investigator assessment are required.

---

# Current Research Direction

The implemented research pipeline connects:
=======
# Current Research Direction

The project focuses on developing an integrated pipeline that connects:
>>>>>>> origin/utkarsh-readme-update

```text
Digital Forensics
       +
<<<<<<< HEAD
Event Normalization
       +
=======
>>>>>>> origin/utkarsh-readme-update
Temporal Reconstruction
       +
Knowledge Graphs
       +
<<<<<<< HEAD
Temporal Anomaly Detection
       +
Graph Anomaly Detection
       +
Graph-only Anomaly Detection
       +
SHAP Explainability
       +
Evidence Attribution
       +
Model Comparison
       +
Investigator-readable Reporting
=======
Anomaly Detection
       +
Explainable AI
       +
Evidence Attribution
>>>>>>> origin/utkarsh-readme-update
       =
ForensiXplain
```

<<<<<<< HEAD
The current experimental focus is on understanding how temporal and graph-derived representations identify and explain unusual process/event behavior within forensic data.
=======
The intended outcome is a research framework capable of connecting **raw forensic observations → temporal events → graph relationships → anomaly detection → explanations → supporting evidence**.
>>>>>>> origin/utkarsh-readme-update

---

# Disclaimer

ForensiXplain is a research-oriented framework.

Anomaly detection results should be treated as analytical indicators rather than definitive conclusions about malicious activity. Final interpretation should consider the underlying forensic evidence, temporal context, model limitations, and investigator judgment.

---

# Project Status

<<<<<<< HEAD
**Research and Development — Core Analytical Pipeline Implemented**

The current implementation includes:

* Event normalization
* Temporal reconstruction
* Temporal feature engineering
* Graph feature engineering
* Temporal anomaly detection
* Graph-aware anomaly detection
* Graph-only anomaly detection
* SHAP-based explanations
* Evidence attribution
* Investigator-readable explanations
* Model comparison reporting

The current experimental case is **M57-Jean**, containing 47 logical timeline events used across the implemented anomaly-detection comparisons.

Additional datasets, experiments, evaluation methodology, reporting, and research validation may be added as development continues.
=======
🚧 **Research and Development**

The framework is under active development. Components, datasets, experiments, and implementation details may change as the research progresses.
>>>>>>> origin/utkarsh-readme-update

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
<<<<<<< HEAD


=======
>>>>>>> origin/utkarsh-readme-update
