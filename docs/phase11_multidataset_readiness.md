# Phase 11 — Multi-Dataset Readiness and Experimental Design

## 1. Objective

Phase 11 evaluates whether the available datasets can support the next stage of ForensiXplain: multi-dataset validation of temporal, graph, fusion, anomaly-detection, and explainability capabilities.

This phase is a readiness/design audit. It does not introduce new implementation changes.

## 2. Dataset Readiness Matrix

| Capability | M57-Jean | MalMem2022 | OpTC |
|---|---|---|---|
| Timeline reconstruction | Supported | Not supported from current CSV | Not currently supported |
| Event-level timestamps | Available | Not available | Raw telemetry unavailable |
| Process relationships | PID/parent-PID available | Aggregate process statistics only | Ground-truth PDF only |
| Temporal anomaly detection | Supported | Sample-level only | Cannot currently run |
| Graph construction | Supported | Not equivalent to event graph | Raw telemetry unavailable |
| Graph anomaly detection | Supported | Not directly comparable | Cannot currently run |
| Temporal + graph fusion | Supported | Not directly supported | Cannot currently run |
| Independent ground truth | Not found in available structured files | Class/Category available | Ground-truth PDF available |
| Event-level labels | Not found | Not available | Not currently available |
| Supervised evaluation | Not defensible currently | Sample-level supervised evaluation | Not currently possible |
| Explainability/evidence attribution | Supported | Feature-level interpretation possible | Cannot reproduce without telemetry |
| Quantitative evaluation | Unsupervised/ranking analysis | Sample-level classification | Not currently possible |
| Primary limitation | No independent ground truth | No event/timeline structure | Raw telemetry unavailable |

## 3. Dataset Roles

### M57-Jean

M57-Jean is the primary end-to-end forensic dataset.

It supports:

- Event extraction
- Event normalization
- Timeline reconstruction
- Temporal feature generation
- Graph construction
- Graph feature generation
- Temporal anomaly detection
- Graph anomaly detection
- Temporal/graph score fusion
- Evidence attribution and explainability

Current M57 experiments should be interpreted as retrospective, unsupervised anomaly/ranking and evidence-attribution analysis because independent ground-truth labels were not identified in the available structured data.

### MalMem2022

MalMem2022 provides an independent supervised validation setting.

The current dataset contains forensic memory-feature aggregates together with:

- Class
- Category
- Filename

It supports sample-level Benign/Malware classification evaluation.

It should not be presented as validation of event-level timeline reconstruction or temporal-graph fusion because the current CSV does not contain the required event-level temporal and relationship structure.

### OpTC

The currently available OpTC resource is the Red Team Ground Truth PDF.

It can provide reference information about ground-truth activity, but raw telemetry required to reproduce event-level quantitative evaluation is not currently available.

Therefore, precision, recall, F1, or accuracy should not be reported for OpTC using the PDF alone.

## 4. Experimental Scope

The next experiments should maintain separate evaluation scopes:

1. M57-Jean:
   Event-level temporal and graph forensic pipeline evaluation.

2. MalMem2022:
   Independent sample-level supervised feature validation.

3. OpTC:
   Reference/ground-truth resource pending availability of raw telemetry.

Results from these datasets must not be combined as though they represent the same prediction task.

## 5. Evaluation Constraints

The following limitations must remain explicit in the research paper:

- M57 does not currently have an identified independent event-level ground-truth label source.
- MalMem2022 is sample-level and does not provide the same event/timeline structure as M57.
- OpTC raw telemetry is not currently available in the repository.
- MalMem2022 supervised results must not be interpreted as temporal or graph detection performance.
- OpTC ground-truth documentation alone is insufficient for reproducible quantitative detection metrics.

## 6. Research Interpretation

The multi-dataset stage should therefore be framed as complementary validation rather than identical cross-dataset benchmarking.

M57-Jean validates the complete temporal-graph forensic workflow.

MalMem2022 provides independent feature-level supervised validation.

OpTC provides a potential future validation resource if the corresponding raw telemetry becomes available.

## 7. Phase 11 Decision

Phase 11 readiness audit is complete.

The project is ready to proceed to multi-dataset experiments with clearly separated evaluation scopes and explicit dataset limitations.

No production pipeline changes are required as part of the readiness audit.

## 8. Next Phase

The next implementation stage is multi-dataset experimentation.

Before reporting final results, the project should preserve the distinction between:

- event-level forensic reconstruction,
- unsupervised anomaly/ranking analysis,
- sample-level supervised classification,
- evidence attribution,
- and ground-truth-based detection evaluation.
