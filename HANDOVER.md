# ForensiXplain — Project Handover Documentation

**Project Title:** ForensiXplain: Explainable Timeline Reconstruction & Anomaly Detection Framework for Digital Forensic Investigations

**Repository:** `D:\ForensiXplain-Project-new`

**Current Git Branch:** `feature/graph-validation`

---

## 1. Project Overview

ForensiXplain is a digital forensic framework designed to reconstruct forensic timelines and identify potentially anomalous activities using temporal and graph-based analysis.

The framework combines event normalization, timeline reconstruction, temporal analysis, graph construction, anomaly detection, score fusion, and explainability.

### Main Objectives

- Normalize forensic events into a consistent structure.
- Reconstruct chronological event timelines.
- Generate logical timelines.
- Construct temporal and process-related graphs.
- Extract temporal and graph-based features.
- Detect anomalous events using machine learning.
- Combine temporal and graph anomaly scores.
- Explain model predictions using SHAP.
- Evaluate the framework using experimental analysis.

**Important research limitation:** M57-Jean anomaly rankings and ranking-overlap results are unsupervised proxy evaluations. They must not be presented as ground-truth precision, recall, F1-score, or proof of malicious activity.

## 2. Technology Stack and Environment

### Technologies

| Component | Technology |
|---|---|
| Programming language | Python |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn |
| Explainability | SHAP |
| Graph processing | NetworkX |
| Visualization | Matplotlib |
| Testing | Pytest |
| IDE | Visual Studio Code |
| Environment | Python virtual environment |

### Development Environment

- Operating System: Windows
- Python version: 3.11.9
- Virtual environment: `.venv`
- Project directory: `D:\ForensiXplain-Project-new`

### Environment Activation

Run in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Check Python:

```powershell
python --version
```

Install dependencies, provided `requirements.txt` is present and complete:

```powershell
python -m pip install -r requirements.txt
```

If creating a fresh environment:

```powershell
py -3.11 -m venv .venv
```

## 3. Repository Structure

The main project directories include:

```text
ForensiXplain-Project-new/
│
├── data/
│   ├── normalized/
│   │   └── M57-Jean/
│   └── features/
│       └── M57-Jean/
│
├── results/
│   ├── M57-Jean/
│   └── MalMem2022/
│
├── scripts/
│   └── generate_figure8.py
│
├── src/
│   ├── anomaly/
│   ├── data/
│   ├── evaluation/
│   ├── explainability/
│   ├── models/
│   ├── normalization/
│   └── timeline/
│
├── tests/
│
├── .venv/
├── requirements.txt
└── HANDOVER.md
```

This is a logical overview. Verify individual filenames and directories against the current repository before running scripts.

## 4. Main Execution Workflow

The main ForensiXplain workflow consists of:

1. Load forensic event data.
2. Normalize forensic events.
3. Reconstruct the chronological timeline.
4. Generate the logical timeline.
5. Perform temporal feature engineering.
6. Construct the temporal graph.
7. Perform graph feature engineering.
8. Run temporal anomaly detection.
9. Run graph anomaly detection.
10. Fuse temporal and graph anomaly scores.
11. Generate explainability outputs.
12. Perform evaluation and robustness analysis.

MalMem2022 is a separate supervised classification and evaluation workflow.

## 5. Important Source Files

The following files are important starting points for understanding the implementation.

| File | Responsibility |
|---|---|
| `src/normalization/event_schema.py` | Forensic event schema |
| `src/normalization/memory_normalizer.py` | Memory event normalization |
| `src/timeline/timeline_builder.py` | Timeline reconstruction |
| `src/temporal/logical_timeline.py` | Logical timeline generation |
| `src/anomaly/temporal_isolation_forest.py` | Temporal anomaly detection |
| `src/anomaly/graph_isolation_forest.py` | Graph anomaly detection |
| `src/explainability/temporal_shap.py` | Temporal explainability |
| `src/explainability/graph_shap.py` | Graph explainability |
| `src/evaluation/temporal_graph_fusion.py` | Temporal-graph score fusion |
| `src/evaluation/generate_table6.py` | Table 6 generation |
| `src/evaluation/generate_table6_normalized.py` | Normalized fusion experiment |
| `src/evaluation/phase10_robustness.py` | Phase 10 robustness analysis |
| `src/evaluation/ablation_study.py` | Ablation study |

## 6. Running Tests

Activate the virtual environment first:

```powershell
.\.venv\Scripts\Activate.ps1
```

Run the complete test suite:

```powershell
python -m pytest -q
```

Run temporal-graph fusion tests:

```powershell
python -m pytest tests/test_temporal_graph_fusion.py -q
```

Run Phase 10 robustness tests:

```powershell
python -m pytest tests/test_phase10_robustness.py -q
```

### Last Recorded Test Result

The complete test suite previously completed with:

- **123 passed**
- **3 warnings**
- Execution time: approximately 32.98 seconds

The warnings were SHAP/Matplotlib colormap deprecation warnings.

These are previously recorded results; rerun the suite on the final handover commit to confirm the current state.

## 7. Temporal-Graph Fusion

ForensiXplain combines temporal and graph-based anomaly scores.

The existing fusion implementation uses:

- Temporal score weight: 0.5
- Graph score weight: 0.5

### Normalized Fusion Experiment

Script:

`src/evaluation/generate_table6_normalized.py`

The experiment:

1. Reads temporal anomaly results.
2. Reads graph anomaly results.
3. Normalizes anomaly scores.
4. Combines scores using equal weights.
5. Produces fused rankings.
6. Calculates ranking-overlap statistics.
7. Compares results against a random baseline.

### Main Outputs

- `results/M57-Jean/table6_normalized_temporal_graph_fusion.csv`
- `results/M57-Jean/normalized_fusion_rankings.csv`

**Important:** This is an unsupervised proxy evaluation. Validate reproducibility and the random-baseline configuration before treating the outputs as final research evidence.

## 8. Explainability

ForensiXplain uses SHAP-based explainability to understand the contribution of features to model predictions.

### Main Source Files

- `src/explainability/temporal_shap.py`
- `src/explainability/graph_shap.py`

### Main Outputs

The results directory contains temporal and graph SHAP explanation CSVs and visualization files, including:

- `temporal_shap_explanations.csv`
- `graph_shap_explanations.csv`
- `figure8_graph_shap_summary.png`

SHAP explains how features influence model behavior. It does not independently establish that an event is malicious.

## 9. Evaluation and Robustness

### Table 7 — Baseline Comparison

The comparison includes:

1. Isolation Forest baseline.
2. Temporal Isolation Forest.
3. Graph-aware Isolation Forest.
4. Temporal + Graph fusion.

The recorded Table 7 comparison contains 46 evaluated events and reports top-five ranking overlap.

The evaluation type is:

**Unsupervised proxy (rank overlap; no ground truth).**

Therefore, the comparison must not be described as ground-truth detection accuracy.

### Phase 10 — Robustness Analysis

Main source:

`src/evaluation/phase10_robustness.py`

The Phase 10 work includes robustness and sensitivity-related checks, artifact validation, configuration checks, and hash-manifest validation.

### Known Manifest Inconsistency

The historical manifest contains a SHA-256 value for `fusion_report.txt` that does not match the current report.

The current report hash matches the version in Git HEAD, while the manifest contains a different historical hash.

A test explicitly handles this known inconsistency. Do not regenerate the report or manifest simply to make the hashes match without investigating the original provenance.

## 10. MalMem2022 Evaluation

MalMem2022 is handled as a separate supervised memory-forensics classification track.

The repository contains experimental grouped evaluation scripts:

- `src/evaluation/generate_table8_source_grouped.py`
- `src/evaluation/generate_table8_strict_grouped.py`

### Source-Grouped Evaluation

The source-grouped experiment groups malware samples by source filename ID and groups benign samples by identical feature vectors.

### Strict-Grouped Evaluation

The strict-grouped experiment uses a Union-Find approach to combine samples sharing malware source IDs or identical feature vectors.

This is intended to reduce potential train/test leakage.

**Validation required:** Verify that the final split has zero shared malware source IDs and zero identical feature vectors. These scripts are experimental; their results should not be treated as validated generalization performance without further checks.

## 11. Runtime and Scalability

The repository contains several runtime-related outputs.

### Table 9 — Runtime and Scalability

The current Table 9 documentation records several pipeline stages as not measured in the reproducible branch.

These include:

- Data normalization
- Temporal feature engineering
- Temporal anomaly detection
- Knowledge graph construction
- Graph anomaly detection
- SHAP explanation generation

Do not replace missing measurements with estimates.

### Table 11 — Runtime Experiments

The repository contains:

- `table11_detection_runtime.csv`
- `table11_final_runtime.csv`
- `table11_runtime_benchmark.csv`

The detection benchmark script is:

`src/evaluation/benchmark_runtime.py`

The final-runtime and runtime-benchmark CSV files are historical measurements. Their generation scripts were not located during handover preparation.

Their reported timings use different stage definitions and should not be combined or directly compared as equivalent end-to-end measurements.

No peak-memory measurement has been established by these reports.

## 12. Known Limitations

1. Independent event-level ground-truth anomaly labels are not available for the evaluated M57-Jean cohort.
2. Ranking overlap is an agreement proxy, not detection accuracy.
3. SHAP explanations describe model behavior and do not establish malicious intent.
4. Runtime measurements do not yet cover a consistently measured, complete end-to-end pipeline.
5. Historical runtime reports lack identifiable generation scripts in the current repository.
6. MalMem2022 grouped evaluation scripts require independent validation of their split methodology and results.
7. The normalized fusion experiment requires reproducibility checks before its outputs are treated as final research evidence.

## 13. Pending Research and Development Tasks

### A. Evaluation Improvements

- Obtain independently reviewed ground-truth labels for the evaluated forensic events.
- Calculate Precision@K, Recall@K, Average Precision, and other suitable metrics once reliable labels are available.
-

