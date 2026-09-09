"""Configuration for the active ForensiXplain M57-Jean experiment."""

from pathlib import Path


CASE_ID = "M57-Jean"

RANDOM_STATE = 42
N_ESTIMATORS = 500
CONTAMINATION = 0.10

TEMPORAL_WINDOWS = [10, 30, 60]

TEMPORAL_MODEL_FEATURES = [
    "gap_log_seconds",
    "local_density_10s",
    "local_density_30s",
    "local_density_60s",
    "process_changed",
]

GRAPH_FEATURES = [
    "parent_count",
    "child_count",
    "graph_degree",
    "in_degree",
    "out_degree",
    "command_line_count",
    "module_count",
    "memory_region_count",
    "relationship_type_count",
]

GRAPH_AWARE_MODEL_FEATURES = [
    *TEMPORAL_MODEL_FEATURES,
    *GRAPH_FEATURES,
]

GRAPH_ONLY_MODEL_FEATURES = [
    *GRAPH_FEATURES,
]

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
NORMALIZED_DIR = DATA_DIR / "normalized" / CASE_ID
FEATURES_DIR = DATA_DIR / "features" / CASE_ID
RESULTS_DIR = PROJECT_ROOT / "results" / CASE_ID

EVENTS_FILE = NORMALIZED_DIR / "events.csv"
TIMELINE_FILE = NORMALIZED_DIR / "timeline.csv"
UNTIMESTAMPED_EVENTS_FILE = NORMALIZED_DIR / "untimestamped_events.csv"
LOGICAL_TIMELINE_FILE = NORMALIZED_DIR / "logical_timeline.csv"
TEMPORAL_GRAPH_FILE = NORMALIZED_DIR / "temporal_graph.graphml"

TEMPORAL_FEATURES_FILE = FEATURES_DIR / "temporal_features.csv"
GRAPH_FEATURES_FILE = FEATURES_DIR / "graph_features.csv"

TEMPORAL_ANOMALIES_FILE = RESULTS_DIR / "temporal_anomalies.csv"
TEMPORAL_SHAP_FILE = RESULTS_DIR / "temporal_shap_explanations.csv"
TEMPORAL_EVIDENCE_FILE = RESULTS_DIR / "temporal_evidence_attribution.csv"
TEMPORAL_EXPLANATIONS_FILE = RESULTS_DIR / "temporal_investigator_explanations.csv"
TEMPORAL_REPORT_FILE = RESULTS_DIR / "temporal_investigator_report.txt"

GRAPH_ANOMALIES_FILE = RESULTS_DIR / "graph_anomalies.csv"
GRAPH_SHAP_FILE = RESULTS_DIR / "graph_shap_explanations.csv"
GRAPH_EVIDENCE_FILE = RESULTS_DIR / "graph_evidence_attribution.csv"
GRAPH_EXPLANATIONS_FILE = RESULTS_DIR / "graph_investigator_explanations.csv"
GRAPH_REPORT_FILE = RESULTS_DIR / "graph_investigator_report.txt"

GRAPH_ONLY_ANOMALIES_FILE = RESULTS_DIR / "graph_only_anomalies.csv"
GRAPH_ONLY_SHAP_FILE = RESULTS_DIR / "graph_only_shap_explanations.csv"
GRAPH_ONLY_EVIDENCE_FILE = RESULTS_DIR / "graph_only_evidence_attribution.csv"

MODEL_COMPARISON_FILE = RESULTS_DIR / "model_comparison.csv"
MODEL_COMPARISON_REPORT_FILE = RESULTS_DIR / "model_comparison_report.txt"
EVALUATION_SUMMARY_FILE = RESULTS_DIR / "evaluation_summary.csv"
EVALUATION_CANDIDATES_FILE = RESULTS_DIR / "evaluation_candidates.csv"
EVALUATION_REPORT_FILE = RESULTS_DIR / "evaluation_report.txt"
