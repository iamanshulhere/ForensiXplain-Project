"""Generate a proxy, ranking-overlap comparison of existing M57-Jean models.

Table 7 compares selected event IDs from saved model outputs. It does not
estimate detection accuracy because independent anomaly labels are unavailable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.anomaly.graph_isolation_forest import (
    CONTAMINATION as GRAPH_CONTAMINATION,
    MODEL_FEATURES as GRAPH_MODEL_FEATURES,
    N_ESTIMATORS as GRAPH_N_ESTIMATORS,
    RANDOM_STATE as GRAPH_RANDOM_STATE,
)
from src.anomaly.isolation_forest import EXCLUDE_COLUMNS
from src.anomaly.temporal_isolation_forest import MODEL_FEATURES as TEMPORAL_MODEL_FEATURES
from src.evaluation.temporal_graph_fusion import (
    DEFAULT_FUSION_WEIGHT,
    build_fused_frame,
    select_top_k_events as _select_top_k_events,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CASE_ID = "M57-Jean"
RESULTS_DIR = PROJECT_ROOT / "results" / CASE_ID
BASELINE_PATH = RESULTS_DIR / "isolation_forest_results.csv"
TEMPORAL_PATH = RESULTS_DIR / "temporal_anomalies.csv"
GRAPH_PATH = RESULTS_DIR / "graph_anomalies.csv"
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / CASE_ID / "features.csv"
OUTPUT_PATH = RESULTS_DIR / "table7_baseline_comparison.csv"
REPORT_PATH = RESULTS_DIR / "table7_baseline_comparison_report.txt"

TOP_K = 5
EVALUATION_TYPE = "Unsupervised proxy (rank overlap; no ground truth)"

OVERLAP_COLUMNS = {
    "Isolation Forest baseline": "top_k_overlap_isolation_forest",
    "Temporal Isolation Forest": "top_k_overlap_temporal",
    "Graph-aware Isolation Forest": "top_k_overlap_graph",
    "Temporal + Graph fused": "top_k_overlap_fused",
}


def _read_csv(path: Path, required_columns: set[str], description: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")

    frame = pd.read_csv(path, low_memory=False)
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(
            f"{description} is missing required columns: {sorted(missing)}"
        )
    return frame


def _validate_unique_event_ids(frame: pd.DataFrame, description: str) -> None:
    if frame["logical_event_id"].isna().any():
        raise ValueError(f"{description} contains an empty logical_event_id")
    if frame["logical_event_id"].duplicated().any():
        raise ValueError(f"{description} contains duplicate logical_event_id values")


def _numeric_scores(frame: pd.DataFrame, column: str, description: str) -> pd.Series:
    scores = pd.to_numeric(frame[column], errors="coerce")
    if scores.isna().any() or not np.isfinite(scores.to_numpy(dtype=float)).all():
        raise ValueError(f"{description} contains missing or non-finite {column} values")
    return scores.astype(float)


def _boolean_flags(frame: pd.DataFrame, column: str, description: str) -> pd.Series:
    values = frame[column].astype(str).str.strip().str.lower()
    flags = values.map({"true": True, "false": False, "1": True, "0": False})
    if flags.isna().any():
        raise ValueError(f"{description} contains invalid {column} values")
    return flags.astype(bool)


def select_top_k_events(
    frame: pd.DataFrame,
    score_column: str,
    k: int = TOP_K,
) -> list[str]:
    """Select a deterministic top-k ranking, breaking ties by event ID."""
    return _select_top_k_events(frame, score_column, k)


def _build_model_inputs(
    baseline_path: Path,
    temporal_path: Path,
    graph_path: Path,
    features_path: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, int], dict[str, str], set[str]]:
    baseline = _read_csv(
        baseline_path,
        {
            "process_id",
            "process_name",
            "create_time",
            "anomaly_score",
            "predicted_anomaly",
        },
        "General Isolation Forest output",
    )
    temporal = _read_csv(
        temporal_path,
        {
            "logical_event_id",
            "process_id",
            "process",
            "timestamp",
            "is_first_event",
            "temporal_anomaly_score",
            "temporal_predicted_anomaly",
            "temporal_anomaly_rank",
        },
        "Temporal Isolation Forest output",
    )
    graph = _read_csv(
        graph_path,
        {
            "logical_event_id",
            "graph_anomaly_score",
            "graph_predicted_anomaly",
            "graph_anomaly_rank",
        },
        "Graph-aware Isolation Forest output",
    )

    _validate_unique_event_ids(temporal, "Temporal output")
    _validate_unique_event_ids(graph, "Graph output")
    if set(temporal["logical_event_id"]) != set(graph["logical_event_id"]):
        raise ValueError("Temporal and graph outputs do not cover the same logical events")

    temporal["_timestamp_utc"] = pd.to_datetime(
        temporal["timestamp"], errors="coerce", utc=True
    )
    if temporal["_timestamp_utc"].isna().any():
        raise ValueError("Temporal output contains an invalid timestamp")

    # The general baseline predates logical event IDs. Link it only when the
    # process ID and UTC timestamp uniquely identify an existing timeline event.
    baseline["_timestamp_utc"] = pd.to_datetime(
        baseline["create_time"], errors="coerce", utc=True
    )
    timestamped_baseline = baseline.loc[baseline["_timestamp_utc"].notna()].copy()
    timeline_identity = temporal[
        ["logical_event_id", "process_id", "process", "_timestamp_utc"]
    ].copy()
    if timestamped_baseline.duplicated(["process_id", "_timestamp_utc"]).any():
        raise ValueError("General baseline has duplicate process ID/timestamp keys")

    mapped_baseline = timestamped_baseline.merge(
        timeline_identity,
        on=["process_id", "_timestamp_utc"],
        how="left",
        validate="one_to_one",
    )
    if mapped_baseline["logical_event_id"].isna().any():
        raise ValueError("Some timestamped baseline rows do not map to a logical event")
    _validate_unique_event_ids(mapped_baseline, "Mapped general baseline output")
    if set(mapped_baseline["logical_event_id"]) != set(temporal["logical_event_id"]):
        raise ValueError(
            "Timestamped general baseline rows do not map one-to-one to temporal events"
        )
    if not (
        mapped_baseline["process_name"].astype(str).to_numpy()
        == mapped_baseline["process"].astype(str).to_numpy()
    ).all():
        raise ValueError("General baseline process names disagree with timeline events")

    temporal["_rank_available"] = temporal["temporal_anomaly_rank"].notna()
    common_ids = set(
        temporal.loc[temporal["_rank_available"], "logical_event_id"].astype(str)
    )
    if len(common_ids) <= TOP_K:
        raise ValueError("The common scored event cohort is too small for top-k comparison")

    baseline_common = mapped_baseline.loc[
        mapped_baseline["logical_event_id"].astype(str).isin(common_ids)
    ].copy()
    temporal_common = temporal.loc[temporal["_rank_available"]].copy()
    graph_common = graph.loc[graph["logical_event_id"].astype(str).isin(common_ids)].copy()

    if not (
        len(baseline_common) == len(temporal_common) == len(graph_common) == len(common_ids)
    ):
        raise ValueError("Model outputs do not cover the same unique scored-event cohort")

    baseline_common["anomaly_score"] = _numeric_scores(
        baseline_common, "anomaly_score", "General Isolation Forest output"
    )
    temporal_common["temporal_anomaly_score"] = _numeric_scores(
        temporal_common, "temporal_anomaly_score", "Temporal Isolation Forest output"
    )
    graph_common["graph_anomaly_score"] = _numeric_scores(
        graph_common, "graph_anomaly_score", "Graph-aware Isolation Forest output"
    )
    baseline_common["_predicted"] = _boolean_flags(
        baseline_common, "predicted_anomaly", "General Isolation Forest output"
    )
    temporal_common["_predicted"] = _boolean_flags(
        temporal_common, "temporal_predicted_anomaly", "Temporal Isolation Forest output"
    )
    graph_common["_predicted"] = _boolean_flags(
        graph_common, "graph_predicted_anomaly", "Graph-aware Isolation Forest output"
    )

    # Reuse the repository's existing fusion implementation and configuration.
    fused = build_fused_frame(
        str(temporal_path),
        str(graph_path),
        temporal_weight=DEFAULT_FUSION_WEIGHT,
    )
    fused = fused.loc[fused["logical_event_id"].astype(str).isin(common_ids)].copy()
    if len(fused) != len(common_ids):
        raise ValueError("Fused scores do not cover the common scored-event cohort")
    fused["fused_score"] = _numeric_scores(fused, "fused_score", "Temporal-graph fusion")

    feature_frame = _read_csv(features_path, set(), "General model feature input")
    baseline_features = [
        column for column in feature_frame.columns if column not in EXCLUDE_COLUMNS
    ]
    if not baseline_features:
        raise ValueError("No general Isolation Forest model features were found")

    model_frames = {
        "Isolation Forest baseline": baseline_common,
        "Temporal Isolation Forest": temporal_common,
        "Graph-aware Isolation Forest": graph_common,
        "Temporal + Graph fused": fused,
    }
    native_counts = {
        "Isolation Forest baseline": int(baseline_common["_predicted"].sum()),
        "Temporal Isolation Forest": int(temporal_common["_predicted"].sum()),
        "Graph-aware Isolation Forest": int(graph_common["_predicted"].sum()),
        "Temporal + Graph fused": -1,
    }
    feature_descriptions = {
        "Isolation Forest baseline": "; ".join(baseline_features),
        "Temporal Isolation Forest": "; ".join(TEMPORAL_MODEL_FEATURES),
        "Graph-aware Isolation Forest": "; ".join(GRAPH_MODEL_FEATURES),
        "Temporal + Graph fused": (
            f"Existing temporal anomaly score + graph anomaly score; "
            f"weights={DEFAULT_FUSION_WEIGHT:.1f}/{1.0 - DEFAULT_FUSION_WEIGHT:.1f}"
        ),
    }
    return model_frames, native_counts, feature_descriptions, common_ids


def build_table7(
    baseline_path: Path = BASELINE_PATH,
    temporal_path: Path = TEMPORAL_PATH,
    graph_path: Path = GRAPH_PATH,
    features_path: Path = FEATURES_PATH,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    """Build the Table 7 method summary from saved model outputs."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    model_frames, native_counts, feature_descriptions, common_ids = _build_model_inputs(
        baseline_path, temporal_path, graph_path, features_path
    )
    if top_k > len(common_ids):
        raise ValueError("top_k cannot exceed the common scored-event cohort")

    score_columns = {
        "Isolation Forest baseline": "anomaly_score",
        "Temporal Isolation Forest": "temporal_anomaly_score",
        "Graph-aware Isolation Forest": "graph_anomaly_score",
        "Temporal + Graph fused": "fused_score",
    }
    top_events = {
        method: select_top_k_events(frame, score_columns[method], k=top_k)
        for method, frame in model_frames.items()
    }
    top_sets = {method: set(events) for method, events in top_events.items()}

    baseline_model = (
        "IsolationForest (n_estimators=300, contamination=auto, random_state=42)"
    )
    temporal_model = (
        "IsolationForest (n_estimators=500, contamination=0.10, random_state=42)"
    )
    graph_model = (
        "IsolationForest "
        f"(n_estimators={GRAPH_N_ESTIMATORS}, contamination={GRAPH_CONTAMINATION}, "
        f"random_state={GRAPH_RANDOM_STATE})"
    )
    models = {
        "Isolation Forest baseline": baseline_model,
        "Temporal Isolation Forest": temporal_model,
        "Graph-aware Isolation Forest": graph_model,
        "Temporal + Graph fused": "Existing weighted anomaly-score fusion (0.5 temporal / 0.5 graph)",
    }
    representations = {
        "Isolation Forest baseline": "General process behavioral features",
        "Temporal Isolation Forest": "Temporal event features",
        "Graph-aware Isolation Forest": "Temporal and process-graph features",
        "Temporal + Graph fused": "Existing temporal and graph anomaly scores",
    }
    score_rules = {
        "Isolation Forest baseline": "Highest baseline anomaly score",
        "Temporal Isolation Forest": "Highest temporal anomaly score",
        "Graph-aware Isolation Forest": "Highest graph-aware anomaly score",
        "Temporal + Graph fused": "Highest existing 0.5/0.5 fused score",
    }

    rows: list[dict[str, Any]] = []
    method_names = list(model_frames)
    for method in method_names:
        other_methods = [name for name in method_names if name != method]
        other_union = set().union(*(top_sets[name] for name in other_methods))
        row: dict[str, Any] = {
            "method": method,
            "feature_representation": representations[method],
            "model_configuration": models[method],
            "features_or_score_inputs": feature_descriptions[method],
            "events_evaluated": len(common_ids),
            "native_flagged_anomalies": (
                "N/A (rank selection only)"
                if native_counts[method] < 0
                else native_counts[method]
            ),
            "top_k": top_k,
            "top_k_selection_rule": score_rules[method],
            "top_k_event_ids": ";".join(top_events[method]),
            "events_selected_only_by_method_in_top_k": len(top_sets[method] - other_union),
            "evaluation_type": EVALUATION_TYPE,
        }
        for other_method, column in OVERLAP_COLUMNS.items():
            row[column] = len(top_sets[method] & top_sets[other_method])
        rows.append(row)

    table = pd.DataFrame(rows)
    if table.isna().any().any():
        raise ValueError("Table 7 unexpectedly contains missing values")
    return table


def build_report(table: pd.DataFrame, top_k: int = TOP_K) -> str:
    """Create a reproducibility and limitations note for the generated table."""
    lines = [
        "Table 7 — Baseline / Model Comparison",
        "",
        "Evaluation type: unsupervised/proxy comparison using ranked event selection and overlap.",
        "No independent anomaly ground-truth labels are available for M57-Jean; this table does not measure detection accuracy.",
        f"Common scored-event cohort: {int(table['events_evaluated'].iloc[0])} logical events.",
        f"Top-k selection: highest within-method anomaly ranking, k={top_k}, with logical_event_id as the deterministic tie-break.",
        "All overlap values count shared event IDs among each method's fixed top-k selection.",
        "Events selected only by a method are top-k IDs absent from the other three methods' top-k sets; this does not mean unique forensic evidence.",
        "",
        "Event alignment and cohort:",
        "- The general Isolation Forest CSV has no logical_event_id. Its 47 timestamped rows map one-to-one to timeline events by process_id and UTC create_time/timestamp.",
        "- One additional baseline row (PID 4, System) has no timestamp and is excluded from event-level comparison; the saved baseline model output was produced from all 48 input rows and was not refit.",
        "- The temporal detector does not score the first event because it has no previous-event context. All methods are compared on the remaining 46 common scored events.",
        "- The graph output covers all 47 timeline events; Table 7 restricts it and the existing fusion ranking to the same 46-event cohort.",
        "",
        "Model selection and interpretation:",
        "- Native anomaly flags are reported separately from fixed top-k selections because the detectors use different thresholds and selection counts.",
        "- Numerical anomaly scores are not compared across methods. The general, temporal, and graph methods use different feature representations.",
        "- Temporal + Graph fused uses the repository's existing 0.5/0.5 weighted average of raw anomaly scores. Its top-k ranking inherits the existing fusion's unnormalized score-scale behavior; Table 7 does not present that score as cross-model evidence.",
        "- Agreement/overlap measures model selection agreement only. Anomaly scores, ranks, and explanations do not establish malicious activity, compromise, attack activity, or intent.",
        "- Model outputs are loaded from the existing result CSVs. Seeds/configurations are recorded from their implementations; Table 7 does not retrain the models or use randomness.",
        "",
        "Table 7 summary:",
        table.to_string(index=False),
        "",
    ]
    return "\n".join(lines)


def write_table7_outputs(
    table: pd.DataFrame,
    output_path: Path = OUTPUT_PATH,
    report_path: Path = REPORT_PATH,
    top_k: int = TOP_K,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    report_path.write_text(build_report(table, top_k=top_k), encoding="utf-8")


def main() -> None:
    table = build_table7()
    write_table7_outputs(table)

    print("=== Table 7 Baseline / Model Comparison ===")
    print(f"Evaluation type: {EVALUATION_TYPE}")
    print(f"Common scored events: {int(table['events_evaluated'].iloc[0])}")
    print(f"Top-k per method: {TOP_K}")
    print("Metric: ranked-event overlap (selection agreement, not detection accuracy)")
    print()
    print(
        table[
            [
                "method",
                "native_flagged_anomalies",
                "top_k",
                "events_selected_only_by_method_in_top_k",
            ]
        ].to_string(index=False)
    )
    print()
    print(f"CSV: {OUTPUT_PATH}")
    print(f"Report: {REPORT_PATH}")
    print(
        "Limitation: M57-Jean has no independent anomaly labels; overlaps measure agreement only."
    )


if __name__ == "__main__":
    main()
