"""Temporal-graph anomaly score fusion and proxy evaluation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.anomaly.graph_features import GRAPH_FEATURE_COLUMNS
from src.anomaly.graph_isolation_forest import (
    CONTAMINATION,
    MODEL_FEATURES as GRAPH_AWARE_MODEL_FEATURES,
    N_ESTIMATORS,
    RANDOM_STATE,
)
from src.anomaly.temporal_isolation_forest import (
    MODEL_FEATURES as TEMPORAL_MODEL_FEATURES,
)


DEFAULT_TOP_K = 5
DEFAULT_RANDOM_TRIALS = 1000
DEFAULT_RANDOM_STATE = 42
DEFAULT_FUSION_WEIGHT = 0.5
FUSION_ALPHAS = (0.00, 0.25, 0.50, 0.75, 1.00)
FUSION_CONTAMINATION = CONTAMINATION
FUSION_N_JOBS = 1
CASE_ID = "M57-Jean"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / CASE_ID
TEMPORAL_RESULTS_PATH = RESULTS_DIR / "temporal_anomalies.csv"
GRAPH_AWARE_RESULTS_PATH = RESULTS_DIR / "graph_anomalies.csv"
GRAPH_FEATURES_PATH = (
    PROJECT_ROOT / "data" / "features" / CASE_ID / "graph_features.csv"
)
TEMPORAL_ATTRIBUTION_PATH = RESULTS_DIR / "temporal_evidence_attribution.csv"
GRAPH_ATTRIBUTION_PATH = RESULTS_DIR / "graph_evidence_attribution.csv"
FUSED_ANOMALIES_PATH = RESULTS_DIR / "fused_anomalies.csv"
FUSION_EVALUATION_PATH = RESULTS_DIR / "fusion_evaluation.csv"
FUSION_REPORT_PATH = RESULTS_DIR / "fusion_report.txt"
CONTROLLED_GRAPH_COMPARISON_PATH = RESULTS_DIR / "controlled_graph_comparison.csv"


def fuse_scores(
    temporal_scores: pd.Series,
    graph_scores: pd.Series,
    temporal_weight: float = DEFAULT_FUSION_WEIGHT,
) -> pd.Series:
    """Combine temporal and graph anomaly scores using a weighted average."""
    if not 0.0 <= temporal_weight <= 1.0:
        raise ValueError("temporal_weight must be between 0 and 1")

    graph_weight = 1.0 - temporal_weight
    return (
        temporal_weight * temporal_scores
        + graph_weight * graph_scores
    )


def select_top_k(
    frame: pd.DataFrame,
    score_column: str,
    k: int = DEFAULT_TOP_K,
) -> set[str]:
    """Return logical event IDs for the top-k highest-scoring events."""
    if k <= 0:
        raise ValueError("k must be greater than 0")

    if score_column not in frame.columns:
        raise KeyError(f"Missing score column: {score_column}")

    if k > len(frame):
        raise ValueError("k cannot exceed the number of events")

    return set(
        frame.nlargest(k, score_column)["logical_event_id"]
    )


def calculate_overlaps(
    temporal_events: Iterable[str],
    graph_events: Iterable[str],
    fused_events: Iterable[str],
) -> dict[str, int]:
    """Calculate overlap counts between detector-selected event sets."""
    temporal = set(temporal_events)
    graph = set(graph_events)
    fused = set(fused_events)

    return {
        "temporal_count": len(temporal),
        "graph_count": len(graph),
        "fused_count": len(fused),
        "temporal_graph_overlap": len(temporal & graph),
        "fused_temporal_overlap": len(fused & temporal),
        "fused_graph_overlap": len(fused & graph),
        "fused_detector_union_overlap": len(
            fused & (temporal | graph)
        ),
        "fused_unique_findings": len(
            fused - (temporal | graph)
        ),
    }


def random_baseline(
    event_ids: Iterable[str],
    temporal_events: Iterable[str],
    graph_events: Iterable[str],
    fused_events: Iterable[str],
    k: int = DEFAULT_TOP_K,
    trials: int = DEFAULT_RANDOM_TRIALS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, float]:
    """Estimate random-selection overlap statistics."""
    if k <= 0:
        raise ValueError("k must be greater than 0")

    if trials <= 0:
        raise ValueError("trials must be greater than 0")

    ids = np.asarray(list(event_ids))

    if k > len(ids):
        raise ValueError("k cannot exceed the number of events")

    temporal = set(temporal_events)
    graph = set(graph_events)
    fused = set(fused_events)

    rng = np.random.default_rng(random_state)

    temporal_overlaps = []
    graph_overlaps = []
    fused_overlaps = []

    for _ in range(trials):
        selected = set(
            rng.choice(ids, size=k, replace=False)
        )

        temporal_overlaps.append(len(selected & temporal))
        graph_overlaps.append(len(selected & graph))
        fused_overlaps.append(len(selected & fused))

    def summarize(values: list[int]) -> tuple[float, float, int]:
        return (
            float(np.mean(values)),
            float(np.std(values)),
            int(np.max(values)),
        )

    temporal_mean, temporal_std, temporal_max = summarize(temporal_overlaps)
    graph_mean, graph_std, graph_max = summarize(graph_overlaps)
    fused_mean, fused_std, fused_max = summarize(fused_overlaps)

    return {
        "trials": float(trials),
        "sample_size": float(k),
        "random_state": float(random_state),
        "random_temporal_overlap_mean": temporal_mean,
        "random_temporal_overlap_std": temporal_std,
        "random_temporal_overlap_max": float(temporal_max),
        "random_graph_overlap_mean": graph_mean,
        "random_graph_overlap_std": graph_std,
        "random_graph_overlap_max": float(graph_max),
        "random_fused_overlap_mean": fused_mean,
        "random_fused_overlap_std": fused_std,
        "random_fused_overlap_max": float(fused_max),
        "random_temporal_zero_overlap_probability": float(
            np.mean(np.asarray(temporal_overlaps) == 0)
        ),
        "random_graph_zero_overlap_probability": float(
            np.mean(np.asarray(graph_overlaps) == 0)
        ),
        "random_fused_zero_overlap_probability": float(
            np.mean(np.asarray(fused_overlaps) == 0)
        ),
    }


def build_fused_frame(
    temporal_path: str,
    graph_path: str,
    temporal_weight: float = DEFAULT_FUSION_WEIGHT,
) -> pd.DataFrame:
    """Load detector results and create the fused anomaly ranking."""
    temporal = pd.read_csv(temporal_path)
    graph = pd.read_csv(graph_path)

    required_temporal = {
        "logical_event_id",
        "temporal_anomaly_score",
        "temporal_predicted_anomaly",
    }
    required_graph = {
        "logical_event_id",
        "graph_anomaly_score",
        "graph_predicted_anomaly",
    }

    missing_temporal = required_temporal - set(temporal.columns)
    missing_graph = required_graph - set(graph.columns)

    if missing_temporal:
        raise KeyError(
            f"Missing temporal columns: {sorted(missing_temporal)}"
        )

    if missing_graph:
        raise KeyError(
            f"Missing graph columns: {sorted(missing_graph)}"
        )

    columns = [
        column
        for column in ["logical_event_id", "process_id", "process"]
        if column in temporal.columns
    ]

    temporal_columns = columns + [
        "temporal_anomaly_score",
        "temporal_predicted_anomaly",
    ]

    graph_columns = [
        "logical_event_id",
        "graph_anomaly_score",
        "graph_predicted_anomaly",
    ]

    frame = temporal[temporal_columns].merge(
        graph[graph_columns],
        on="logical_event_id",
        how="inner",
        validate="one_to_one",
    )

    if frame.empty:
        raise ValueError("Temporal and graph results have no matching events")

    frame["fused_score"] = fuse_scores(
        frame["temporal_anomaly_score"],
        frame["graph_anomaly_score"],
        temporal_weight=temporal_weight,
    )

    return frame.sort_values(
        "fused_score",
        ascending=False,
    ).reset_index(drop=True)


def select_top_k_events(
    frame: pd.DataFrame,
    score_column: str,
    k: int = DEFAULT_TOP_K,
) -> list[str]:
    """Select a stable top-k event ranking, breaking score ties by event ID."""
    if k <= 0:
        raise ValueError("k must be greater than 0")
    if "logical_event_id" not in frame.columns:
        raise KeyError("Missing logical_event_id column")
    if score_column not in frame.columns:
        raise KeyError(f"Missing score column: {score_column}")
    if frame["logical_event_id"].isna().any():
        raise ValueError("Cannot rank rows with empty logical_event_id values")
    if frame["logical_event_id"].duplicated().any():
        raise ValueError("Cannot rank duplicate logical_event_id values")
    if k > len(frame):
        raise ValueError("k cannot exceed the number of events")

    scores = pd.to_numeric(frame[score_column], errors="coerce")
    if scores.isna().any() or not np.isfinite(scores.to_numpy(dtype=float)).all():
        raise ValueError(
            f"Ranking input contains missing or non-finite {score_column} values"
        )
    ranked = frame.assign(**{score_column: scores.astype(float)}).sort_values(
        [score_column, "logical_event_id"],
        ascending=[False, True],
        kind="mergesort",
    )
    return ranked.head(k)["logical_event_id"].astype(str).tolist()


def _event_key_info(
    frame: pd.DataFrame,
    description: str,
) -> tuple[set[str], set[str]]:
    if "logical_event_id" not in frame.columns:
        raise KeyError(f"{description} is missing logical_event_id")
    keys = frame["logical_event_id"]
    if keys.isna().any() or keys.astype(str).str.strip().eq("").any():
        raise ValueError(f"{description} contains an empty logical_event_id")
    keys = keys.astype(str)
    return set(keys), set(keys.loc[keys.duplicated(keep=False)])


def validate_event_alignment(
    temporal: pd.DataFrame,
    graph: pd.DataFrame,
    graph_aware: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Summarize logical-event coverage and reject ambiguous duplicate keys."""
    temporal_keys, temporal_duplicates = _event_key_info(
        temporal, "Temporal output"
    )
    graph_keys, graph_duplicates = _event_key_info(graph, "Graph output")
    graph_aware_keys: set[str] = set()
    graph_aware_duplicates: set[str] = set()
    if graph_aware is not None:
        graph_aware_keys, graph_aware_duplicates = _event_key_info(
            graph_aware, "Graph-aware output"
        )

    duplicates = {
        "temporal": temporal_duplicates,
        "graph": graph_duplicates,
        "graph_aware": graph_aware_duplicates,
    }
    duplicate_summary = "; ".join(
        f"{name}={sorted(keys)}"
        for name, keys in duplicates.items()
        if keys
    )
    if duplicate_summary:
        raise ValueError(
            f"Duplicate logical event keys detected: {duplicate_summary}"
        )

    matched = temporal_keys & graph_keys
    summary: dict[str, object] = {
        "total_temporal_events": len(temporal),
        "total_graph_events": len(graph),
        "matched_events": len(matched),
        "temporal_only_events": len(temporal_keys - graph_keys),
        "graph_only_events": len(graph_keys - temporal_keys),
        "duplicate_temporal_keys": 0,
        "duplicate_graph_keys": 0,
        "duplicate_graph_aware_keys": 0,
        "duplicate_keys": 0,
        "temporal_only_logical_event_ids": sorted(temporal_keys - graph_keys),
        "graph_only_logical_event_ids": sorted(graph_keys - temporal_keys),
    }
    if graph_aware is not None:
        summary.update(
            {
                "total_graph_aware_events": len(graph_aware),
                "matched_graph_aware_events": len(temporal_keys & graph_aware_keys),
                "temporal_only_vs_graph_aware_events": len(
                    temporal_keys - graph_aware_keys
                ),
                "temporal_only_vs_graph_aware_logical_event_ids": sorted(
                    temporal_keys - graph_aware_keys
                ),
                "graph_aware_only_events": len(graph_aware_keys - temporal_keys),
                "graph_aware_only_logical_event_ids": sorted(
                    graph_aware_keys - temporal_keys
                ),
                "matched_all_model_events": len(
                    temporal_keys & graph_keys & graph_aware_keys
                ),
            }
        )
    return summary


def align_event_outputs(
    temporal: pd.DataFrame,
    graph_features: pd.DataFrame,
    graph_aware: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Join one-to-one on logical_event_id and verify PID and UTC timestamps."""
    summary = validate_event_alignment(temporal, graph_features, graph_aware)
    temporal_columns = [
        "logical_event_id",
        "process_id",
        "timestamp",
        "temporal_anomaly_score",
        "temporal_predicted_anomaly",
        "temporal_anomaly_rank",
    ]
    graph_columns = [
        "logical_event_id",
        "process_id",
        "timestamp",
        "process",
        *GRAPH_FEATURE_COLUMNS,
    ]
    missing_temporal = set(temporal_columns) - set(temporal.columns)
    missing_graph = set(graph_columns) - set(graph_features.columns)
    if missing_temporal:
        raise KeyError(
            f"Temporal output is missing columns: {sorted(missing_temporal)}"
        )
    if missing_graph:
        raise KeyError(f"Graph features are missing columns: {sorted(missing_graph)}")

    temporal_view = temporal[temporal_columns].rename(
        columns={
            "process_id": "process_id_temporal",
            "timestamp": "timestamp_temporal",
        }
    )
    graph_view = graph_features[graph_columns].rename(
        columns={
            "process_id": "process_id_graph",
            "timestamp": "timestamp_graph",
        }
    )
    aligned = temporal_view.merge(
        graph_view,
        on="logical_event_id",
        how="inner",
        validate="one_to_one",
    )

    if graph_aware is not None:
        graph_aware_columns = [
            "logical_event_id",
            "process_id",
            "timestamp",
            "graph_anomaly_score",
            "graph_predicted_anomaly",
        ]
        missing_graph_aware = set(graph_aware_columns) - set(graph_aware.columns)
        if missing_graph_aware:
            raise KeyError(
                "Graph-aware output is missing columns: "
                f"{sorted(missing_graph_aware)}"
            )
        graph_aware_view = graph_aware[graph_aware_columns].rename(
            columns={
                "process_id": "process_id_graph_aware",
                "timestamp": "timestamp_graph_aware",
                "graph_anomaly_score": "graph_aware_score",
                "graph_predicted_anomaly": "graph_aware_predicted_anomaly",
            }
        )
        aligned = aligned.merge(
            graph_aware_view,
            on="logical_event_id",
            how="inner",
            validate="one_to_one",
        )
        summary["matched_all_model_events"] = len(aligned)

    pid_columns = ["process_id_temporal", "process_id_graph"]
    timestamp_columns = ["timestamp_temporal", "timestamp_graph"]
    if graph_aware is not None:
        pid_columns.append("process_id_graph_aware")
        timestamp_columns.append("timestamp_graph_aware")
    pids = [
        pd.to_numeric(aligned[column], errors="coerce")
        for column in pid_columns
    ]
    if any(series.isna().any() for series in pids) or any(
        not series.equals(pids[0]) for series in pids[1:]
    ):
        raise ValueError("PID/process_id values disagree across aligned outputs")
    timestamps = [
        pd.to_datetime(aligned[column], errors="coerce", utc=True)
        for column in timestamp_columns
    ]
    if any(series.isna().any() for series in timestamps) or any(
        not series.equals(timestamps[0]) for series in timestamps[1:]
    ):
        raise ValueError("UTC timestamps disagree across aligned outputs")

    aligned["process_id"] = pids[0].astype("Int64")
    aligned["timestamp"] = timestamps[0].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    aligned = aligned.sort_values(
        "logical_event_id", kind="mergesort"
    ).reset_index(drop=True)
    return aligned, summary


def normalize_scores(scores: pd.Series) -> pd.Series:
    """Min-max normalize finite anomaly scores to [0, 1], preserving rank."""
    values = pd.to_numeric(scores, errors="coerce").astype(float)
    if values.empty or values.isna().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError("Scores must be non-empty, finite numeric values")
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum == minimum:
        return pd.Series(0.0, index=scores.index, name=scores.name)
    return (values - minimum) / (maximum - minimum)


def _prepare_model_feature_matrix(
    feature_frame: pd.DataFrame,
    logical_event_ids: Iterable[str],
    feature_columns: Iterable[str],
    description: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Select an exact event cohort and explicit, cleaned model features."""
    selected_ids = set(str(event_id) for event_id in logical_event_ids)
    if not selected_ids:
        raise ValueError("At least one logical event is required")
    if any(not event_id.strip() for event_id in selected_ids):
        raise ValueError("Selected logical_event_id values cannot be empty")
    _, duplicate_keys = _event_key_info(feature_frame, description)
    if duplicate_keys:
        raise ValueError(
            f"Duplicate {description.lower()} logical_event_id values: "
            f"{sorted(duplicate_keys)}"
        )
    columns = list(feature_columns)
    if not columns or len(columns) != len(set(columns)):
        raise ValueError("Model feature columns must be non-empty and unique")
    missing = set(columns) - set(feature_frame.columns)
    if missing:
        raise KeyError(f"{description} is missing columns: {sorted(missing)}")
    cohort = feature_frame.loc[
        feature_frame["logical_event_id"].astype(str).isin(selected_ids)
    ].copy()
    if set(cohort["logical_event_id"].astype(str)) != selected_ids:
        raise ValueError(f"{description} do not cover every selected logical event")
    cohort = cohort.sort_values(
        "logical_event_id", kind="mergesort"
    ).reset_index(drop=True)
    features = cohort[columns].apply(
        pd.to_numeric, errors="coerce"
    )
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median().fillna(0.0))
    values = features.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Model feature matrix contains non-finite values")
    return cohort, values


def _score_model_feature_family(
    feature_frame: pd.DataFrame,
    logical_event_ids: Iterable[str],
    feature_columns: Iterable[str],
    description: str,
    score_column: str,
    prediction_column: str,
) -> pd.DataFrame:
    """Fit the fixed Isolation Forest pipeline for an explicit feature set."""
    cohort, values = _prepare_model_feature_matrix(
        feature_frame,
        logical_event_ids,
        feature_columns,
        description,
    )
    scaled = StandardScaler().fit_transform(values)
    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=FUSION_N_JOBS,
    )
    model.fit(scaled)
    scores = -model.decision_function(scaled)
    predictions = model.predict(scaled) == -1
    result = pd.DataFrame(
        {
            "logical_event_id": cohort["logical_event_id"].astype(str),
            score_column: scores,
            prediction_column: predictions,
        }
    )
    result[f"{score_column}_rank"] = result[score_column].rank(
        ascending=False, method="first"
    ).astype(int)
    return result


def score_graph_only_features(
    graph_features: pd.DataFrame,
    logical_event_ids: Iterable[str],
) -> pd.DataFrame:
    """Fit the configured Isolation Forest using only the nine graph metrics."""
    return _score_model_feature_family(
        graph_features,
        logical_event_ids,
        GRAPH_FEATURE_COLUMNS,
        "Graph feature output",
        "graph_score",
        "graph_predicted_anomaly",
    )


def score_controlled_graph_aware_features(
    graph_features: pd.DataFrame,
    logical_event_ids: Iterable[str],
) -> pd.DataFrame:
    """Refit the existing 14-feature graph-aware model on a supplied cohort."""
    return _score_model_feature_family(
        graph_features,
        logical_event_ids,
        GRAPH_AWARE_MODEL_FEATURES,
        "Graph feature output",
        "controlled_graph_aware_score",
        "controlled_graph_aware_predicted_anomaly",
    )


def build_alpha_rankings(
    events: pd.DataFrame,
    alphas: Iterable[float] = FUSION_ALPHAS,
    contamination: float = FUSION_CONTAMINATION,
) -> dict[float, pd.DataFrame]:
    """Build deterministic per-alpha scores, ranks, and anomaly flags."""
    required = {
        "logical_event_id",
        "normalized_temporal_score",
        "normalized_graph_score",
    }
    missing = required - set(events.columns)
    if missing:
        raise KeyError(f"Fusion input is missing columns: {sorted(missing)}")
    if events.empty:
        raise ValueError("Fusion input must contain at least one event")
    if events["logical_event_id"].isna().any():
        raise ValueError("Fusion input contains empty logical_event_id values")
    if events["logical_event_id"].duplicated().any():
        raise ValueError("Fusion input contains duplicate logical_event_id values")
    alpha_values = tuple(float(alpha) for alpha in alphas)
    events = events.copy()
    for column in ["normalized_temporal_score", "normalized_graph_score"]:
        values = pd.to_numeric(events[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Fusion input contains invalid {column} values")
        events[column] = values
    if 1.0 in alpha_values:
        if "temporal_anomaly_rank" not in events.columns:
            raise KeyError(
                "Fusion input is missing temporal_anomaly_rank required for alpha=1"
            )
        temporal_ranks = pd.to_numeric(
            events["temporal_anomaly_rank"], errors="coerce"
        ).to_numpy(dtype=float)
        if (
            not np.isfinite(temporal_ranks).all()
            or (temporal_ranks <= 0).any()
            or len(np.unique(temporal_ranks)) != len(temporal_ranks)
        ):
            raise ValueError(
                "alpha=1 requires unique positive finite temporal_anomaly_rank values"
            )
        events["temporal_anomaly_rank"] = temporal_ranks
    if not 0.0 < contamination < 1.0:
        raise ValueError("contamination must be between 0 and 1")
    anomaly_count = max(1, int(np.ceil(contamination * len(events))))
    rankings: dict[float, pd.DataFrame] = {}
    for alpha in alpha_values:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Each alpha must be between 0 and 1")
        ranked = events.copy()
        ranked["alpha"] = alpha
        ranked["fusion_score"] = (
            alpha * ranked["normalized_temporal_score"]
            + (1.0 - alpha) * ranked["normalized_graph_score"]
        )
        tie_breakers = (
            ["temporal_anomaly_rank", "logical_event_id"]
            if alpha == 1.0
            else ["logical_event_id"]
        )
        ranked = ranked.sort_values(
            ["fusion_score", *tie_breakers],
            ascending=[False, *([True] * len(tie_breakers))],
            kind="mergesort",
        ).reset_index(drop=True)
        ranked["fusion_rank"] = np.arange(1, len(ranked) + 1, dtype=int)
        if alpha == 1.0 and "temporal_predicted_anomaly" in ranked.columns:
            ranked["fusion_predicted_anomaly"] = _parse_boolean_flags(
                ranked, "temporal_predicted_anomaly"
            )
        else:
            ranked["fusion_predicted_anomaly"] = (
                ranked["fusion_rank"] <= anomaly_count
            )
        rankings[alpha] = ranked
    return rankings


def build_fused_anomaly_output(
    events: pd.DataFrame,
    rankings: dict[float, pd.DataFrame],
    primary_alpha: float = DEFAULT_FUSION_WEIGHT,
) -> pd.DataFrame:
    """Create one row per event with the primary and all tested alpha results."""
    if primary_alpha not in rankings:
        raise ValueError("primary_alpha must be included in the rankings")
    if events["logical_event_id"].duplicated().any():
        raise ValueError(
            "Fusion output would contain duplicate logical_event_id values"
        )
    primary = rankings[primary_alpha][
        [
            "logical_event_id",
            "alpha",
            "fusion_score",
            "fusion_rank",
            "fusion_predicted_anomaly",
        ]
    ].copy()
    output = events.merge(
        primary,
        on="logical_event_id",
        how="inner",
        validate="one_to_one",
    )
    if len(output) != len(events):
        raise ValueError("Primary fusion ranking does not cover every event")
    for alpha, ranking in rankings.items():
        suffix = f"{alpha:.2f}".replace(".", "_")
        alpha_columns = ranking[
            [
                "logical_event_id",
                "fusion_score",
                "fusion_rank",
                "fusion_predicted_anomaly",
            ]
        ].rename(
            columns={
                "fusion_score": f"fusion_score_alpha_{suffix}",
                "fusion_rank": f"fusion_rank_alpha_{suffix}",
                "fusion_predicted_anomaly": (
                    f"fusion_predicted_anomaly_alpha_{suffix}"
                ),
            }
        )
        output = output.merge(
            alpha_columns,
            on="logical_event_id",
            how="left",
            validate="one_to_one",
        )
    if output["logical_event_id"].duplicated().any():
        raise ValueError(
            "Fused anomaly output contains duplicate logical_event_id values"
        )
    return output.sort_values(
        ["fusion_rank", "logical_event_id"], kind="mergesort"
    ).reset_index(drop=True)


def _parse_boolean_flags(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column].astype(str).str.strip().str.lower()
    parsed = values.map({"true": True, "false": False, "1": True, "0": False})
    if parsed.isna().any():
        raise ValueError(f"Invalid boolean values in {column}")
    return parsed.astype(bool)


def _attributed_event_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    frame = pd.read_csv(path, low_memory=False)
    if "logical_event_id" not in frame.columns:
        raise KeyError(
            f"Evidence attribution file is missing logical_event_id: {path}"
        )
    return set(frame["logical_event_id"].dropna().astype(str))


def _evaluation_rows(
    events: pd.DataFrame,
    rankings: dict[float, pd.DataFrame],
    evidence_linked_ids: set[str],
    graph_aware_training_events: int | None = None,
) -> pd.DataFrame:
    top_k = min(DEFAULT_TOP_K, len(events))
    model_specs = [
        (
            "Temporal-only",
            "temporal_score",
            "temporal_predicted_anomaly",
            len(events),
            TEMPORAL_MODEL_FEATURES,
        ),
        (
            "Graph-only",
            "graph_score",
            "graph_predicted_anomaly",
            len(events),
            GRAPH_FEATURE_COLUMNS,
        ),
        (
            "Existing Graph-aware",
            "graph_aware_score",
            "graph_aware_predicted_anomaly",
            graph_aware_training_events or len(events),
            GRAPH_AWARE_MODEL_FEATURES,
        ),
        (
            "Controlled Graph-only",
            "graph_score",
            "graph_predicted_anomaly",
            len(events),
            GRAPH_FEATURE_COLUMNS,
        ),
        (
            "Controlled Graph-aware",
            "controlled_graph_aware_score",
            "controlled_graph_aware_predicted_anomaly",
            len(events),
            GRAPH_AWARE_MODEL_FEATURES,
        ),
    ]
    models: dict[str, dict[str, object]] = {}
    comparison_keys = {
        "Temporal-only": "temporal_only",
        "Graph-only": "graph_only",
        "Existing Graph-aware": "graph_aware",
        "Controlled Graph-only": "controlled_graph_only",
        "Controlled Graph-aware": "controlled_graph_aware",
    }
    model_origins = {
        "Temporal-only": "Existing temporal model output",
        "Graph-only": "Fusion graph-only fit; also used as Controlled Graph-only",
        "Existing Graph-aware": "Saved graph_anomalies.csv; model fit on the full output cohort",
        "Controlled Graph-only": "Same fit as Graph-only; evaluated as the common-cohort control",
        "Controlled Graph-aware": "New refit of the existing 14-feature family on the common cohort",
    }
    for method, score_column, flag_column, fit_events, feature_columns in model_specs:
        if method == "Temporal-only" and "temporal_anomaly_rank" in events:
            top_ids = (
                events.sort_values("temporal_anomaly_rank", kind="mergesort")
                .head(top_k)["logical_event_id"]
                .astype(str)
                .tolist()
            )
        else:
            top_ids = select_top_k_events(events, score_column, top_k)
        models[method] = {
            "top_ids": top_ids,
            "top_set": set(top_ids),
            "flag_set": set(
                events.loc[events[flag_column], "logical_event_id"].astype(str)
            ),
            "fit_events": fit_events,
            "feature_columns": feature_columns,
        }

    rows: list[dict[str, object]] = []
    for method, info in models.items():
        top_ids = info["top_ids"]
        top_events = events.set_index("logical_event_id").loc[top_ids]
        row: dict[str, object] = {
            "method": method,
            "alpha": pd.NA,
            "number_of_events": len(events),
            "training_cohort_events": info["fit_events"],
            "input_feature_columns": ";".join(info["feature_columns"]),
            "model_origin": model_origins[method],
            "number_of_anomalies": len(info["flag_set"]),
            "top_5_event_ids": ";".join(top_ids),
            "top_5_pids": ";".join(str(pid) for pid in top_events["process_id"]),
            "evidence_linked_top_5_count": len(set(top_ids) & evidence_linked_ids),
            "evidence_linked_top_5_event_ids": ";".join(
                event_id for event_id in top_ids if event_id in evidence_linked_ids
            ),
        }
        for reference in models:
            key = comparison_keys[reference]
            row[f"top_5_overlap_with_{key}"] = len(
                info["top_set"] & models[reference]["top_set"]
            )
            row[f"anomaly_overlap_with_{key}"] = len(
                info["flag_set"] & models[reference]["flag_set"]
            )
        rows.append(row)

    temporal_top = models["Temporal-only"]["top_set"]
    graph_top = models["Graph-only"]["top_set"]
    for alpha, ranking in rankings.items():
        top_ids = select_top_k_events(ranking, "fusion_score", top_k)
        top_set = set(top_ids)
        top_events = ranking.set_index("logical_event_id").loc[top_ids]
        flag_set = set(
            ranking.loc[
                ranking["fusion_predicted_anomaly"], "logical_event_id"
            ].astype(str)
        )
        row = {
            "method": "Fused",
            "alpha": alpha,
            "number_of_events": len(events),
            "training_cohort_events": pd.NA,
            "input_feature_columns": "normalized_temporal_score;normalized_graph_score",
            "model_origin": "Weighted score fusion; no separately fit detector",
            "number_of_anomalies": len(flag_set),
            "top_5_event_ids": ";".join(top_ids),
            "top_5_pids": ";".join(str(pid) for pid in top_events["process_id"]),
            "evidence_linked_top_5_count": len(top_set & evidence_linked_ids),
            "evidence_linked_top_5_event_ids": ";".join(
                event_id for event_id in top_ids if event_id in evidence_linked_ids
            ),
            "top_5_overlap_temporal_graph_only": len(temporal_top & graph_top),
        }
        for reference in models:
            key = comparison_keys[reference]
            row[f"top_5_overlap_with_{key}"] = len(
                top_set & models[reference]["top_set"]
            )
            row[f"anomaly_overlap_with_{key}"] = len(
                flag_set & models[reference]["flag_set"]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _controlled_graph_comparison(events: pd.DataFrame) -> pd.DataFrame:
    """Build an auditable per-event comparison for the common model cohort."""
    required = {
        "logical_event_id",
        "process_id",
        "timestamp",
        "graph_score",
        "graph_score_rank",
        "graph_predicted_anomaly",
        "controlled_graph_aware_score",
        "controlled_graph_aware_score_rank",
        "controlled_graph_aware_predicted_anomaly",
    }
    missing = required - set(events.columns)
    if missing:
        raise KeyError(
            f"Controlled graph comparison is missing columns: {sorted(missing)}"
        )
    if events["logical_event_id"].duplicated().any():
        raise ValueError("Controlled graph comparison contains duplicate event IDs")
    cohort_size = len(events)
    result = events[
        [
            "logical_event_id",
            "process_id",
            "timestamp",
            "graph_score",
            "graph_score_rank",
            "graph_predicted_anomaly",
            "controlled_graph_aware_score",
            "controlled_graph_aware_score_rank",
            "controlled_graph_aware_predicted_anomaly",
        ]
    ].copy()
    result = result.rename(
        columns={
            "graph_score": "controlled_graph_only_score",
            "graph_score_rank": "controlled_graph_only_rank",
            "graph_predicted_anomaly": "controlled_graph_only_predicted_anomaly",
            "controlled_graph_aware_score_rank": "controlled_graph_aware_rank",
        }
    )
    result["graph_only_training_cohort_events"] = cohort_size
    result["graph_aware_training_cohort_events"] = cohort_size
    result["graph_only_feature_columns"] = ";".join(GRAPH_FEATURE_COLUMNS)
    result["graph_aware_feature_columns"] = ";".join(GRAPH_AWARE_MODEL_FEATURES)
    return result.sort_values("logical_event_id", kind="mergesort").reset_index(drop=True)


def _format_report(
    events: pd.DataFrame,
    alignment: dict[str, object],
    rankings: dict[float, pd.DataFrame],
    evaluation: pd.DataFrame,
    unscored_temporal_events: int,
) -> str:
    temporal_scores = events["temporal_score"]
    graph_scores = events["graph_score"]
    report = [
        "ForensiXplain temporal + graph score fusion experiment",
        f"Dataset: {CASE_ID}",
        "",
        "Method",
        "------",
        "Graph-only scores are produced by the configured Isolation Forest using only the nine graph metrics in graph_features.csv.",
        f"Graph-only configuration: n_estimators={N_ESTIMATORS}, contamination={CONTAMINATION}, random_state={RANDOM_STATE}, n_jobs={FUSION_N_JOBS}.",
        "Existing Graph-aware scores come from the unchanged graph_isolation_forest.py output, which uses its 14 temporal and graph features and was fit on all 47 events.",
        "The fusion Graph-only fit uses 46 scored events and is also the Controlled Graph-only comparator; it uses exactly the nine graph metrics. Controlled Graph-aware is a new refit of the existing 14-feature family on the same 46 events.",
        "This controlled comparison holds the event cohort constant across feature families. The saved existing Graph-aware output remains a separate 47-event-fit reference and is not overwritten.",
        "Fusion is alpha * normalized_temporal_score + (1 - alpha) * normalized_graph_score.",
        "Both component scores use min-max normalization over the common scored-event cohort.",
        f"Alpha values: {', '.join(f'{alpha:.2f}' for alpha in rankings)}",
        "Higher scores indicate more anomalous events. Ranks break ties by logical_event_id, except alpha=1 uses temporal_anomaly_rank to reproduce the temporal detector's native tie order.",
        f"Fused predicted-anomaly flags select the top ceil({FUSION_CONTAMINATION:.2f} * N) events; alpha=1 preserves temporal detector flags when supplied.",
        "",
        "Score ranges before normalization",
        "-------------------------------",
        f"Temporal score: min={temporal_scores.min():.9f}, max={temporal_scores.max():.9f}",
        f"Graph-only score: min={graph_scores.min():.9f}, max={graph_scores.max():.9f}",
        "",
        "Event alignment",
        "---------------",
        f"Temporal events: {alignment['total_temporal_events']}",
        f"Graph-feature events: {alignment['total_graph_events']}",
        f"Graph-aware model events: {alignment.get('total_graph_aware_events', 'not provided')}",
        f"Matched temporal/graph keys: {alignment['matched_events']}",
        f"Temporal-only keys: {alignment['temporal_only_events']} {alignment['temporal_only_logical_event_ids']}",
        f"Graph-only keys: {alignment['graph_only_events']} {alignment['graph_only_logical_event_ids']}",
        (
            "Temporal-only vs graph-aware keys: "
            f"{alignment.get('temporal_only_vs_graph_aware_events', 0)} "
            f"{alignment.get('temporal_only_vs_graph_aware_logical_event_ids', [])}"
        ),
        (
            "Graph-aware-only keys: "
            f"{alignment.get('graph_aware_only_events', 0)} "
            f"{alignment.get('graph_aware_only_logical_event_ids', [])}"
        ),
        f"Duplicate keys: {alignment['duplicate_keys']}",
        f"Matched keys across temporal, graph features, and graph-aware output: {alignment.get('matched_all_model_events', alignment['matched_events'])}",
        f"Unscored temporal events excluded from fusion: {unscored_temporal_events}",
        f"Scored fusion cohort: {len(events)} events",
        "Join key: logical_event_id; PID and UTC timestamps were checked across inputs.",
        "",
        "Model comparison",
        "----------------",
    ]
    model_rows = evaluation.loc[evaluation["method"] != "Fused"].set_index("method")
    graph_row = model_rows.loc["Graph-only"]
    graph_aware_row = model_rows.loc["Existing Graph-aware"]
    report.extend(
        [
            "Top-5 overlap among component detectors:",
            (
                "  Temporal / graph-only top-5 intersection: "
                f"{graph_row['top_5_overlap_with_temporal_only']}"
            ),
            (
                "  Temporal / graph-aware top-5 intersection: "
                f"{graph_aware_row['top_5_overlap_with_temporal_only']}"
            ),
            (
                "  Graph-only / graph-aware top-5 intersection: "
                f"{graph_aware_row['top_5_overlap_with_graph_only']}"
            ),
            (
                "  Controlled graph-only / controlled graph-aware top-5 intersection: "
                f"{model_rows.loc['Controlled Graph-aware', 'top_5_overlap_with_controlled_graph_only']}"
            ),
            "",
        ]
    )
    for _, row in evaluation.iterrows():
        alpha = "" if pd.isna(row["alpha"]) else f" (alpha={row['alpha']:.2f})"
        report.extend(
            [
                f"{row['method']}{alpha}: {row['number_of_events']} events, {row['number_of_anomalies']} predicted anomalies",
                f"  Top 5: {row['top_5_event_ids']}",
                (
                    f"  Evidence-linked top-5 candidates: "
                    f"{row['evidence_linked_top_5_count']} "
                    f"[{row['evidence_linked_top_5_event_ids']}]"
                ),
            ]
        )
        if row["method"] == "Fused":
            report.append(
                "  Top-5 overlap with temporal / graph-only / existing graph-aware / controlled graph-only / controlled graph-aware: "
                f"{row['top_5_overlap_with_temporal_only']} / "
                f"{row['top_5_overlap_with_graph_only']} / "
                f"{row['top_5_overlap_with_graph_aware']} / "
                f"{row['top_5_overlap_with_controlled_graph_only']} / "
                f"{row['top_5_overlap_with_controlled_graph_aware']}"
            )
            report.append(
                "  Predicted-anomaly overlap with temporal / graph-only / existing graph-aware / controlled graph-only / controlled graph-aware: "
                f"{row['anomaly_overlap_with_temporal_only']} / "
                f"{row['anomaly_overlap_with_graph_only']} / "
                f"{row['anomaly_overlap_with_graph_aware']} / "
                f"{row['anomaly_overlap_with_controlled_graph_only']} / "
                f"{row['anomaly_overlap_with_controlled_graph_aware']}"
            )
    report.extend(
        [
            "",
            "Interpretation and limits",
            "--------------------------",
            "Top-5 overlap describes ranking agreement; predicted-anomaly overlap describes detector-flag agreement.",
            "Evidence-linked candidates measure traceability to existing temporal or graph attribution records. No fused SHAP or attribution is generated.",
            "No independent M57-Jean ground-truth anomaly labels were found in the evaluation inputs. These results describe score behavior, overlap, deterministic ranking, and evidence traceability; they do not measure detection accuracy.",
            "Retrospective scope: temporal local-density features include events_next_* context, graph metrics are computed from the full reconstructed process graph, and min-max normalization uses the complete scored cohort.",
            "The experiment does not claim that scores can be computed causally at the exact event time. A future causal evaluation requires time-aware feature construction and time-local normalization.",
            "",
            "Reproduce with: python -m src.evaluation.temporal_graph_fusion",
        ]
    )
    return "\n".join(report) + "\n"


def run_fusion_experiment() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Run and save the deterministic M57-Jean temporal + graph experiment."""
    temporal = pd.read_csv(TEMPORAL_RESULTS_PATH, low_memory=False)
    graph_features = pd.read_csv(GRAPH_FEATURES_PATH, low_memory=False)
    graph_aware = pd.read_csv(GRAPH_AWARE_RESULTS_PATH, low_memory=False)
    aligned, alignment = align_event_outputs(
        temporal, graph_features, graph_aware
    )

    temporal_rank = pd.to_numeric(aligned["temporal_anomaly_rank"], errors="coerce")
    unscored_count = int(temporal_rank.isna().sum())
    scored = aligned.loc[temporal_rank.notna()].copy()
    if len(scored) <= DEFAULT_TOP_K:
        raise ValueError(
            "The aligned scored-event cohort is too small for top-5 analysis"
        )
    if scored["logical_event_id"].duplicated().any():
        raise ValueError("Aligned scored cohort contains duplicate logical_event_id values")

    scored["temporal_score"] = pd.to_numeric(
        scored["temporal_anomaly_score"], errors="coerce"
    )
    scored["graph_aware_score"] = pd.to_numeric(
        scored["graph_aware_score"], errors="coerce"
    )
    for column in ["temporal_score", "graph_aware_score"]:
        if not np.isfinite(scored[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Aligned events contain invalid {column} values")
    scored["temporal_predicted_anomaly"] = _parse_boolean_flags(
        scored, "temporal_predicted_anomaly"
    )
    scored["graph_aware_predicted_anomaly"] = _parse_boolean_flags(
        scored, "graph_aware_predicted_anomaly"
    )

    graph_only = score_graph_only_features(
        graph_features, scored["logical_event_id"].astype(str)
    )
    controlled_graph_aware = score_controlled_graph_aware_features(
        graph_features, scored["logical_event_id"].astype(str)
    )
    expected_event_ids = set(scored["logical_event_id"].astype(str))
    for description, scores in [
        ("Graph-only", graph_only),
        ("Controlled graph-aware", controlled_graph_aware),
    ]:
        actual_event_ids = set(scores["logical_event_id"].astype(str))
        if actual_event_ids != expected_event_ids:
            raise ValueError(
                f"{description} model did not score exactly the common scored-event cohort"
            )
    scored = scored.merge(
        graph_only,
        on="logical_event_id",
        how="inner",
        validate="one_to_one",
    )
    if len(scored) != len(graph_only):
        raise ValueError("Graph-only scores do not align one-to-one with temporal events")
    scored = scored.merge(
        controlled_graph_aware,
        on="logical_event_id",
        how="inner",
        validate="one_to_one",
    )
    if len(scored) != len(expected_event_ids):
        raise ValueError(
            "Controlled graph-aware scores do not align one-to-one with temporal events"
        )
    scored["normalized_temporal_score"] = normalize_scores(
        scored["temporal_score"]
    )
    scored["normalized_graph_score"] = normalize_scores(scored["graph_score"])

    rankings = build_alpha_rankings(scored)
    evidence_linked_ids = (
        _attributed_event_ids(TEMPORAL_ATTRIBUTION_PATH)
        | _attributed_event_ids(GRAPH_ATTRIBUTION_PATH)
    )
    evaluation = _evaluation_rows(
        scored,
        rankings,
        evidence_linked_ids,
        graph_aware_training_events=int(alignment["total_graph_aware_events"]),
    )
    fused_output = build_fused_anomaly_output(scored, rankings)
    controlled_comparison = _controlled_graph_comparison(scored)
    report = _format_report(
        scored, alignment, rankings, evaluation, unscored_count
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fused_output.to_csv(FUSED_ANOMALIES_PATH, index=False, encoding="utf-8")
    evaluation.to_csv(FUSION_EVALUATION_PATH, index=False, encoding="utf-8")
    controlled_comparison.to_csv(
        CONTROLLED_GRAPH_COMPARISON_PATH, index=False, encoding="utf-8"
    )
    FUSION_REPORT_PATH.write_text(report, encoding="utf-8")
    return fused_output, evaluation, report


def main() -> None:
    """Run the reproducible M57-Jean fusion experiment and print its report."""
    _, _, report = run_fusion_experiment()
    print(report)
    print(f"Fused anomalies: {FUSED_ANOMALIES_PATH}")
    print(f"Fusion evaluation: {FUSION_EVALUATION_PATH}")
    print(f"Controlled graph comparison: {CONTROLLED_GRAPH_COMPARISON_PATH}")
    print(f"Fusion report: {FUSION_REPORT_PATH}")


if __name__ == "__main__":
    main()
