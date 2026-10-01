"""Temporal-graph anomaly score fusion and proxy evaluation utilities."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_TOP_K = 5
DEFAULT_RANDOM_TRIALS = 1000
DEFAULT_RANDOM_STATE = 42
DEFAULT_FUSION_WEIGHT = 0.5


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
