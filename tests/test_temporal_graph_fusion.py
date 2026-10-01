import pandas as pd
import pytest

from src.evaluation.temporal_graph_fusion import (
    build_fused_frame,
    calculate_overlaps,
    fuse_scores,
    random_baseline,
    select_top_k,
)


def test_fuse_scores():
    temporal = pd.Series([0.10, 0.00, -0.10])
    graph = pd.Series([0.00, 0.20, -0.20])

    result = fuse_scores(temporal, graph)

    expected = pd.Series([0.05, 0.10, -0.15])

    pd.testing.assert_series_equal(
        result,
        expected,
        check_names=False,
    )


def test_fuse_scores_custom_weight():
    temporal = pd.Series([0.10])
    graph = pd.Series([0.00])

    result = fuse_scores(temporal, graph, temporal_weight=0.75)

    assert result.iloc[0] == pytest.approx(0.075)


def test_select_top_k():
    frame = pd.DataFrame(
        {
            "logical_event_id": ["A", "B", "C"],
            "score": [0.1, 0.3, 0.2],
        }
    )

    result = select_top_k(frame, "score", k=2)

    assert result == {"B", "C"}


def test_calculate_overlaps():
    result = calculate_overlaps(
        {"A", "B", "C"},
        {"B", "C", "D"},
        {"B", "D"},
    )

    assert result["temporal_count"] == 3
    assert result["graph_count"] == 3
    assert result["fused_count"] == 2
    assert result["temporal_graph_overlap"] == 2
    assert result["fused_temporal_overlap"] == 1
    assert result["fused_graph_overlap"] == 2
    assert result["fused_detector_union_overlap"] == 2
    assert result["fused_unique_findings"] == 0


def test_random_baseline_is_reproducible():
    event_ids = [f"E{i}" for i in range(47)]
    temporal = {"E1", "E2", "E3", "E4", "E5"}
    graph = {"E6", "E7", "E8", "E9", "E10"}
    fused = {"E1", "E6", "E11", "E12", "E13"}

    first = random_baseline(
        event_ids,
        temporal,
        graph,
        fused,
        k=5,
        trials=100,
        random_state=42,
    )

    second = random_baseline(
        event_ids,
        temporal,
        graph,
        fused,
        k=5,
        trials=100,
        random_state=42,
    )

    assert first == second
    assert first["trials"] == 100.0
    assert first["sample_size"] == 5.0
    assert first["random_state"] == 42.0


def test_build_fused_frame():
    frame = build_fused_frame(
        "results/M57-Jean/temporal_anomalies.csv",
        "results/M57-Jean/graph_anomalies.csv",
    )

    assert len(frame) == 47
    assert "fused_score" in frame.columns

    top_five = frame.head(5)["process_id"].tolist()

    assert top_five == [3992, 1372, 3560, 2892, 2004]
