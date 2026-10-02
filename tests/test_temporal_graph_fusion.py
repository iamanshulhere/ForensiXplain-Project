import pandas as pd
import pytest

from src.evaluation.temporal_graph_fusion import (
    FUSION_ALPHAS,
    GRAPH_AWARE_MODEL_FEATURES,
    TEMPORAL_RESULTS_PATH,
    GRAPH_FEATURES_PATH,
    _format_report,
    _prepare_model_feature_matrix,
    _controlled_graph_comparison,
    _evaluation_rows,
    align_event_outputs,
    build_alpha_rankings,
    build_fused_anomaly_output,
    build_fused_frame,
    calculate_overlaps,
    fuse_scores,
    normalize_scores,
    random_baseline,
    score_controlled_graph_aware_features,
    score_graph_only_features,
    select_top_k_events,
    select_top_k,
    validate_event_alignment,
)
from src.anomaly.graph_features import GRAPH_FEATURE_COLUMNS
from src.anomaly.temporal_isolation_forest import (
    MODEL_FEATURES as TEMPORAL_MODEL_FEATURES,
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


def _alignment_frames(temporal_ids, graph_ids):
    temporal = pd.DataFrame(
        {
            "logical_event_id": temporal_ids,
            "process_id": [int(event_id[-1]) for event_id in temporal_ids],
            "timestamp": [
                f"2020-01-01T00:00:0{event_id[-1]}+00:00"
                for event_id in temporal_ids
            ],
            "temporal_anomaly_score": range(len(temporal_ids)),
            "temporal_predicted_anomaly": [False] * len(temporal_ids),
            "temporal_anomaly_rank": range(1, len(temporal_ids) + 1),
        }
    )
    graph_features = pd.DataFrame(
        {
            "logical_event_id": graph_ids,
            "process_id": [int(event_id[-1]) for event_id in graph_ids],
            "timestamp": [
                f"2020-01-01T00:00:0{event_id[-1]}Z"
                for event_id in graph_ids
            ],
            "process": [f"process-{event_id}" for event_id in graph_ids],
            **{
                column: [float(index) for index in range(len(graph_ids))]
                for column in GRAPH_FEATURE_COLUMNS
            },
        }
    )
    return temporal, graph_features


def test_event_alignment_uses_logical_event_id_not_row_order():
    temporal, graph_features = _alignment_frames(["E2", "E1"], ["E1", "E2"])

    aligned, summary = align_event_outputs(temporal, graph_features)

    assert aligned["logical_event_id"].tolist() == ["E1", "E2"]
    assert aligned["temporal_anomaly_score"].tolist() == [1, 0]
    assert aligned["process_id"].tolist() == [1, 2]
    assert summary["matched_events"] == 2


def test_duplicate_logical_event_keys_are_rejected():
    temporal, graph_features = _alignment_frames(["E1", "E1"], ["E1"])

    with pytest.raises(ValueError, match="Duplicate logical event keys"):
        validate_event_alignment(temporal, graph_features)


def test_unmatched_events_are_reported_and_not_silently_joined():
    temporal, graph_features = _alignment_frames(["E1", "E2"], ["E1", "E3"])

    aligned, summary = align_event_outputs(temporal, graph_features)

    assert aligned["logical_event_id"].tolist() == ["E1"]
    assert summary["total_temporal_events"] == 2
    assert summary["total_graph_events"] == 2
    assert summary["matched_events"] == 1
    assert summary["temporal_only_events"] == 1
    assert summary["graph_only_events"] == 1
    assert summary["temporal_only_logical_event_ids"] == ["E2"]
    assert summary["graph_only_logical_event_ids"] == ["E3"]


def test_min_max_normalization_maps_scores_to_zero_one_range():
    result = normalize_scores(pd.Series([-2.0, 0.0, 2.0]))

    assert result.tolist() == [0.0, 0.5, 1.0]
    assert result.is_monotonic_increasing


def test_alpha_endpoints_and_midpoint_use_expected_signals():
    events = pd.DataFrame(
        {
            "logical_event_id": ["A", "B", "C"],
            "temporal_anomaly_rank": [1, 3, 2],
            "normalized_temporal_score": [1.0, 0.0, 0.5],
            "normalized_graph_score": [0.0, 0.5, 1.0],
        }
    )

    rankings = build_alpha_rankings(events, alphas=[0.0, 0.5, 1.0])
    by_id = {
        alpha: ranking.set_index("logical_event_id")
        for alpha, ranking in rankings.items()
    }

    assert by_id[1.0]["fusion_score"].to_dict() == {
        "A": 1.0,
        "B": 0.0,
        "C": 0.5,
    }
    assert by_id[0.0]["fusion_score"].to_dict() == {
        "A": 0.0,
        "B": 0.5,
        "C": 1.0,
    }
    assert by_id[0.5]["fusion_score"].to_dict() == {
        "A": 0.5,
        "B": 0.25,
        "C": 0.75,
    }


def test_fusion_ranking_is_deterministic_and_uses_event_id_for_ties():
    events = pd.DataFrame(
        {
            "logical_event_id": ["C", "A", "B"],
            "temporal_anomaly_rank": [2, 3, 1],
            "normalized_temporal_score": [0.5, 0.5, 0.5],
            "normalized_graph_score": [0.5, 0.5, 0.5],
        }
    )

    first = build_alpha_rankings(events, alphas=FUSION_ALPHAS)
    second = build_alpha_rankings(events, alphas=FUSION_ALPHAS)

    assert first[0.5]["logical_event_id"].tolist() == ["A", "B", "C"]
    assert first[1.0]["logical_event_id"].tolist() == ["B", "C", "A"]
    pd.testing.assert_frame_equal(first[0.5], second[0.5])
    assert select_top_k_events(first[0.5], "fusion_score", k=2) == ["A", "B"]


def test_fused_output_has_audit_columns_and_unique_event_ids():
    events = pd.DataFrame(
        {
            "logical_event_id": ["A", "B", "C"],
            "temporal_anomaly_rank": [1, 3, 2],
            "process_id": [10, 20, 30],
            "timestamp": ["2020-01-01T00:00:00Z"] * 3,
            "temporal_score": [1.0, 0.0, 0.5],
            "graph_score": [0.0, 0.5, 1.0],
            "normalized_temporal_score": [1.0, 0.0, 0.5],
            "normalized_graph_score": [0.0, 0.5, 1.0],
        }
    )
    rankings = build_alpha_rankings(events)

    output = build_fused_anomaly_output(events, rankings)

    required = {
        "logical_event_id",
        "process_id",
        "timestamp",
        "temporal_score",
        "graph_score",
        "normalized_temporal_score",
        "normalized_graph_score",
        "alpha",
        "fusion_score",
        "fusion_rank",
        "fusion_predicted_anomaly",
        "fusion_score_alpha_0_00",
        "fusion_rank_alpha_1_00",
    }
    assert required <= set(output.columns)
    assert output["logical_event_id"].is_unique
    assert len(output) == len(events)


def _model_feature_frame(event_count=46):
    ids = [f"E{index:02d}" for index in range(event_count)]
    frame = pd.DataFrame(
        {
            "logical_event_id": ids,
            "process_id": range(1000, 1000 + event_count),
            "timestamp": ["2020-01-01T00:00:00Z"] * event_count,
        }
    )
    for feature_index, column in enumerate(GRAPH_AWARE_MODEL_FEATURES):
        frame[column] = [
            float((row + 1) * (feature_index + 2) + (row % 3))
            for row in range(event_count)
        ]
    return frame


def test_explicit_feature_matrix_ignores_labels_ranks_and_other_scores():
    frame = _model_feature_frame(8)
    frame["temporal_anomaly_score"] = range(8)
    frame["graph_anomaly_score"] = range(8, 16)
    frame["controlled_graph_aware_score"] = range(16, 24)
    frame["temporal_anomaly_rank"] = range(1, 9)
    frame["graph_anomaly_rank"] = range(8, 0, -1)
    frame["temporal_predicted_anomaly"] = [False] * 8
    frame["graph_predicted_anomaly"] = [True] * 8
    frame["ground_truth_label"] = [1] * 8

    _, graph_values = _prepare_model_feature_matrix(
        frame, frame["logical_event_id"], GRAPH_FEATURE_COLUMNS, "Graph feature output"
    )
    _, aware_values = _prepare_model_feature_matrix(
        frame,
        frame["logical_event_id"],
        GRAPH_AWARE_MODEL_FEATURES,
        "Graph feature output",
    )
    poisoned = frame.copy()
    poisoned["temporal_anomaly_score"] = -10000
    poisoned["graph_anomaly_score"] = 10000
    poisoned["controlled_graph_aware_score"] = -20000
    poisoned["temporal_anomaly_rank"] = -5
    poisoned["graph_anomaly_rank"] = -6
    poisoned["temporal_predicted_anomaly"] = True
    poisoned["graph_predicted_anomaly"] = False
    poisoned["ground_truth_label"] = 0
    _, poisoned_graph_values = _prepare_model_feature_matrix(
        poisoned,
        poisoned["logical_event_id"],
        GRAPH_FEATURE_COLUMNS,
        "Graph feature output",
    )
    _, poisoned_aware_values = _prepare_model_feature_matrix(
        poisoned,
        poisoned["logical_event_id"],
        GRAPH_AWARE_MODEL_FEATURES,
        "Graph feature output",
    )

    assert GRAPH_FEATURE_COLUMNS == [
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
    assert len(GRAPH_FEATURE_COLUMNS) == 9
    assert len(GRAPH_AWARE_MODEL_FEATURES) == 14
    assert set(GRAPH_FEATURE_COLUMNS).isdisjoint(
        {
            "temporal_anomaly_score",
            "graph_anomaly_score",
            "controlled_graph_aware_score",
            "temporal_anomaly_rank",
            "graph_anomaly_rank",
            "temporal_predicted_anomaly",
            "graph_predicted_anomaly",
            "ground_truth_label",
        }
    )
    assert graph_values.shape == (8, 9)
    assert aware_values.shape == (8, 14)
    assert (graph_values == poisoned_graph_values).all()
    assert (aware_values == poisoned_aware_values).all()


def test_controlled_models_fit_and_report_exactly_the_same_46_event_cohort():
    frame = _model_feature_frame()
    selected_ids = frame["logical_event_id"].tolist()

    graph_only = score_graph_only_features(frame, selected_ids)
    graph_aware = score_controlled_graph_aware_features(frame, selected_ids)
    repeated = score_controlled_graph_aware_features(frame, selected_ids)
    combined = frame.merge(graph_only, on="logical_event_id").merge(
        graph_aware, on="logical_event_id"
    )
    comparison = _controlled_graph_comparison(combined)

    assert len(graph_only) == len(graph_aware) == len(comparison) == 46
    assert set(graph_only["logical_event_id"]) == set(selected_ids)
    assert set(graph_aware["logical_event_id"]) == set(selected_ids)
    pd.testing.assert_frame_equal(graph_aware, repeated)
    assert comparison["graph_only_training_cohort_events"].eq(46).all()
    assert comparison["graph_aware_training_cohort_events"].eq(46).all()
    assert comparison["graph_only_feature_columns"].eq(
        ";".join(GRAPH_FEATURE_COLUMNS)
    ).all()
    assert comparison["graph_aware_feature_columns"].eq(
        ";".join(GRAPH_AWARE_MODEL_FEATURES)
    ).all()


def test_alpha_one_matches_all_saved_temporal_ranks_scores_and_flags():
    temporal = pd.read_csv(TEMPORAL_RESULTS_PATH, low_memory=False)
    graph_features = pd.read_csv(GRAPH_FEATURES_PATH, low_memory=False)
    aligned, _ = align_event_outputs(temporal, graph_features)
    scored = aligned.loc[
        pd.to_numeric(aligned["temporal_anomaly_rank"], errors="coerce").notna()
    ].copy()
    assert len(scored) == 46
    scored["normalized_temporal_score"] = normalize_scores(
        pd.to_numeric(scored["temporal_anomaly_score"])
    )
    scored["normalized_graph_score"] = 0.0

    alpha_one = build_alpha_rankings(scored, alphas=[1.0])[1.0]
    expected = scored.sort_values(
        "temporal_anomaly_rank", kind="mergesort"
    ).reset_index(drop=True)

    assert alpha_one["logical_event_id"].tolist() == expected[
        "logical_event_id"
    ].tolist()
    assert alpha_one["fusion_rank"].tolist() == list(range(1, 47))
    assert alpha_one["fusion_score"].tolist() == pytest.approx(
        expected["normalized_temporal_score"].tolist()
    )
    assert alpha_one["fusion_predicted_anomaly"].tolist() == expected[
        "temporal_predicted_anomaly"
    ].astype(bool).tolist()
    assert alpha_one.loc[
        alpha_one["fusion_rank"].le(5), "logical_event_id"
    ].tolist() == expected.loc[
        pd.to_numeric(expected["temporal_anomaly_rank"]).le(5), "logical_event_id"
    ].tolist()


def test_report_states_retrospective_scope_and_controlled_cohorts():
    events = _model_feature_frame(6)
    events["temporal_score"] = [0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    events["graph_score"] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    events["graph_aware_score"] = events["graph_score"]
    events["controlled_graph_aware_score"] = events["temporal_score"]
    events["normalized_temporal_score"] = normalize_scores(events["temporal_score"])
    events["normalized_graph_score"] = normalize_scores(events["graph_score"])
    events["temporal_anomaly_rank"] = range(1, 7)
    events["temporal_predicted_anomaly"] = [True, False, False, False, False, False]
    events["graph_predicted_anomaly"] = [False, False, False, False, False, True]
    events["graph_aware_predicted_anomaly"] = events["graph_predicted_anomaly"]
    events["controlled_graph_aware_predicted_anomaly"] = events[
        "temporal_predicted_anomaly"
    ]
    rankings = build_alpha_rankings(events)
    evaluation = _evaluation_rows(events, rankings, set(), graph_aware_training_events=47)
    alignment = {
        "total_temporal_events": 47,
        "total_graph_events": 47,
        "total_graph_aware_events": 47,
        "matched_events": 47,
        "temporal_only_events": 0,
        "temporal_only_logical_event_ids": [],
        "graph_only_events": 0,
        "graph_only_logical_event_ids": [],
        "temporal_only_vs_graph_aware_events": 0,
        "temporal_only_vs_graph_aware_logical_event_ids": [],
        "graph_aware_only_events": 0,
        "graph_aware_only_logical_event_ids": [],
        "duplicate_keys": 0,
        "matched_all_model_events": 47,
    }

    report = _format_report(events, alignment, rankings, evaluation, 1)

    assert "Retrospective scope" in report
    assert "events_next_*" in report
    assert "full reconstructed process graph" in report
    assert "complete scored cohort" in report
    assert "time-aware feature construction and time-local normalization" in report
    assert "fit on all 47 events" in report
    assert "new refit of the existing 14-feature family on the same 46 events" in report
    assert "also the Controlled Graph-only comparator" in report
    graph_only_row = evaluation.set_index("method").loc["Graph-only"]
    controlled_graph_only_row = evaluation.set_index("method").loc[
        "Controlled Graph-only"
    ]
    assert graph_only_row["top_5_event_ids"] == controlled_graph_only_row[
        "top_5_event_ids"
    ]
    assert "Same fit as Graph-only" in controlled_graph_only_row["model_origin"]
