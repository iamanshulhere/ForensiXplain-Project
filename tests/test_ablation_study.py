import hashlib
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.evaluation.ablation_study import (
    ABLATION_COMPARISON_PATH,
    ABLATION_REPORT_PATH,
    ABLATION_RESULTS_PATH,
    ABLATION_SUMMARY_PATH,
    ALL_MODEL_FEATURES,
    CONTAMINATION,
    EXPECTED_COHORT_SIZE,
    FUSION_ALPHA,
    GRAPH_FEATURES_PATH,
    GRAPH_FULL_FEATURES,
    GRAPH_STRUCTURAL_FEATURES,
    METADATA_FEATURES,
    RESULTS_DIR,
    TEMPORAL_FEATURES,
    TEMPORAL_FEATURES_PATH,
    build_feature_matrices,
    load_repository_cohort,
    run_ablation_study,
    save_ablation_outputs,
)
from src.evaluation.temporal_graph_fusion import normalize_scores


@pytest.fixture(scope="module")
def repository_inputs():
    return load_repository_cohort()


@pytest.fixture(scope="module")
def matrices(repository_inputs):
    temporal, graph, cohort_ids = repository_inputs
    return build_feature_matrices(temporal, graph, cohort_ids)


@pytest.fixture(scope="module")
def ablation_run(repository_inputs):
    temporal, graph, cohort_ids = repository_inputs
    return run_ablation_study(temporal, graph, cohort_ids)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_feature_matrix_shapes_match_e1_to_e6_definitions(matrices):
    _, experiment_matrices = matrices

    assert experiment_matrices["E1"]["joint"].shape == (46, 16)
    assert experiment_matrices["E2"]["joint"].shape == (46, 5)
    assert experiment_matrices["E3"]["joint"].shape == (46, 9)
    assert experiment_matrices["E4"]["joint"].shape == (46, 25)
    assert experiment_matrices["E5"]["temporal"].shape == (46, 16)
    assert experiment_matrices["E5"]["graph"].shape == (46, 5)
    assert experiment_matrices["E6"]["temporal"].shape == (46, 16)
    assert experiment_matrices["E6"]["graph"].shape == (46, 9)
    assert "joint" not in experiment_matrices["E5"]
    assert "joint" not in experiment_matrices["E6"]


def test_each_experiment_uses_exact_approved_feature_membership(matrices):
    _, experiment_matrices = matrices
    assert experiment_matrices["E1"]["joint"].columns.tolist() == list(TEMPORAL_FEATURES)
    assert experiment_matrices["E2"]["joint"].columns.tolist() == list(GRAPH_STRUCTURAL_FEATURES)
    assert experiment_matrices["E3"]["joint"].columns.tolist() == list(GRAPH_FULL_FEATURES)
    assert experiment_matrices["E4"]["joint"].columns.tolist() == list(ALL_MODEL_FEATURES)
    assert experiment_matrices["E5"]["temporal"].columns.tolist() == list(TEMPORAL_FEATURES)
    assert experiment_matrices["E5"]["graph"].columns.tolist() == list(GRAPH_STRUCTURAL_FEATURES)
    assert experiment_matrices["E6"]["temporal"].columns.tolist() == list(TEMPORAL_FEATURES)
    assert experiment_matrices["E6"]["graph"].columns.tolist() == list(GRAPH_FULL_FEATURES)
    for components in experiment_matrices.values():
        for matrix in components.values():
            assert not (set(matrix.columns) & set(METADATA_FEATURES))


def test_all_experiments_use_same_46_unique_scored_logical_events(ablation_run):
    results, comparisons, summary, _ = ablation_run
    assert summary["cohort_size"] == EXPECTED_COHORT_SIZE == 46
    expected_ids = set(summary["cohort_logical_event_ids"])
    assert len(expected_ids) == 46
    assert summary["excluded_boundary_event"] not in expected_ids
    assert len(results) == 6 * 46
    for experiment_id, rows in results.groupby("experiment_id"):
        assert len(rows) == 46, experiment_id
        assert rows["logical_event_id"].is_unique
        assert set(rows["logical_event_id"]) == expected_ids
    assert len(comparisons) == 4
    assert comparisons["cohort_size"].eq(46).all()


def test_deterministic_reproduction(repository_inputs, ablation_run):
    temporal, graph, cohort_ids = repository_inputs
    first_results, first_comparison, first_summary, first_report = ablation_run
    second_results, second_comparison, second_summary, second_report = run_ablation_study(
        temporal, graph, cohort_ids
    )

    assert_frame_equal(first_results, second_results, check_exact=True)
    assert_frame_equal(first_comparison, second_comparison, check_exact=True)
    assert first_summary == second_summary
    assert first_report == second_report


def test_no_supervised_classification_metrics_are_generated(ablation_run):
    results, comparisons, summary, report = ablation_run
    forbidden = {"accuracy", "precision", "recall", "f1", "f1_score", "roc_auc", "pr_auc"}
    assert not (forbidden & {column.lower() for column in results.columns})
    assert not (forbidden & {column.lower() for column in comparisons.columns})
    assert summary["ground_truth_anomaly_labels_available"] is False
    assert summary["supervised_classification_metrics_generated"] is False
    assert "No supervised classification metrics are calculated or reported." in report
    assert "Ground Truth" not in report


def test_alpha_050_fusion_uses_min_max_component_scores(ablation_run):
    results, _, _, _ = ablation_run
    assert FUSION_ALPHA == 0.50
    event_ids = results.loc[results["experiment_id"] == "E1", "logical_event_id"].tolist()
    assert len(event_ids) == 46

    for experiment_id, graph_reference in (("E5", "E2"), ("E6", "E3")):
        temporal_base = results.loc[results["experiment_id"] == "E1"].set_index("logical_event_id")
        graph_base = results.loc[results["experiment_id"] == graph_reference].set_index("logical_event_id")
        fused = results.loc[results["experiment_id"] == experiment_id].set_index("logical_event_id")
        expected_temporal = normalize_scores(temporal_base["score"]).sort_index()
        expected_graph = normalize_scores(graph_base["score"]).sort_index()
        assert fused.loc[expected_temporal.index, "temporal_component_score"].tolist() == pytest.approx(
            expected_temporal.tolist()
        )
        assert fused.loc[expected_graph.index, "graph_component_score"].tolist() == pytest.approx(
            expected_graph.tolist()
        )
        assert fused.loc[expected_temporal.index, "score"].tolist() == pytest.approx(
            (0.50 * expected_temporal + 0.50 * expected_graph).tolist()
        )
        assert fused["fusion_alpha"].eq(0.50).all()


def test_existing_validated_fusion_artifacts_remain_unchanged(ablation_run, tmp_path):
    results, comparisons, summary, report = ablation_run
    protected = [
        RESULTS_DIR / "fused_anomalies.csv",
        RESULTS_DIR / "fusion_report.txt",
        RESULTS_DIR / "fusion_evaluation.csv",
    ]
    before = {path: _sha256(path) for path in protected}

    save_ablation_outputs(results, comparisons, summary, report, output_dir=tmp_path)

    after = {path: _sha256(path) for path in protected}
    assert after == before
    assert (tmp_path / ABLATION_RESULTS_PATH.name).exists()
    assert (tmp_path / ABLATION_COMPARISON_PATH.name).exists()
    assert (tmp_path / ABLATION_REPORT_PATH.name).exists()
    assert (tmp_path / ABLATION_SUMMARY_PATH.name).exists()


def test_duplicate_logical_event_ids_are_rejected(repository_inputs):
    temporal, graph, cohort_ids = repository_inputs
    duplicate_temporal = pd.concat([temporal, temporal.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate logical_event_id"):
        build_feature_matrices(duplicate_temporal, graph, cohort_ids)


def test_repository_cohort_is_exactly_the_validated_fusion_cohort(repository_inputs):
    temporal, graph, cohort_ids = repository_inputs
    assert len(cohort_ids) == 46
    assert temporal["logical_event_id"].is_unique
    assert graph["logical_event_id"].is_unique
    fusion = pd.read_csv(RESULTS_DIR / "fused_anomalies.csv", low_memory=False)
    assert set(cohort_ids) == set(fusion["logical_event_id"].astype(str))
