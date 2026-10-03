"""Tests for the Phase 10 robustness analysis and its generated audit files."""

import json
import math

import numpy as np
import pandas as pd

from src.evaluation import phase10_robustness as phase10


def test_phase10_parameter_grid_is_limited_one_factor_at_a_time():
    assert phase10.ALPHAS == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert len(phase10.CONFIGS) == 7
    assert phase10.CONFIGS[0] == ("baseline", 500, 0.10, 42)
    assert {item[1] for item in phase10.CONFIGS} == {250, 500, 1000}
    assert {item[2] for item in phase10.CONFIGS} == {0.05, 0.10, 0.15}
    assert {item[3] for item in phase10.CONFIGS} == {0, 42, 123}
    for _, estimators, contamination, seed in phase10.CONFIGS[1:]:
        changed = sum((estimators != 500, contamination != 0.10, seed != 42))
        assert changed == 1


def test_top_k_and_contamination_count_rules():
    assert phase10.TOP_KS == (3, 5, 10)
    assert [max(1, math.ceil(c * 46)) for c in (0.05, 0.10, 0.15)] == [3, 5, 7]


def test_stability_classifications_follow_approved_cutoffs():
    assert phase10._classify(4) == "stable"
    assert phase10._classify(3) == "stable"
    assert phase10._classify(2) == "mixed"
    assert phase10._classify(1) == "sensitive"
    assert phase10._classify(0) == "sensitive"


def test_phase9_and_legacy_feature_matrices_have_expected_shapes_and_same_cohort():
    temporal, graph, ids = phase10.phase9.load_repository_cohort()
    assert len(ids) == 46
    assert "LEVT-M57-Jean-PROCESS-812-20091121013230" not in ids
    phase_matrices, legacy_matrices, _ = phase10._load_matrices(ids)
    assert {name: matrix.shape for name, matrix in phase_matrices.items()} == {
        "E1": (46, 16), "E2": (46, 5), "E3": (46, 9), "E4": (46, 25)
    }
    assert {name: matrix.shape for name, matrix in legacy_matrices.items()} == {
        "LegacyTemporal5": (46, 5), "LegacyGraphOnly9": (46, 9), "LegacyGraphAware14": (46, 14)
    }
    assert np.array_equal(phase_matrices["E3"].to_numpy(), legacy_matrices["LegacyGraphOnly9"].to_numpy())
    assert phase_matrices["E1"].columns.tolist() == list(phase10.phase9.TEMPORAL_FEATURES)
    assert phase_matrices["E2"].columns.tolist() == list(phase10.phase9.GRAPH_STRUCTURAL_FEATURES)
    assert phase_matrices["E3"].columns.tolist() == list(phase10.phase9.GRAPH_FULL_FEATURES)
    assert phase_matrices["E4"].columns.tolist() == list(phase10.phase9.ALL_MODEL_FEATURES)
    assert not (set(phase_matrices["E4"].columns) & set(phase10.phase9.METADATA_FEATURES))
    assert temporal is not None and graph is not None


def test_fusion_alpha_endpoints_preserve_component_rank_and_flags():
    ids = [f"event-{ix:02d}" for ix in range(46)]
    temporal_scores = pd.Series(np.linspace(-0.2, 0.2, 46), index=ids)
    graph_scores = pd.Series(np.linspace(0.3, -0.1, 46), index=ids)
    temporal = {
        "scores": temporal_scores,
        "rank": phase10._rank(temporal_scores, ids, ids),
        "candidates": pd.Series([ix < 5 for ix in range(46)], index=ids),
    }
    graph = {
        "scores": graph_scores,
        "rank": phase10._rank(graph_scores, ids, ids),
        "candidates": pd.Series([ix >= 41 for ix in range(46)], index=ids),
    }
    for alpha, component in ((0.0, graph), (1.0, temporal)):
        fused = phase10._fusion_result(temporal, graph, alpha, 0.10)
        phase10._endpoint_check(fused, component, f"alpha={alpha}")
    midpoint = phase10._fusion_result(temporal, graph, 0.5, 0.10)
    assert int(midpoint["candidates"].sum()) == 5
    expected_midpoint = 0.5 * phase10.normalize_scores(temporal_scores) + 0.5 * phase10.normalize_scores(graph_scores)
    pd.testing.assert_series_equal(midpoint["scores"], expected_midpoint)


def test_alpha_one_keeps_native_temporal_order_for_tied_scores():
    ids = [f"event-{ix:02d}" for ix in reversed(range(46))]
    tied = pd.Series(np.ones(46), index=ids, dtype=float)
    temporal = {
        "scores": tied,
        "rank": phase10._rank(tied, ids, ids),
        "candidates": pd.Series([ix < 5 for ix in range(46)], index=ids, dtype=bool),
    }
    graph_scores = pd.Series(np.arange(46, dtype=float), index=ids)
    graph = {
        "scores": graph_scores,
        "rank": phase10._rank(graph_scores, ids, ids),
        "candidates": pd.Series([ix < 5 for ix in range(46)], index=ids, dtype=bool),
    }
    result = phase10._fusion_result(temporal, graph, 1.0, 0.10)
    assert phase10._ranked_ids(result) == ids
    assert phase10._ranked_ids(result) != sorted(ids)
    phase10._endpoint_check(result, temporal, "alpha=1 native temporal tie order")


def test_pair_metrics_use_same_46_ids_and_report_proxy_measures_only():
    ids = [f"event-{ix:02d}" for ix in range(46)]
    left_scores = pd.Series(np.arange(46, dtype=float), index=ids)
    left = {"scores": left_scores, "rank": phase10._rank(left_scores, ids), "candidates": pd.Series([ix >= 41 for ix in range(46)], index=ids)}
    right_scores = left_scores.copy()
    right = {"scores": right_scores, "rank": phase10._rank(right_scores, ids), "candidates": left["candidates"].copy()}
    for k in (3, 5, 10):
        result = phase10._metric_pair(left, right, k)
        assert result["cohort_size"] == 46
        assert result["top_k_overlap_count"] == k
        assert result["top_k_jaccard"] == 1.0
        assert result["kendall_tau"] == 1.0
        assert result["candidate_overlap_count"] == 5
        assert result["candidate_overlap_event_ids"] == ";".join(ids[-5:])
        assert "accuracy" not in result


def test_generated_phase10_artifacts_and_read_only_hash_manifest():
    expected_names = {
        "phase10_sensitivity_rankings.csv",
        "phase10_sensitivity_comparisons.csv",
        "phase10_artifact_manifest.json",
        "phase10_summary.json",
        "phase10_report.txt",
    }
    assert {path.name for path in (phase10.RANKINGS_PATH, phase10.COMPARISONS_PATH, phase10.MANIFEST_PATH, phase10.SUMMARY_PATH, phase10.REPORT_PATH)} == expected_names
    assert all(path.is_file() for path in (phase10.RANKINGS_PATH, phase10.COMPARISONS_PATH, phase10.MANIFEST_PATH, phase10.SUMMARY_PATH, phase10.REPORT_PATH))
    manifest = json.loads(phase10.MANIFEST_PATH.read_text(encoding="utf-8"))
    summary = json.loads(phase10.SUMMARY_PATH.read_text(encoding="utf-8"))
    report = phase10.REPORT_PATH.read_text(encoding="utf-8")
    assert manifest["protected_artifacts_byte_stable"] is True
    assert manifest["protected_artifact_sha256_before"] == manifest["protected_artifact_sha256_after"]
    assert manifest["deterministic_repeat_checks_passed"] is True
    assert manifest["cohort_size"] == 46
    assert manifest["logical_event_id_definition"] == "logical_event_id is the unique event identity used for all cohort alignment and ranking comparisons"
    assert manifest["excluded_boundary_event"] == "LEVT-M57-Jean-PROCESS-812-20091121013230"
    assert manifest["alpha_grid"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert manifest["estimator_settings"]["n_estimators_grid"] == [250, 500, 1000]
    assert manifest["estimator_settings"]["contamination_grid"] == [0.05, 0.1, 0.15]
    assert manifest["estimator_settings"]["random_state_grid"] == [0, 42, 123]
    assert manifest["scikit_learn_version"] == summary["environment"]["scikit_learn"]
    for artifact_name, digest in manifest["protected_artifact_sha256_after"].items():
        assert phase10._sha256(phase10.phase9.PROJECT_ROOT / artifact_name) == digest
    assert summary["cohort"]["size"] == 46
    assert summary["statistical_interpretation"]["supervised_metrics_calculated"] is False
    assert len(summary["determinism"]["checks"]) == 7
    assert all(check["comparison_metrics_exact_at_k_3_5_10"] and check["cohort_ids_exact"] for check in summary["determinism"]["checks"])
    for required_disclosure in ("events_next_*", "complete reconstructed graph", "full 46-event cohort", "retrospective/post-mortem", "no causal, online, real-time, streaming"):
        assert any(required_disclosure in item for item in summary["retrospective_leakage_disclosures"])
        assert required_disclosure in report
    for family in ("E1", "E2", "E3", "E4", "LegacyTemporal5", "LegacyGraphOnly9", "LegacyGraphAware14"):
        audit = summary["contamination_audit"][family]
        assert audit["contamination_005"]["candidate_count"] == 3
        assert audit["contamination_015"]["candidate_count"] == 7
        assert audit["contamination_005"]["score_order_unchanged_from_baseline"] is True
        assert audit["contamination_015"]["score_order_unchanged_from_baseline"] is True
        assert audit["contamination_005"]["absolute_decision_scores_unchanged_from_baseline"] is False
        assert audit["contamination_015"]["absolute_decision_scores_unchanged_from_baseline"] is False
        for condition in ("contamination_005", "contamination_015"):
            assert audit[condition]["candidate_overlap_count"] == len(audit[condition]["candidate_overlap_event_ids"])
            assert 0.0 <= audit[condition]["candidate_jaccard"] <= 1.0
    rankings = pd.read_csv(phase10.RANKINGS_PATH, low_memory=False)
    comparisons = pd.read_csv(phase10.COMPARISONS_PATH, low_memory=False)
    assert set(rankings["track"]) == {"phase9", "legacy"}
    assert set(comparisons["track"]) == {"phase9", "legacy"}
    family_track = rankings.groupby("model_family")["track"].agg(lambda values: set(values))
    for family in ("E1", "E2", "E3", "E4", "E5", "E6"):
        assert family_track[family] == {"phase9"}
    for family in ("LegacyTemporal5", "LegacyGraphOnly9", "LegacyGraphAware14"):
        assert family_track[family] == {"legacy"}
    prohibited = {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"}
    assert not prohibited.intersection(column.lower() for column in rankings.columns)
    assert not prohibited.intersection(column.lower() for column in comparisons.columns)
    assert "stable" in report and "sensitive" in report and "mixed" in report
    per_cohort = rankings.groupby(["track", "model_family", "condition_id", "alpha"], dropna=False).size()
    assert (per_cohort == 46).all()

