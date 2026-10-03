"""Tests for ground-truth evaluation protocol and model comparison modules."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.dataset_label_mapping import (
    DATASET_REGISTRY,
    get_dataset_metadata,
    normalize_binary_labels,
    validate_label_alignment,
)
from src.evaluation.evaluation_protocol import (
    compute_supervised_metrics,
    compute_unsupervised_ranking_metrics,
    enforce_same_test_cohort,
)
from src.evaluation.ground_truth_evaluation import run_ground_truth_evaluation
from src.evaluation.model_comparison import load_and_compare_models


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "M57-Jean"


def test_dataset_metadata_registry():
    """Verify metadata retrieval for supported datasets."""
    m57 = get_dataset_metadata("M57-Jean")
    assert not m57.has_ground_truth_labels
    assert m57.mappable_to_logical_event_id
    assert not m57.suitable_for_supervised_evaluation

    malmem = get_dataset_metadata("MalMem2022")
    assert malmem.has_ground_truth_labels
    assert not malmem.mappable_to_logical_event_id
    assert not malmem.suitable_for_supervised_evaluation

    optc = get_dataset_metadata("OpTC")
    assert optc.has_ground_truth_labels
    assert optc.mappable_to_logical_event_id
    assert not optc.suitable_for_supervised_evaluation
    assert optc.ground_truth_doc_available
    assert not optc.telemetry_ingested
    assert not optc.supervised_evaluation_ready

    with pytest.raises(ValueError, match="Unknown dataset"):
        get_dataset_metadata("InvalidDataset")


def test_label_alignment_success():
    """Test successful label alignment with unique keys."""
    event_ids = pd.Series(["E1", "E2", "E3"])
    labels = pd.Series([0, 1, 0])
    e_out, l_out = validate_label_alignment(event_ids, labels)
    assert len(e_out) == 3
    assert len(l_out) == 3


def test_missing_labels():
    """Test label alignment failure when missing or NaN values are present."""
    event_ids = pd.Series(["E1", "E2", None])
    labels = pd.Series([0, 1, 0])
    with pytest.raises(ValueError, match="Null values found in logical_event_id"):
        validate_label_alignment(event_ids, labels)

    event_ids2 = pd.Series(["E1", "E2", "E3"])
    labels2 = pd.Series([0, 1, None])
    with pytest.raises(ValueError, match="Null values found in labels series"):
        validate_label_alignment(event_ids2, labels2)


def test_duplicate_labels():
    """Test label alignment failure when duplicate keys exist."""
    event_ids = pd.Series(["E1", "E2", "E1"])
    labels = pd.Series([0, 1, 0])
    with pytest.raises(ValueError, match="Duplicate logical_event_id keys found"):
        validate_label_alignment(event_ids, labels)


def test_same_cohort_enforcement():
    """Test enforcement of identical test cohort across multiple model frames."""
    df1 = pd.DataFrame({"logical_event_id": ["E1", "E2", "E3"], "score": [0.1, 0.2, 0.3]})
    df2 = pd.DataFrame({"logical_event_id": ["E2", "E3", "E4"], "score": [0.5, 0.6, 0.7]})
    df3 = pd.DataFrame({"logical_event_id": ["E1", "E2", "E3", "E4"], "score": [0.9, 0.8, 0.7, 0.6]})

    aligned, common_keys, excluded = enforce_same_test_cohort(
        model_frames={"M1": df1, "M2": df2, "M3": df3},
        join_key="logical_event_id",
    )

    assert common_keys == {"E2", "E3"}
    assert excluded["M1"] == {"E1"}
    assert excluded["M2"] == {"E4"}
    assert excluded["M3"] == {"E1", "E4"}
    assert len(aligned["M1"]) == 2
    assert len(aligned["M2"]) == 2
    assert len(aligned["M3"]) == 2
    assert list(aligned["M1"]["logical_event_id"]) == ["E2", "E3"]


def test_key_mismatches_raises():
    """Test error handling when join keys are missing or unaligned."""
    df1 = pd.DataFrame({"wrong_key": ["E1"], "score": [0.1]})
    df2 = pd.DataFrame({"logical_event_id": ["E1"], "score": [0.2]})
    with pytest.raises(ValueError, match="missing join key"):
        enforce_same_test_cohort({"M1": df1, "M2": df2}, join_key="logical_event_id")


def test_metric_calculations():
    """Test calculation of supervised metrics on known ground truth."""
    y_true = pd.Series([1, 1, 0, 0, 1])
    y_pred = pd.Series([1, 0, 0, 1, 1])
    y_scores = pd.Series([0.9, 0.4, 0.1, 0.8, 0.95])

    metrics = compute_supervised_metrics(y_true, y_pred, y_scores=y_scores, k=2)

    assert metrics.cohort_size == 5
    assert metrics.n_positives == 3
    assert metrics.n_negatives == 2
    assert metrics.tp == 2
    assert metrics.fp == 1
    assert metrics.tn == 1
    assert metrics.fn == 1
    assert pytest.approx(metrics.precision) == 2 / 3
    assert pytest.approx(metrics.recall) == 2 / 3
    assert metrics.roc_auc is not None and metrics.roc_auc > 0.5
    assert metrics.precision_at_k == 1.0  # Top 2 scores are 0.95 and 0.9, both y_true=1


def test_empty_positive_class():
    """Test metric evaluation when ground truth has zero positives."""
    y_true = pd.Series([0, 0, 0, 0])
    y_pred = pd.Series([0, 1, 0, 0])
    y_scores = pd.Series([0.1, 0.9, 0.2, 0.3])

    metrics = compute_supervised_metrics(y_true, y_pred, y_scores=y_scores)

    assert metrics.n_positives == 0
    assert metrics.n_negatives == 4
    assert metrics.tp == 0
    assert metrics.fp == 1
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0


def test_empty_negative_class():
    """Test metric evaluation when ground truth has zero negatives."""
    y_true = pd.Series([1, 1, 1, 1])
    y_pred = pd.Series([1, 1, 0, 1])
    y_scores = pd.Series([0.8, 0.7, 0.2, 0.9])

    metrics = compute_supervised_metrics(y_true, y_pred, y_scores=y_scores)

    assert metrics.n_positives == 4
    assert metrics.n_negatives == 0
    assert metrics.tp == 3
    assert metrics.fn == 1
    assert metrics.precision == 1.0
    assert metrics.recall == 0.75


def test_deterministic_ground_truth_evaluation():
    """Test that ground-truth evaluation runs deterministically."""
    summary1 = run_ground_truth_evaluation("M57-Jean")
    summary2 = run_ground_truth_evaluation("M57-Jean")

    json1 = json.dumps(summary1, sort_keys=True)
    json2 = json.dumps(summary2, sort_keys=True)

    assert json1 == json2


def test_no_accidental_modification_of_existing_outputs():
    """Verify that existing M57-Jean fusion artifacts are unchanged."""
    fused_csv = RESULTS_DIR / "fused_anomalies.csv"
    fusion_report = RESULTS_DIR / "fusion_report.txt"

    assert fused_csv.exists()
    assert fusion_report.exists()

    hash_before_csv = hashlib.sha256(fused_csv.read_bytes()).hexdigest()
    hash_before_txt = hashlib.sha256(fusion_report.read_bytes()).hexdigest()

    # Run evaluation protocol
    run_ground_truth_evaluation("M57-Jean")

    hash_after_csv = hashlib.sha256(fused_csv.read_bytes()).hexdigest()
    hash_after_txt = hashlib.sha256(fusion_report.read_bytes()).hexdigest()

    assert hash_before_csv == hash_after_csv, "fused_anomalies.csv was modified!"
    assert hash_before_txt == hash_after_txt, "fusion_report.txt was modified!"


def test_four_model_comparison():
    """Verify that model comparison aligns all four model variants on the exact same cohort."""
    temporal_path = RESULTS_DIR / "temporal_anomalies.csv"
    graph_path = RESULTS_DIR / "graph_anomalies.csv"
    fused_path = RESULTS_DIR / "fused_anomalies.csv"

    comp_df, summary_df, report_text = load_and_compare_models(
        temporal_path=temporal_path,
        graph_path=graph_path,
        fused_path=fused_path,
        dataset_name="M57-Jean",
    )

    models_in_summary = summary_df["model_name"].unique().tolist()
    assert "Temporal-only" in models_in_summary
    assert "Graph-only" in models_in_summary
    assert "Graph-aware" in models_in_summary
    assert "Fused (alpha=0.50)" in models_in_summary
    assert len(models_in_summary) == 4

    # Verify that all 4 models have the exact same cohort size
    cohort_sizes = summary_df.groupby("model_name").size().to_dict()
    assert cohort_sizes["Temporal-only"] == 46
    assert cohort_sizes["Graph-only"] == 46
    assert cohort_sizes["Graph-aware"] == 46
    assert cohort_sizes["Fused (alpha=0.50)"] == 46

    # Verify that Graph-aware vs Fused (alpha=0.50) Kendall Tau is approximately 0.5652
    ga_fused_row = comp_df[comp_df["model_pair"] == "Graph-aware vs Fused (alpha=0.50)"]
    assert not ga_fused_row.empty
    tau = ga_fused_row["kendall_tau"].values[0]
    assert pytest.approx(tau, abs=1e-3) == 0.5652


def test_explicit_score_column_resolution_regression():
    """Regression test proving explicit score-column mapping prevents column collision."""
    from src.evaluation.model_comparison import _find_score_col

    fused_path = RESULTS_DIR / "fused_anomalies.csv"
    fused_df = pd.read_csv(fused_path)

    ga_col = _find_score_col("Graph-aware", list(fused_df.columns))
    fused_col = _find_score_col("Fused (alpha=0.50)", list(fused_df.columns))

    assert ga_col == "controlled_graph_aware_score"
    assert fused_col == "fusion_score_alpha_0_50"
    assert ga_col != fused_col

    # Verify actual score values are non-identical
    ga_scores = fused_df[ga_col].to_numpy()
    fused_scores = fused_df[fused_col].to_numpy()
    assert not np.array_equal(ga_scores, fused_scores)


def test_retrospective_disclosure_presence():
    """Verify that generated report text contains the retrospective scope disclosure section."""
    run_ground_truth_evaluation("M57-Jean")
    report_file = RESULTS_DIR / "ground_truth_evaluation_report.txt"

    assert report_file.exists()
    report_content = report_file.read_text(encoding="utf-8")

    assert "Retrospective Scope & Data Leakage Audit" in report_content
    assert "events_next_*" in report_content
    assert "causal detection" in report_content
    assert "online detection" in report_content
    assert "real-time detection" in report_content


def test_missing_model_file_handling():
    """Verify clean FileNotFoundError when a required model file is missing."""
    with pytest.raises(FileNotFoundError, match="model output file missing"):
        load_and_compare_models(
            temporal_path=Path("non_existent_temporal.csv"),
            graph_path=RESULTS_DIR / "graph_anomalies.csv",
            fused_path=RESULTS_DIR / "fused_anomalies.csv",
            dataset_name="M57-Jean",
        )
