import pandas as pd

from src.explainability.graph_explanation_generator import (
    FEATURE_DESCRIPTIONS,
    feature_explanation,
)


OUTPUT_FILE = "results/M57-Jean/graph_investigator_explanations.csv"
REPORT_FILE = "results/M57-Jean/graph_investigator_report.txt"


def test_graph_explanation_output_has_expected_rows():
    df = pd.read_csv(OUTPUT_FILE)

    assert len(df) == 5


def test_graph_explanation_output_has_expected_columns():
    df = pd.read_csv(OUTPUT_FILE)

    required_columns = [
        "case_id",
        "logical_event_id",
        "graph_anomaly_rank",
        "process_id",
        "process",
        "timestamp",
        "graph_anomaly_score",
        "top_shap_features",
        "top_shap_values",
        "top_feature_values",
        "parent_process_ids",
        "child_process_ids",
        "previous_process_id",
        "previous_process",
        "time_since_previous_event_seconds",
        "raw_event_count",
        "artifact_type_counts",
        "event_type_counts",
        "evidence_by_artifact",
        "evidence_ids",
        "timeline_evidence_ids",
        "provenance",
        "command_lines",
        "assessment",
        "limitation",
        "investigator_explanation",
    ]

    assert list(df.columns) == required_columns


def test_graph_explanation_ranks_are_sequential():
    df = pd.read_csv(OUTPUT_FILE)

    assert df["graph_anomaly_rank"].astype(int).tolist() == [1, 2, 3, 4, 5]


def test_graph_explanation_scores_are_descending():
    df = pd.read_csv(OUTPUT_FILE)

    scores = pd.to_numeric(df["graph_anomaly_score"])

    assert scores.is_monotonic_decreasing


def test_graph_explanation_logical_event_ids_are_unique():
    df = pd.read_csv(OUTPUT_FILE)

    assert df["logical_event_id"].is_unique


def test_graph_explanation_contains_shap_summary_fields():
    df = pd.read_csv(OUTPUT_FILE)

    assert df["top_shap_features"].notna().all()
    assert df["top_shap_values"].notna().all()
    assert df["top_feature_values"].notna().all()


def test_graph_explanation_contains_graph_and_forensic_context():
    df = pd.read_csv(OUTPUT_FILE)

    required_context = [
        "parent_process_ids",
        "child_process_ids",
        "raw_event_count",
        "artifact_type_counts",
        "event_type_counts",
        "evidence_ids",
        "provenance",
    ]

    for column in required_context:
        assert df[column].notna().all()


def test_graph_explanation_contains_investigator_explanation():
    df = pd.read_csv(OUTPUT_FILE)

    assert df["investigator_explanation"].notna().all()
    assert df["investigator_explanation"].str.len().gt(0).all()


def test_graph_explanation_preserves_limitation():
    df = pd.read_csv(OUTPUT_FILE)

    for limitation in df["limitation"]:
        assert "do not establish malicious activity" in limitation
        assert "Investigator review" in limitation


def test_graph_explanation_report_exists():
    with open(REPORT_FILE, encoding="utf-8") as file:
        report = file.read()

    assert "Graph Investigator Explanation Report" in report
    assert "IMPORTANT INTERPRETATION NOTE" in report
    assert "Neither anomaly scores nor SHAP values establish malicious activity." in report


def test_feature_explanation_positive_shap():
    result = feature_explanation(
        "graph_degree",
        200,
        0.5,
    )

    assert "graph_degree=200.000" in result
    assert "positive contribution" in result
    assert "SHAP=+0.500000" in result
    assert FEATURE_DESCRIPTIONS["graph_degree"] in result


def test_feature_explanation_negative_shap():
    result = feature_explanation(
        "out_degree",
        199,
        -0.25,
    )

    assert "out_degree=199.000" in result
    assert "negative contribution" in result
    assert "SHAP=-0.250000" in result


def test_feature_explanation_invalid_values():
    result = feature_explanation(
        "unknown_feature",
        "invalid",
        "invalid",
    )

    assert "unknown_feature=unknown" in result
    assert "negative contribution" not in result
    assert "positive contribution" in result
    assert "SHAP=+0.000000" in result