import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "M57-Jean"
EXPLAINABILITY_DIR = PROJECT_ROOT / "src" / "explainability"

GRAPH_SHAP_FILE = RESULTS_DIR / "graph_shap_explanations.csv"
GRAPH_ATTRIBUTION_FILE = RESULTS_DIR / "graph_evidence_attribution.csv"
GRAPH_EXPLANATION_FILE = RESULTS_DIR / "graph_investigator_explanations.csv"
GRAPH_REPORT_FILE = RESULTS_DIR / "graph_investigator_report.txt"

GRAPH_SHAP_SCRIPT = EXPLAINABILITY_DIR / "graph_shap.py"
GRAPH_ATTRIBUTION_SCRIPT = EXPLAINABILITY_DIR / "graph_evidence_attribution.py"
GRAPH_EXPLANATION_SCRIPT = EXPLAINABILITY_DIR / "graph_explanation_generator.py"


def run_script(script_path):
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"{script_path.name} failed.\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )


@pytest.fixture(scope="module")
def graph_shap_output():
    run_script(GRAPH_SHAP_SCRIPT)
    assert GRAPH_SHAP_FILE.exists()
    return pd.read_csv(GRAPH_SHAP_FILE)


@pytest.fixture(scope="module")
def graph_attribution_output():
    run_script(GRAPH_ATTRIBUTION_SCRIPT)
    assert GRAPH_ATTRIBUTION_FILE.exists()
    return pd.read_csv(GRAPH_ATTRIBUTION_FILE)


@pytest.fixture(scope="module")
def graph_explanation_output():
    run_script(GRAPH_EXPLANATION_SCRIPT)
    assert GRAPH_EXPLANATION_FILE.exists()
    return pd.read_csv(GRAPH_EXPLANATION_FILE)


def test_graph_shap_input_script_exists():
    assert GRAPH_SHAP_SCRIPT.exists()


def test_graph_shap_output_has_all_events(graph_shap_output):
    assert len(graph_shap_output) == 47


def test_graph_shap_output_contains_required_columns(graph_shap_output):
    required_columns = {
        "case_id",
        "logical_event_id",
        "temporal_sequence",
        "timestamp",
        "process_id",
        "process",
        "graph_anomaly_score",
        "graph_predicted_anomaly",
        "graph_anomaly_rank",
        "shap_parent_count",
        "shap_child_count",
        "shap_graph_degree",
        "shap_in_degree",
        "shap_out_degree",
        "shap_command_line_count",
        "shap_module_count",
        "shap_memory_region_count",
        "shap_relationship_type_count",
    }

    assert required_columns.issubset(graph_shap_output.columns)


def test_graph_shap_values_are_present(graph_shap_output):
    shap_columns = [
        column
        for column in graph_shap_output.columns
        if column.startswith("shap_")
    ]

    assert shap_columns

    for column in shap_columns:
        assert graph_shap_output[column].notna().all()


def test_graph_shap_anomaly_ranking_is_valid(graph_shap_output):
    ranks = sorted(graph_shap_output["graph_anomaly_rank"].tolist())

    assert ranks == list(range(1, len(graph_shap_output) + 1))


def test_graph_shap_scores_match_graph_anomalies(graph_shap_output):
    graph_anomalies = pd.read_csv(
        RESULTS_DIR / "graph_anomalies.csv"
    )

    merged = graph_shap_output.merge(
        graph_anomalies[
            [
                "logical_event_id",
                "graph_anomaly_score",
                "graph_anomaly_rank",
            ]
        ],
        on="logical_event_id",
        suffixes=("_shap", "_original"),
    )

    assert len(merged) == 47

    assert merged["graph_anomaly_score_shap"].tolist() == pytest.approx(
        merged["graph_anomaly_score_original"].tolist()
    )

    assert (
        merged["graph_anomaly_rank_shap"]
        == merged["graph_anomaly_rank_original"]
    ).all()


def test_graph_attribution_output_has_five_rows(graph_attribution_output):
    assert len(graph_attribution_output) == 5


def test_graph_attribution_contains_required_columns(
    graph_attribution_output,
):
    required_columns = {
        "case_id",
        "logical_event_id",
        "temporal_sequence",
        "timestamp",
        "process_id",
        "process",
        "graph_anomaly_score",
        "graph_anomaly_rank",
        "top_shap_features",
        "top_shap_values",
        "top_feature_values",
        "parent_process_ids",
        "child_process_ids",
        "evidence_ids",
        "timeline_evidence_ids",
        "provenance",
        "command_lines",
        "assessment",
        "limitation",
    }

    assert required_columns.issubset(graph_attribution_output.columns)


def test_graph_attribution_ranks_are_valid(graph_attribution_output):
    ranks = sorted(
        graph_attribution_output["graph_anomaly_rank"].tolist()
    )

    assert ranks == [1, 2, 3, 4, 5]


def test_graph_attribution_contains_evidence(
    graph_attribution_output,
):
    assert (
        graph_attribution_output["evidence_ids"]
        .fillna("")
        .str.len()
        .gt(0)
        .all()
    )

    assert (
        graph_attribution_output["timeline_evidence_ids"]
        .fillna("")
        .str.len()
        .gt(0)
        .all()
    )


def test_graph_attribution_contains_assessment_and_limitation(
    graph_attribution_output,
):
    assert (
        graph_attribution_output["assessment"]
        .fillna("")
        .str.len()
        .gt(0)
        .all()
    )

    assert (
        graph_attribution_output["limitation"]
        .fillna("")
        .str.len()
        .gt(0)
        .all()
    )


def test_graph_explanation_output_has_five_rows(graph_explanation_output):
    assert len(graph_explanation_output) == 5


def test_graph_explanation_contains_required_columns(
    graph_explanation_output,
):
    required_columns = {
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
        "evidence_ids",
        "timeline_evidence_ids",
        "provenance",
        "command_lines",
        "assessment",
        "limitation",
        "investigator_explanation",
    }

    assert required_columns.issubset(graph_explanation_output.columns)


def test_graph_explanation_ranks_are_valid(graph_explanation_output):
    ranks = sorted(
        graph_explanation_output["graph_anomaly_rank"].tolist()
    )

    assert ranks == [1, 2, 3, 4, 5]


def test_graph_explanation_scores_are_descending(
    graph_explanation_output,
):
    scores = graph_explanation_output[
        "graph_anomaly_score"
    ].tolist()

    assert scores == sorted(scores, reverse=True)


def test_graph_explanation_text_is_present(graph_explanation_output):
    assert (
        graph_explanation_output["investigator_explanation"]
        .fillna("")
        .str.len()
        .gt(0)
        .all()
    )


def test_graph_explanation_preserves_evidence(
    graph_explanation_output,
):
    assert (
        graph_explanation_output["evidence_ids"]
        .fillna("")
        .str.len()
        .gt(0)
        .all()
    )

    assert (
        graph_explanation_output["timeline_evidence_ids"]
        .fillna("")
        .str.len()
        .gt(0)
        .all()
    )


def test_graph_explanation_contains_assessment_and_limitation(
    graph_explanation_output,
):
    assert (
        graph_explanation_output["assessment"]
        .fillna("")
        .str.len()
        .gt(0)
        .all()
    )

    assert (
        graph_explanation_output["limitation"]
        .fillna("")
        .str.len()
        .gt(0)
        .all()
    )


def test_graph_investigator_report_exists():
    assert GRAPH_REPORT_FILE.exists()


def test_graph_investigator_report_contains_required_sections():
    report = GRAPH_REPORT_FILE.read_text(encoding="utf-8")

    required_sections = [
        "Graph Investigator Explanation Report",
        "ASSESSMENT",
        "Evidence",
        "LIMITATION",
    ]

    for section in required_sections:
        assert section in report