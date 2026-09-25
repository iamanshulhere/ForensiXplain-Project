from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results" / "M57-Jean"

ATTRIBUTION_FILE = RESULTS_DIR / "temporal_evidence_attribution.csv"
SHAP_FILE = RESULTS_DIR / "temporal_shap_explanations.csv"
OUTPUT_CSV = RESULTS_DIR / "temporal_investigator_explanations.csv"
OUTPUT_REPORT = RESULTS_DIR / "temporal_investigator_report.txt"

SCRIPT = (
    PROJECT_ROOT
    / "src"
    / "explainability"
    / "temporal_explanation_generator.py"
)


@pytest.fixture(scope="module")
def explanation_output():
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        "Temporal explanation generator failed.\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )

    assert OUTPUT_CSV.exists()
    assert OUTPUT_REPORT.exists()

    return pd.read_csv(OUTPUT_CSV)


def test_temporal_explanation_input_files_exist():
    assert ATTRIBUTION_FILE.exists()
    assert SHAP_FILE.exists()


def test_temporal_explanation_output_files_exist(explanation_output):
    assert OUTPUT_CSV.exists()
    assert OUTPUT_REPORT.exists()


def test_temporal_explanation_row_count(explanation_output):
    assert len(explanation_output) == 5


def test_temporal_explanation_required_columns(explanation_output):
    required_columns = {
        "logical_event_id",
        "temporal_anomaly_rank",
        "temporal_anomaly_score",
        "temporal_sequence",
        "timestamp",
        "process_id",
        "process",
        "previous_process_id",
        "previous_process",
        "process_transition",
        "time_since_previous_event_seconds",
        "local_density_10s",
        "local_density_30s",
        "local_density_60s",
        "process_changed",
        "shap_explanation",
        "evidence_summary",
        "evidence_ids",
        "relationship_summary",
        "assessment",
        "limitation",
    }

    assert required_columns.issubset(
        set(explanation_output.columns)
    )


def test_temporal_anomaly_ranks_are_sequential(explanation_output):
    assert explanation_output["temporal_anomaly_rank"].tolist() == [
        1, 2, 3, 4, 5
    ]


def test_temporal_anomaly_scores_are_descending(explanation_output):
    scores = explanation_output["temporal_anomaly_score"].tolist()

    assert all(
        scores[i] >= scores[i + 1]
        for i in range(len(scores) - 1)
    )


def test_explanations_contain_shap_evidence_and_relationships(
    explanation_output,
):
    for _, row in explanation_output.iterrows():
        assert str(row["shap_explanation"]).strip()
        assert str(row["evidence_summary"]).strip()
        assert str(row["evidence_ids"]).strip()
        assert str(row["relationship_summary"]).strip()


def test_explanations_contain_assessment_and_limitation(
    explanation_output,
):
    for _, row in explanation_output.iterrows():
        assert str(row["assessment"]).strip()
        assert str(row["limitation"]).strip()

        assert "maliciousness" in row["limitation"]
        assert "causality" in row["limitation"]


def test_top_temporal_explanation_is_deterministic(explanation_output):
    top = explanation_output.iloc[0]

    assert top["temporal_anomaly_rank"] == 1
    assert top["process_id"] == 3992
    assert top["process"] == "iexplore.exe"
    assert top["temporal_sequence"] == 36

    assert top["temporal_anomaly_score"] == pytest.approx(
        0.066266,
        abs=1e-6,
    )


def test_temporal_report_contains_required_sections(
    explanation_output,
):
    report = OUTPUT_REPORT.read_text(encoding="utf-8")

    required_sections = [
        "ForensiXplain - Temporal Investigator Report",
        "Purpose:",
        "Important interpretation:",
        "Temporal anomaly does not mean malicious activity.",
        "Chronological adjacency is not treated as causality.",
        "Temporal Context:",
        "SHAP Explanation:",
        "Evidence Summary:",
        "Process Relationships:",
        "Evidence IDs:",
        "Assessment:",
        "Limitation:",
        "Total temporal anomalies: 5",
    ]

    for section in required_sections:
        assert section in report