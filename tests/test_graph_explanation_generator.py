import pandas as pd
import pytest

from src.explainability import graph_explanation_generator as geg


FEATURES = [
    "gap_log_seconds",
    "local_density_10s",
    "local_density_30s",
    "local_density_60s",
    "process_changed",
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


def make_row(
    logical_event_id="LEVT-1",
    rank=1,
    process_id=100,
    process="test.exe",
    timestamp="2026-01-01T10:00:00Z",
    score=0.8,
    time_gap=12.5,
):
    row = {
        "case_id": "M57-Jean",
        "logical_event_id": logical_event_id,
        "graph_anomaly_rank": rank,
        "process_id": process_id,
        "process": process,
        "timestamp": timestamp,
        "graph_anomaly_score": score,
        "top_shap_features": "out_degree;graph_degree;module_count",
        "top_shap_values": "-0.500000;-0.400000;-0.300000",
        "top_feature_values": "5.000;7.000;3.000",
        "parent_process_ids": "50",
        "child_process_ids": "75",
        "previous_process_id": "90",
        "previous_process": "parent.exe",
        "time_since_previous_event_seconds": time_gap,
        "raw_event_count": "5",
        "artifact_type_counts": "pstree=2;pslist=1;cmdline=1",
        "event_type_counts": "process=1;relationship=2;command_line=1;malfind=1",
        "evidence_by_artifact": "pstree=2;pslist=1;cmdline=1;malfind=1",
        "evidence_ids": "REL-1;REL-2;PROC-1;CMD-1;MAL-1",
        "timeline_evidence_ids": "TL-1",
        "provenance": "M57-Jean|memory",
        "command_lines": "test.exe --sample",
    }

    shap_values = {
        "gap_log_seconds": 0.10,
        "local_density_10s": 0.05,
        "local_density_30s": 0.04,
        "local_density_60s": 0.03,
        "process_changed": 0.02,
        "parent_count": 0.20,
        "child_count": 0.15,
        "graph_degree": -0.40,
        "in_degree": 0.10,
        "out_degree": -0.50,
        "command_line_count": 0.12,
        "module_count": -0.30,
        "memory_region_count": 0.08,
        "relationship_type_count": 0.07,
    }

    feature_values = {
        "gap_log_seconds": 2.5,
        "local_density_10s": 1,
        "local_density_30s": 2,
        "local_density_60s": 3,
        "process_changed": 1,
        "parent_count": 1,
        "child_count": 1,
        "graph_degree": 7,
        "in_degree": 2,
        "out_degree": 5,
        "command_line_count": 1,
        "module_count": 3,
        "memory_region_count": 2,
        "relationship_type_count": 4,
    }

    for feature in FEATURES:
        row[f"value_{feature}"] = feature_values[feature]
        row[f"shap_{feature}"] = shap_values[feature]

    return row


@pytest.fixture
def configure_paths(tmp_path, monkeypatch):
    input_file = tmp_path / "graph_evidence_attribution.csv"
    results_dir = tmp_path / "results"
    csv_output = results_dir / "graph_investigator_explanations.csv"
    report_output = results_dir / "graph_investigator_report.txt"

    monkeypatch.setattr(geg, "INPUT_FILE", input_file)
    monkeypatch.setattr(geg, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(geg, "CSV_OUTPUT", csv_output)
    monkeypatch.setattr(geg, "REPORT_OUTPUT", report_output)

    return input_file, csv_output, report_output


def test_format_number():
    assert geg.format_number(12.34567, 3) == "12.346"
    assert geg.format_number("1.23456", 2) == "1.23"
    assert geg.format_number("invalid") == "unknown"


def test_clean_text_and_format_list():
    assert geg.clean_text("  hello  ") == "hello"
    assert geg.clean_text(float("nan")) == ""

    assert geg.format_list("  a;b  ") == "a;b"
    assert geg.format_list("") == "none observed"
    assert geg.format_list(" ", "missing") == "missing"


def test_feature_explanation_positive_contribution():
    text = geg.feature_explanation(
        "out_degree",
        5,
        0.25,
    )

    assert "out_degree=5.000" in text
    assert "positive contribution to the tree-model output" in text
    assert "SHAP=+0.250000" in text
    assert "number of outgoing graph relationships" in text


def test_feature_explanation_negative_contribution():
    text = geg.feature_explanation(
        "graph_degree",
        7,
        -0.4,
    )

    assert "graph_degree=7.000" in text
    assert "negative contribution to the tree-model output" in text
    assert "SHAP=-0.400000" in text
    assert "total graph connectivity of the process node" in text


def test_main_requires_input_file(configure_paths):
    input_file, _, _ = configure_paths

    with pytest.raises(FileNotFoundError):
        geg.main()

    assert not input_file.exists()


def test_main_rejects_empty_attribution_file(configure_paths):
    input_file, _, _ = configure_paths
    pd.DataFrame(columns=["case_id"]).to_csv(input_file, index=False)

    with pytest.raises(ValueError, match="No graph evidence attribution records found"):
        geg.main()


def test_main_creates_csv_and_report(configure_paths):
    input_file, csv_output, report_output = configure_paths

    pd.DataFrame([make_row()]).to_csv(input_file, index=False)

    geg.main()

    assert csv_output.exists()
    assert report_output.exists()


def test_main_sorts_records_by_anomaly_rank(configure_paths):
    input_file, csv_output, _ = configure_paths

    rows = [
        make_row(
            logical_event_id="LEVT-2",
            rank=2,
            process_id=200,
            process="second.exe",
            score=0.4,
        ),
        make_row(
            logical_event_id="LEVT-1",
            rank=1,
            process_id=100,
            process="first.exe",
            score=0.8,
        ),
    ]

    pd.DataFrame(rows).to_csv(input_file, index=False)

    geg.main()

    result = pd.read_csv(csv_output)

    assert list(result["logical_event_id"]) == ["LEVT-1", "LEVT-2"]
    assert list(result["graph_anomaly_rank"]) == [1, 2]


def test_output_contains_expected_columns(configure_paths):
    input_file, csv_output, _ = configure_paths

    pd.DataFrame([make_row()]).to_csv(input_file, index=False)

    geg.main()

    result = pd.read_csv(csv_output)

    expected_columns = {
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
    }

    assert set(result.columns) == expected_columns
    assert len(result.columns) == 26


def test_main_preserves_core_attribution_fields(configure_paths):
    input_file, csv_output, _ = configure_paths

    pd.DataFrame([make_row()]).to_csv(input_file, index=False)

    geg.main()

    row = pd.read_csv(csv_output).iloc[0]

    assert row["case_id"] == "M57-Jean"
    assert row["logical_event_id"] == "LEVT-1"
    assert str(row["graph_anomaly_rank"]) == "1"
    assert row["process_id"] == 100
    assert row["process"] == "test.exe"
    assert row["timestamp"] == "2026-01-01T10:00:00Z"
    assert row["graph_anomaly_score"] == pytest.approx(0.8)


def test_top_three_feature_contributors_are_sorted_by_absolute_shap(
    configure_paths,
):
    input_file, csv_output, _ = configure_paths

    pd.DataFrame([make_row()]).to_csv(input_file, index=False)

    geg.main()

    row = pd.read_csv(csv_output).iloc[0]
    explanation = row["investigator_explanation"]

    out_pos = explanation.index("out_degree=5.000")
    graph_pos = explanation.index("graph_degree=7.000")
    module_pos = explanation.index("module_count=3.000")

    assert out_pos < graph_pos < module_pos
    assert "parent_count=1.000" not in explanation
    assert "child_count=1.000" not in explanation


def test_positive_and_negative_shap_directions_are_reported(
    configure_paths,
):
    input_file, csv_output, _ = configure_paths

    pd.DataFrame([make_row()]).to_csv(input_file, index=False)

    geg.main()

    row = pd.read_csv(csv_output).iloc[0]
    explanation = row["investigator_explanation"]

    assert "negative contribution to the tree-model output" in explanation


def test_time_gap_is_formatted(configure_paths):
    input_file, csv_output, _ = configure_paths

    pd.DataFrame([
        make_row(time_gap=12.5)
    ]).to_csv(input_file, index=False)

    geg.main()

    row = pd.read_csv(csv_output).iloc[0]
    explanation = row["investigator_explanation"]

    assert "with a time gap of 12.500 seconds" in explanation


def test_missing_time_gap_gets_safe_message(configure_paths):
    input_file, csv_output, _ = configure_paths

    pd.DataFrame([
        make_row(time_gap=None)
    ]).to_csv(input_file, index=False)

    geg.main()

    row = pd.read_csv(csv_output).iloc[0]
    explanation = row["investigator_explanation"]

    assert "No previous logical event available." in explanation


def test_evidence_and_command_line_context_are_preserved(configure_paths):
    input_file, csv_output, _ = configure_paths

    pd.DataFrame([make_row()]).to_csv(input_file, index=False)

    geg.main()

    row = pd.read_csv(csv_output).iloc[0]

    assert row["parent_process_ids"] == 50
    assert row["child_process_ids"] == 75
    assert row["raw_event_count"] == 5
    assert row["artifact_type_counts"] == "pstree=2;pslist=1;cmdline=1"
    assert row["evidence_ids"] == "REL-1;REL-2;PROC-1;CMD-1;MAL-1"
    assert row["timeline_evidence_ids"] == "TL-1"
    assert row["provenance"] == "M57-Jean|memory"
    assert row["command_lines"] == "test.exe --sample"


def test_assessment_and_limitation_are_present(configure_paths):
    input_file, csv_output, _ = configure_paths

    pd.DataFrame([make_row()]).to_csv(input_file, index=False)

    geg.main()

    row = pd.read_csv(csv_output).iloc[0]

    assert "Isolation Forest baseline" in row["assessment"]
    assert "SHAP results identify features" in row["assessment"]

    assert "do not establish malicious activity" in row["limitation"]
    assert "Investigator review" in row["limitation"]


def test_report_contains_required_sections(configure_paths):
    input_file, _, report_output = configure_paths

    pd.DataFrame([make_row()]).to_csv(input_file, index=False)

    geg.main()

    report = report_output.read_text(encoding="utf-8")

    assert "FORENSIXPLAIN" in report
    assert "Graph Investigator Explanation Report" in report
    assert "IMPORTANT INTERPRETATION NOTE" in report
    assert "GRAPH ANOMALY CANDIDATE #1" in report
    assert "TOP SHAP CONTRIBUTORS" in report
    assert "TEMPORAL CONTEXT" in report
    assert "GRAPH CONTEXT" in report
    assert "FORENSIC EVIDENCE" in report
    assert "COMMAND-LINE OBSERVATIONS" in report
    assert "ASSESSMENT" in report
    assert "LIMITATION" in report


def test_report_preserves_non_malicious_interpretation(configure_paths):
    input_file, _, report_output = configure_paths

    pd.DataFrame([make_row()]).to_csv(input_file, index=False)

    geg.main()

    report = report_output.read_text(encoding="utf-8")

    assert "statistical anomaly candidates" in report
    assert "feature contributions to the tree-model output" in report
    assert "Neither anomaly scores nor SHAP values establish malicious activity." in report
    assert "All candidates require investigator review" in report