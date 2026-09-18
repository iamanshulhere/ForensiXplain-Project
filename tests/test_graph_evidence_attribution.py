import numpy as np
import pandas as pd
import pytest

from src.explainability import graph_evidence_attribution as gea


def make_graph_features():
    return pd.DataFrame(
        [
            {
                "logical_event_id": "LEVT-1",
                "process_id": 100,
                "process": "test.exe",
                "parent_count": 1,
                "child_count": 2,
                "graph_degree": 4,
                "in_degree": 1,
                "out_degree": 3,
                "command_line_count": 1,
                "module_count": 1,
                "memory_region_count": 1,
                "relationship_type_count": 3,
            },
            {
                "logical_event_id": "LEVT-2",
                "process_id": 200,
                "process": "normal.exe",
                "parent_count": 1,
                "child_count": 0,
                "graph_degree": 1,
                "in_degree": 1,
                "out_degree": 0,
                "command_line_count": 0,
                "module_count": 0,
                "memory_region_count": 0,
                "relationship_type_count": 1,
            },
        ]
    )


def make_raw_events():
    return pd.DataFrame(
        [
            {
                "process_id": 100,
                "parent_process_id": 50,
                "relationship": "parent_of",
                "evidence_id": "REL-1",
                "artifact_type": "pstree",
                "event_type": "process_relationship",
                "provenance": "pstree-source",
                "command_line": "-",
            },
            {
                "process_id": 100,
                "parent_process_id": 50,
                "relationship": "parent_of",
                "evidence_id": "REL-2",
                "artifact_type": "pstree",
                "event_type": "process_relationship",
                "provenance": "pstree-source",
                "command_line": "-",
            },
            {
                "process_id": 75,
                "parent_process_id": 100,
                "relationship": "parent_of",
                "evidence_id": "REL-3",
                "artifact_type": "pstree",
                "event_type": "process_relationship",
                "provenance": "pstree-child",
                "command_line": "-",
            },
            {
                "process_id": 100,
                "parent_process_id": np.nan,
                "relationship": "",
                "evidence_id": "PROC-1",
                "artifact_type": "pslist",
                "event_type": "process",
                "provenance": "pslist-source",
                "command_line": "-",
            },
            {
                "process_id": 100,
                "parent_process_id": np.nan,
                "relationship": "",
                "evidence_id": "CMD-1",
                "artifact_type": "cmdline",
                "event_type": "command_line",
                "provenance": "cmd-source",
                "command_line": "test.exe -arg",
            },
            {
                "process_id": 100,
                "parent_process_id": np.nan,
                "relationship": "",
                "evidence_id": "MAL-1",
                "artifact_type": "malfind",
                "event_type": "memory_region",
                "provenance": "malfind-source",
                "command_line": "-",
            },
            {
                "process_id": 200,
                "parent_process_id": 50,
                "relationship": "parent_of",
                "evidence_id": "REL-4",
                "artifact_type": "pstree",
                "event_type": "process_relationship",
                "provenance": "normal-source",
                "command_line": "-",
            },
        ]
    )


def make_logical_timeline():
    return pd.DataFrame(
        [
            {
                "logical_event_id": "LEVT-1",
                "process_id": 100,
                "timestamp": "2009-11-23 12:00:00+00:00",
                "previous_process_id": 90,
                "previous_process": "prev.exe",
                "time_since_previous_event_seconds": 12.5,
                "parent_process_ids": "50",
                "evidence_ids": "PROC-1;REL-1",
            },
            {
                "logical_event_id": "LEVT-2",
                "process_id": 200,
                "timestamp": "2009-11-23 12:01:00+00:00",
                "previous_process_id": 100,
                "previous_process": "test.exe",
                "time_since_previous_event_seconds": 60.0,
                "parent_process_ids": "50",
                "evidence_ids": "PROC-2;REL-4",
            },
        ]
    )


def make_shap():
    return pd.DataFrame(
        [
            {
                "case_id": "TEST",
                "logical_event_id": "LEVT-1",
                "temporal_sequence": 1,
                "process_id": 100,
                "process": "test.exe",
                "graph_anomaly_score": 0.8,
                "graph_predicted_anomaly": True,
                "graph_anomaly_rank": 1,
                "value_gap_log_seconds": 3.0,
                "shap_gap_log_seconds": 0.10,
                "value_local_density_10s": 2.0,
                "shap_local_density_10s": -0.20,
                "value_local_density_30s": 3.0,
                "shap_local_density_30s": 0.05,
                "value_local_density_60s": 4.0,
                "shap_local_density_60s": 0.01,
                "value_process_changed": 1.0,
                "shap_process_changed": 0.02,
                "value_parent_count": 1.0,
                "shap_parent_count": 0.03,
                "value_child_count": 2.0,
                "shap_child_count": 0.20,
                "value_graph_degree": 4.0,
                "shap_graph_degree": -0.40,
                "value_in_degree": 1.0,
                "shap_in_degree": 0.01,
                "value_out_degree": 3.0,
                "shap_out_degree": -0.50,
                "value_command_line_count": 1.0,
                "shap_command_line_count": 0.01,
                "value_module_count": 1.0,
                "shap_module_count": -0.30,
                "value_memory_region_count": 1.0,
                "shap_memory_region_count": -0.10,
                "value_relationship_type_count": 3.0,
                "shap_relationship_type_count": 0.02,
            },
            {
                "case_id": "TEST",
                "logical_event_id": "LEVT-2",
                "temporal_sequence": 2,
                "process_id": 200,
                "process": "normal.exe",
                "graph_anomaly_score": -0.2,
                "graph_predicted_anomaly": False,
                "graph_anomaly_rank": 2,
            },
        ]
    )


def configure_paths(tmp_path, monkeypatch):
    graph_file = tmp_path / "graph_features.csv"
    raw_file = tmp_path / "events.csv"
    logical_file = tmp_path / "logical_timeline.csv"
    shap_file = tmp_path / "graph_shap_explanations.csv"
    anomaly_file = tmp_path / "graph_anomalies.csv"
    output_file = tmp_path / "graph_evidence_attribution.csv"

    make_graph_features().to_csv(graph_file, index=False)
    make_raw_events().to_csv(raw_file, index=False)
    make_logical_timeline().to_csv(logical_file, index=False)
    make_shap().to_csv(shap_file, index=False)
    make_shap()[
        [
            "logical_event_id",
            "graph_anomaly_score",
            "graph_predicted_anomaly",
            "graph_anomaly_rank",
        ]
    ].to_csv(anomaly_file, index=False)

    monkeypatch.setattr(gea, "GRAPH_FEATURE_FILE", graph_file)
    monkeypatch.setattr(gea, "RAW_EVENT_FILE", raw_file)
    monkeypatch.setattr(gea, "LOGICAL_TIMELINE_FILE", logical_file)
    monkeypatch.setattr(gea, "SHAP_FILE", shap_file)
    monkeypatch.setattr(gea, "ANOMALY_FILE", anomaly_file)
    monkeypatch.setattr(gea, "OUTPUT_FILE", output_file)
    monkeypatch.setattr(gea, "RESULTS_DIR", tmp_path)

    return output_file


def test_safe_float_valid_value():
    assert gea.safe_float("3.14") == pytest.approx(3.14)


def test_safe_float_invalid_value():
    assert np.isnan(gea.safe_float("invalid"))


def test_safe_float_missing_value():
    assert np.isnan(gea.safe_float(None))


def test_unique_join_removes_duplicates_and_blank_values():
    result = gea.unique_join(
        ["A", " A ", "", "B", "A", None, "B"]
    )

    assert result == "A;B"


def test_main_attributes_only_anomalous_events(
    tmp_path,
    monkeypatch,
):
    output_file = configure_paths(
        tmp_path,
        monkeypatch,
    )

    gea.main()

    result = pd.read_csv(output_file)

    assert len(result) == 1
    assert result.iloc[0]["logical_event_id"] == "LEVT-1"
    assert len(result) == 1
    assert result.iloc[0]["logical_event_id"] == "LEVT-1"
    assert result.iloc[0]["graph_anomaly_rank"] == 1


def test_anomaly_and_shap_alignment_is_preserved(
    tmp_path,
    monkeypatch,
):
    output_file = configure_paths(
        tmp_path,
        monkeypatch,
    )

    gea.main()

    result = pd.read_csv(output_file)

    row = result.iloc[0]

    assert row["logical_event_id"] == "LEVT-1"
    assert row["graph_anomaly_score"] == pytest.approx(0.8)
    assert row["graph_anomaly_rank"] == 1


def test_top_shap_features_are_sorted_by_absolute_contribution(
    tmp_path,
    monkeypatch,
):
    output_file = configure_paths(
        tmp_path,
        monkeypatch,
    )

    gea.main()

    result = pd.read_csv(output_file)
    row = result.iloc[0]

    assert row["top_shap_features"].split(";") == [
        "out_degree",
        "graph_degree",
        "module_count",
    ]


def test_parent_and_child_relationships_are_attributed(
    tmp_path,
    monkeypatch,
):
    output_file = configure_paths(
        tmp_path,
        monkeypatch,
    )

    gea.main()

    row = pd.read_csv(output_file).iloc[0]

    assert str(row["parent_process_ids"]) == "50"
    assert str(row["child_process_ids"]) == "75"


def test_raw_evidence_is_counted_and_grouped(
    tmp_path,
    monkeypatch,
):
    output_file = configure_paths(
        tmp_path,
        monkeypatch,
    )

    gea.main()

    row = pd.read_csv(output_file).iloc[0]

    assert row["raw_event_count"] == 5
    assert "pslist=1" in row["artifact_type_counts"]
    assert "pstree=2" in row["artifact_type_counts"]
    assert "cmdline=1" in row["artifact_type_counts"]
    assert "malfind=1" in row["artifact_type_counts"]


def test_evidence_ids_and_provenance_are_preserved(
    tmp_path,
    monkeypatch,
):
    output_file = configure_paths(
        tmp_path,
        monkeypatch,
    )

    gea.main()

    row = pd.read_csv(output_file).iloc[0]

    evidence_ids = set(
        row["evidence_ids"].split(";")
    )

    assert {
        "REL-1",
        "REL-2",
        "PROC-1",
        "CMD-1",
        "MAL-1",
    }.issubset(evidence_ids)

    provenance = set(
        row["provenance"].split(";")
    )

    assert {
        "pstree-source",
        "pslist-source",
        "cmd-source",
        "malfind-source",
    }.issubset(provenance)


def test_command_lines_are_preserved(
    tmp_path,
    monkeypatch,
):
    output_file = configure_paths(
        tmp_path,
        monkeypatch,
    )

    gea.main()

    row = pd.read_csv(output_file).iloc[0]

    assert "test.exe -arg" in row["command_lines"]


def test_logical_timeline_context_is_preserved(
    tmp_path,
    monkeypatch,
):
    output_file = configure_paths(
        tmp_path,
        monkeypatch,
    )

    gea.main()

    row = pd.read_csv(output_file).iloc[0]

    assert row["timestamp"] == "2009-11-23 12:00:00+00:00"
    assert row["previous_process_id"] == 90
    assert row["previous_process"] == "prev.exe"
    assert row["time_since_previous_event_seconds"] == pytest.approx(12.5)
    assert row["timeline_evidence_ids"] == "PROC-1;REL-1"


def test_graph_feature_values_are_attached(
    tmp_path,
    monkeypatch,
):
    output_file = configure_paths(
        tmp_path,
        monkeypatch,
    )

    gea.main()

    row = pd.read_csv(output_file).iloc[0]

    assert row["parent_count"] == 1
    assert row["child_count"] == 2
    assert row["graph_degree"] == 4
    assert row["in_degree"] == 1
    assert row["out_degree"] == 3
    assert row["command_line_count"] == 1
    assert row["module_count"] == 1
    assert row["memory_region_count"] == 1
    assert row["relationship_type_count"] == 3


def test_output_contains_required_investigator_fields(
    tmp_path,
    monkeypatch,
):
    output_file = configure_paths(
        tmp_path,
        monkeypatch,
    )

    gea.main()

    result = pd.read_csv(output_file)

    required = [
        "logical_event_id",
        "graph_anomaly_score",
        "graph_anomaly_rank",
        "top_shap_features",
        "parent_process_ids",
        "child_process_ids",
        "raw_event_count",
        "artifact_type_counts",
        "event_type_counts",
        "evidence_by_artifact",
        "evidence_ids",
        "provenance",
        "command_lines",
        "timeline_evidence_ids",
        "assessment",
        "limitation",
    ]

    for column in required:
        assert column in result.columns


def test_output_contains_non_maliciousness_limitation(
    tmp_path,
    monkeypatch,
):
    output_file = configure_paths(
        tmp_path,
        monkeypatch,
    )

    gea.main()

    row = pd.read_csv(output_file).iloc[0]

    assert "do not establish malicious activity" in row["limitation"]