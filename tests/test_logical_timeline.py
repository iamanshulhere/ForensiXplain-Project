import pandas as pd

from src.temporal.logical_timeline import build_logical_timeline


def make_timeline(rows):
    """Create a timeline DataFrame with all required columns."""
    return pd.DataFrame(rows)


def test_invalid_and_missing_timestamps_are_removed():
    timeline = make_timeline(
        [
            {
                "timestamp": "2009-11-21 01:32:30",
                "process_id": "100",
                "parent_process_id": "4",
                "event_type": "process",
                "process": "test.exe",
                "evidence_id": "E1",
                "provenance": "pslist",
            },
            {
                "timestamp": "not-a-date",
                "process_id": "101",
                "parent_process_id": "4",
                "event_type": "process",
                "process": "bad.exe",
                "evidence_id": "E2",
                "provenance": "pslist",
            },
            {
                "timestamp": None,
                "process_id": "102",
                "parent_process_id": "4",
                "event_type": "process",
                "process": "missing.exe",
                "evidence_id": "E3",
                "provenance": "pslist",
            },
        ]
    )

    result = build_logical_timeline(
        timeline,
        "TEST",
    )

    assert len(result) == 1
    assert result.iloc[0]["process"] == "test.exe"


def test_process_ids_are_normalized():
    timeline = make_timeline(
        [
            {
                "timestamp": "2009-11-21 01:32:30",
                "process_id": "123",
                "parent_process_id": "4",
                "event_type": "process",
                "process": "test.exe",
                "evidence_id": "E1",
                "provenance": "pslist",
            },
        ]
    )

    result = build_logical_timeline(
        timeline,
        "TEST",
    )

    assert result.iloc[0]["process_id"] == 123


def test_invalid_process_id_does_not_create_logical_event():
    timeline = make_timeline(
        [
            {
                "timestamp": "2009-11-21 01:32:30",
                "process_id": "invalid",
                "parent_process_id": "4",
                "event_type": "process",
                "process": "bad.exe",
                "evidence_id": "E1",
                "provenance": "pslist",
            },
        ]
    )

    result = build_logical_timeline(
        timeline,
        "TEST",
    )

    assert len(result) == 0


def test_events_are_sorted_by_timestamp_and_process_id():
    timeline = make_timeline(
        [
            {
                "timestamp": "2009-11-21 01:32:32",
                "process_id": "200",
                "parent_process_id": "4",
                "event_type": "process",
                "process": "second.exe",
                "evidence_id": "E2",
                "provenance": "pslist",
            },
            {
                "timestamp": "2009-11-21 01:32:30",
                "process_id": "100",
                "parent_process_id": "4",
                "event_type": "process",
                "process": "first.exe",
                "evidence_id": "E1",
                "provenance": "pslist",
            },
            {
                "timestamp": "2009-11-21 01:32:32",
                "process_id": "150",
                "parent_process_id": "4",
                "event_type": "process",
                "process": "middle.exe",
                "evidence_id": "E3",
                "provenance": "pslist",
            },
        ]
    )

    result = build_logical_timeline(
        timeline,
        "TEST",
    )

    assert result["process_id"].tolist() == [
        100,
        150,
        200,
    ]


def test_relationship_is_matched_by_process_id_and_timestamp():
    timeline = make_timeline(
        [
            {
                "timestamp": "2009-11-21 01:32:30",
                "process_id": "100",
                "parent_process_id": "",
                "event_type": "process",
                "process": "test.exe",
                "evidence_id": "E1",
                "provenance": "pslist",
            },
            {
                "timestamp": "2009-11-21 01:32:30",
                "process_id": "100",
                "parent_process_id": "50",
                "event_type": "process_relationship",
                "process": "test.exe",
                "evidence_id": "E2",
                "provenance": "pstree",
            },
            {
                "timestamp": "2009-11-21 01:32:31",
                "process_id": "100",
                "parent_process_id": "999",
                "event_type": "process_relationship",
                "process": "test.exe",
                "evidence_id": "E3",
                "provenance": "pstree",
            },
        ]
    )

    result = build_logical_timeline(
        timeline,
        "TEST",
    )

    assert result.iloc[0]["parent_process_ids"] == "50"


def test_multiple_parent_ids_are_unique_and_stable():
    timeline = make_timeline(
        [
            {
                "timestamp": "2009-11-21 01:32:30",
                "process_id": "100",
                "parent_process_id": "",
                "event_type": "process",
                "process": "test.exe",
                "evidence_id": "E1",
                "provenance": "pslist",
            },
            {
                "timestamp": "2009-11-21 01:32:30",
                "process_id": "100",
                "parent_process_id": "50",
                "event_type": "process_relationship",
                "process": "test.exe",
                "evidence_id": "E2",
                "provenance": "pstree",
            },
            {
                "timestamp": "2009-11-21 01:32:30",
                "process_id": "100",
                "parent_process_id": "60",
                "event_type": "process_relationship",
                "process": "test.exe",
                "evidence_id": "E3",
                "provenance": "pstree",
            },
            {
                "timestamp": "2009-11-21 01:32:30",
                "process_id": "100",
                "parent_process_id": "50",
                "event_type": "process_relationship",
                "process": "test.exe",
                "evidence_id": "E4",
                "provenance": "pstree",
            },
        ]
    )

    result = build_logical_timeline(
        timeline,
        "TEST",
    )

    assert result.iloc[0]["parent_process_ids"] == "50;60"


def test_evidence_ids_are_combined_without_duplicates():
    timeline = make_timeline(
        [
            {
                "timestamp": "2009-11-21 01:32:30",
                "process_id": "100",
                "parent_process_id": "",
                "event_type": "process",
                "process": "test.exe",
                "evidence_id": "E1",
                "provenance": "pslist",
            },
            {
                "timestamp": "2009-11-21 01:32:30",
                "process_id": "100",
                "parent_process_id": "50",
                "event_type": "process_relationship",
                "process": "test.exe",
                "evidence_id": "E1",
                "provenance": "pstree",
            },
            {
                "timestamp": "2009-11-21 01:32:30",
                "process_id": "100",
                "parent_process_id": "50",
                "event_type": "process_relationship",
                "process": "test.exe",
                "evidence_id": "E2",
                "provenance": "pstree",
            },
        ]
    )

    result = build_logical_timeline(
        timeline,
        "TEST",
    )

    assert result.iloc[0]["evidence_ids"] == "E1;E2"


def test_provenance_values_are_combined_without_duplicates():
    timeline = make_timeline(
        [
            {
                "timestamp": "2009-11-21 01:32:30",
                "process_id": "100",
                "parent_process_id": "",
                "event_type": "process",
                "process": "test.exe",
                "evidence_id": "E1",
                "provenance": "pslist",
            },
            {
                "timestamp": "2009-11-21 01:32:30",
                "process_id": "100",
                "parent_process_id": "50",
                "event_type": "process_relationship",
                "process": "test.exe",
                "evidence_id": "E2",
                "provenance": "pstree",
            },
            {
                "timestamp": "2009-11-21 01:32:30",
                "process_id": "100",
                "parent_process_id": "50",
                "event_type": "process_relationship",
                "process": "test.exe",
                "evidence_id": "E3",
                "provenance": "pstree",
            },
        ]
    )

    result = build_logical_timeline(
        timeline,
        "TEST",
    )

    assert result.iloc[0]["provenance"] == "pstree | pslist"


def test_logical_event_id_is_generated_correctly():
    timeline = make_timeline(
        [
            {
                "timestamp": "2009-11-21 01:32:30",
                "process_id": "812",
                "parent_process_id": "4",
                "event_type": "process",
                "process": "smss.exe",
                "evidence_id": "E1",
                "provenance": "pslist",
            },
        ]
    )

    result = build_logical_timeline(
        timeline,
        "M57-Jean",
    )

    assert (
        result.iloc[0]["logical_event_id"]
        == "LEVT-M57-Jean-PROCESS-812-20091121013230"
    )


def test_temporal_sequence_starts_at_one():
    timeline = make_timeline(
        [
            {
                "timestamp": "2009-11-21 01:32:30",
                "process_id": "100",
                "parent_process_id": "4",
                "event_type": "process",
                "process": "first.exe",
                "evidence_id": "E1",
                "provenance": "pslist",
            },
            {
                "timestamp": "2009-11-21 01:32:32",
                "process_id": "200",
                "parent_process_id": "4",
                "event_type": "process",
                "process": "second.exe",
                "evidence_id": "E2",
                "provenance": "pslist",
            },
        ]
    )

    result = build_logical_timeline(
        timeline,
        "TEST",
    )

    assert result["temporal_sequence"].tolist() == [1, 2]


def test_time_gap_is_calculated_correctly():
    timeline = make_timeline(
        [
            {
                "timestamp": "2009-11-21 01:32:30",
                "process_id": "100",
                "parent_process_id": "4",
                "event_type": "process",
                "process": "first.exe",
                "evidence_id": "E1",
                "provenance": "pslist",
            },
            {
                "timestamp": "2009-11-21 01:32:35",
                "process_id": "200",
                "parent_process_id": "4",
                "event_type": "process",
                "process": "second.exe",
                "evidence_id": "E2",
                "provenance": "pslist",
            },
            {
                "timestamp": "2009-11-21 01:33:35",
                "process_id": "300",
                "parent_process_id": "4",
                "event_type": "process",
                "process": "third.exe",
                "evidence_id": "E3",
                "provenance": "pslist",
            },
        ]
    )

    result = build_logical_timeline(
        timeline,
        "TEST",
    )

    assert result[
        "time_since_previous_event_seconds"
    ].tolist() == [0.0, 5.0, 60.0]


def test_first_event_has_zero_time_gap():
    timeline = make_timeline(
        [
            {
                "timestamp": "2009-11-21 01:32:30",
                "process_id": "100",
                "parent_process_id": "4",
                "event_type": "process",
                "process": "first.exe",
                "evidence_id": "E1",
                "provenance": "pslist",
            },
        ]
    )

    result = build_logical_timeline(
        timeline,
        "TEST",
    )

    assert (
        result.iloc[0][
            "time_since_previous_event_seconds"
        ]
        == 0
    )


def test_relationship_only_observation_is_not_emitted():
    timeline = make_timeline(
        [
            {
                "timestamp": "2009-11-21 01:32:30",
                "process_id": "100",
                "parent_process_id": "50",
                "event_type": "process_relationship",
                "process": "test.exe",
                "evidence_id": "E1",
                "provenance": "pstree",
            },
        ]
    )

    result = build_logical_timeline(
        timeline,
        "TEST",
    )

    assert result.empty