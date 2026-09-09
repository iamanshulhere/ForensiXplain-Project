from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.temporal.temporal_features import (
    build_temporal_features,
    count_events_within_window,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGICAL_TIMELINE_PATH = (
    PROJECT_ROOT
    / "data"
    / "normalized"
    / "M57-Jean"
    / "logical_timeline.csv"
)
TEMPORAL_FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "M57-Jean"
    / "temporal_features.csv"
)

EXPECTED_FEATURE_COLUMNS = [
    "case_id",
    "logical_event_id",
    "timestamp",
    "timestamp_confidence",
    "logical_event_type",
    "action",
    "process_id",
    "process",
    "parent_process_ids",
    "evidence_ids",
    "source_observation_count",
    "provenance",
    "temporal_sequence",
    "previous_timestamp",
    "time_since_previous_event_seconds",
    "is_first_event",
    "gap_log_seconds",
    "event_hour",
    "event_day_of_week",
    "after_hours",
    "events_prev_10s",
    "events_next_10s",
    "local_density_10s",
    "events_prev_30s",
    "events_next_30s",
    "local_density_30s",
    "events_prev_60s",
    "events_next_60s",
    "local_density_60s",
    "previous_process_id",
    "previous_process",
    "process_changed",
    "process_transition",
    "parent_process_id",
    "rapid_event",
    "short_event_gap",
    "medium_event_gap",
    "long_event_gap",
]


def make_logical_timeline(rows):
    """Create a logical timeline with the fields required by the feature code."""
    return pd.DataFrame(
        [
            {
                "timestamp": row["timestamp"],
                "process_id": row["process_id"],
                "process": row["process"],
                "parent_process_ids": row.get("parent_process_ids", "4"),
                "source_observation_count": row.get(
                    "source_observation_count",
                    1,
                ),
            }
            for row in rows
        ]
    )


def test_timestamp_preparation_removes_invalid_events_and_sorts():
    timeline = make_logical_timeline(
        [
            {
                "timestamp": "2009-11-21T00:00:10+00:00",
                "process_id": 30,
                "process": "third.exe",
            },
            {
                "timestamp": "not-a-date",
                "process_id": 10,
                "process": "invalid.exe",
            },
            {
                "timestamp": "2009-11-21T00:00:00+00:00",
                "process_id": 20,
                "process": "first.exe",
            },
            {
                "timestamp": None,
                "process_id": 40,
                "process": "missing.exe",
            },
            {
                "timestamp": "2009-11-21T00:00:10+00:00",
                "process_id": 10,
                "process": "second.exe",
            },
        ]
    )
    original = timeline.copy(deep=True)

    result = build_temporal_features(timeline)

    assert result["process"].tolist() == [
        "first.exe",
        "second.exe",
        "third.exe",
    ]
    assert str(result["timestamp"].dtype) == "datetime64[ns, UTC]"
    assert result["timestamp"].tolist() == [
        pd.Timestamp("2009-11-21T00:00:00+00:00"),
        pd.Timestamp("2009-11-21T00:00:10+00:00"),
        pd.Timestamp("2009-11-21T00:00:10+00:00"),
    ]
    assert_frame_equal(timeline, original)


def test_time_gaps_process_transitions_and_gap_indicators():
    timeline = make_logical_timeline(
        [
            {
                "timestamp": "2009-11-21T08:00:00+00:00",
                "process_id": 10,
                "process": "alpha.exe",
            },
            {
                "timestamp": "2009-11-21T08:00:05+00:00",
                "process_id": 20,
                "process": "alpha.exe",
            },
            {
                "timestamp": "2009-11-21T08:00:35+00:00",
                "process_id": 30,
                "process": "beta.exe",
            },
            {
                "timestamp": "2009-11-21T08:05:35+00:00",
                "process_id": 40,
                "process": "beta.exe",
            },
            {
                "timestamp": "2009-11-21T08:10:36+00:00",
                "process_id": 50,
                "process": "gamma.exe",
            },
        ]
    )

    result = build_temporal_features(timeline)

    assert result["temporal_sequence"].tolist() == [1, 2, 3, 4, 5]
    assert result["is_first_event"].tolist() == [1, 0, 0, 0, 0]
    assert pd.isna(result.loc[0, "previous_timestamp"])
    assert np.isnan(result.loc[0, "time_since_previous_event_seconds"])
    assert np.isnan(result.loc[0, "gap_log_seconds"])
    np.testing.assert_allclose(
        result.loc[1:, "time_since_previous_event_seconds"],
        [5.0, 30.0, 300.0, 301.0],
    )
    np.testing.assert_allclose(
        result.loc[1:, "gap_log_seconds"],
        np.log1p([5.0, 30.0, 300.0, 301.0]),
    )
    assert pd.isna(result.loc[0, "previous_process_id"])
    assert result.loc[1:, "previous_process_id"].tolist() == [
        10.0,
        20.0,
        30.0,
        40.0,
    ]
    assert result["process_changed"].tolist() == [1, 0, 1, 0, 1]
    assert result["process_transition"].tolist() == [
        "START -> alpha.exe",
        "alpha.exe -> alpha.exe",
        "alpha.exe -> beta.exe",
        "beta.exe -> beta.exe",
        "beta.exe -> gamma.exe",
    ]
    assert result["rapid_event"].tolist() == [0, 1, 0, 0, 0]
    assert result["short_event_gap"].tolist() == [0, 1, 1, 0, 0]
    assert result["medium_event_gap"].tolist() == [0, 1, 1, 1, 0]
    assert result["long_event_gap"].tolist() == [0, 0, 0, 0, 1]


def test_temporal_density_uses_inclusive_window_boundaries():
    timeline = make_logical_timeline(
        [
            {
                "timestamp": "2009-11-21T08:00:00+00:00",
                "process_id": 10,
                "process": "one.exe",
            },
            {
                "timestamp": "2009-11-21T08:00:05+00:00",
                "process_id": 20,
                "process": "two.exe",
            },
            {
                "timestamp": "2009-11-21T08:00:20+00:00",
                "process_id": 30,
                "process": "three.exe",
            },
            {
                "timestamp": "2009-11-21T08:01:00+00:00",
                "process_id": 40,
                "process": "four.exe",
            },
        ]
    )

    result = build_temporal_features(timeline)

    assert result["events_prev_10s"].tolist() == [1, 2, 1, 1]
    assert result["events_next_10s"].tolist() == [2, 1, 1, 1]
    assert result["local_density_10s"].tolist() == [2, 2, 1, 1]
    assert result["events_prev_30s"].tolist() == [1, 2, 3, 1]
    assert result["events_next_30s"].tolist() == [3, 2, 1, 1]
    assert result["local_density_30s"].tolist() == [3, 3, 3, 1]
    assert result["events_prev_60s"].tolist() == [1, 2, 3, 4]
    assert result["events_next_60s"].tolist() == [4, 3, 2, 1]
    assert result["local_density_60s"].tolist() == [4, 4, 4, 4]


def test_window_count_rejects_unknown_direction():
    timestamps_ns = np.array([0, 5_000_000_000])

    with pytest.raises(ValueError, match="direction must be"):
        count_events_within_window(
            timestamps_ns,
            10,
            direction="sideways",
        )


def test_m57_jean_output_matches_regression_baseline_and_is_deterministic():
    logical_timeline = pd.read_csv(LOGICAL_TIMELINE_PATH)
    baseline = pd.read_csv(TEMPORAL_FEATURES_PATH)

    result = build_temporal_features(logical_timeline)
    repeated_result = build_temporal_features(logical_timeline)

    assert len(result) == 47
    assert len(baseline) == 47
    assert result.columns.tolist() == EXPECTED_FEATURE_COLUMNS
    assert baseline.columns.tolist() == EXPECTED_FEATURE_COLUMNS
    assert result["source_observation_count"].sum() == 94
    assert result[
        [
            "is_first_event",
            "rapid_event",
            "short_event_gap",
            "medium_event_gap",
            "long_event_gap",
        ]
    ].sum().to_dict() == {
        "is_first_event": 1,
        "rapid_event": 31,
        "short_event_gap": 34,
        "medium_event_gap": 36,
        "long_event_gap": 10,
    }
    assert_frame_equal(result, repeated_result)
