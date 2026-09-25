from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = ROOT / "data" / "features" / "M57-Jean" / "temporal_features.csv"
OUTPUT_PATH = ROOT / "results" / "M57-Jean" / "temporal_anomalies.csv"
SCRIPT_PATH = ROOT / "src" / "anomaly" / "temporal_isolation_forest.py"


@pytest.fixture(scope="module")
def temporal_output():
    """Run Temporal Isolation Forest once and return its output."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        "Temporal Isolation Forest failed.\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )

    assert OUTPUT_PATH.exists(), (
        f"Expected output was not created: {OUTPUT_PATH}"
    )

    return pd.read_csv(OUTPUT_PATH)


def test_temporal_feature_input_exists():
    assert INPUT_PATH.exists()


def test_temporal_feature_input_has_expected_rows():
    df = pd.read_csv(INPUT_PATH)

    assert len(df) == 47


def test_temporal_output_preserves_all_events(temporal_output):
    assert len(temporal_output) == 47


def test_temporal_output_contains_anomaly_columns(temporal_output):
    required_columns = {
        "temporal_anomaly_score",
        "temporal_predicted_anomaly",
        "temporal_anomaly_rank",
    }

    assert required_columns.issubset(temporal_output.columns)


def test_first_event_is_not_scored_as_anomaly(temporal_output):
    first_event = temporal_output.iloc[0]

    assert first_event["is_first_event"] == 1
    assert first_event["temporal_anomaly_score"] == 0.0
    assert first_event["temporal_predicted_anomaly"] is False or (
        first_event["temporal_predicted_anomaly"] == False
    )
    assert pd.isna(first_event["temporal_anomaly_rank"])


def test_only_non_first_events_receive_scores(temporal_output):
    first_events = temporal_output[temporal_output["is_first_event"] == 1]
    scored_events = temporal_output[
        temporal_output["temporal_anomaly_score"] != 0
    ]

    assert len(first_events) == 1
    assert len(scored_events) == 46


def test_expected_temporal_anomaly_count(temporal_output):
    anomalies = temporal_output[
        temporal_output["temporal_predicted_anomaly"] == True
    ]

    assert len(anomalies) == 5


def test_anomaly_ranking_is_valid(temporal_output):
    anomalies = temporal_output[
        temporal_output["temporal_predicted_anomaly"] == True
    ].copy()

    anomalies = anomalies.sort_values(
        "temporal_anomaly_score",
        ascending=False,
    )

    expected_ranks = list(range(1, len(anomalies) + 1))

    assert anomalies["temporal_anomaly_rank"].tolist() == expected_ranks


def test_top_anomaly_has_rank_one(temporal_output):
    top_ranked = temporal_output[
        temporal_output["temporal_anomaly_rank"] == 1
    ]

    assert len(top_ranked) == 1

    assert top_ranked.iloc[0]["temporal_anomaly_score"] == pytest.approx(
        0.053829,
        abs=1e-6,
    )