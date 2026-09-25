from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "data"
    / "features"
    / "M57-Jean"
    / "graph_features.csv"
)

OUTPUT_PATH = (
    ROOT
    / "results"
    / "M57-Jean"
    / "graph_anomalies.csv"
)

SCRIPT_PATH = (
    ROOT
    / "src"
    / "anomaly"
    / "graph_isolation_forest.py"
)


EXPECTED_MODEL_FEATURES = {
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
}


@pytest.fixture(scope="module")
def graph_output():
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        "Graph Isolation Forest failed.\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )

    assert OUTPUT_PATH.exists(), (
        f"Expected output was not created: {OUTPUT_PATH}"
    )

    return pd.read_csv(OUTPUT_PATH)


def test_graph_feature_input_exists():
    assert INPUT_PATH.exists()


def test_graph_feature_input_has_expected_rows():
    df = pd.read_csv(INPUT_PATH)

    assert len(df) == 47


def test_graph_model_features_exist():
    df = pd.read_csv(INPUT_PATH)

    assert EXPECTED_MODEL_FEATURES.issubset(df.columns)


def test_graph_output_preserves_all_events(graph_output):
    assert len(graph_output) == 47


def test_graph_output_contains_required_columns(graph_output):
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
    }

    assert required_columns.issubset(graph_output.columns)


def test_all_events_receive_anomaly_scores(graph_output):
    assert graph_output["graph_anomaly_score"].notna().all()


def test_expected_graph_anomaly_count(graph_output):
    anomalies = graph_output[
        graph_output["graph_predicted_anomaly"] == True
    ]

    assert len(anomalies) == 5


def test_graph_anomaly_ranking_is_valid(graph_output):
    ranks = graph_output["graph_anomaly_rank"].tolist()

    assert ranks == list(range(1, len(graph_output) + 1))


def test_top_graph_anomaly_is_deterministic(graph_output):
    top_ranked = graph_output[
        graph_output["graph_anomaly_rank"] == 1
    ]

    assert len(top_ranked) == 1

    row = top_ranked.iloc[0]

    assert row["process_id"] == 3560
    assert row["process"] == "soffice.bin"
    assert row["graph_predicted_anomaly"] == True

    assert row["graph_anomaly_score"] == pytest.approx(
        0.0738711086,
        abs=1e-8,
    )