import pandas as pd

from src.anomaly.graph_isolation_forest import (
    MODEL_FEATURES,
    RANDOM_STATE,
    N_ESTIMATORS,
    CONTAMINATION,
)


def test_graph_isolation_forest_configuration():
    assert RANDOM_STATE == 42
    assert N_ESTIMATORS == 500
    assert CONTAMINATION == 0.10


def test_model_features_are_present_in_graph_features():
    path = "data/features/M57-Jean/graph_features.csv"

    df = pd.read_csv(path)

    missing_features = [
        feature
        for feature in MODEL_FEATURES
        if feature not in df.columns
    ]

    assert missing_features == []


def test_graph_features_have_expected_count():
    assert len(MODEL_FEATURES) == 14


def test_graph_features_can_be_preprocessed_for_model():
    path = "data/features/M57-Jean/graph_features.csv"

    df = pd.read_csv(path)

    features = df[MODEL_FEATURES].apply(
        pd.to_numeric,
        errors="coerce",
    )

    features = features.replace(
        [float("inf"), float("-inf")],
        pd.NA,
    )

    features = features.fillna(
        features.median()
    )

    assert not features.isna().any().any()


def test_graph_features_have_expected_rows():
    path = "data/features/M57-Jean/graph_features.csv"

    df = pd.read_csv(path)

    assert len(df) == 47


def test_graph_anomaly_output_schema_if_available():
    path = "results/M57-Jean/graph_anomalies.csv"

    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return

    required_columns = [
        "case_id",
        "logical_event_id",
        "temporal_sequence",
        "timestamp",
        "process_id",
        "process",
        "graph_anomaly_score",
        "graph_predicted_anomaly",
        "graph_anomaly_rank",
    ]

    for column in required_columns:
        assert column in df.columns