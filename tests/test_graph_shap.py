import pandas as pd

from src.explainability.graph_shap import (
    MODEL_FEATURES,
    RANDOM_STATE,
    N_ESTIMATORS,
    CONTAMINATION,
)


SHAP_OUTPUT = "results/M57-Jean/graph_shap_explanations.csv"


def test_graph_shap_configuration_matches_graph_model():
    assert RANDOM_STATE == 42
    assert N_ESTIMATORS == 500
    assert CONTAMINATION == 0.10


def test_graph_shap_has_14_model_features():
    assert len(MODEL_FEATURES) == 14


def test_graph_shap_output_has_expected_rows():
    df = pd.read_csv(SHAP_OUTPUT)

    assert len(df) == 47


def test_graph_shap_output_has_required_metadata():
    df = pd.read_csv(SHAP_OUTPUT)

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


def test_graph_shap_output_has_value_and_shap_columns():
    df = pd.read_csv(SHAP_OUTPUT)

    for feature in MODEL_FEATURES:
        assert f"value_{feature}" in df.columns
        assert f"shap_{feature}" in df.columns


def test_graph_shap_output_has_expected_column_count():
    df = pd.read_csv(SHAP_OUTPUT)

    # 9 metadata/anomaly columns +
    # 14 original feature values +
    # 14 SHAP values
    assert len(df.columns) == 37


def test_graph_shap_scores_are_sorted_descending():
    df = pd.read_csv(SHAP_OUTPUT)

    scores = df["graph_anomaly_score"]

    assert scores.is_monotonic_decreasing


def test_graph_shap_ranks_are_sequential():
    df = pd.read_csv(SHAP_OUTPUT)

    assert df["graph_anomaly_rank"].tolist() == list(
        range(1, len(df) + 1)
    )


def test_graph_shap_contains_five_predicted_anomalies():
    df = pd.read_csv(SHAP_OUTPUT)

    assert df["graph_predicted_anomaly"].sum() == 5


def test_graph_shap_values_are_numeric():
    df = pd.read_csv(SHAP_OUTPUT)

    shap_columns = [
        f"shap_{feature}"
        for feature in MODEL_FEATURES
    ]

    shap_values = df[shap_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    assert not shap_values.isna().any().any()


def test_graph_shap_logical_event_ids_are_unique():
    df = pd.read_csv(SHAP_OUTPUT)

    assert df["logical_event_id"].is_unique