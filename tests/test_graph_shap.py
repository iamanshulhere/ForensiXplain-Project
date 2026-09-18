import numpy as np
import pandas as pd
import pytest

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from src.explainability import graph_shap as gshap


EXPECTED_FEATURES = [
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


def make_features(rows=12):
    data = []

    for i in range(rows):
        data.append(
            {
                "case_id": "TEST",
                "logical_event_id": f"LEVT-{i + 1}",
                "temporal_sequence": i + 1,
                "timestamp": f"2009-11-23 12:{i:02d}:00+00:00",
                "process_id": 1000 + i,
                "process": f"proc{i}.exe",

                "gap_log_seconds": float(i + 1),
                "local_density_10s": float((i % 5) + 1),
                "local_density_30s": float((i % 6) + 2),
                "local_density_60s": float((i % 8) + 3),
                "process_changed": i % 2,

                "parent_count": i % 3,
                "child_count": i % 4,
                "graph_degree": i + 1,
                "in_degree": i % 3,
                "out_degree": i % 5,
                "command_line_count": i % 2,
                "module_count": i + 2,
                "memory_region_count": i % 3,
                "relationship_type_count": (i % 4) + 1,
            }
        )

    return pd.DataFrame(data)


def make_anomalies(features):
    """Create anomaly results using the exact model contract."""
    X = features[gshap.MODEL_FEATURES].copy()

    X = X.apply(
        pd.to_numeric,
        errors="coerce",
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    X = X.fillna(
        X.median()
    )

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_scaled = np.asarray(
        X_scaled,
        dtype=np.float64,
    )

    model = IsolationForest(
        n_estimators=gshap.N_ESTIMATORS,
        contamination=gshap.CONTAMINATION,
        random_state=gshap.RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_scaled)

    anomaly_score = -model.decision_function(
        X_scaled
    )

    prediction = model.predict(
        X_scaled
    )

    predicted_anomaly = prediction == -1

    result = features[
        ["logical_event_id"]
    ].copy()

    result["graph_anomaly_score"] = anomaly_score
    result["graph_predicted_anomaly"] = predicted_anomaly

    result = result.sort_values(
        by="graph_anomaly_score",
        ascending=False,
    ).reset_index(drop=True)

    result["graph_anomaly_rank"] = range(
        1,
        len(result) + 1,
    )

    return result

def configure_paths(tmp_path, monkeypatch, features, anomalies):
    feature_file = tmp_path / "graph_features.csv"
    anomaly_file = tmp_path / "graph_anomalies.csv"
    results_dir = tmp_path / "results"
    output_file = results_dir / "graph_shap_explanations.csv"

    features.to_csv(feature_file, index=False)
    anomalies.to_csv(anomaly_file, index=False)

    monkeypatch.setattr(
        gshap,
        "FEATURE_FILE",
        feature_file,
    )

    monkeypatch.setattr(
        gshap,
        "ANOMALY_FILE",
        anomaly_file,
    )

    monkeypatch.setattr(
        gshap,
        "RESULTS_DIR",
        results_dir,
    )

    monkeypatch.setattr(
        gshap,
        "OUTPUT_FILE",
        output_file,
    )

    return output_file

def test_model_features_exactly_match_expected_contract():
    assert gshap.MODEL_FEATURES == EXPECTED_FEATURES
    assert len(gshap.MODEL_FEATURES) == 14


def test_model_features_have_no_duplicates():
    assert len(gshap.MODEL_FEATURES) == len(
        set(gshap.MODEL_FEATURES)
    )


def test_missing_required_feature_raises_value_error(
    tmp_path,
    monkeypatch,
):
    features = make_features().drop(
        columns=["memory_region_count"]
    )
    anomalies = make_anomalies(make_features())

    configure_paths(
        tmp_path,
        monkeypatch,
        features,
        anomalies,
    )

    with pytest.raises(ValueError, match="Missing model features"):
        gshap.main()


def test_missing_anomaly_score_raises_value_error(
    tmp_path,
    monkeypatch,
):
    features = make_features()
    anomalies = make_anomalies(features).drop(
        columns=["graph_anomaly_score"]
    )

    configure_paths(
        tmp_path,
        monkeypatch,
        features,
        anomalies,
    )

    with pytest.raises(
        KeyError,
        match="graph_anomaly_score",
    ):
        gshap.main()

def test_one_to_one_event_alignment_is_required(
    tmp_path,
    monkeypatch,
):
    features = make_features()
    anomalies = make_anomalies(features)

    duplicate = anomalies.iloc[[0]].copy()
    anomalies = pd.concat(
        [anomalies, duplicate],
        ignore_index=True,
    )

    configure_paths(
        tmp_path,
        monkeypatch,
        features,
        anomalies,
    )

    with pytest.raises(
        Exception,
        match="one_to_one|Merge keys",
    ):
        gshap.main()


def test_full_graph_shap_run_produces_aligned_output(
    tmp_path,
    monkeypatch,
):
    features = make_features()
    anomalies = make_anomalies(features)

    output_file = configure_paths(
        tmp_path,
        monkeypatch,
        features,
        anomalies,
    )

    gshap.main()

    result = pd.read_csv(output_file)

    assert output_file.exists()
    assert len(result) == len(features)

    assert result["logical_event_id"].is_unique
    assert set(result["logical_event_id"]) == set(
        features["logical_event_id"]
    )

    assert result["graph_anomaly_score"].notna().all()
    assert result["graph_predicted_anomaly"].notna().all()
    assert result["graph_anomaly_rank"].notna().all()


def test_shap_columns_exist_for_all_model_features(
    tmp_path,
    monkeypatch,
):
    features = make_features()
    anomalies = make_anomalies(features)

    output_file = configure_paths(
        tmp_path,
        monkeypatch,
        features,
        anomalies,
    )

    gshap.main()

    result = pd.read_csv(output_file)

    for feature in EXPECTED_FEATURES:
        assert f"value_{feature}" in result.columns
        assert f"shap_{feature}" in result.columns


def test_shap_matrix_dimensions_match_rows_and_features(
    tmp_path,
    monkeypatch,
):
    features = make_features()
    anomalies = make_anomalies(features)

    output_file = configure_paths(
        tmp_path,
        monkeypatch,
        features,
        anomalies,
    )

    gshap.main()

    result = pd.read_csv(output_file)

    shap_columns = [
        f"shap_{feature}"
        for feature in EXPECTED_FEATURES
    ]

    assert result[shap_columns].shape == (
        len(features),
        len(EXPECTED_FEATURES),
    )


def test_output_is_sorted_by_anomaly_score(
    tmp_path,
    monkeypatch,
):
    features = make_features()
    anomalies = make_anomalies(features)

    output_file = configure_paths(
        tmp_path,
        monkeypatch,
        features,
        anomalies,
    )

    gshap.main()

    result = pd.read_csv(output_file)

    assert result["graph_anomaly_score"].is_monotonic_decreasing


def test_anomaly_metadata_aligns_with_each_event(
    tmp_path,
    monkeypatch,
):
    features = make_features()
    anomalies = make_anomalies(features)

    output_file = configure_paths(
        tmp_path,
        monkeypatch,
        features,
        anomalies,
    )

    gshap.main()

    result = pd.read_csv(output_file)

    expected_scores = dict(
        zip(
            anomalies["logical_event_id"],
            anomalies["graph_anomaly_score"],
        )
    )

    for _, row in result.iterrows():
        expected = expected_scores[row["logical_event_id"]]
        assert row["graph_anomaly_score"] == pytest.approx(
            expected
        )