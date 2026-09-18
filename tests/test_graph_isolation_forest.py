import pandas as pd
import pytest

from src.anomaly import graph_isolation_forest as gif


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


def make_features(rows=20):
    """Create deterministic synthetic graph-feature data."""
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
                "local_density_30s": float((i % 7) + 2),
                "local_density_60s": float((i % 9) + 3),
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


def run_model(tmp_path, df):
    """Run the existing main() against a temporary feature file."""
    feature_file = tmp_path / "graph_features.csv"
    results_dir = tmp_path / "results"
    output_file = results_dir / "graph_anomalies.csv"

    df.to_csv(feature_file, index=False)

    monkeypatch_values = {
        "FEATURE_FILE": feature_file,
        "RESULTS_DIR": results_dir,
        "OUTPUT_FILE": output_file,
    }

    return feature_file, results_dir, output_file, monkeypatch_values


def test_model_features_exactly_match_expected_contract():
    assert gif.MODEL_FEATURES == EXPECTED_FEATURES
    assert len(gif.MODEL_FEATURES) == 14


def test_model_features_have_no_duplicates():
    assert len(gif.MODEL_FEATURES) == len(set(gif.MODEL_FEATURES))


def test_all_required_features_exist_in_synthetic_input():
    df = make_features()

    missing = [
        feature
        for feature in gif.MODEL_FEATURES
        if feature not in df.columns
    ]

    assert missing == []


def test_missing_required_feature_raises_value_error(tmp_path, monkeypatch):
    df = make_features().drop(columns=["graph_degree"])

    feature_file = tmp_path / "graph_features.csv"
    results_dir = tmp_path / "results"
    output_file = results_dir / "graph_anomalies.csv"

    df.to_csv(feature_file, index=False)

    monkeypatch.setattr(gif, "FEATURE_FILE", feature_file)
    monkeypatch.setattr(gif, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(gif, "OUTPUT_FILE", output_file)

    with pytest.raises(ValueError, match="Missing required model features"):
        gif.main()


def test_nonnumeric_values_are_coerced_and_model_completes(
    tmp_path,
    monkeypatch,
):
    df = make_features()

    df["graph_degree"] = df["graph_degree"].astype(object)
    df["module_count"] = df["module_count"].astype(object)
    df["gap_log_seconds"] = df["gap_log_seconds"].astype(object)

    df.loc[0, "graph_degree"] = "invalid"
    df.loc[1, "module_count"] = "123"
    df.loc[2, "gap_log_seconds"] = "4.5"

    feature_file = tmp_path / "graph_features.csv"
    results_dir = tmp_path / "results"
    output_file = results_dir / "graph_anomalies.csv"

    df.to_csv(feature_file, index=False)

    monkeypatch.setattr(gif, "FEATURE_FILE", feature_file)
    monkeypatch.setattr(gif, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(gif, "OUTPUT_FILE", output_file)

    gif.main()

    result = pd.read_csv(output_file)

    assert len(result) == len(df)
    assert result["graph_anomaly_score"].notna().all()


def test_infinite_values_are_handled(tmp_path, monkeypatch):
    df = make_features()

    df["gap_log_seconds"] = df["gap_log_seconds"].astype(float)

    df.loc[0, "gap_log_seconds"] = float("inf")
    df.loc[1, "gap_log_seconds"] = float("-inf")

    feature_file = tmp_path / "graph_features.csv"
    results_dir = tmp_path / "results"
    output_file = results_dir / "graph_anomalies.csv"

    df.to_csv(feature_file, index=False)

    monkeypatch.setattr(gif, "FEATURE_FILE", feature_file)
    monkeypatch.setattr(gif, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(gif, "OUTPUT_FILE", output_file)

    gif.main()

    result = pd.read_csv(output_file)

    assert len(result) == len(df)
    assert result["graph_anomaly_score"].notna().all()


def test_missing_values_are_median_imputed(tmp_path, monkeypatch):
    df = make_features()

    df.loc[0, "graph_degree"] = None
    df.loc[1, "module_count"] = None

    feature_file = tmp_path / "graph_features.csv"
    results_dir = tmp_path / "results"
    output_file = results_dir / "graph_anomalies.csv"

    df.to_csv(feature_file, index=False)

    monkeypatch.setattr(gif, "FEATURE_FILE", feature_file)
    monkeypatch.setattr(gif, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(gif, "OUTPUT_FILE", output_file)

    gif.main()

    result = pd.read_csv(output_file)

    assert len(result) == len(df)
    assert result["graph_anomaly_score"].notna().all()


def test_output_preserves_all_input_events(tmp_path, monkeypatch):
    df = make_features()

    _, _, output_file, values = run_model(tmp_path, df)

    monkeypatch.setattr(gif, "FEATURE_FILE", values["FEATURE_FILE"])
    monkeypatch.setattr(gif, "RESULTS_DIR", values["RESULTS_DIR"])
    monkeypatch.setattr(gif, "OUTPUT_FILE", values["OUTPUT_FILE"])

    gif.main()

    result = pd.read_csv(output_file)

    assert len(result) == len(df)
    assert set(result["logical_event_id"]) == set(
        df["logical_event_id"]
    )


def test_output_preserves_metadata(tmp_path, monkeypatch):
    df = make_features()

    feature_file = tmp_path / "graph_features.csv"
    results_dir = tmp_path / "results"
    output_file = results_dir / "graph_anomalies.csv"

    df.to_csv(feature_file, index=False)

    monkeypatch.setattr(gif, "FEATURE_FILE", feature_file)
    monkeypatch.setattr(gif, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(gif, "OUTPUT_FILE", output_file)

    gif.main()

    result = pd.read_csv(output_file)

    metadata = [
        "case_id",
        "logical_event_id",
        "temporal_sequence",
        "timestamp",
        "process_id",
        "process",
    ]

    for column in metadata:
        assert column in result.columns

    assert result["case_id"].eq("TEST").all()


def test_scores_are_generated_for_every_event(tmp_path, monkeypatch):
    df = make_features()

    feature_file = tmp_path / "graph_features.csv"
    results_dir = tmp_path / "results"
    output_file = results_dir / "graph_anomalies.csv"

    df.to_csv(feature_file, index=False)

    monkeypatch.setattr(gif, "FEATURE_FILE", feature_file)
    monkeypatch.setattr(gif, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(gif, "OUTPUT_FILE", output_file)

    gif.main()

    result = pd.read_csv(output_file)

    assert result["graph_anomaly_score"].notna().all()
    assert result["graph_predicted_anomaly"].notna().all()


def test_anomaly_predictions_are_boolean(tmp_path, monkeypatch):
    df = make_features()

    feature_file = tmp_path / "graph_features.csv"
    results_dir = tmp_path / "results"
    output_file = results_dir / "graph_anomalies.csv"

    df.to_csv(feature_file, index=False)

    monkeypatch.setattr(gif, "FEATURE_FILE", feature_file)
    monkeypatch.setattr(gif, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(gif, "OUTPUT_FILE", output_file)

    gif.main()

    result = pd.read_csv(output_file)

    assert set(result["graph_predicted_anomaly"].unique()).issubset(
        {True, False}
    )


def test_ranks_are_unique_and_cover_all_rows(tmp_path, monkeypatch):
    df = make_features()

    feature_file = tmp_path / "graph_features.csv"
    results_dir = tmp_path / "results"
    output_file = results_dir / "graph_anomalies.csv"

    df.to_csv(feature_file, index=False)

    monkeypatch.setattr(gif, "FEATURE_FILE", feature_file)
    monkeypatch.setattr(gif, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(gif, "OUTPUT_FILE", output_file)

    gif.main()

    result = pd.read_csv(output_file)

    assert result["graph_anomaly_rank"].is_unique
    assert sorted(result["graph_anomaly_rank"].tolist()) == list(
        range(1, len(df) + 1)
    )


def test_results_are_sorted_by_descending_anomaly_score(
    tmp_path,
    monkeypatch,
):
    df = make_features()

    feature_file = tmp_path / "graph_features.csv"
    results_dir = tmp_path / "results"
    output_file = results_dir / "graph_anomalies.csv"

    df.to_csv(feature_file, index=False)

    monkeypatch.setattr(gif, "FEATURE_FILE", feature_file)
    monkeypatch.setattr(gif, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(gif, "OUTPUT_FILE", output_file)

    gif.main()

    result = pd.read_csv(output_file)

    assert result["graph_anomaly_score"].is_monotonic_decreasing


def test_first_event_is_scored_like_other_events(
    tmp_path,
    monkeypatch,
):
    df = make_features()

    feature_file = tmp_path / "graph_features.csv"
    results_dir = tmp_path / "results"
    output_file = results_dir / "graph_anomalies.csv"

    df.to_csv(feature_file, index=False)

    monkeypatch.setattr(gif, "FEATURE_FILE", feature_file)
    monkeypatch.setattr(gif, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(gif, "OUTPUT_FILE", output_file)

    gif.main()

    result = pd.read_csv(output_file)

    first_event = result[result["temporal_sequence"] == 1].iloc[0]

    assert pd.notna(first_event["graph_anomaly_score"])
    assert pd.notna(first_event["graph_anomaly_rank"])


def test_model_outputs_are_not_taken_from_input_columns(
    tmp_path,
    monkeypatch,
):
    df = make_features()

    # These simulate already-existing model-output columns.
    df["graph_anomaly_score"] = 999.0
    df["graph_predicted_anomaly"] = True
    df["graph_anomaly_rank"] = 999

    feature_file = tmp_path / "graph_features.csv"
    results_dir = tmp_path / "results"
    output_file = results_dir / "graph_anomalies.csv"

    df.to_csv(feature_file, index=False)

    monkeypatch.setattr(gif, "FEATURE_FILE", feature_file)
    monkeypatch.setattr(gif, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(gif, "OUTPUT_FILE", output_file)

    gif.main()

    result = pd.read_csv(output_file)

    assert not result["graph_anomaly_score"].eq(999.0).all()
    assert not result["graph_anomaly_rank"].eq(999).all()


def test_model_is_reproducible_with_fixed_random_state(
    tmp_path,
    monkeypatch,
):
    df = make_features()

    feature_file = tmp_path / "graph_features.csv"
    results_dir = tmp_path / "results"
    output_file_1 = results_dir / "graph_anomalies_1.csv"
    output_file_2 = results_dir / "graph_anomalies_2.csv"

    df.to_csv(feature_file, index=False)

    monkeypatch.setattr(gif, "FEATURE_FILE", feature_file)
    monkeypatch.setattr(gif, "RESULTS_DIR", results_dir)

    monkeypatch.setattr(gif, "OUTPUT_FILE", output_file_1)
    gif.main()

    first = pd.read_csv(output_file_1)

    monkeypatch.setattr(gif, "OUTPUT_FILE", output_file_2)
    gif.main()

    second = pd.read_csv(output_file_2)

    pd.testing.assert_frame_equal(first, second)