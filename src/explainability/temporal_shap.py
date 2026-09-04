from pathlib import Path

import pandas as pd
import shap
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


CASE_ID = "M57-Jean"

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    BASE_DIR
    / "data"
    / "features"
    / CASE_ID
    / "temporal_features.csv"
)

ANOMALY_PATH = (
    BASE_DIR
    / "results"
    / CASE_ID
    / "temporal_anomalies.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / CASE_ID
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "temporal_shap_explanations.csv"
)


MODEL_FEATURES = [
    "gap_log_seconds",
    "local_density_10s",
    "local_density_30s",
    "local_density_60s",
    "process_changed",
]


def main():

    print("=== ForensiXplain Temporal SHAP ===")

    # -----------------------------------------------------
    # Load temporal features
    # -----------------------------------------------------

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Temporal features not found:\n{INPUT_PATH}"
        )

    if not ANOMALY_PATH.exists():
        raise FileNotFoundError(
            f"Temporal anomaly results not found:\n{ANOMALY_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    anomalies = pd.read_csv(
        ANOMALY_PATH
    )

    print(
        f"Temporal feature rows: {len(df)}"
    )

    # -----------------------------------------------------
    # Only score events eligible for temporal modeling
    # -----------------------------------------------------

    scoring_df = df[
        df["is_first_event"] == 0
    ].copy()

    X = scoring_df[
        MODEL_FEATURES
    ].copy()

    X = X.fillna(
        X.median(numeric_only=True)
    )

    # -----------------------------------------------------
    # Scaling
    # -----------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # -----------------------------------------------------
    # Recreate the exact Isolation Forest
    # -----------------------------------------------------

    model = IsolationForest(
        n_estimators=500,
        contamination=0.10,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_scaled)

    # -----------------------------------------------------
    # Anomaly score function
    #
    # SHAP explains the same score used by the detector.
    # -----------------------------------------------------

    def anomaly_score_function(X_input):

        return -model.decision_function(
            X_input
        )

    # -----------------------------------------------------
    # SHAP
    #
    # Permutation is model-agnostic and appropriate here
    # because Isolation Forest does not have a simple
    # native SHAP TreeExplainer formulation for our
    # transformed anomaly-score function.
    # -----------------------------------------------------

    explainer = shap.Explainer(
        anomaly_score_function,
        X_scaled,
        algorithm="permutation",
    )

    shap_result = explainer(
        X_scaled
    )

    shap_values = shap_result.values

    # -----------------------------------------------------
    # Create SHAP dataframe
    # -----------------------------------------------------

    shap_df = pd.DataFrame(
        shap_values,
        columns=[
            f"shap_{feature}"
            for feature in MODEL_FEATURES
        ],
    )

    # -----------------------------------------------------
    # Add feature metadata
    # -----------------------------------------------------

    for feature in MODEL_FEATURES:

        shap_df[
            f"value_{feature}"
        ] = X[feature].values

    # -----------------------------------------------------
    # Add temporal event identifiers
    # -----------------------------------------------------

    metadata_columns = [
        "case_id",
        "logical_event_id",
        "temporal_sequence",
        "timestamp",
        "process_id",
        "process",
        "time_since_previous_event_seconds",
        "temporal_anomaly_score",
        "temporal_predicted_anomaly",
        "temporal_anomaly_rank",
    ]

    metadata = scoring_df[
        metadata_columns
    ].reset_index(drop=True)

    result = pd.concat(
        [
            metadata,
            shap_df.reset_index(drop=True),
        ],
        axis=1,
    )

    # -----------------------------------------------------
    # Keep anomaly results
    # -----------------------------------------------------

    result = result.sort_values(
        "temporal_anomaly_score",
        ascending=False,
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # -----------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------

    print()
    print(
        "=== Temporal SHAP Complete ==="
    )

    print(
        f"Rows explained: {len(result)}"
    )

    print(
        f"Columns: {len(result.columns)}"
    )

    print(
        f"Output: {OUTPUT_PATH}"
    )

    # -----------------------------------------------------
    # Print explanations for anomalies
    # -----------------------------------------------------

    anomaly_results = result[
        result[
            "temporal_predicted_anomaly"
        ] == True
    ].copy()

    print()
    print(
        f"Anomalous events explained: "
        f"{len(anomaly_results)}"
    )

    for _, row in anomaly_results.iterrows():

        print()
        print(
            "--------------------------------------------------"
        )

        print(
            f"Rank: {int(row['temporal_anomaly_rank'])}"
        )

        print(
            f"Process: {row['process']} "
            f"(PID {int(row['process_id'])})"
        )

        print(
            f"Timestamp: {row['timestamp']}"
        )

        print(
            f"Anomaly score: "
            f"{row['temporal_anomaly_score']:.6f}"
        )

        contributions = []

        for feature in MODEL_FEATURES:

            value = row[
                f"shap_{feature}"
            ]

            contributions.append(
                (
                    feature,
                    value,
                    row[
                        f"value_{feature}"
                    ],
                )
            )

        contributions.sort(
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        print(
            "Top temporal contributors:"
        )

        for feature, shap_value, feature_value in (
            contributions[:3]
        ):

            direction = (
                "increased"
                if shap_value > 0
                else "decreased"
            )

            print(
                f"  {feature}: "
                f"{direction} anomaly score "
                f"(SHAP={shap_value:+.6f}, "
                f"value={feature_value:.4f})"
            )


if __name__ == "__main__":
    main()