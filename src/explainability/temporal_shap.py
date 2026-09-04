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


# ---------------------------------------------------------
# These MUST match temporal_isolation_forest.py exactly.
# ---------------------------------------------------------

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

    anomaly_df = pd.read_csv(
        ANOMALY_PATH
    )

    print(
        f"Temporal feature rows: {len(df)}"
    )

    print(
        f"Temporal anomaly rows: {len(anomaly_df)}"
    )

    # -----------------------------------------------------
    # Only events eligible for temporal scoring
    # -----------------------------------------------------

    scoring_df = df[
        df["is_first_event"] == 0
    ].copy()

    scoring_df = scoring_df.reset_index(
        drop=True
    )

    print(
        f"Events eligible for SHAP: "
        f"{len(scoring_df)}"
    )

    # -----------------------------------------------------
    # Prepare model features
    # -----------------------------------------------------

    X = scoring_df[
        MODEL_FEATURES
    ].copy()

    X = X.fillna(
        X.median(numeric_only=True)
    )

    # -----------------------------------------------------
    # Standardization
    # -----------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # -----------------------------------------------------
    # Recreate the exact Temporal Isolation Forest
    # -----------------------------------------------------

    model = IsolationForest(
        n_estimators=500,
        contamination=0.10,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_scaled)

    # -----------------------------------------------------
    # Same anomaly score used by detector
    # -----------------------------------------------------

    def anomaly_score_function(X_input):

        return -model.decision_function(
            X_input
        )

    # -----------------------------------------------------
    # SHAP
    # -----------------------------------------------------

    print()
    print(
        "Running permutation SHAP..."
    )

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
    # SHAP dataframe
    # -----------------------------------------------------

    shap_df = pd.DataFrame(
        shap_values,
        columns=[
            f"shap_{feature}"
            for feature in MODEL_FEATURES
        ],
    )

    # Add actual feature values

    for feature in MODEL_FEATURES:

        shap_df[
            f"value_{feature}"
        ] = X[feature].values

    # -----------------------------------------------------
    # Event metadata
    # -----------------------------------------------------

    metadata_columns = [
        "case_id",
        "logical_event_id",
        "temporal_sequence",
        "timestamp",
        "process_id",
        "process",
        "time_since_previous_event_seconds",
    ]

    metadata = scoring_df[
        metadata_columns
    ].reset_index(drop=True)

    # -----------------------------------------------------
    # IMPORTANT FIX
    #
    # Get anomaly score / prediction / rank from the
    # already-generated temporal anomaly results.
    #
    # Match using logical_event_id.
    # -----------------------------------------------------

    anomaly_columns = [
        "logical_event_id",
        "temporal_anomaly_score",
        "temporal_predicted_anomaly",
        "temporal_anomaly_rank",
    ]

    anomaly_metadata = anomaly_df[
        anomaly_columns
    ].copy()

    result = metadata.merge(
        anomaly_metadata,
        on="logical_event_id",
        how="left",
        validate="one_to_one",
    )

    # -----------------------------------------------------
    # Validate merge
    # -----------------------------------------------------

    missing_scores = result[
        "temporal_anomaly_score"
    ].isna().sum()

    if missing_scores > 0:

        raise ValueError(
            f"{missing_scores} SHAP events could not "
            "be matched with temporal anomaly results."
        )

    # -----------------------------------------------------
    # Combine metadata + SHAP
    # -----------------------------------------------------

    result = pd.concat(
        [
            result.reset_index(drop=True),
            shap_df.reset_index(drop=True),
        ],
        axis=1,
    )

    # -----------------------------------------------------
    # Sort by anomaly score
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

    anomaly_results = result[
        result[
            "temporal_predicted_anomaly"
        ] == True
    ].copy()

    print()
    print(
        "=== Temporal SHAP Complete ==="
    )

    print(
        f"Rows explained: {len(result)}"
    )

    print(
        f"Anomalous events explained: "
        f"{len(anomaly_results)}"
    )

    print(
        f"Output: {OUTPUT_PATH}"
    )

    # -----------------------------------------------------
    # Print explanations
    # -----------------------------------------------------

    for _, row in anomaly_results.iterrows():

        print()
        print(
            "--------------------------------------------------"
        )

        print(
            f"Rank: "
            f"{int(row['temporal_anomaly_rank'])}"
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

            shap_value = row[
                f"shap_{feature}"
            ]

            feature_value = row[
                f"value_{feature}"
            ]

            contributions.append(
                (
                    feature,
                    shap_value,
                    feature_value,
                )
            )

        contributions.sort(
            key=lambda item: abs(item[1]),
            reverse=True,
        )

        print(
            "Top temporal contributors:"
        )

        for (
            feature,
            shap_value,
            feature_value,
        ) in contributions[:3]:

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