
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


CASE_ID = "M57-Jean"

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    BASE_DIR / "data" / "features" / CASE_ID
    / "temporal_features.csv"
)

ANOMALY_PATH = (
    BASE_DIR / "results" / CASE_ID
    / "temporal_anomalies.csv"
)

OUTPUT_DIR = BASE_DIR / "results" / CASE_ID

OUTPUT_PATH = (
    OUTPUT_DIR / "temporal_shap_explanations.csv"
)

MODEL_FEATURES = [
    "gap_log_seconds",
    "local_density_10s",
    "local_density_30s",
    "local_density_60s",
    "process_changed",
]

SCORE_TOLERANCE = 1e-10
SHAP_TOLERANCE = 1e-6


def main():

    print("=== ForensiXplain Temporal SHAP ===")

    # 1. Load inputs

    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    if not ANOMALY_PATH.exists():
        raise FileNotFoundError(ANOMALY_PATH)

    df = pd.read_csv(INPUT_PATH)
    anomaly_df = pd.read_csv(ANOMALY_PATH)

    print(f"Temporal feature rows: {len(df)}")
    print(f"Temporal anomaly rows: {len(anomaly_df)}")

    # 2. Select eligible events

    scoring_df = (
        df[df["is_first_event"] == 0]
        .copy()
        .reset_index(drop=True)
    )

    if scoring_df.empty:
        raise ValueError("No eligible temporal events found.")

    print(f"Events eligible for SHAP: {len(scoring_df)}")

    # 3. Validate required columns

    required_features = MODEL_FEATURES + [
        "logical_event_id",
        "is_first_event",
    ]

    missing = [
        col for col in required_features
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    anomaly_columns = [
        "logical_event_id",
        "temporal_anomaly_score",
        "temporal_predicted_anomaly",
        "temporal_anomaly_rank",
    ]

    missing = [
        col for col in anomaly_columns
        if col not in anomaly_df.columns
    ]

    if missing:
        raise ValueError(f"Missing anomaly columns: {missing}")

    # 4. Prepare features exactly as the detector does

    X = scoring_df[MODEL_FEATURES].copy()

    X = X.fillna(
        X.median(numeric_only=True)
    )

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError(
            "Temporal feature matrix contains non-finite values."
        )

    # 5. Recreate the detector

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=500,
        contamination=0.10,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_scaled)

    recreated_scores = -model.decision_function(X_scaled)

    # 6. Match and validate detector scores

    anomaly_scored = anomaly_df[
        anomaly_df["logical_event_id"].isin(
            scoring_df["logical_event_id"]
        )
    ].copy()

    if anomaly_scored["logical_event_id"].duplicated().any():
        raise ValueError("Duplicate event IDs in anomaly results.")

    result = scoring_df[
        [
            col for col in [
                "case_id",
                "logical_event_id",
                "temporal_sequence",
                "timestamp",
                "process_id",
                "process",
                "time_since_previous_event_seconds",
            ]
            if col in scoring_df.columns
        ]
    ].copy()

    result["recreated_score"] = recreated_scores

    result = result.merge(
        anomaly_scored[anomaly_columns],
        on="logical_event_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    if not (result["_merge"] == "both").all():
        raise ValueError(
            "Some SHAP events could not be matched to detector results."
        )

    result = result.drop(columns="_merge")

    score_difference = np.abs(
        result["recreated_score"].to_numpy()
        - result["temporal_anomaly_score"].to_numpy()
    )

    max_score_difference = float(score_difference.max())

    print(
        "Maximum detector score difference: "
        f"{max_score_difference:.12g}"
    )

    if max_score_difference > SCORE_TOLERANCE:
        raise ValueError(
            "Detector score verification failed. "
            "SHAP generation stopped."
        )

    print("PASS: Detector scores reproduced.")

    # 7. Define anomaly score function

    def anomaly_score_function(data):
        return -model.decision_function(data)

    # 8. Generate permutation SHAP

    print("Running permutation SHAP...")

    explainer = shap.Explainer(
        anomaly_score_function,
        X_scaled,
        algorithm="permutation",
    )

    shap_result = explainer(X_scaled)

    shap_values = np.asarray(shap_result.values)
    base_values = np.asarray(shap_result.base_values).reshape(-1)

    if shap_values.shape != (len(X), len(MODEL_FEATURES)):
        raise ValueError(
            f"Unexpected SHAP shape: {shap_values.shape}"
        )

    if base_values.size == 1:
        base_values = np.repeat(base_values, len(X))

    if base_values.size != len(X):
        raise ValueError(
            f"Unexpected SHAP base-value shape: {base_values.shape}"
        )

    # 9. Verify local accuracy

    reconstructed_scores = (
        base_values + shap_values.sum(axis=1)
    )

    shap_score_difference = np.abs(
        reconstructed_scores - recreated_scores
    )

    max_shap_error = float(shap_score_difference.max())

    print(
        "Maximum SHAP reconstruction error: "
        f"{max_shap_error:.12g}"
    )

    if max_shap_error > SHAP_TOLERANCE:
        raise ValueError(
            "SHAP local accuracy verification failed."
        )

    print("PASS: SHAP local accuracy verified.")

    # 10. Build explanation output

    shap_df = pd.DataFrame(
        shap_values,
        columns=[
            f"shap_{feature}"
            for feature in MODEL_FEATURES
        ],
    )

    for feature in MODEL_FEATURES:
        shap_df[f"value_{feature}"] = X[feature].to_numpy()

    result["shap_base_value"] = base_values
    result["shap_reconstructed_score"] = reconstructed_scores
    result["shap_reconstruction_error"] = (
        reconstructed_scores - recreated_scores
    )

    result = pd.concat(
        [
            result.reset_index(drop=True),
            shap_df.reset_index(drop=True),
        ],
        axis=1,
    )

    # 11. Sort and save

    result = result.sort_values(
        "temporal_anomaly_score",
        ascending=False,
    ).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result.to_csv(OUTPUT_PATH, index=False)

    # 12. Diagnostics

    anomalous = result[
        result["temporal_predicted_anomaly"].astype(bool)
    ]

    print("\n=== Temporal SHAP Complete ===")
    print(f"Rows explained: {len(result)}")
    print(f"Anomalous events explained: {len(anomalous)}")
    print(f"Maximum detector score difference: {max_score_difference:.12g}")
    print(f"Maximum SHAP reconstruction error: {max_shap_error:.12g}")
    print(f"Output: {OUTPUT_PATH}")

    # 13. Print top anomaly explanations

    for _, row in anomalous.iterrows():

        print("\n" + "-" * 55)

        print(f"Rank: {int(row['temporal_anomaly_rank'])}")
        print(
            f"Process: {row['process']} "
            f"(PID {int(row['process_id'])})"
        )
        print(f"Timestamp: {row['timestamp']}")
        print(f"Anomaly score: {row['temporal_anomaly_score']:.6f}")
        print(f"Base value: {row['shap_base_value']:.6f}")

        contributions = []

        for feature in MODEL_FEATURES:
            contributions.append(
                (
                    feature,
                    row[f"shap_{feature}"],
                    row[f"value_{feature}"],
                )
            )

        contributions.sort(
            key=lambda item: abs(item[1]),
            reverse=True,
        )

        print("Top temporal contributors:")

        for feature, shap_value, feature_value in contributions[:3]:

            direction = (
                "increased" if shap_value > 0
                else "decreased"
            )

            print(
                f"  {feature}: {direction} anomaly score "
                f"(SHAP={shap_value:+.6f}, "
                f"value={feature_value:.4f})"
            )


if __name__ == "__main__":
    main()
