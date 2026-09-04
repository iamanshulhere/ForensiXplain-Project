from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


CASE_ID = "M57-Jean"

FEATURES_PATH = Path(
    f"data/features/{CASE_ID}/features.csv"
)

OUTPUT_PATH = Path(
    f"results/{CASE_ID}/isolation_forest_results.csv"
)


# These are identifiers / metadata, not model features.
EXCLUDE_COLUMNS = [
    "case_id",
    "process_id",
    "process_name",
    "create_time",
    "parent_count",
    "timestamp_available",
]


def main():

    print("=== ForensiXplain Isolation Forest Baseline ===")

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Features file not found: {FEATURES_PATH}"
        )

    features = pd.read_csv(
        FEATURES_PATH,
        low_memory=False,
    )

    print(
        f"Feature rows loaded: {len(features)}"
    )

    # Select numerical behavioral features.
    model_columns = [
        column
        for column in features.columns
        if column not in EXCLUDE_COLUMNS
    ]

    X = features[model_columns].copy()

    # Replace any remaining missing values.
    X = X.fillna(0)

    print("\nModel features:")

    for column in model_columns:
        print(f"  {column}")

    # Standardization makes feature scales comparable.
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # Unsupervised anomaly detection.
    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_scaled)

    # Isolation Forest:
    # higher decision_function = more normal
    # lower decision_function = more anomalous
    decision_score = model.decision_function(
        X_scaled
    )

    # Convert to an anomaly score where
    # higher = more anomalous.
    anomaly_score = -decision_score

    prediction = model.predict(
        X_scaled
    )

    predicted_anomaly = (
        prediction == -1
    ).astype(int)

    results = features[
        [
            "case_id",
            "process_id",
            "process_name",
            "create_time",
        ]
    ].copy()

    results["anomaly_score"] = anomaly_score
    results["predicted_anomaly"] = predicted_anomaly

    results = results.sort_values(
        "anomaly_score",
        ascending=False,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nResults saved to: {OUTPUT_PATH}"
    )

    print(
        f"\nPredicted anomalies: "
        f"{predicted_anomaly.sum()}"
    )

    print(
        f"Normal processes: "
        f"{len(predicted_anomaly) - predicted_anomaly.sum()}"
    )

    print("\nTop 10 anomaly scores:")

    print(
        results.head(10).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()