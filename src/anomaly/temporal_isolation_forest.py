from pathlib import Path

import pandas as pd
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

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / CASE_ID
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "temporal_anomalies.csv"
)


# ---------------------------------------------------------
# Features used by the temporal anomaly detector
# ---------------------------------------------------------

MODEL_FEATURES = [
    "gap_log_seconds",
    "event_hour",
    "event_day_of_week",
    "after_hours",

    "events_prev_10s",
    "events_next_10s",
    "local_density_10s",

    "events_prev_30s",
    "events_next_30s",
    "local_density_30s",

    "events_prev_60s",
    "events_next_60s",
    "local_density_60s",

    "process_changed",

    "rapid_event",
    "short_event_gap",
    "medium_event_gap",
    "long_event_gap",
]


def main():

    print("=== ForensiXplain Temporal Isolation Forest ===")

    # -----------------------------------------------------
    # Load features
    # -----------------------------------------------------

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Temporal features not found:\n{INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    print(
        f"Temporal feature rows loaded: {len(df)}"
    )

    # -----------------------------------------------------
    # Validate required features
    # -----------------------------------------------------

    missing_features = [
        feature
        for feature in MODEL_FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing model features:\n"
            + "\n".join(missing_features)
        )

    # -----------------------------------------------------
    # Prepare numerical features
    # -----------------------------------------------------

    X = df[MODEL_FEATURES].copy()

    # First event has no previous event.
    # gap_log_seconds is therefore NaN.
    #
    # Use median of observed gaps for modeling.
    # The original NaN is preserved separately in the
    # metadata dataframe.

    X = X.fillna(
        X.median(numeric_only=True)
    )

    # -----------------------------------------------------
    # Scaling
    # -----------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # -----------------------------------------------------
    # Isolation Forest
    # -----------------------------------------------------

    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_scaled)

    predictions = model.predict(X_scaled)

    # Isolation Forest decision_function:
    # larger = more normal
    #
    # We invert it so that:
    # larger anomaly_score = more anomalous

    anomaly_scores = (
        -model.decision_function(X_scaled)
    )

    df["temporal_anomaly_score"] = (
        anomaly_scores
    )

    df["temporal_predicted_anomaly"] = (
        predictions == -1
    )

    # -----------------------------------------------------
    # Rank anomalies
    # -----------------------------------------------------

    df["temporal_anomaly_rank"] = (
        df["temporal_anomaly_score"]
        .rank(
            ascending=False,
            method="first",
        )
        .astype(int)
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # -----------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------

    anomaly_count = int(
        df["temporal_predicted_anomaly"].sum()
    )

    normal_count = (
        len(df) - anomaly_count
    )

    print()
    print(
        "=== Temporal Anomaly Detection Complete ==="
    )

    print(
        f"Total events: {len(df)}"
    )

    print(
        f"Temporal anomalies: {anomaly_count}"
    )

    print(
        f"Normal events: {normal_count}"
    )

    print(
        f"Output: {OUTPUT_PATH}"
    )

    # -----------------------------------------------------
    # Top anomalies
    # -----------------------------------------------------

    top = (
        df.sort_values(
            "temporal_anomaly_score",
            ascending=False,
        )
        [
            [
                "temporal_anomaly_rank",
                "temporal_sequence",
                "timestamp",
                "process_id",
                "process",
                "time_since_previous_event_seconds",
                "local_density_10s",
                "local_density_30s",
                "local_density_60s",
                "after_hours",
                "temporal_anomaly_score",
                "temporal_predicted_anomaly",
            ]
        ]
        .head(15)
    )

    print()
    print("Top temporal anomalies:")

    print(
        top.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()