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
# Temporal anomaly features
#
# These describe temporal behavior rather than
# forensic identity or evidence.
# ---------------------------------------------------------

MODEL_FEATURES = [
    "gap_log_seconds",
    "local_density_10s",
    "local_density_30s",
    "local_density_60s",
    "process_changed",
]


def main():

    print(
        "=== ForensiXplain Temporal Isolation Forest v2 ==="
    )

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
    # Exclude first timeline event
    #
    # It has no previous temporal context and therefore
    # should not be scored by a gap-based detector.
    # -----------------------------------------------------

    scoring_df = df[
        df["is_first_event"] == 0
    ].copy()

    print(
        f"Events available for temporal scoring: "
        f"{len(scoring_df)}"
    )

    # -----------------------------------------------------
    # Validate features
    # -----------------------------------------------------

    missing_features = [
        feature
        for feature in MODEL_FEATURES
        if feature not in scoring_df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing model features:\n"
            + "\n".join(missing_features)
        )

    # -----------------------------------------------------
    # Prepare numerical matrix
    # -----------------------------------------------------

    X = scoring_df[
        MODEL_FEATURES
    ].copy()

    # First event has already been removed, but NaN
    # protection remains useful for robustness.

    X = X.fillna(
        X.median(numeric_only=True)
    )

    # -----------------------------------------------------
    # Standardization
    # -----------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # -----------------------------------------------------
    # Isolation Forest
    #
    # 10% contamination gives a conservative candidate
    # ranking for this small proof-of-concept dataset.
    # -----------------------------------------------------

    model = IsolationForest(
        n_estimators=500,
        contamination=0.10,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_scaled)

    predictions = model.predict(
        X_scaled
    )

    anomaly_scores = (
        -model.decision_function(X_scaled)
    )

    scoring_df[
        "temporal_anomaly_score"
    ] = anomaly_scores

    scoring_df[
        "temporal_predicted_anomaly"
    ] = (
        predictions == -1
    )

    # -----------------------------------------------------
    # Rank
    # -----------------------------------------------------

    scoring_df[
        "temporal_anomaly_rank"
    ] = (
        scoring_df[
            "temporal_anomaly_score"
        ]
        .rank(
            ascending=False,
            method="first",
        )
        .astype(int)
    )

    # -----------------------------------------------------
    # Restore first event
    #
    # It remains in the output for timeline completeness,
    # but is NOT classified as anomalous.
    # -----------------------------------------------------

    first_event = df[
        df["is_first_event"] == 1
    ].copy()

    if not first_event.empty:

        first_event[
            "temporal_anomaly_score"
        ] = 0.0

        first_event[
            "temporal_predicted_anomaly"
        ] = False

        first_event[
            "temporal_anomaly_rank"
        ] = pd.NA

    # -----------------------------------------------------
    # Combine
    # -----------------------------------------------------

    result = pd.concat(
        [
            scoring_df,
            first_event,
        ],
        ignore_index=True,
    )

    result = result.sort_values(
        "temporal_sequence"
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

    anomaly_count = int(
        result[
            "temporal_predicted_anomaly"
        ].sum()
    )

    normal_count = (
        len(result) - anomaly_count
    )

    print()
    print(
        "=== Temporal Anomaly Detection Complete ==="
    )

    print(
        f"Total timeline events: {len(result)}"
    )

    print(
        f"Events scored: {len(scoring_df)}"
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
    # Top candidates
    # -----------------------------------------------------

    top = (
        result[
            result[
                "temporal_predicted_anomaly"
            ]
        ]
        .sort_values(
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
                "gap_log_seconds",
                "local_density_10s",
                "local_density_30s",
                "local_density_60s",
                "process_changed",
                "temporal_anomaly_score",
            ]
        ]
        .head(15)
    )

    print()
    print(
        "Top temporal anomaly candidates:"
    )

    print(
        top.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()