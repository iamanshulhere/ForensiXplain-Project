"""
ForensiXplain - Graph-Aware Isolation Forest

Purpose:
    Detect anomalous process-start events using a combination
    of temporal and forensic graph features.

Input:
    data/features/M57-Jean/graph_features.csv

Output:
    results/M57-Jean/graph_anomalies.csv

Important:
    - Model outputs are never used as input features.
    - This model detects unusual feature profiles.
    - Anomaly does NOT mean malicious activity.
"""

from pathlib import Path

import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "M57-Jean"
    / "graph_features.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "M57-Jean"
)

OUTPUT_FILE = (
    RESULTS_DIR
    / "graph_anomalies.csv"
)


# ============================================================
# Model features
# ============================================================

MODEL_FEATURES = [
    # -------------------------
    # Temporal features
    # -------------------------
    "gap_log_seconds",
    "local_density_10s",
    "local_density_30s",
    "local_density_60s",
    "process_changed",

    # -------------------------
    # Graph features
    # -------------------------
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


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42

N_ESTIMATORS = 500

CONTAMINATION = 0.10


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=== ForensiXplain Graph-Aware Isolation Forest ==="
    )

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not FEATURE_FILE.exists():

        raise FileNotFoundError(
            f"Feature file not found:\n{FEATURE_FILE}"
        )

    # --------------------------------------------------------
    # Load features
    # --------------------------------------------------------

    df = pd.read_csv(
        FEATURE_FILE
    )

    print(
        f"Feature rows: {len(df)}"
    )

    # --------------------------------------------------------
    # Validate model features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in MODEL_FEATURES
        if feature not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing required model features:\n"
            + "\n".join(
                f"  - {feature}"
                for feature in missing_features
            )
        )

    # --------------------------------------------------------
    # Prepare model matrix
    # --------------------------------------------------------

    X = df[
        MODEL_FEATURES
    ].copy()

    # --------------------------------------------------------
    # Handle missing / invalid values
    # --------------------------------------------------------

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Replace infinite values
    X = X.replace(
        [float("inf"), float("-inf")],
        pd.NA
    )

    # Median imputation
    X = X.fillna(
        X.median()
    )

    # --------------------------------------------------------
    # Scale features
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )

    # --------------------------------------------------------
    # Isolation Forest
    # --------------------------------------------------------

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        X_scaled
    )

    # --------------------------------------------------------
    # Calculate anomaly score
    # --------------------------------------------------------

    # Larger value = more anomalous
    anomaly_score = (
        -model.decision_function(
            X_scaled
        )
    )

    prediction = model.predict(
        X_scaled
    )

    predicted_anomaly = (
        prediction == -1
    )

    # --------------------------------------------------------
    # Build result dataframe
    # --------------------------------------------------------

    results = df[
        [
            "case_id",
            "logical_event_id",
            "temporal_sequence",
            "timestamp",
            "process_id",
            "process",
        ]
    ].copy()

    results[
        "graph_anomaly_score"
    ] = anomaly_score

    results[
        "graph_predicted_anomaly"
    ] = predicted_anomaly

    # --------------------------------------------------------
    # Rank anomalies
    # --------------------------------------------------------

    results = results.sort_values(
        by="graph_anomaly_score",
        ascending=False
    ).reset_index(
        drop=True
    )

    results[
        "graph_anomaly_rank"
    ] = range(
        1,
        len(results) + 1
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    anomaly_count = int(
        predicted_anomaly.sum()
    )

    normal_count = (
        len(results)
        - anomaly_count
    )

    print("")
    print(
        "=== Graph-Aware Isolation Forest Complete ==="
    )

    print(
        f"Total events: {len(results)}"
    )

    print(
        f"Events scored: {len(results)}"
    )

    print(
        f"Graph anomalies: {anomaly_count}"
    )

    print(
        f"Normal events: {normal_count}"
    )

    print(
        f"Model features: {len(MODEL_FEATURES)}"
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # Top anomalies
    # --------------------------------------------------------

    print("")
    print(
        "Top graph anomaly candidates:"
    )

    top_columns = [
        "graph_anomaly_rank",
        "process_id",
        "process",
        "timestamp",
        "graph_anomaly_score",
        "graph_predicted_anomaly",
    ]

    print(
        results[
            top_columns
        ].head(10).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()