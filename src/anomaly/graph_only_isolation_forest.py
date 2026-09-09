"""
ForensiXplain - Graph-Only Isolation Forest

Purpose:
    Detect anomalous process-start events using only
    forensic graph-derived features.

Input:
    data/features/M57-Jean/graph_features.csv

Output:
    results/M57-Jean/graph_only_anomalies.csv

Important:
    - Temporal features are intentionally excluded.
    - Model outputs are never used as input features.
    - Anomaly means an unusual graph-feature profile,
      not necessarily malicious activity.
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ============================================================
# Configuration
# ============================================================

CASE_ID = "M57-Jean"

RANDOM_STATE = 42
N_ESTIMATORS = 500
CONTAMINATION = 0.10


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / CASE_ID
    / "graph_features.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / CASE_ID
)

OUTPUT_FILE = (
    RESULTS_DIR
    / "graph_only_anomalies.csv"
)


# ============================================================
# Graph-only model features
# ============================================================

# IMPORTANT:
# These are graph-derived features only.
#
# Temporal features such as:
#   gap_log_seconds
#   local_density_10s
#   local_density_30s
#   local_density_60s
#   process_changed
#
# are intentionally NOT included.

GRAPH_MODEL_FEATURES = [
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
# Main
# ============================================================

def main():

    print(
        "=== ForensiXplain Graph-Only Isolation Forest ==="
    )

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not FEATURE_FILE.exists():

        raise FileNotFoundError(
            f"Graph feature file not found:\n{FEATURE_FILE}"
        )

    # --------------------------------------------------------
    # Load graph features
    # --------------------------------------------------------

    df = pd.read_csv(
        FEATURE_FILE
    )

    print(
        f"Feature rows: {len(df)}"
    )

    # --------------------------------------------------------
    # Validate graph features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in GRAPH_MODEL_FEATURES
        if feature not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing required graph features:\n"
            + "\n".join(
                f"  - {feature}"
                for feature in missing_features
            )
        )

    # --------------------------------------------------------
    # Prepare model matrix
    # --------------------------------------------------------

    X = df[
        GRAPH_MODEL_FEATURES
    ].copy()

    # Convert values to numeric.
    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Replace infinite values.
    X = X.replace(
        [float("inf"), float("-inf")],
        pd.NA
    )

    # --------------------------------------------------------
    # Check feature variability
    # --------------------------------------------------------

    print()
    print("Graph-only model features:")

    for feature in GRAPH_MODEL_FEATURES:

        unique_count = X[feature].nunique(
            dropna=False
        )

        print(
            f"  {feature}: "
            f"{unique_count} unique values"
        )

    # --------------------------------------------------------
    # Median imputation
    # --------------------------------------------------------

    X = X.fillna(
        X.median()
    )

    # --------------------------------------------------------
    # Standardization
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
    # Anomaly score
    # --------------------------------------------------------

    # Isolation Forest decision_function:
    # higher = more normal
    # lower = more anomalous
    #
    # Therefore negate it so:
    # higher = more anomalous

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

    result_columns = [
        "case_id",
        "logical_event_id",
        "temporal_sequence",
        "timestamp",
        "process_id",
        "process",
    ]

    results = df[
        result_columns
    ].copy()

    results[
        "graph_only_anomaly_score"
    ] = anomaly_score

    results[
        "graph_only_predicted_anomaly"
    ] = predicted_anomaly

    # --------------------------------------------------------
    # Rank
    # --------------------------------------------------------

    results = results.sort_values(
        by="graph_only_anomaly_score",
        ascending=False,
    ).reset_index(
        drop=True
    )

    results[
        "graph_only_anomaly_rank"
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

    print()
    print(
        "=== Graph-Only Isolation Forest Complete ==="
    )

    print(
        f"Total events: {len(results)}"
    )

    print(
        f"Events scored: {len(results)}"
    )

    print(
        f"Graph-only anomalies: {anomaly_count}"
    )

    print(
        f"Normal events: {normal_count}"
    )

    print(
        f"Model features: "
        f"{len(GRAPH_MODEL_FEATURES)}"
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # Top candidates
    # --------------------------------------------------------

    print()
    print(
        "Top graph-only anomaly candidates:"
    )

    top_columns = [
        "graph_only_anomaly_rank",
        "process_id",
        "process",
        "timestamp",
        "graph_only_anomaly_score",
        "graph_only_predicted_anomaly",
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