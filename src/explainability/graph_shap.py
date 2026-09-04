"""
ForensiXplain - Graph-Aware Isolation Forest SHAP

Purpose:
    Explain graph-aware anomaly scores using fast Tree SHAP.

Input:
    data/features/M57-Jean/graph_features.csv
    results/M57-Jean/graph_anomalies.csv

Output:
    results/M57-Jean/graph_shap_explanations.csv

Important:
    The Isolation Forest configuration and model features must
    exactly match graph_isolation_forest.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import shap


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

ANOMALY_FILE = (
    PROJECT_ROOT
    / "results"
    / "M57-Jean"
    / "graph_anomalies.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "M57-Jean"
)

OUTPUT_FILE = (
    RESULTS_DIR
    / "graph_shap_explanations.csv"
)


# ============================================================
# Exact model features
# ============================================================

MODEL_FEATURES = [
    # Temporal
    "gap_log_seconds",
    "local_density_10s",
    "local_density_30s",
    "local_density_60s",
    "process_changed",

    # Graph
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
# Exact model configuration
# ============================================================

RANDOM_STATE = 42
N_ESTIMATORS = 500
CONTAMINATION = 0.10


# ============================================================
# Main
# ============================================================

def main():

    print("=== ForensiXplain Graph SHAP ===")

    # --------------------------------------------------------
    # Validate files
    # --------------------------------------------------------

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature file not found:\n{FEATURE_FILE}"
        )

    if not ANOMALY_FILE.exists():
        raise FileNotFoundError(
            f"Anomaly file not found:\n{ANOMALY_FILE}"
        )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    feature_df = pd.read_csv(FEATURE_FILE)

    anomaly_df = pd.read_csv(ANOMALY_FILE)

    print(f"Feature rows: {len(feature_df)}")
    print(f"Anomaly rows: {len(anomaly_df)}")

    # --------------------------------------------------------
    # Validate features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in MODEL_FEATURES
        if feature not in feature_df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing model features:\n"
            + "\n".join(
                f"  - {feature}"
                for feature in missing_features
            )
        )

    # --------------------------------------------------------
    # Prepare feature matrix
    # --------------------------------------------------------

    X = feature_df[MODEL_FEATURES].copy()

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(
        X.median()
    )

    # --------------------------------------------------------
    # Scale
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # IMPORTANT:
    # Use a NumPy array for the Isolation Forest.
    # This avoids sklearn feature-name warnings.
    X_scaled = np.asarray(
        X_scaled,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Recreate EXACT Isolation Forest
    # --------------------------------------------------------

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_scaled)

    # --------------------------------------------------------
    # Verify scores against graph_anomalies.csv
    # --------------------------------------------------------

    recreated_scores = (
        -model.decision_function(
            X_scaled
        )
    )

    verification_df = feature_df[
        ["logical_event_id"]
    ].copy()

    verification_df[
        "recreated_score"
    ] = recreated_scores

    verification_df = verification_df.merge(
        anomaly_df[
            [
                "logical_event_id",
                "graph_anomaly_score",
            ]
        ],
        on="logical_event_id",
        how="left",
        validate="one_to_one",
    )

    if verification_df[
        "graph_anomaly_score"
    ].isna().any():

        raise ValueError(
            "Some logical_event_id values could not "
            "be matched with graph anomaly scores."
        )

    score_difference = np.abs(
        verification_df[
            "recreated_score"
        ]
        -
        verification_df[
            "graph_anomaly_score"
        ]
    )

    max_difference = score_difference.max()

    print(
        f"Maximum score reproduction difference: "
        f"{max_difference:.12f}"
    )

    if max_difference > 1e-10:

        raise ValueError(
            "Recreated Isolation Forest scores do not "
            "match graph_anomalies.csv."
        )

    print(
        "Isolation Forest score reproduction: PASS"
    )

    # ========================================================
    # Fast SHAP
    # ========================================================

    print("")
    print("Running Tree SHAP...")

    # --------------------------------------------------------
    # TreeExplainer
    # --------------------------------------------------------
    #
    # Isolation Forest is an ensemble of decision trees.
    # TreeExplainer avoids the extremely expensive permutation
    # process that previously caused KeyboardInterrupt.
    #
    # We explain the model's raw tree output. The anomaly
    # score from graph_anomalies.csv remains the authoritative
    # score used in the final results.
    # --------------------------------------------------------

    try:

        explainer = shap.TreeExplainer(
            model
        )

        shap_output = explainer.shap_values(
            X_scaled
        )

    except Exception as exc:

        raise RuntimeError(
            "Tree SHAP failed for the Isolation Forest.\n"
            f"Error: {exc}"
        ) from exc

    # --------------------------------------------------------
    # Normalize SHAP output
    # --------------------------------------------------------

    if isinstance(shap_output, list):

        if len(shap_output) != 1:
            raise ValueError(
                "Unexpected multi-output SHAP result."
            )

        shap_matrix = np.asarray(
            shap_output[0]
        )

    else:

        shap_matrix = np.asarray(
            shap_output
        )

    # --------------------------------------------------------
    # Validate SHAP shape
    # --------------------------------------------------------

    expected_shape = (
        len(X_scaled),
        len(MODEL_FEATURES),
    )

    if shap_matrix.shape != expected_shape:

        raise ValueError(
            "Unexpected SHAP output shape: "
            f"{shap_matrix.shape}; "
            f"expected {expected_shape}"
        )

    print(
        f"SHAP matrix shape: {shap_matrix.shape}"
    )

    # ========================================================
    # Metadata
    # ========================================================

    metadata_columns = [
        "case_id",
        "logical_event_id",
        "temporal_sequence",
        "timestamp",
        "process_id",
        "process",
    ]

    metadata_columns = [
        column
        for column in metadata_columns
        if column in feature_df.columns
    ]

    results = feature_df[
        metadata_columns
    ].copy()

    # ========================================================
    # Anomaly metadata
    # ========================================================

    anomaly_metadata_columns = [
        "logical_event_id",
        "graph_anomaly_score",
        "graph_predicted_anomaly",
        "graph_anomaly_rank",
    ]

    anomaly_metadata_columns = [
        column
        for column in anomaly_metadata_columns
        if column in anomaly_df.columns
    ]

    results = results.merge(
        anomaly_df[
            anomaly_metadata_columns
        ],
        on="logical_event_id",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Validate anomaly metadata
    # --------------------------------------------------------

    if results[
        "graph_anomaly_score"
    ].isna().any():

        raise ValueError(
            "Missing graph anomaly scores after merge."
        )

    # ========================================================
    # Original feature values + SHAP values
    # ========================================================

    for index, feature in enumerate(
        MODEL_FEATURES
    ):

        results[
            f"value_{feature}"
        ] = X[
            feature
        ].values

        results[
            f"shap_{feature}"
        ] = shap_matrix[
            :,
            index
        ]

    # ========================================================
    # Sort by anomaly score
    # ========================================================

    results = results.sort_values(
        by="graph_anomaly_score",
        ascending=False,
    ).reset_index(
        drop=True
    )

    # ========================================================
    # Save
    # ========================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    # ========================================================
    # Print summary
    # ========================================================

    anomalous_rows = results[
        results[
            "graph_predicted_anomaly"
        ] == True
    ]

    print("")
    print(
        "=== Graph SHAP Complete ==="
    )

    print(
        f"Rows explained: {len(results)}"
    )

    print(
        f"Anomalous events explained: "
        f"{len(anomalous_rows)}"
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )

    # ========================================================
    # Top SHAP contributors
    # ========================================================

    print("")
    print(
        "Top SHAP contributors for graph anomalies:"
    )

    for _, row in anomalous_rows.iterrows():

        contributions = []

        for feature in MODEL_FEATURES:

            shap_column = (
                f"shap_{feature}"
            )

            value_column = (
                f"value_{feature}"
            )

            contributions.append(
                (
                    feature,
                    float(
                        row[
                            value_column
                        ]
                    ),
                    float(
                        row[
                            shap_column
                        ]
                    ),
                )
            )

        contributions.sort(
            key=lambda item: abs(item[2]),
            reverse=True
        )

        print("")

        print(
            f"Rank {int(row['graph_anomaly_rank'])}: "
            f"PID {int(row['process_id'])} "
            f"{row['process']} "
            f"score={row['graph_anomaly_score']:.6f}"
        )

        for feature, value, contribution in (
            contributions[:3]
        ):

            print(
                f"  {feature}: "
                f"value={value:.6f}, "
                f"SHAP={contribution:+.6f}"
            )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()