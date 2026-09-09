"""
ForensiXplain - Graph-Only SHAP Explainability

Purpose:
    Explain Graph-only Isolation Forest anomaly scores
    using only forensic graph-derived features.

Input:
    data/features/M57-Jean/graph_features.csv
    results/M57-Jean/graph_only_anomalies.csv

Output:
    results/M57-Jean/graph_only_shap_explanations.csv

Important:
    - Uses the same 9 graph-only features as graph_only_isolation_forest.py.
    - Temporal features are intentionally excluded.
    - Recreates the same Isolation Forest configuration.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import shap
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

ANOMALY_FILE = (
    PROJECT_ROOT
    / "results"
    / CASE_ID
    / "graph_only_anomalies.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / CASE_ID
    / "graph_only_shap_explanations.csv"
)


# ============================================================
# Graph-only model features
# ============================================================

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

    print("=== ForensiXplain Graph-Only SHAP ===")

    # --------------------------------------------------------
    # Validate input files
    # --------------------------------------------------------

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Graph feature file not found:\n{FEATURE_FILE}"
        )

    if not ANOMALY_FILE.exists():
        raise FileNotFoundError(
            f"Graph-only anomaly file not found:\n{ANOMALY_FILE}"
        )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = pd.read_csv(FEATURE_FILE)
    anomalies = pd.read_csv(ANOMALY_FILE)

    print(f"Feature rows: {len(df)}")
    print(f"Anomaly rows: {len(anomalies)}")

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

    X = df[GRAPH_MODEL_FEATURES].copy()

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.replace(
        [float("inf"), float("-inf")],
        np.nan
    )

    X = X.fillna(
        X.median()
    )

    # --------------------------------------------------------
    # Standardization
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_scaled_df = pd.DataFrame(
        X_scaled,
        columns=GRAPH_MODEL_FEATURES
    )

    # --------------------------------------------------------
    # Recreate Graph-only Isolation Forest
    # --------------------------------------------------------

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_scaled)

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    print()
    print("Creating SHAP explanations...")

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer.shap_values(
        X_scaled_df
    )

    # --------------------------------------------------------
    # Handle SHAP output format
    # --------------------------------------------------------

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.asarray(
        shap_values
    )

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]

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

    # Add graph-only anomaly metadata

    anomaly_metadata = anomalies[
        [
            "logical_event_id",
            "graph_only_anomaly_rank",
            "graph_only_anomaly_score",
            "graph_only_predicted_anomaly",
        ]
    ].copy()

    results = results.merge(
        anomaly_metadata,
        on="logical_event_id",
        how="left",
    )

    # --------------------------------------------------------
    # Add SHAP values
    # --------------------------------------------------------

    for index, feature in enumerate(
        GRAPH_MODEL_FEATURES
    ):

        results[
            f"shap_{feature}"
        ] = shap_values[:, index]

    # --------------------------------------------------------
    # SHAP magnitude
    # --------------------------------------------------------

    shap_feature_columns = [
        f"shap_{feature}"
        for feature in GRAPH_MODEL_FEATURES
    ]

    results[
        "shap_total_abs"
    ] = results[
        shap_feature_columns
    ].abs().sum(axis=1)

    # --------------------------------------------------------
    # Top contributing feature
    # --------------------------------------------------------

    results[
        "top_graph_feature"
    ] = results[
        shap_feature_columns
    ].abs().idxmax(axis=1)

    results[
        "top_graph_feature"
    ] = results[
        "top_graph_feature"
    ].str.replace(
        "shap_",
        "",
        regex=False
    )

    results[
        "top_graph_feature_shap"
    ] = results.apply(
        lambda row: row[
            f"shap_{row['top_graph_feature']}"
        ],
        axis=1
    )

    # --------------------------------------------------------
    # Sort by anomaly rank
    # --------------------------------------------------------

    results = results.sort_values(
        by="graph_only_anomaly_rank",
        ascending=True,
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
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

    print()
    print(
        "=== Graph-Only SHAP Complete ==="
    )

    print(
        f"Total events explained: {len(results)}"
    )

    print(
        f"SHAP features: {len(GRAPH_MODEL_FEATURES)}"
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # Top anomaly explanations
    # --------------------------------------------------------

    print()
    print(
        "Top graph-only anomaly explanations:"
    )

    display_columns = [
        "graph_only_anomaly_rank",
        "process_id",
        "process",
        "graph_only_anomaly_score",
        "top_graph_feature",
        "top_graph_feature_shap",
    ]

    print(
        results[
            display_columns
        ].head(10).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()