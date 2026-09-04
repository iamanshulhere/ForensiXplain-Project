from pathlib import Path

import pandas as pd
import shap

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

FEATURE_PATH = (
    BASE_DIR
    / "data"
    / "features"
    / "M57-Jean"
    / "features.csv"
)

RESULT_PATH = (
    BASE_DIR
    / "results"
    / "M57-Jean"
    / "isolation_forest_results.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "M57-Jean"
)

OUTPUT_PATH = OUTPUT_DIR / "shap_explanations.csv"


# =========================================================
# Features used by Isolation Forest
# =========================================================

MODEL_FEATURES = [
    "child_count",
    "command_line_count",
    "module_count",
    "memory_region_count",
    "graph_degree",
    "hour",
    "day_of_week",
    "after_hours",
]


# =========================================================
# Main
# =========================================================

def main():

    print("=== ForensiXplain SHAP Explainability ===")

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------

    features = pd.read_csv(FEATURE_PATH)
    results = pd.read_csv(RESULT_PATH)

    print(f"Feature rows: {len(features)}")
    print(f"Result rows: {len(results)}")

    # -----------------------------------------------------
    # Prepare model input
    # -----------------------------------------------------

    X = features[MODEL_FEATURES].copy()

    # Safely handle missing numeric values
    X = X.fillna(0)

    # -----------------------------------------------------
    # Standardize features
    # -----------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # -----------------------------------------------------
    # Recreate the exact Isolation Forest
    # -----------------------------------------------------

    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_scaled)

    # -----------------------------------------------------
    # Define ForensiXplain anomaly score
    #
    # The original model uses:
    #
    # anomaly_score = -decision_function(X)
    #
    # Therefore SHAP should explain this exact function.
    # -----------------------------------------------------

    def anomaly_score_function(X_input):

        return -model.decision_function(X_input)

    # -----------------------------------------------------
    # SHAP explanation
    # -----------------------------------------------------

    print("Creating SHAP explanations...")

    explainer = shap.Explainer(
        anomaly_score_function,
        X_scaled,
        algorithm="permutation",
    )

    shap_result = explainer(X_scaled)

    shap_values = shap_result.values

    # -----------------------------------------------------
    # Convert SHAP values into DataFrame
    # -----------------------------------------------------

    shap_df = pd.DataFrame(
        shap_values,
        columns=MODEL_FEATURES,
    )

    # -----------------------------------------------------
    # Build base output table
    # -----------------------------------------------------

    output = features[
        [
            "case_id",
            "process_id",
            "process_name",
        ]
    ].copy()

    # -----------------------------------------------------
    # Align model results using process_id
    #
    # This prevents row-order mismatches.
    # -----------------------------------------------------

    result_lookup = results[
        [
            "process_id",
            "anomaly_score",
            "predicted_anomaly",
        ]
    ].copy()

    output = output.merge(
        result_lookup,
        on="process_id",
        how="left",
        validate="one_to_one",
    )

    # -----------------------------------------------------
    # Add SHAP values
    # -----------------------------------------------------

    for feature in MODEL_FEATURES:

        output[f"{feature}_shap"] = shap_df[feature]

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {len(output)}")
    print(f"Columns: {len(output.columns)}")

    # -----------------------------------------------------
    # Display top anomalies
    # -----------------------------------------------------

    anomalies = output[
        output["predicted_anomaly"] == 1
    ].copy()

    # IMPORTANT:
    # Sort by anomaly score rather than feature-row order.
    anomalies = anomalies.sort_values(
        "anomaly_score",
        ascending=False,
    )

    print("\nTop anomaly explanations:")

    for _, row in anomalies.head(10).iterrows():

        # Signed SHAP values
        contributions = {
            feature: row[f"{feature}_shap"]
            for feature in MODEL_FEATURES
        }

        # Rank using absolute magnitude
        top_features = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:3]

        print(
            f"\nPID {int(row['process_id'])} "
            f"({row['process_name']})"
        )

        print(
            f"Anomaly score: "
            f"{row['anomaly_score']:.6f}"
        )

        print("Top contributing features:")

        for feature, value in top_features:

            direction = (
                "increases anomaly score"
                if value > 0
                else "decreases anomaly score"
            )

            print(
                f"  {feature}: "
                f"{value:+.6f} "
                f"({direction})"
            )


if __name__ == "__main__":
    main()