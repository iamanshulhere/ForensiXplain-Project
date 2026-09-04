from pathlib import Path

import pandas as pd
import shap


from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parents[2]

FEATURE_PATH = BASE_DIR / "data" / "features" / "M57-Jean" / "features.csv"
RESULT_PATH = BASE_DIR / "results" / "M57-Jean" / "isolation_forest_results.csv"

OUTPUT_DIR = BASE_DIR / "results" / "M57-Jean"
OUTPUT_PATH = OUTPUT_DIR / "shap_explanations.csv"


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


def main():

    print("=== ForensiXplain SHAP Explainability ===")

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    features = pd.read_csv(FEATURE_PATH)
    results = pd.read_csv(RESULT_PATH)

    print(f"Feature rows: {len(features)}")

    # ---------------------------------------------------------
    # Prepare model input
    # ---------------------------------------------------------

    X = features[MODEL_FEATURES].copy()

    # Handle any missing numeric values safely
    X = X.fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---------------------------------------------------------
    # Recreate Isolation Forest
    # ---------------------------------------------------------

    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_scaled)

    # ---------------------------------------------------------
    # SHAP explanation
    # ---------------------------------------------------------

    print("Creating SHAP explanations...")

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_scaled)

    shap_df = pd.DataFrame(
        shap_values,
        columns=MODEL_FEATURES
    )

    # ---------------------------------------------------------
    # Build explanation table
    # ---------------------------------------------------------

    output = features[
        [
            "case_id",
            "process_id",
            "process_name",
        ]
    ].copy()

    # Align model results with feature rows using process_id
    result_lookup = results[
        ["process_id", "anomaly_score", "predicted_anomaly"]
    ].copy()

    output = output.merge(
        result_lookup,
        on="process_id",
        how="left",
        validate="one_to_one"
    )

    for feature in MODEL_FEATURES:
        output[f"{feature}_shap"] = shap_df[feature]

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {len(output)}")

    # ---------------------------------------------------------
    # Display top anomalies
    # ---------------------------------------------------------

    anomalies = output[
        output["predicted_anomaly"] == 1
    ].copy()

    print("\nTop anomaly explanations:")

    for _, row in anomalies.head(10).iterrows():

        contributions = {
            feature: abs(row[f"{feature}_shap"])
            for feature in MODEL_FEATURES
        }

        top_features = sorted(
            contributions.items(),
            key=lambda x: x[1],
            reverse=True
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
            print(
                f"  {feature}: "
                f"{value:.6f}"
            )


if __name__ == "__main__":
    main()