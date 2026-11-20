from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


DATASET = "M57-Jean"

INPUT_FILE = (
    Path("results")
    / DATASET
    / "graph_shap_explanations.csv"
)

OUTPUT_FIGURE = (
    Path("results")
    / DATASET
    / "graph_shap_feature_attribution.png"
)

OUTPUT_CSV = (
    Path("results")
    / DATASET
    / "graph_shap_feature_summary.csv"
)


def main():
    print("=== ForensiXplain Graph-SHAP Feature Attribution ===")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"Input rows: {len(df)}")
    print(f"Input columns: {len(df.columns)}")

    # ---------------------------------------------------------
    # Find SHAP columns
    # ---------------------------------------------------------
    shap_columns = [
        column
        for column in df.columns
        if column.startswith("shap_")
    ]

    if not shap_columns:
        raise ValueError(
            "No SHAP columns found. "
            f"Available columns: {list(df.columns)}"
        )

    print("SHAP columns:")
    for column in shap_columns:
        print(f"  {column}")

    # ---------------------------------------------------------
    # Calculate mean absolute SHAP
    # ---------------------------------------------------------
    summary_rows = []

    for column in shap_columns:

        feature_name = column.replace("shap_", "")

        values = pd.to_numeric(
            df[column],
            errors="coerce"
        ).dropna()

        if values.empty:
            continue

        summary_rows.append(
            {
                "feature": feature_name,
                "shap_column": column,
                "mean_absolute_shap": values.abs().mean(),
                "mean_shap": values.mean(),
                "max_absolute_shap": values.abs().max(),
            }
        )

    summary = pd.DataFrame(summary_rows)

    if summary.empty:
        raise ValueError(
            "No valid SHAP values were found."
        )

    # ---------------------------------------------------------
    # Sort by mean absolute SHAP
    # ---------------------------------------------------------
    summary = summary.sort_values(
        "mean_absolute_shap",
        ascending=False,
    ).reset_index(drop=True)

    # Keep top features for the paper figure
    top_n = min(10, len(summary))

    plot_data = summary.head(top_n).copy()

    # Reverse for horizontal bar chart
    plot_data = plot_data.iloc[::-1]

    # ---------------------------------------------------------
    # Save summary CSV
    # ---------------------------------------------------------
    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # ---------------------------------------------------------
    # Create figure
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))

    plt.barh(
        plot_data["feature"],
        plot_data["mean_absolute_shap"],
    )

    plt.xlabel("Mean Absolute SHAP Value")
    plt.ylabel("Graph Feature")
    plt.title(
        "Graph-SHAP Feature Attribution — M57-Jean"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_FIGURE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ---------------------------------------------------------
    # Terminal summary
    # ---------------------------------------------------------
    print("")
    print("=== Graph-SHAP Figure Complete ===")
    print("")
    print("Top graph features:")

    print(
        summary[
            [
                "feature",
                "mean_absolute_shap",
                "mean_shap",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("")
    print(f"Figure: {OUTPUT_FIGURE}")
    print(f"Summary: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()