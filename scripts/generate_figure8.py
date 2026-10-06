from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


CASE_ID = "M57-Jean"

INPUT_FILE = (
    Path("results")
    / CASE_ID
    / "graph_shap_explanations.csv"
)

OUTPUT_FILE = (
    Path("results")
    / CASE_ID
    / "figure8_graph_shap_summary.png"
)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Graph SHAP input not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    shap_columns = [
        column
        for column in df.columns
        if column.startswith("shap_")
    ]

    if not shap_columns:
        raise ValueError(
            "No SHAP columns found in the input CSV."
        )

    importance = (
        df[shap_columns]
        .abs()
        .mean()
        .sort_values(ascending=True)
    )

    feature_names = [
        column[len("shap_"):]
        for column in importance.index
    ]

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.barh(
        feature_names,
        importance.values
    )

    ax.set_title(
        f"Graph-SHAP Feature Attribution - {CASE_ID}"
    )

    ax.set_xlabel(
        "Mean Absolute SHAP Value"
    )

    ax.set_ylabel(
        "Graph/Temporal Feature"
    )

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.3
    )

    fig.tight_layout()

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"Figure 8 generated successfully: {OUTPUT_FILE}"
    )

    print(
        f"Rows analyzed: {len(df)}"
    )

    print(
        f"Features analyzed: {len(shap_columns)}"
    )

    print("Mean absolute SHAP values:")

    for feature, value in zip(
        feature_names,
        importance.values
    ):
        print(
            f"  {feature}: {value:.9f}"
        )


if __name__ == "__main__":
    main()
