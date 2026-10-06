from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data" / "features" / "M57-Jean" / "temporal_features.csv"
OUTPUT = ROOT / "results" / "M57-Jean" / "figure6_temporal_feature_correlation_new.png"


EXCLUDE = {
    "process_id",
    "parent_process_ids",
    "previous_process_id",
    "parent_process_id",
}


def main():
    df = pd.read_csv(INPUT)

    numeric = df.select_dtypes(include="number").columns.tolist()

    features = [
        column
        for column in numeric
        if column not in EXCLUDE
    ]

    if len(features) != 22:
        raise ValueError(
            f"Expected 22 correlation-eligible features, found {len(features)}."
        )

    correlation = df[features].corr(method="pearson")

    fig, ax = plt.subplots(
        figsize=(15, 13)
    )

    image = ax.imshow(
        correlation,
        vmin=-1,
        vmax=1,
        aspect="auto",
    )

    ax.set_xticks(range(len(features)))
    ax.set_yticks(range(len(features)))

    ax.set_xticklabels(
        features,
        rotation=75,
        ha="right",
        fontsize=8,
    )

    ax.set_yticklabels(
        features,
        fontsize=8,
    )

    ax.set_title(
        "Temporal Feature Correlation Heatmap – M57-Jean",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    colorbar.set_label(
        "Pearson correlation",
        rotation=270,
        labelpad=18,
    )

    ax.set_xlabel(
        "22 correlation-eligible numeric temporal/behavioral features"
    )

    ax.set_ylabel(
        "22 correlation-eligible numeric temporal/behavioral features"
    )

    fig.text(
        0.5,
        0.01,
        "Identifier-like fields and non-numeric context fields excluded from correlation analysis.",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(
        rect=[0, 0.03, 1, 1]
    )

    fig.savefig(
        OUTPUT,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Figure 6 generated successfully.")
    print(f"Matrix shape: {df.shape}")
    print(f"Correlation features: {len(features)}")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
