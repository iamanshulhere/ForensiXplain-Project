"""
Generate Figure 9: Global Graph SHAP Feature Importance.

Input:
    results/M57-Jean/graph_shap_explanations.csv

Output:
    results/M57-Jean/figure9_graph_shap_global_importance.png

Global importance is calculated as mean(abs(SHAP)) over all logical events.
The script reads the existing SHAP export; it does not rerun the explainer.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "results" / "M57-Jean" / "graph_shap_explanations.csv"
OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "M57-Jean"
    / "figure9_graph_shap_global_importance.png"
)

EXPECTED_FEATURES = [
    "gap_log_seconds",
    "local_density_10s",
    "local_density_30s",
    "local_density_60s",
    "process_changed",
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


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"SHAP CSV not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    if len(df) != 47:
        raise ValueError(
            f"Expected 47 logical-event rows, found {len(df)}. "
            "Check the experiment/export before generating the paper figure."
        )

    shap_columns = [f"shap_{feature}" for feature in EXPECTED_FEATURES]
    missing = [column for column in shap_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing SHAP columns: {missing}")

    shap_values = df[shap_columns].apply(pd.to_numeric, errors="coerce")

    if shap_values.isna().any().any():
        raise ValueError("SHAP matrix contains missing or non-numeric values.")

    if not np.isfinite(shap_values.to_numpy()).all():
        raise ValueError("SHAP matrix contains infinite values.")

    # Global feature importance: mean absolute attribution across 47 events.
    importance = shap_values.abs().mean(axis=0)
    importance.index = EXPECTED_FEATURES
    importance = importance.sort_values(ascending=True)

    print("Global mean absolute SHAP importance (47 events):")
    for feature, value in importance.sort_values(ascending=False).items():
        print(f"{feature:30s} {value:.10f}")

    for feature in ("in_degree", "parent_count"):
        value = float(importance[feature])
        if not np.isclose(value, 0.0, atol=1e-12):
            print(
                f"WARNING: {feature} mean absolute SHAP is {value:.12g}, "
                "not zero. Check the manuscript's claim."
            )
        else:
            print(f"Verified zero attribution: {feature}")

    # Publication-quality horizontal bar chart.
    fig_height = max(6.0, 0.42 * len(importance) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    ax.barh(importance.index, importance.values)
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_ylabel("Graph-aware feature")
    ax.set_title(
        "Global Feature Importance — Graph-Aware Isolation Forest",
        pad=14,
    )
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)

    max_value = float(importance.max())
    ax.set_xlim(0, max_value * 1.15 if max_value > 0 else 1)
    fig.tight_layout()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"\nFigure 9 saved to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
