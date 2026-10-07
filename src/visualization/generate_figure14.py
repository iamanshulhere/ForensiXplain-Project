from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


OUTPUT = Path("results/M57-Jean/figure14_end_to_end_explanation_flow.png")


STEPS = [
    ("Raw forensic\nEvent", "Forensic artifacts"),
    ("Temporal\nFeatures", "Timeline + context"),
    ("Anomaly\nScore", "Unsupervised score"),
    ("SHAP\nExplanation", "Feature contribution"),
    ("Graph\nContext", "Process relationships"),
    ("Supporting Forensic\nEvidence", "Evidence + provenance"),
    ("Investigator-Readable\nExplanation", "Ranked, contextual finding"),
]


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 4.8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")

    x_positions = [1, 3, 5, 7, 9, 11, 13]
    y = 2.5
    width = 1.65
    height = 1.35

    for i, ((title, subtitle), x) in enumerate(zip(STEPS, x_positions)):
        box = FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.05,rounding_size=0.08",
            linewidth=1.5,
            facecolor="white",
            edgecolor="black",
        )
        ax.add_patch(box)

        ax.text(
            x,
            y + 0.20,
            title,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
        ax.text(
            x,
            y - 0.38,
            subtitle,
            ha="center",
            va="center",
            fontsize=8,
        )

        if i < len(STEPS) - 1:
            arrow = FancyArrowPatch(
                (x + width / 2 + 0.08, y),
                (x_positions[i + 1] - width / 2 - 0.08, y),
                arrowstyle="-|>",
                mutation_scale=16,
                linewidth=1.3,
                color="black",
            )
            ax.add_patch(arrow)

    ax.text(
        7,
        4.45,
        "ForensiXplain: End-to-End Explainable Forensic Investigation Flow",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )

    ax.text(
        7,
        0.55,
        "Evidence is progressively transformed into an anomaly finding with temporal, "
        "graph, explainability, and provenance context.",
        ha="center",
        va="center",
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Figure 14 generated: {OUTPUT}")


if __name__ == "__main__":
    main()
