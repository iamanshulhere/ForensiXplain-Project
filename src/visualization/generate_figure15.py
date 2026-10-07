from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


OUTPUT = Path("results/M57-Jean/figure15_system_architecture.png")


LAYERS = [
    ("Raw Forensic Artifacts", "Memory image / disk image / forensic events"),
    ("Data Ingestion & Extraction", "Event parsing and artifact extraction"),
    ("Normalization & Logical Timeline", "Canonical schema + ordered logical events"),
    ("Temporal Analysis", "Temporal features + anomaly detection"),
    ("Knowledge Graph Construction", "Processes + commands + modules + memory"),
    ("Graph Anomaly Detection", "Graph features + anomaly scoring"),
    ("Temporal + Graph Fusion", "Controlled score fusion and ranking"),
    ("SHAP + Evidence Attribution", "Feature contribution + forensic evidence"),
    ("Investigator-Readable Output", "Explanations, reports and traceable findings"),
]


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 15))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 18)
    ax.axis("off")

    ax.text(
        5,
        17.35,
        "ForensiXplain System Architecture",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
    )

    ax.text(
        5,
        16.85,
        "Explainable temporal and graph-based forensic investigation pipeline",
        ha="center",
        va="center",
        fontsize=10,
    )

    x = 5
    start_y = 15.8
    step = 1.65
    width = 7.2
    height = 1.05

    for i, (title, description) in enumerate(LAYERS):
        y = start_y - i * step

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
            y + 0.18,
            title,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

        ax.text(
            x,
            y - 0.22,
            description,
            ha="center",
            va="center",
            fontsize=8.5,
        )

        if i < len(LAYERS) - 1:
            next_y = start_y - (i + 1) * step

            arrow = FancyArrowPatch(
                (x, y - height / 2 - 0.05),
                (x, next_y + height / 2 + 0.05),
                arrowstyle="-|>",
                mutation_scale=15,
                linewidth=1.3,
                color="black",
            )
            ax.add_patch(arrow)

    ax.text(
        5,
        0.45,
        "The architecture separates forensic preprocessing, temporal and graph analysis, "
        "fusion, explainability, and investigator-facing evidence reporting.",
        ha="center",
        va="center",
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Figure 15 generated: {OUTPUT}")


if __name__ == "__main__":
    main()
