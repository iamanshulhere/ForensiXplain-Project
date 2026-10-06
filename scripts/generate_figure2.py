from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "results" / "M57-Jean" / "temporal_anomalies.csv"
OUTPUT = ROOT / "results" / "M57-Jean" / "figure2_temporal_anomaly_score_distribution.png"


def main():
    df = pd.read_csv(INPUT)

    scored = df[df["temporal_anomaly_score"].notna()].copy()

    scores = scored["temporal_anomaly_score"]
    flagged = scored["temporal_predicted_anomaly"].astype(bool)

    total_events = len(df)
    scored_events = len(scored)
    flagged_events = int(flagged.sum())

    # The saved score representation separates normal events at <= 0
    # from the five model-flagged events at > 0.
    effective_boundary = 0.0

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.hist(
        scores[~flagged],
        bins=20,
        alpha=0.75,
        label=f"Normal events (n={int((~flagged).sum())})",
    )

    ax.hist(
        scores[flagged],
        bins=10,
        alpha=0.9,
        label=f"Flagged events (n={flagged_events})",
    )

    ax.axvline(
        effective_boundary,
        linestyle="--",
        linewidth=2,
        label="Effective decision boundary (0.0)",
    )

    ax.set_title(
        "Temporal Anomaly Score Distribution – M57-Jean",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xlabel("Temporal Anomaly Score")
    ax.set_ylabel("Number of Events")

    ax.text(
        0.98,
        0.95,
        f"Flagged: {flagged_events}/{total_events} events (10.6%)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", alpha=0.9),
    )

    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Figure 2 generated successfully.")
    print(f"Total events: {total_events}")
    print(f"Scored events: {scored_events}")
    print(f"Flagged events: {flagged_events}")
    print(f"Effective boundary: {effective_boundary:.1f}")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
