from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "results" / "M57-Jean" / "temporal_shap_explanations.csv"
OUTPUT = ROOT / "results" / "M57-Jean" / "figure5_temporal_shap_top_anomaly.png"


FEATURES = [
    "gap_log_seconds",
    "local_density_10s",
    "local_density_30s",
    "local_density_60s",
    "process_changed",
]

TOP_EVENT_ID = "LEVT-M57-Jean-PROCESS-3992-20091123170451"


def main():
    df = pd.read_csv(INPUT)

    row = df[df["logical_event_id"] == TOP_EVENT_ID]

    if row.empty:
        raise ValueError(
            f"Top anomaly event not found: {TOP_EVENT_ID}"
        )

    row = row.iloc[0]

    contributions = [
        (feature, float(row[f"shap_{feature}"]))
        for feature in FEATURES
    ]

    # Display largest contribution first.
    contributions.sort(
        key=lambda item: abs(item[1]),
        reverse=True,
    )

    # The permutation SHAP explanation has a base value.
    # Obtain it from the SHAP output if available; otherwise
    # use the anomaly score minus the total SHAP contribution.
    anomaly_score = float(
        row["temporal_anomaly_score"]
    )

    shap_sum = sum(
        value for _, value in contributions
    )

    base_value = anomaly_score - shap_sum

    labels = ["Base value"]
    values = [base_value]

    for feature, value in contributions:
        labels.append(feature)
        values.append(value)

    labels.append("Anomaly score")
    values.append(anomaly_score)

    # Build cumulative waterfall positions.
    cumulative = [base_value]

    for value in [v for _, v in contributions]:
        cumulative.append(cumulative[-1] + value)

    cumulative.append(anomaly_score)

    fig, ax = plt.subplots(figsize=(12, 7))

    # Base value
    ax.bar(
        0,
        base_value,
        width=0.65,
        alpha=0.8,
    )

    # SHAP contribution bars
    running = base_value

    for i, (_, value) in enumerate(contributions, start=1):
        start = running
        end = running + value

        bottom = min(start, end)
        height = abs(value)

        ax.bar(
            i,
            height,
            bottom=bottom,
            width=0.65,
            alpha=0.85,
        )

        ax.text(
            i,
            end + (0.003 if value >= 0 else -0.003),
            f"{value:+.4f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=10,
        )

        running = end

    # Final anomaly score
    ax.bar(
        len(labels) - 1,
        anomaly_score,
        width=0.65,
        alpha=0.8,
    )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(
        labels,
        rotation=25,
        ha="right",
    )

    ax.set_ylabel(
        "Anomaly Score Contribution"
    )

    ax.set_title(
        "SHAP Waterfall Explanation – Top Temporal Anomaly",
        fontsize=16,
        fontweight="bold",
    )

    ax.text(
        0.02,
        0.95,
        (
            f"Process: {row['process']} "
            f"(PID {int(row['process_id'])})\n"
            f"Anomaly score: {anomaly_score:.6f}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox=dict(
            boxstyle="round,pad=0.4",
            alpha=0.9,
        ),
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Figure 5 generated successfully.")
    print(f"Process: {row['process']}")
    print(f"PID: {int(row['process_id'])}")
    print(f"Anomaly score: {anomaly_score:.6f}")
    print(f"SHAP base value: {base_value:.6f}")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
