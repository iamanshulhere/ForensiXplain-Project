import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVENTS = PROJECT_ROOT / "data" / "normalized" / "M57-Jean" / "events.csv"
TIMELINE = PROJECT_ROOT / "data" / "normalized" / "M57-Jean" / "timeline.csv"
LOGICAL_TIMELINE = PROJECT_ROOT / "data" / "normalized" / "M57-Jean" / "logical_timeline.csv"

OUTPUT = PROJECT_ROOT / "results" / "M57-Jean" / "figure1_pipeline_reduction_funnel.png"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

raw_count = len(pd.read_csv(EVENTS))
timestamped_count = len(pd.read_csv(TIMELINE))
logical_count = len(pd.read_csv(LOGICAL_TIMELINE))

stages = [
    "Raw observations",
    "Timestamped events",
    "Logical events",
]

counts = [
    raw_count,
    timestamped_count,
    logical_count,
]

fig, ax = plt.subplots(figsize=(12, 7))

bars = ax.barh(stages, counts)

ax.invert_yaxis()
ax.set_xlabel("Number of observations/events")
ax.set_title("Figure 1. M57-Jean Pipeline Reduction Funnel")

for bar, value in zip(bars, counts):
    ax.text(
        value + max(counts) * 0.015,
        bar.get_y() + bar.get_height() / 2,
        f"{value:,}",
        va="center",
        fontsize=12,
        fontweight="bold",
    )

ax.text(
    0.5,
    -0.16,
    "Raw observations -> timestamped events -> logical timeline events",
    transform=ax.transAxes,
    ha="center",
    fontsize=11,
)

plt.tight_layout()
fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
plt.close(fig)

print("Figure 1 generated successfully.")
print(f"Raw observations: {raw_count:,}")
print(f"Timestamped events: {timestamped_count:,}")
print(f"Logical events: {logical_count:,}")
print(f"Output: {OUTPUT}")

