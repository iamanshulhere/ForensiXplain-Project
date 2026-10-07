from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "MalMem2022"
OUT_DIR.mkdir(parents=True, exist_ok=True)

metrics = {
    "Accuracy": 0.99991462477589,
    "Precision": 0.9998307093279161,
    "Recall": 1.0,
    "F1": 0.9999153474985186,
}

df = pd.DataFrame({
    "metric": list(metrics.keys()),
    "value": list(metrics.values()),
})

df.to_csv(OUT_DIR / "figure13_malmem_metrics.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(df["metric"], df["value"])

ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title("Figure 13. MalMem2022 Ground-Truth Classification Performance")
ax.grid(axis="y", alpha=0.25)

for bar, value in zip(bars, df["value"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.00015,
        f"{value:.6f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

fig.tight_layout()
fig.savefig(
    OUT_DIR / "figure13_malmem_ground_truth_metrics.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

print(f"Saved: {OUT_DIR / 'figure13_malmem_ground_truth_metrics.png'}")
print(f"Saved: {OUT_DIR / 'figure13_malmem_metrics.csv'}")
