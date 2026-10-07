import pandas as pd

path = "results/M57-Jean/table9_runtime_memory.csv"
df = pd.read_csv(path)

paper = pd.DataFrame({
    "Pipeline Stage": df["stage"],
    "Runtime (s)": df["runtime_seconds"],
    "Peak Memory (MB)": ["N/R"] * len(df),
})

paper["Hardware / Software"] = (
    "Intel Core i5-12500H (12 cores, 16 logical processors), "
    "15.70 GB RAM, Windows 11, Python 3.12.7"
)

paper.to_csv("results/M57-Jean/table9_paper_ready.csv", index=False)

print(paper.to_string(index=False))
print("\nSaved: results/M57-Jean/table9_paper_ready.csv")
