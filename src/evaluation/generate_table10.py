from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / "results" / "M57-Jean"


def evidence_id_count(series: pd.Series) -> pd.Series:
    return series.fillna("").apply(
        lambda value: 0 if not value else len(str(value).split(";"))
    )


def coverage(df: pd.DataFrame, column: str) -> int:
    return int((df[column] > 0).sum())


def main() -> None:
    fused = pd.read_csv(
        RESULTS_DIR / "evidence_attribution.csv"
    )
    graph = pd.read_csv(
        RESULTS_DIR / "graph_evidence_attribution.csv"
    )
    temporal = pd.read_csv(
        RESULTS_DIR / "temporal_evidence_attribution.csv"
    )

    fused_evidence_cols = [
        "pslist_count",
        "pstree_count",
        "cmdline_count",
        "dlllist_count",
        "malfind_count",
    ]

    # Process/fused attribution statistics
    fused_id_counts = evidence_id_count(fused["evidence_ids"])

    fused_rows = len(fused)

    fused_stats = {
        "candidates": fused_rows,
        "linked_candidates": int((fused_id_counts > 0).sum()),
        "linked_percentage": (
            100.0 * (fused_id_counts > 0).sum() / fused_rows
        ),
        "mean_evidence_ids": fused_id_counts.mean(),
        "median_evidence_ids": fused_id_counts.median(),
        "pslist_coverage": coverage(fused, "pslist_count"),
        "pstree_coverage": coverage(fused, "pstree_count"),
        "cmdline_coverage": coverage(fused, "cmdline_count"),
        "dlllist_coverage": coverage(fused, "dlllist_count"),
        "malfind_coverage": coverage(fused, "malfind_count"),
    }

    # Graph-aware top-5 statistics
    graph_id_counts = evidence_id_count(graph["evidence_ids"])
    graph_rows = len(graph)
    graph_artifacts = graph["evidence_by_artifact"].fillna("").str.lower()

    graph_stats = {
        "candidates": graph_rows,
        "linked_candidates": int((graph_id_counts > 0).sum()),
        "linked_percentage": (
            100.0 * (graph_id_counts > 0).sum() / graph_rows
        ),
        "mean_evidence_ids": graph_id_counts.mean(),
        "median_evidence_ids": graph_id_counts.median(),
        "pslist_coverage": int(
            graph_artifacts.str.contains(r"\bpslist\s*:", regex=True).sum()
        ),
        "pstree_coverage": int(
            graph_artifacts.str.contains(r"\bpstree\s*:", regex=True).sum()
        ),
        "cmdline_coverage": int(
            graph_artifacts.str.contains(r"\bcmdline\s*:", regex=True).sum()
        ),
        "dlllist_coverage": int(
            graph_artifacts.str.contains(r"\bdlllist\s*:", regex=True).sum()
        ),
        "malfind_coverage": int(
            graph_artifacts.str.contains(r"\bmalfind\s*:", regex=True).sum()
        ),
    }

    # Temporal top-5 statistics
    temporal_id_counts = temporal["evidence_id_count"]

    temporal_rows = len(temporal)

    temporal_stats = {
        "candidates": temporal_rows,
        "linked_candidates": int((temporal_id_counts > 0).sum()),
        "linked_percentage": (
            100.0 * (temporal_id_counts > 0).sum() / temporal_rows
        ),
        "mean_evidence_ids": temporal_id_counts.mean(),
        "median_evidence_ids": temporal_id_counts.median(),
        "pslist_coverage": coverage(temporal, "pslist_count"),
        "pstree_coverage": coverage(temporal, "pstree_count"),
        "cmdline_coverage": coverage(temporal, "cmdline_count"),
        "dlllist_coverage": coverage(temporal, "dlllist_count"),
        "malfind_coverage": coverage(temporal, "malfind_count"),
    }

    def pct(count: int, total: int) -> str:
        return f"{count}/{total} ({100.0 * count / total:.2f}%)"

    lines = [
        "Table 10. Evidence Attribution and Forensic Traceability",
        "",
        "Metric | Process/Fused | Graph-aware Top-5 | Temporal Top-5",
        "--- | ---: | ---: | ---:",
        (
            f"Attributed candidates | {fused_stats['candidates']} | "
            f"{graph_stats['candidates']} | {temporal_stats['candidates']}"
        ),
        (
            f"Candidates with linked evidence | "
            f"{pct(fused_stats['linked_candidates'], fused_stats['candidates'])} | "
            f"{pct(graph_stats['linked_candidates'], graph_stats['candidates'])} | "
            f"{pct(temporal_stats['linked_candidates'], temporal_stats['candidates'])}"
        ),
        (
            f"Mean evidence IDs/candidate | "
            f"{fused_stats['mean_evidence_ids']:.2f} | "
            f"{graph_stats['mean_evidence_ids']:.2f} | "
            f"{temporal_stats['mean_evidence_ids']:.2f}"
        ),
        (
            f"Median evidence IDs/candidate | "
            f"{fused_stats['median_evidence_ids']:.0f} | "
            f"{graph_stats['median_evidence_ids']:.0f} | "
            f"{temporal_stats['median_evidence_ids']:.0f}"
        ),
        (
            f"PSList coverage | "
            f"{pct(fused_stats['pslist_coverage'], fused_stats['candidates'])} | "
            f"{pct(graph_stats['pslist_coverage'], graph_stats['candidates'])} | "
            f"{pct(temporal_stats['pslist_coverage'], temporal_stats['candidates'])}"
        ),
        (
            f"PSTree coverage | "
            f"{pct(fused_stats['pstree_coverage'], fused_stats['candidates'])} | "
            f"{pct(graph_stats['pstree_coverage'], graph_stats['candidates'])} | "
            f"{pct(temporal_stats['pstree_coverage'], temporal_stats['candidates'])}"
        ),
        (
            f"Cmdline coverage | "
            f"{pct(fused_stats['cmdline_coverage'], fused_stats['candidates'])} | "
            f"{pct(graph_stats['cmdline_coverage'], graph_stats['candidates'])} | "
            f"{pct(temporal_stats['cmdline_coverage'], temporal_stats['candidates'])}"
        ),
        (
            f"DLLList coverage | "
            f"{pct(fused_stats['dlllist_coverage'], fused_stats['candidates'])} | "
            f"{pct(graph_stats['dlllist_coverage'], graph_stats['candidates'])} | "
            f"{pct(temporal_stats['dlllist_coverage'], temporal_stats['candidates'])}"
        ),
        (
            f"Malfind coverage | "
            f"{pct(fused_stats['malfind_coverage'], fused_stats['candidates'])} | "
            f"{pct(graph_stats['malfind_coverage'], graph_stats['candidates'])} | "
            f"{pct(temporal_stats['malfind_coverage'], temporal_stats['candidates'])}"
        ),
        "",
        "Interpretation:",
        (
            "All evaluated anomaly candidates had explicit links to underlying "
            "forensic evidence. Evidence attribution varies by analytical view "
            "and evidence granularity."
        ),
        (
            "These measurements evaluate evidence traceability and attribution "
            "coverage. They do not constitute ground-truth detection accuracy "
            "or establish that an attributed process is malicious."
        ),
    ]

    output_path = RESULTS_DIR / "table10_evidence_attribution.txt"
    output_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
