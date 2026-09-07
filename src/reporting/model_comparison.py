"""
ForensiXplain - Model Comparison

Purpose:
    Compare anomaly candidates produced by:
        1. Temporal-only Isolation Forest
        2. Graph-aware Isolation Forest
        3. Graph-only Isolation Forest

This module does NOT determine whether an event is malicious.
It summarizes agreement and differences between detection approaches.

Input:
    results/M57-Jean/temporal_anomalies.csv
    results/M57-Jean/graph_anomalies.csv
    results/M57-Jean/graph_only_anomalies.csv

Output:
    results/M57-Jean/model_comparison.csv
    results/M57-Jean/model_comparison_report.txt
"""

from pathlib import Path

import pandas as pd


# ============================================================
# Configuration
# ============================================================

CASE_ID = "M57-Jean"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / CASE_ID
)

TEMPORAL_FILE = RESULTS_DIR / "temporal_anomalies.csv"
GRAPH_FILE = RESULTS_DIR / "graph_anomalies.csv"
GRAPH_ONLY_FILE = RESULTS_DIR / "graph_only_anomalies.csv"

CSV_OUTPUT = RESULTS_DIR / "model_comparison.csv"
REPORT_OUTPUT = RESULTS_DIR / "model_comparison_report.txt"


# ============================================================
# Main
# ============================================================

def main():

    print("=== ForensiXplain Model Comparison ===")

    # --------------------------------------------------------
    # Validate inputs
    # --------------------------------------------------------

    for file_path in [
        TEMPORAL_FILE,
        GRAPH_FILE,
        GRAPH_ONLY_FILE,
    ]:

        if not file_path.exists():
            raise FileNotFoundError(
                f"Required input not found:\n{file_path}"
            )

    # --------------------------------------------------------
    # Load results
    # --------------------------------------------------------

    temporal = pd.read_csv(TEMPORAL_FILE)
    graph = pd.read_csv(GRAPH_FILE)
    graph_only = pd.read_csv(GRAPH_ONLY_FILE)

    print(f"Temporal rows: {len(temporal)}")
    print(f"Graph-aware rows: {len(graph)}")
    print(f"Graph-only rows: {len(graph_only)}")

    # --------------------------------------------------------
    # Create anomaly sets
    # --------------------------------------------------------

    temporal_ids = set(
        temporal.loc[
            temporal["temporal_predicted_anomaly"],
            "process_id"
        ]
    )

    graph_ids = set(
        graph.loc[
            graph["graph_predicted_anomaly"],
            "process_id"
        ]
    )

    graph_only_ids = set(
        graph_only.loc[
            graph_only["graph_only_predicted_anomaly"],
            "process_id"
        ]
    )

    # --------------------------------------------------------
    # Build union of process IDs
    # --------------------------------------------------------

    all_process_ids = sorted(
        temporal_ids
        | graph_ids
        | graph_only_ids
    )

    records = []

    for process_id in all_process_ids:

        temporal_row = temporal[
            temporal["process_id"] == process_id
        ]

        graph_row = graph[
            graph["process_id"] == process_id
        ]

        graph_only_row = graph_only[
            graph_only["process_id"] == process_id
        ]

        process_name = ""

        if not temporal_row.empty:
            process_name = str(
                temporal_row.iloc[0]["process"]
            )
        elif not graph_row.empty:
            process_name = str(
                graph_row.iloc[0]["process"]
            )
        elif not graph_only_row.empty:
            process_name = str(
                graph_only_row.iloc[0]["process"]
            )

        temporal_anomaly = process_id in temporal_ids
        graph_anomaly = process_id in graph_ids
        graph_only_anomaly = process_id in graph_only_ids

        detection_count = sum(
            [
                temporal_anomaly,
                graph_anomaly,
                graph_only_anomaly,
            ]
        )

        records.append(
            {
                "case_id": CASE_ID,
                "process_id": process_id,
                "process": process_name,
                "temporal_anomaly": temporal_anomaly,
                "graph_anomaly": graph_anomaly,
                "graph_only_anomaly": graph_only_anomaly,
                "detection_count": detection_count,
            }
        )

    comparison = pd.DataFrame(records)

    # --------------------------------------------------------
    # Agreement category
    # --------------------------------------------------------

    def category(row):

        count = row["detection_count"]

        if count == 3:
            return "all_three"

        if (
            row["temporal_anomaly"]
            and row["graph_anomaly"]
        ):
            return "temporal_and_graph"

        if (
            row["temporal_anomaly"]
            and row["graph_only_anomaly"]
        ):
            return "temporal_and_graph_only"

        if (
            row["graph_anomaly"]
            and row["graph_only_anomaly"]
        ):
            return "graph_and_graph_only"

        if count == 1:

            if row["temporal_anomaly"]:
                return "temporal_only"

            if row["graph_anomaly"]:
                return "graph_only"

            if row["graph_only_anomaly"]:
                return "graph_only_model_only"

        return "not_anomalous"

    comparison["agreement_category"] = comparison.apply(
        category,
        axis=1
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    comparison.to_csv(
        CSV_OUTPUT,
        index=False,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Counts
    # --------------------------------------------------------

    temporal_count = len(temporal_ids)
    graph_count = len(graph_ids)
    graph_only_count = len(graph_only_ids)

    temporal_graph = temporal_ids & graph_ids
    temporal_graph_only = temporal_ids & graph_only_ids
    graph_graph_only = graph_ids & graph_only_ids
    all_three = (
        temporal_ids
        & graph_ids
        & graph_only_ids
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report_lines = [
        "ForensiXplain Model Comparison Report",
        "=" * 45,
        "",
        f"Case: {CASE_ID}",
        "",
        "Anomaly counts:",
        f"  Temporal-only model: {temporal_count}",
        f"  Graph-aware model: {graph_count}",
        f"  Graph-only model: {graph_only_count}",
        "",
        "Overlap:",
        f"  Temporal AND Graph-aware: "
        f"{sorted(temporal_graph)}",
        f"  Temporal AND Graph-only: "
        f"{sorted(temporal_graph_only)}",
        f"  Graph-aware AND Graph-only: "
        f"{sorted(graph_graph_only)}",
        f"  All three: "
        f"{sorted(all_three)}",
        "",
        "Temporal-only candidates:",
        f"  {sorted(temporal_ids - graph_ids - graph_only_ids)}",
        "",
        "Graph-aware-only candidates:",
        f"  {sorted(graph_ids - temporal_ids - graph_only_ids)}",
        "",
        "Graph-only-model candidates:",
        f"  {sorted(graph_only_ids - temporal_ids - graph_ids)}",
        "",
        "Important:",
        "An anomaly indicates an unusual feature profile",
        "under the corresponding model. It does not by itself",
        "establish malicious activity.",
        "",
    ]

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print("=== Model Comparison Complete ===")
    print(f"Temporal anomalies: {temporal_count}")
    print(f"Graph-aware anomalies: {graph_count}")
    print(f"Graph-only anomalies: {graph_only_count}")
    print()
    print(
        "Temporal ∩ Graph-aware:",
        sorted(temporal_graph)
    )
    print(
        "Temporal ∩ Graph-only:",
        sorted(temporal_graph_only)
    )
    print(
        "Graph-aware ∩ Graph-only:",
        sorted(graph_graph_only)
    )
    print(
        "All three:",
        sorted(all_three)
    )
    print()
    print(f"CSV output: {CSV_OUTPUT}")
    print(f"Report output: {REPORT_OUTPUT}")


if __name__ == "__main__":
    main()