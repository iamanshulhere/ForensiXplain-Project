"""
ForensiXplain - Graph Statistics Evaluation

Generates reproducible statistics for the forensic
temporal knowledge graph.

Output:
    results/M57-Jean/graph_statistics.csv
    results/M57-Jean/graph_statistics_report.txt
"""

from pathlib import Path
from collections import Counter

import networkx as nx
import pandas as pd


CASE_ID = "M57-Jean"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GRAPH_FILE = (
    PROJECT_ROOT
    / "data"
    / "normalized"
    / CASE_ID
    / "temporal_graph.graphml"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / CASE_ID
)

CSV_OUTPUT = RESULTS_DIR / "graph_statistics.csv"
REPORT_OUTPUT = RESULTS_DIR / "graph_statistics_report.txt"


def main():
    print("=== ForensiXplain Graph Statistics Evaluation ===")

    if not GRAPH_FILE.exists():
        raise FileNotFoundError(
            f"Graph file not found:\n{GRAPH_FILE}"
        )

    graph = nx.read_graphml(GRAPH_FILE)

    # ---------------------------------------------------------
    # Basic graph statistics
    # ---------------------------------------------------------

    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()

    average_degree = (
        sum(dict(graph.degree()).values()) / node_count
        if node_count > 0
        else 0.0
    )

    density = nx.density(graph)

    # ---------------------------------------------------------
    # Node type statistics
    # ---------------------------------------------------------

    node_types = Counter(
        data.get("node_type", "unknown")
        for _, data in graph.nodes(data=True)
    )

    # ---------------------------------------------------------
    # Edge type statistics
    # ---------------------------------------------------------

    edge_types = Counter(
        data.get("relationship", "unknown")
        for _, _, data in graph.edges(data=True)
    )

    # ---------------------------------------------------------
    # Timestamp statistics
    # ---------------------------------------------------------

    timestamps = []

    for _, _, data in graph.edges(data=True):
        timestamp = data.get("timestamp")

        if timestamp:
            timestamps.append(str(timestamp))

    timestamp_count = len(timestamps)

    # ---------------------------------------------------------
    # Create table row
    # ---------------------------------------------------------

    row = {
        "dataset": CASE_ID,
        "node_count": node_count,
        "edge_count": edge_count,
        "process_nodes": node_types.get("process", 0),
        "module_nodes": node_types.get("module", 0),
        "memory_region_nodes": node_types.get(
            "memory_region", 0
        ),
        "command_line_nodes": node_types.get(
            "command_line", 0
        ),
        "parent_of_edges": edge_types.get(
            "parent_of", 0
        ),
        "loaded_module_edges": edge_types.get(
            "loaded_module", 0
        ),
        "has_memory_region_edges": edge_types.get(
            "has_memory_region", 0
        ),
        "has_command_line_edges": edge_types.get(
            "has_command_line", 0
        ),
        "average_degree": average_degree,
        "graph_density": density,
        "timestamped_edges": timestamp_count,
    }

    statistics_df = pd.DataFrame([row])

    # ---------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    statistics_df.to_csv(
        CSV_OUTPUT,
        index=False,
        encoding="utf-8"
    )

    # ---------------------------------------------------------
    # Human-readable report
    # ---------------------------------------------------------

    report_lines = [
        "FORENSIXPLAIN",
        "Knowledge Graph Statistics Report",
        "=" * 70,
        "",
        f"Dataset: {CASE_ID}",
        "",
        "GRAPH SUMMARY",
        "-" * 70,
        f"Nodes                 : {node_count}",
        f"Edges                 : {edge_count}",
        f"Average degree        : {average_degree:.6f}",
        f"Graph density         : {density:.9f}",
        f"Timestamped edges     : {timestamp_count}",
        "",
        "NODE TYPES",
        "-" * 70,
    ]

    for node_type, count in sorted(node_types.items()):
        report_lines.append(
            f"{node_type:22}: {count}"
        )

    report_lines.extend(
        [
            "",
            "EDGE TYPES",
            "-" * 70,
        ]
    )

    for edge_type, count in sorted(edge_types.items()):
        report_lines.append(
            f"{edge_type:22}: {count}"
        )

    report_lines.extend(
        [
            "",
            "INTERPRETATION",
            "-" * 70,
            (
                "The graph represents process-centric forensic "
                "relationships extracted from the normalized "
                "M57-Jean event data. Node and edge counts "
                "describe the constructed knowledge graph and "
                "do not by themselves indicate malicious activity."
            ),
            "",
        ]
    )

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8"
    )

    # ---------------------------------------------------------
    # Console summary
    # ---------------------------------------------------------

    print("")
    print("=== Graph Statistics Complete ===")
    print(f"Dataset: {CASE_ID}")
    print(f"Nodes: {node_count}")
    print(f"Edges: {edge_count}")
    print(f"Average degree: {average_degree:.6f}")
    print(f"Graph density: {density:.9f}")
    print(f"Timestamped edges: {timestamp_count}")

    print("")
    print("Node types:")

    for node_type, count in sorted(node_types.items()):
        print(f"  {node_type}: {count}")

    print("")
    print("Edge types:")

    for edge_type, count in sorted(edge_types.items()):
        print(f"  {edge_type}: {count}")

    print("")
    print(f"CSV output: {CSV_OUTPUT}")
    print(f"Report output: {REPORT_OUTPUT}")


if __name__ == "__main__":
    main()