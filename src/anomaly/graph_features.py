"""
ForensiXplain - Graph Feature Engineering

Purpose:
    Extract process-level graph features from the temporal forensic
    knowledge graph and combine them with temporal features.

Input:
    data/normalized/M57-Jean/temporal_graph.graphml
    data/features/M57-Jean/temporal_features.csv

Output:
    data/features/M57-Jean/graph_features.csv

Important:
    Model outputs such as anomaly scores or predicted anomaly labels
    are NOT used as input features.
"""

from pathlib import Path
import pandas as pd
import networkx as nx


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GRAPH_FILE = (
    PROJECT_ROOT
    / "data"
    / "normalized"
    / "M57-Jean"
    / "temporal_graph.graphml"
)

TEMPORAL_FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "M57-Jean"
    / "temporal_features.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "M57-Jean"
)

OUTPUT_FILE = OUTPUT_DIR / "graph_features.csv"


# ============================================================
# Configuration
# ============================================================

GRAPH_FEATURE_COLUMNS = [
    "parent_count",
    "child_count",
    "graph_degree",
    "in_degree",
    "out_degree",
    "command_line_count",
    "module_count",
    "memory_region_count",
    "relationship_type_count",
]


# ============================================================
# Utility
# ============================================================

def safe_int(value):
    """Convert a graph attribute to integer safely."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def get_process_nodes(graph):
    """Return process nodes from the graph."""
    process_nodes = []

    for node, data in graph.nodes(data=True):
        if data.get("node_type") == "process":
            process_nodes.append(node)

    return process_nodes


# ============================================================
# Feature extraction
# ============================================================

def extract_process_graph_features(graph):
    """
    Extract graph features for every process node.
    """

    rows = []

    process_nodes = get_process_nodes(graph)

    for node in process_nodes:

        node_data = graph.nodes[node]

        pid = safe_int(
            node_data.get("pid", 0)
        )

        process_name = str(
            node_data.get("name", "")
        )

        # ----------------------------------------------------
        # Neighbour relationships
        # ----------------------------------------------------

        parent_count = 0
        child_count = 0
        command_line_count = 0
        module_count = 0
        memory_region_count = 0

        relationship_types = set()

        # ----------------------------------------------------
        # Incoming edges
        # ----------------------------------------------------

        for _, _, edge_data in graph.in_edges(
            node,
            data=True
        ):

            relationship = str(
                edge_data.get(
                    "relationship",
                    ""
                )
            )

            relationship_types.add(
                relationship
            )

            if relationship == "parent_of":
                parent_count += 1

        # ----------------------------------------------------
        # Outgoing edges
        # ----------------------------------------------------

        for _, _, edge_data in graph.out_edges(
            node,
            data=True
        ):

            relationship = str(
                edge_data.get(
                    "relationship",
                    ""
                )
            )

            relationship_types.add(
                relationship
            )

            if relationship == "parent_of":
                child_count += 1

            elif relationship == "has_command_line":
                command_line_count += 1

            elif relationship == "loaded_module":
                module_count += 1

            elif relationship == "has_memory_region":
                memory_region_count += 1

        # ----------------------------------------------------
        # Degree
        # ----------------------------------------------------

        in_degree = graph.in_degree(node)
        out_degree = graph.out_degree(node)
        graph_degree = graph.degree(node)

        # ----------------------------------------------------
        # Save process feature row
        # ----------------------------------------------------

        rows.append(
            {
                "process_node": node,
                "process_id": pid,
                "process": process_name,

                "parent_count": parent_count,
                "child_count": child_count,

                "graph_degree": graph_degree,
                "in_degree": in_degree,
                "out_degree": out_degree,

                "command_line_count": command_line_count,
                "module_count": module_count,
                "memory_region_count": memory_region_count,

                "relationship_type_count": len(
                    relationship_types
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Combine temporal + graph features
# ============================================================

def combine_with_temporal_features(
    graph_features,
    temporal_features
):
    """
    Join graph features to temporal features using process_id.

    The temporal feature table contains logical process-start
    events. Graph features are process-level. Therefore each
    process-start event receives the graph features associated
    with its PID.
    """

    # --------------------------------------------------------
    # Normalize process IDs
    # --------------------------------------------------------

    graph_features["process_id"] = pd.to_numeric(
        graph_features["process_id"],
        errors="coerce"
    )

    temporal_features["process_id"] = pd.to_numeric(
        temporal_features["process_id"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Keep temporal information
    # --------------------------------------------------------

    combined = temporal_features.merge(
        graph_features,
        on="process_id",
        how="left",
        suffixes=(
            "",
            "_graph"
        )
    )

    # --------------------------------------------------------
    # Fill graph counts
    # --------------------------------------------------------

    count_columns = [
        "parent_count",
        "child_count",
        "graph_degree",
        "in_degree",
        "out_degree",
        "command_line_count",
        "module_count",
        "memory_region_count",
        "relationship_type_count",
    ]

    for column in count_columns:

        if column in combined.columns:
            combined[column] = combined[column].fillna(0)

    # --------------------------------------------------------
    # Graph match indicator
    # --------------------------------------------------------

    combined["graph_node_available"] = (
        combined["graph_degree"] > 0
    ).astype(int)

    return combined


# ============================================================
# Main
# ============================================================

def main():

    print("=== ForensiXplain Graph Feature Engineering ===")

    # --------------------------------------------------------
    # Validate input files
    # --------------------------------------------------------

    if not GRAPH_FILE.exists():
        raise FileNotFoundError(
            f"Graph file not found:\n{GRAPH_FILE}"
        )

    if not TEMPORAL_FEATURE_FILE.exists():
        raise FileNotFoundError(
            "Temporal feature file not found:\n"
            f"{TEMPORAL_FEATURE_FILE}"
        )

    # --------------------------------------------------------
    # Load graph
    # --------------------------------------------------------

    print(f"Loading graph:\n{GRAPH_FILE}")

    graph = nx.read_graphml(
        GRAPH_FILE
    )

    print(
        f"Graph nodes: {graph.number_of_nodes()}"
    )

    print(
        f"Graph edges: {graph.number_of_edges()}"
    )

    # --------------------------------------------------------
    # Load temporal features
    # --------------------------------------------------------

    temporal_features = pd.read_csv(
        TEMPORAL_FEATURE_FILE
    )

    print(
        f"Temporal feature rows: "
        f"{len(temporal_features)}"
    )

    # --------------------------------------------------------
    # Extract graph features
    # --------------------------------------------------------

    graph_features = extract_process_graph_features(
        graph
    )

    print(
        f"Process graph feature rows: "
        f"{len(graph_features)}"
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    combined = combine_with_temporal_features(
        graph_features,
        temporal_features
    )

    # --------------------------------------------------------
    # Remove model-output columns if present
    # --------------------------------------------------------

    leakage_columns = [
        "temporal_anomaly_score",
        "temporal_predicted_anomaly",
        "temporal_anomaly_rank",
        "anomaly_score",
        "predicted_anomaly",
        "anomaly_rank",
        "ground_truth",
        "anomaly_label",
        "attack_stage",
    ]

    leakage_columns_present = [
        column
        for column in leakage_columns
        if column in combined.columns
    ]

    if leakage_columns_present:

        print(
            "Removing potential leakage columns:"
        )

        for column in leakage_columns_present:
            print(f"  - {column}")

        combined = combined.drop(
            columns=leakage_columns_present
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    combined.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("")
    print(
        "=== Graph Feature Engineering Complete ==="
    )

    print(
        f"Feature rows: {len(combined)}"
    )

    print(
        f"Feature columns: {len(combined.columns)}"
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )

    print("")
    print("Graph feature columns:")

    for column in GRAPH_FEATURE_COLUMNS:
        print(
            f"  {column}"
        )

    print("")
    print("Feature preview:")

    preview_columns = [
        "process_id",
        "process",
        "parent_count",
        "child_count",
        "graph_degree",
        "in_degree",
        "out_degree",
        "command_line_count",
        "module_count",
        "memory_region_count",
        "relationship_type_count",
    ]

    available_preview_columns = [
        column
        for column in preview_columns
        if column in combined.columns
    ]

    print(
        combined[
            available_preview_columns
        ].head(10).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()