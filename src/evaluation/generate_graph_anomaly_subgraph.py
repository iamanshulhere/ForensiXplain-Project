from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


# ============================================================
# Paths
# ============================================================

GRAPH_FILE = Path(
    "data/normalized/M57-Jean/temporal_graph.graphml"
)

GRAPH_ANOMALIES_FILE = Path(
    "results/M57-Jean/graph_anomalies.csv"
)

GRAPH_SHAP_FILE = Path(
    "results/M57-Jean/graph_shap_explanations.csv"
)

OUTPUT_DIR = Path("results/M57-Jean")

FIGURE_FILE = OUTPUT_DIR / "graph_anomaly_subgraph.png"
SUMMARY_FILE = OUTPUT_DIR / "graph_anomaly_subgraph_summary.csv"


# ============================================================
# Configuration
# ============================================================

ANOMALY_NODE = "process:3560"


# ============================================================
# Helpers
# ============================================================

def clean_number(value, digits=4):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def get_first_matching_row(df, column, value):
    if column not in df.columns:
        return None

    rows = df[df[column].astype(str) == str(value)]

    if rows.empty:
        return None

    return rows.iloc[0]


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== ForensiXplain Graph Anomaly Subgraph ===")

    # --------------------------------------------------------
    # Load graph
    # --------------------------------------------------------

    graph = nx.read_graphml(GRAPH_FILE)

    if ANOMALY_NODE not in graph:
        raise ValueError(
            f"Anomaly node not found in graph: {ANOMALY_NODE}"
        )

    anomaly_data = dict(graph.nodes[ANOMALY_NODE])

    process_name = anomaly_data.get("name", "unknown")
    pid = anomaly_data.get("pid", "unknown")
    create_time = anomaly_data.get("create_time", "unknown")

    # --------------------------------------------------------
    # Load anomaly results
    # --------------------------------------------------------

    anomalies = pd.read_csv(GRAPH_ANOMALIES_FILE)

    anomaly_row = None

    if "process_id" in anomalies.columns:
        anomaly_row = get_first_matching_row(
            anomalies,
            "process_id",
            pid
        )

    if anomaly_row is None and "pid" in anomalies.columns:
        anomaly_row = get_first_matching_row(
            anomalies,
            "pid",
            pid
        )

    # --------------------------------------------------------
    # Load SHAP results
    # --------------------------------------------------------

    shap = pd.read_csv(GRAPH_SHAP_FILE)

    shap_row = None

    if "process_id" in shap.columns:
        shap_row = get_first_matching_row(
            shap,
            "process_id",
            pid
        )

    if shap_row is None and "pid" in shap.columns:
        shap_row = get_first_matching_row(
            shap,
            "pid",
            pid
        )

    # --------------------------------------------------------
    # Read anomaly values
    # --------------------------------------------------------

    anomaly_score = None
    anomaly_rank = None
    prediction = None

    if anomaly_row is not None:

        if "graph_anomaly_score" in anomaly_row:
            anomaly_score = anomaly_row["graph_anomaly_score"]

        elif "anomaly_score" in anomaly_row:
            anomaly_score = anomaly_row["anomaly_score"]

        if "graph_anomaly_rank" in anomaly_row:
            anomaly_rank = anomaly_row["graph_anomaly_rank"]

        elif "rank" in anomaly_row:
            anomaly_rank = anomaly_row["rank"]

        if "graph_anomaly_prediction" in anomaly_row:
            prediction = anomaly_row["graph_anomaly_prediction"]

        elif "prediction" in anomaly_row:
            prediction = anomaly_row["prediction"]

    # --------------------------------------------------------
    # Read SHAP values
    # --------------------------------------------------------

    shap_features = []
    shap_values = []

    if shap_row is not None:

        if "top_shap_features" in shap_row:
            raw_features = str(shap_row["top_shap_features"])
            shap_features = [
                x.strip()
                for x in raw_features.split(";")
                if x.strip()
            ]

        if "top_shap_values" in shap_row:
            raw_values = str(shap_row["top_shap_values"])
            shap_values = [
                x.strip()
                for x in raw_values.split(";")
                if x.strip()
            ]

    # --------------------------------------------------------
    # Inspect immediate neighborhood
    # --------------------------------------------------------

    parent_nodes = []
    command_nodes = []
    module_nodes = []
    memory_nodes = []
    child_nodes = []

    for source, target, data in graph.in_edges(
        ANOMALY_NODE,
        data=True
    ):

        relationship = data.get("relationship", "")

        if relationship == "parent_of":
            parent_nodes.append(source)

    for source, target, data in graph.out_edges(
        ANOMALY_NODE,
        data=True
    ):

        relationship = data.get("relationship", "")

        target_type = graph.nodes[target].get(
            "node_type",
            ""
        )

        if relationship == "has_command_line":
            command_nodes.append(target)

        elif relationship == "loaded_module":
            module_nodes.append(target)

        elif relationship == "has_memory_region":
            memory_nodes.append(target)

        elif target_type == "process":
            child_nodes.append(target)

    # --------------------------------------------------------
    # Create summarized visualization graph
    # --------------------------------------------------------

    G = nx.DiGraph()

    root = "ANOMALY"

    G.add_node(
        root,
        node_type="anomaly"
    )

    # Parent
    if parent_nodes:

        parent = parent_nodes[0]

        parent_name = graph.nodes[parent].get(
            "name",
            parent
        )

        parent_pid = graph.nodes[parent].get(
            "pid",
            "unknown"
        )

        parent_label = (
            f"{parent_name}\nPID {parent_pid}"
        )

        G.add_node(
            "PARENT",
            node_type="process"
        )

        G.add_edge(
            "PARENT",
            root,
            relationship="parent_of"
        )

    # Command line
    if command_nodes:

        G.add_node(
            "COMMAND",
            node_type="command_line"
        )

        G.add_edge(
            root,
            "COMMAND",
            relationship="has_command_line"
        )

    # Modules
    if module_nodes:

        G.add_node(
            "MODULES",
            node_type="module"
        )

        G.add_edge(
            root,
            "MODULES",
            relationship="loaded_module"
        )

    # Memory region
    if memory_nodes:

        G.add_node(
            "MEMORY",
            node_type="memory_region"
        )

        G.add_edge(
            root,
            "MEMORY",
            relationship="has_memory_region"
        )

    # Children
    for index, child in enumerate(child_nodes[:5]):

        child_name = graph.nodes[child].get(
            "name",
            child
        )

        child_pid = graph.nodes[child].get(
            "pid",
            "unknown"
        )

        child_node = f"CHILD_{index}"

        G.add_node(
            child_node,
            node_type="process"
        )

        G.add_edge(
            root,
            child_node,
            relationship="child_process"
        )

    # --------------------------------------------------------
    # Draw figure
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(14, 9)
    )

    pos = {
        root: (0, 0),
    }

    if "PARENT" in G:
        pos["PARENT"] = (-4, 1.5)

    if "COMMAND" in G:
        pos["COMMAND"] = (-3, -2)

    if "MODULES" in G:
        pos["MODULES"] = (0, -2)

    if "MEMORY" in G:
        pos["MEMORY"] = (3, -2)

    for index, child in enumerate(
        [n for n in G.nodes if n.startswith("CHILD_")]
    ):
        pos[child] = (
            4,
            1.5 - index
        )

    # --------------------------------------------------------
    # Draw edges
    # --------------------------------------------------------

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        arrows=True,
        arrowsize=20,
        width=2
    )

    # --------------------------------------------------------
    # Draw nodes
    # --------------------------------------------------------

    node_positions = []

    labels = {}

    for node in G.nodes:

        node_type = G.nodes[node].get(
            "node_type"
        )

        if node == root:

            label = (
                f"ANOMALY CANDIDATE\n"
                f"{process_name}\n"
                f"PID {pid}\n"
                f"Rank {anomaly_rank}\n"
                f"Score {clean_number(anomaly_score)}"
            )

        elif node == "PARENT":

            parent = parent_nodes[0]

            parent_name = graph.nodes[parent].get(
                "name",
                parent
            )

            parent_pid = graph.nodes[parent].get(
                "pid",
                "unknown"
            )

            label = (
                f"PARENT PROCESS\n"
                f"{parent_name}\n"
                f"PID {parent_pid}"
            )

        elif node == "COMMAND":

            label = (
                "COMMAND LINE\n"
                f"{len(command_nodes)} evidence node"
            )

        elif node == "MODULES":

            label = (
                "LOADED MODULES\n"
                f"{len(module_nodes)} modules"
            )

        elif node == "MEMORY":

            label = (
                "MEMORY REGION\n"
                f"{len(memory_nodes)} evidence node"
            )

        else:

            index = int(
                node.split("_")[1]
            )

            child = child_nodes[index]

            child_name = graph.nodes[child].get(
                "name",
                child
            )

            child_pid = graph.nodes[child].get(
                "pid",
                "unknown"
            )

            label = (
                f"CHILD PROCESS\n"
                f"{child_name}\n"
                f"PID {child_pid}"
            )

        labels[node] = label

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=5000,
        node_shape="o",
        linewidths=2
    )

    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        ax=ax,
        font_size=9,
        font_weight="bold"
    )

    # --------------------------------------------------------
    # Edge labels
    # --------------------------------------------------------

    edge_labels = {
        (u, v): data["relationship"]
        for u, v, data in G.edges(data=True)
    }

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        ax=ax,
        font_size=8
    )

    # --------------------------------------------------------
    # SHAP annotation
    # --------------------------------------------------------

    shap_lines = []

    for feature, value in zip(
        shap_features[:3],
        shap_values[:3]
    ):

        feature_value = "N/A"

        feature_column = feature

        if shap_row is not None:

            if feature_column in shap_row.index:
                feature_value = shap_row[
                    feature_column
                ]

            value_column = (
                f"value_{feature}"
            )

            if value_column in shap_row.index:
                feature_value = shap_row[
                    value_column
                ]

        shap_lines.append(
            f"{feature}: "
            f"value={feature_value}, "
            f"SHAP={value}"
        )

    shap_text = (
        "Top Graph-SHAP Features\n"
        + "\n".join(shap_lines)
    )

    ax.text(
        0.02,
        0.02,
        shap_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.9
        )
    )

    # --------------------------------------------------------
    # Metadata annotation
    # --------------------------------------------------------

    metadata = (
        f"Dataset: M57-Jean\n"
        f"Process: {process_name} (PID {pid})\n"
        f"Creation time: {create_time}\n"
        f"Parent edges: {len(parent_nodes)}\n"
        f"Child process edges: {len(child_nodes)}\n"
        f"Loaded modules: {len(module_nodes)}\n"
        f"Command-line nodes: {len(command_nodes)}\n"
        f"Memory-region nodes: {len(memory_nodes)}"
    )

    ax.text(
        0.98,
        0.02,
        metadata,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.9
        )
    )

    ax.set_title(
        "Figure 7. Knowledge Graph Anomaly Subgraph — M57-Jean",
        fontsize=15,
        fontweight="bold",
        pad=20
    )

    ax.axis("off")

    plt.tight_layout()

    plt.savefig(
        FIGURE_FILE,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Summary CSV
    # --------------------------------------------------------

    summary = pd.DataFrame(
        [
            {
                "dataset": "M57-Jean",
                "anomaly_process": process_name,
                "pid": pid,
                "anomaly_rank": anomaly_rank,
                "anomaly_score": anomaly_score,
                "parent_process_count": len(parent_nodes),
                "child_process_count": len(child_nodes),
                "command_line_nodes": len(command_nodes),
                "loaded_module_nodes": len(module_nodes),
                "memory_region_nodes": len(memory_nodes),
                "shap_features": ";".join(shap_features),
                "shap_values": ";".join(shap_values),
            }
        ]
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print()
    print("=== Graph Anomaly Subgraph Complete ===")
    print(f"Process: {process_name}")
    print(f"PID: {pid}")
    print(f"Rank: {anomaly_rank}")
    print(f"Score: {anomaly_score}")
    print(f"Parent processes: {len(parent_nodes)}")
    print(f"Child processes: {len(child_nodes)}")
    print(f"Command-line nodes: {len(command_nodes)}")
    print(f"Loaded modules: {len(module_nodes)}")
    print(f"Memory regions: {len(memory_nodes)}")
    print()
    print(f"Figure: {FIGURE_FILE}")
    print(f"Summary: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()