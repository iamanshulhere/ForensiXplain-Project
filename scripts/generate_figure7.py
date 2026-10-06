import os
import networkx as nx
import matplotlib.pyplot as plt

GRAPH_PATH = r"data\normalized\M57-Jean\temporal_graph.graphml"
OUTPUT_PATH = r"results\M57-Jean\figure7_graph_anomaly_subgraph.png"

ANOMALY_NODE = "process:3560"
PARENT_NODE = "process:3908"

# Load the existing knowledge graph
G = nx.read_graphml(GRAPH_PATH)

# Build a small anomaly-centered subgraph
selected_nodes = {ANOMALY_NODE, PARENT_NODE}

# Add command-line and memory-region nodes directly connected to the anomaly
for _, target, data in G.out_edges(ANOMALY_NODE, data=True):
    relationship = data.get("relationship", "")
    if relationship in {"has_command_line", "has_memory_region"}:
        selected_nodes.add(target)

# Add three representative loaded modules
module_nodes = []
for _, target, data in G.out_edges(ANOMALY_NODE, data=True):
    if data.get("relationship") == "loaded_module":
        module_nodes.append(target)

for node in sorted(module_nodes)[:3]:
    selected_nodes.add(node)

# Create the subgraph
SG = G.subgraph(selected_nodes).copy()

# Node-type colors
node_colors = {
    "process": "#4C78A8",
    "module": "#59A14F",
    "command_line": "#F2CF5B",
    "memory_region": "#E15759",
}

# Node sizes
node_sizes = []
colors = []

for node, data in SG.nodes(data=True):
    node_type = data.get("node_type", "unknown")
    colors.append(node_colors.get(node_type, "#9D9D9D"))

    if node == ANOMALY_NODE:
        node_sizes.append(1800)
    else:
        node_sizes.append(1100)

# Human-readable node labels
labels = {}

for node, data in SG.nodes(data=True):
    node_type = data.get("node_type", "")

    if node == ANOMALY_NODE:
        labels[node] = "soffice.bin\nPID 3560\nANOMALY"
    elif node == PARENT_NODE:
        labels[node] = f"{data.get('name', 'parent process')}\nPID {data.get('pid', '?')}"
    elif node_type == "module":
        name = data.get("name", node.split(":")[-1])
        labels[node] = str(name)
    elif node_type == "command_line":
        labels[node] = "Command line"
    elif node_type == "memory_region":
        labels[node] = "Memory region"
    else:
        labels[node] = node.split(":")[-1]

# Layout
pos = nx.spring_layout(
    SG,
    seed=42,
    k=1.8,
    iterations=200
)

plt.figure(figsize=(13, 8))

# Draw nodes
nx.draw_networkx_nodes(
    SG,
    pos,
    node_color=colors,
    node_size=node_sizes,
    edgecolors="black",
    linewidths=1.2
)

# Draw edges
nx.draw_networkx_edges(
    SG,
    pos,
    arrows=True,
    arrowsize=18,
    width=1.5,
    edge_color="#666666",
    connectionstyle="arc3,rad=0.05"
)

# Draw node labels
nx.draw_networkx_labels(
    SG,
    pos,
    labels=labels,
    font_size=9,
    font_weight="bold"
)

# Edge relationship labels
edge_labels = {
    (u, v): data.get("relationship", "")
    for u, v, data in SG.edges(data=True)
}

nx.draw_networkx_edge_labels(
    SG,
    pos,
    edge_labels=edge_labels,
    font_size=8,
    label_pos=0.5,
    rotate=False,
    bbox=dict(
        alpha=0.85,
        color="white",
        pad=0.2
    )
)

plt.title(
    "Figure 7 - Graph Anomaly Subgraph: soffice.bin (PID 3560)",
    fontsize=14,
    fontweight="bold"
)

# Legend
legend_items = [
    plt.Line2D(
        [0], [0],
        marker="o",
        color="w",
        label="Process",
        markerfacecolor=node_colors["process"],
        markeredgecolor="black",
        markersize=10
    ),
    plt.Line2D(
        [0], [0],
        marker="o",
        color="w",
        label="Module",
        markerfacecolor=node_colors["module"],
        markeredgecolor="black",
        markersize=10
    ),
    plt.Line2D(
        [0], [0],
        marker="o",
        color="w",
        label="Command line",
        markerfacecolor=node_colors["command_line"],
        markeredgecolor="black",
        markersize=10
    ),
    plt.Line2D(
        [0], [0],
        marker="o",
        color="w",
        label="Memory region",
        markerfacecolor=node_colors["memory_region"],
        markeredgecolor="black",
        markersize=10
    ),
]

plt.legend(
    handles=legend_items,
    loc="upper left",
    frameon=True
)

plt.axis("off")
plt.tight_layout()

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Figure 7 generated successfully: {OUTPUT_PATH}")
print(f"Nodes displayed: {SG.number_of_nodes()}")
print(f"Edges displayed: {SG.number_of_edges()}")
print("Displayed edge relationships:")
for u, v, data in SG.edges(data=True):
    print(f"  {u} -> {v}: {data.get('relationship', '')}")
