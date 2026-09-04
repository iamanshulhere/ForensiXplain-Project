from pathlib import Path
from collections import Counter

import networkx as nx
import pandas as pd


CASE_ID = "M57-Jean"

EVENTS_PATH = Path(
    f"data/normalized/{CASE_ID}/events.csv"
)

OUTPUT_PATH = Path(
    f"data/normalized/{CASE_ID}/temporal_graph.graphml"
)


def clean_value(value):
    """Convert pandas values into GraphML-safe values."""

    if pd.isna(value):
        return ""

    return str(value)


def add_process_node(graph, row):
    """Create or update a process node."""

    pid = clean_value(row["process_id"])

    if not pid:
        return None

    # Convert 812.0 -> 812
    try:
        pid = str(int(float(pid)))
    except ValueError:
        pass

    node_id = f"process:{pid}"

    if node_id not in graph:
        graph.add_node(
            node_id,
            node_type="process",
            pid=pid,
            name=clean_value(row["process"]),
            create_time="",
            create_time_confidence="unknown",
            process_evidence_id="",
        )

    # Process creation observation
    timestamp = clean_value(row["timestamp"])

    if timestamp:
        graph.nodes[node_id]["create_time"] = timestamp
        graph.nodes[node_id]["create_time_confidence"] = (
            clean_value(row["timestamp_confidence"])
        )

    evidence_id = clean_value(row["evidence_id"])

    if evidence_id:
        graph.nodes[node_id]["process_evidence_id"] = evidence_id

    return node_id


def add_process_relationship(graph, row):
    """Add parent-child process relationship."""

    pid = clean_value(row["process_id"])
    ppid = clean_value(row["parent_process_id"])

    if not pid or not ppid:
        return

    try:
        pid = str(int(float(pid)))
        ppid = str(int(float(ppid)))
    except ValueError:
        return

    child = f"process:{pid}"
    parent = f"process:{ppid}"

    # Ensure both processes exist.
    if parent not in graph:
        graph.add_node(
            parent,
            node_type="process",
            pid=ppid,
            name="",
            create_time="",
            create_time_confidence="unknown",
            process_evidence_id="",
        )

    if child not in graph:
        graph.add_node(
            child,
            node_type="process",
            pid=pid,
            name=clean_value(row["process"]),
            create_time="",
            create_time_confidence="unknown",
            process_evidence_id="",
        )

    graph.add_edge(
        parent,
        child,
        relationship="parent_of",
        timestamp=clean_value(row["timestamp"]),
        timestamp_confidence=clean_value(
            row["timestamp_confidence"]
        ),
        evidence_id=clean_value(row["evidence_id"]),
        source=clean_value(row["source"]),
        artifact_type=clean_value(row["artifact_type"]),
        provenance=clean_value(row["provenance"]),
    )


def add_command_line(graph, row):
    """Attach command-line observation to a process."""

    pid = clean_value(row["process_id"])
    command_line = clean_value(row["command_line"])

    if not pid or not command_line:
        return

    try:
        pid = str(int(float(pid)))
    except ValueError:
        return

    process_node = f"process:{pid}"

    if process_node not in graph:
        graph.add_node(
            process_node,
            node_type="process",
            pid=pid,
            name=clean_value(row["process"]),
            create_time="",
            create_time_confidence="unknown",
            process_evidence_id="",
        )

    evidence_id = clean_value(row["evidence_id"])

    command_node = f"command:{evidence_id}"

    graph.add_node(
        command_node,
        node_type="command_line",
        command_line=command_line,
        evidence_id=evidence_id,
        source=clean_value(row["source"]),
        artifact_type=clean_value(row["artifact_type"]),
        provenance=clean_value(row["provenance"]),
    )

    graph.add_edge(
        process_node,
        command_node,
        relationship="has_command_line",
        timestamp=clean_value(row["timestamp"]),
        timestamp_confidence=clean_value(
            row["timestamp_confidence"]
        ),
        evidence_id=evidence_id,
        source=clean_value(row["source"]),
        artifact_type=clean_value(row["artifact_type"]),
        provenance=clean_value(row["provenance"]),
    )


def add_module(graph, row):
    """Attach loaded module observation to a process."""

    pid = clean_value(row["process_id"])
    module_name = clean_value(row["file"])
    module_path = clean_value(row["file_path"])

    if not pid:
        return

    try:
        pid = str(int(float(pid)))
    except ValueError:
        return

    process_node = f"process:{pid}"

    if process_node not in graph:
        graph.add_node(
            process_node,
            node_type="process",
            pid=pid,
            name=clean_value(row["process"]),
            create_time="",
            create_time_confidence="unknown",
            process_evidence_id="",
        )

    evidence_id = clean_value(row["evidence_id"])

    module_key = module_path or module_name

    if not module_key:
        module_key = evidence_id

    module_node = f"module:{evidence_id}"

    graph.add_node(
        module_node,
        node_type="module",
        name=module_name,
        path=module_path,
        evidence_id=evidence_id,
        source=clean_value(row["source"]),
        artifact_type=clean_value(row["artifact_type"]),
        provenance=clean_value(row["provenance"]),
    )

    graph.add_edge(
        process_node,
        module_node,
        relationship="loaded_module",
        timestamp=clean_value(row["timestamp"]),
        timestamp_confidence=clean_value(
            row["timestamp_confidence"]
        ),
        evidence_id=evidence_id,
        source=clean_value(row["source"]),
        artifact_type=clean_value(row["artifact_type"]),
        provenance=clean_value(row["provenance"]),
    )


def add_memory_region(graph, row):
    """Attach memory-region observation to a process."""

    pid = clean_value(row["process_id"])

    if not pid:
        return

    try:
        pid = str(int(float(pid)))
    except ValueError:
        return

    process_node = f"process:{pid}"

    if process_node not in graph:
        graph.add_node(
            process_node,
            node_type="process",
            pid=pid,
            name=clean_value(row["process"]),
            create_time="",
            create_time_confidence="unknown",
            process_evidence_id="",
        )

    evidence_id = clean_value(row["evidence_id"])

    memory_node = f"memory_region:{evidence_id}"

    graph.add_node(
        memory_node,
        node_type="memory_region",
        evidence_id=evidence_id,
        source=clean_value(row["source"]),
        artifact_type=clean_value(row["artifact_type"]),
        provenance=clean_value(row["provenance"]),
    )

    graph.add_edge(
        process_node,
        memory_node,
        relationship="has_memory_region",
        timestamp=clean_value(row["timestamp"]),
        timestamp_confidence=clean_value(
            row["timestamp_confidence"]
        ),
        evidence_id=evidence_id,
        source=clean_value(row["source"]),
        artifact_type=clean_value(row["artifact_type"]),
        provenance=clean_value(row["provenance"]),
    )


def build_graph(events):
    """Build the ForensiXplain temporal knowledge graph."""

    graph = nx.MultiDiGraph(
        case_id=CASE_ID,
        graph_type="temporal_forensic_knowledge_graph",
    )

    for _, row in events.iterrows():

        event_type = clean_value(row["event_type"])

        if event_type == "process":
            add_process_node(graph, row)

        elif event_type == "process_relationship":
            add_process_relationship(graph, row)

        elif event_type == "command_line":
            add_command_line(graph, row)

        elif event_type == "module":
            add_module(graph, row)

        elif event_type == "memory_region":
            add_memory_region(graph, row)

    return graph


def make_graphml_safe(graph):
    """Remove unsupported None values before GraphML export."""

    for node, attributes in graph.nodes(data=True):
        for key, value in list(attributes.items()):
            if value is None:
                attributes[key] = ""

    for source, target, key, attributes in graph.edges(
        keys=True,
        data=True,
    ):
        for attr_key, value in list(attributes.items()):
            if value is None:
                attributes[attr_key] = ""


def save_graph(graph):
    """Save graph to GraphML."""

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    make_graphml_safe(graph)

    nx.write_graphml(
        graph,
        OUTPUT_PATH,
    )

    print(
        f"\nGraph saved to: {OUTPUT_PATH}"
    )


def print_summary(graph):
    """Print graph statistics."""

    print(
        "\n=== ForensiXplain Temporal Knowledge Graph ==="
    )

    print(
        f"Nodes: {graph.number_of_nodes()}"
    )

    print(
        f"Edges: {graph.number_of_edges()}"
    )

    node_types = Counter(
        data.get("node_type", "unknown")
        for _, data in graph.nodes(data=True)
    )

    edge_types = Counter(
        data.get("relationship", "unknown")
        for _, _, data in graph.edges(data=True)
    )

    print("\nNode Types:")

    for node_type, count in node_types.items():
        print(
            f"  {node_type}: {count}"
        )

    print("\nRelationships:")

    for relationship, count in edge_types.items():
        print(
            f"  {relationship}: {count}"
        )


def main():

    if not EVENTS_PATH.exists():
        raise FileNotFoundError(
            f"Events file not found: {EVENTS_PATH}"
        )

    events = pd.read_csv(
        EVENTS_PATH,
        low_memory=False,
    )

    graph = build_graph(events)

    print_summary(graph)

    save_graph(graph)


if __name__ == "__main__":
    main()