from pathlib import Path

import networkx as nx
import pandas as pd


CASE_ID = "M57-Jean"

EVENTS_PATH = Path(
    f"data/normalized/{CASE_ID}/events.csv"
)

GRAPH_PATH = Path(
    f"data/normalized/{CASE_ID}/temporal_graph.graphml"
)

OUTPUT_PATH = Path(
    f"data/features/{CASE_ID}/features.csv"
)


def safe_int(value):
    """Convert a value to integer when possible."""

    try:
        return int(float(value))
    except (ValueError, TypeError):
        return 0


def build_process_features(events, graph):
    """
    Build process-level behavioral features.

    These features are derived only from observed forensic
    events and graph structure.
    """

    process_events = events[
        events["event_type"] == "process"
    ].copy()

    relationship_events = events[
        events["event_type"] == "process_relationship"
    ].copy()

    command_events = events[
        events["event_type"] == "command_line"
    ].copy()

    module_events = events[
        events["event_type"] == "module"
    ].copy()

    memory_events = events[
        events["event_type"] == "memory_region"
    ].copy()

    rows = []

    process_ids = set()

    for _, row in process_events.iterrows():
        pid = safe_int(row.get("process_id"))

        if pid:
            process_ids.add(pid)

    for _, row in relationship_events.iterrows():
        pid = safe_int(row.get("process_id"))

        if pid:
            process_ids.add(pid)

    for pid in sorted(process_ids):

        node_id = f"process:{pid}"

        node_data = graph.nodes.get(
            node_id,
            {}
        )

        process_name = node_data.get(
            "name",
            ""
        )

        # Parent count
        parent_count = graph.in_degree(
            node_id
        )

        # Child count
        child_count = graph.out_degree(
            node_id
        )

        # Command-line observations
        command_count = len(
            command_events[
                command_events["process_id"]
                .apply(safe_int)
                == pid
            ]
        )

        # Loaded modules
        module_count = len(
            module_events[
                module_events["process_id"]
                .apply(safe_int)
                == pid
            ]
        )

        # Memory regions
        memory_region_count = len(
            memory_events[
                memory_events["process_id"]
                .apply(safe_int)
                == pid
            ]
        )

        # Process creation observations
        process_rows = process_events[
            process_events["process_id"]
            .apply(safe_int)
            == pid
        ]

        create_time = ""

        if not process_rows.empty:
            create_time = str(
                process_rows.iloc[0]["timestamp"]
            )

        rows.append(
            {
                "case_id": CASE_ID,
                "process_id": pid,
                "process_name": process_name,
                "create_time": create_time,
                "parent_count": parent_count,
                "child_count": child_count,
                "command_line_count": command_count,
                "module_count": module_count,
                "memory_region_count": memory_region_count,
                "graph_degree": (
                    parent_count + child_count
                ),
            }
        )

    return pd.DataFrame(rows)


def add_temporal_features(features):

    """
    Add simple temporal features.

    Missing timestamps remain missing rather than being inferred.
    """

    features = features.copy()

    features["create_time"] = pd.to_datetime(
        features["create_time"],
        errors="coerce",
        utc=True,
    )

    features["hour"] = (
        features["create_time"]
        .dt.hour
        .fillna(-1)
        .astype(int)
    )

    features["day_of_week"] = (
        features["create_time"]
        .dt.dayofweek
        .fillna(-1)
        .astype(int)
    )

    features["after_hours"] = (
        (features["hour"] >= 22)
        | (features["hour"] < 6)
    ).astype(int)

    features["timestamp_available"] = (
        features["create_time"]
        .notna()
        .astype(int)
    )

    return features


def save_features(features):

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nFeatures saved to: {OUTPUT_PATH}"
    )


def main():

    if not EVENTS_PATH.exists():
        raise FileNotFoundError(
            f"Events file not found: {EVENTS_PATH}"
        )

    if not GRAPH_PATH.exists():
        raise FileNotFoundError(
            f"Graph file not found: {GRAPH_PATH}"
        )

    print(
        "=== ForensiXplain Feature Engineering ==="
    )

    events = pd.read_csv(
        EVENTS_PATH,
        low_memory=False,
    )

    graph = nx.read_graphml(
        GRAPH_PATH
    )

    print(
        f"Events loaded: {len(events)}"
    )

    print(
        f"Graph nodes: {graph.number_of_nodes()}"
    )

    print(
        f"Graph edges: {graph.number_of_edges()}"
    )

    features = build_process_features(
        events,
        graph,
    )

    features = add_temporal_features(
        features
    )

    save_features(features)

    print(
        f"Feature rows: {len(features)}"
    )

    print(
        f"Feature columns: {len(features.columns)}"
    )

    print("\nFeatures:")

    for column in features.columns:
        print(
            f"  {column}"
        )


if __name__ == "__main__":
    main()