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

    Features are derived only from observed forensic
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

    # -----------------------------------------------------
    # Collect process IDs from process observations
    # -----------------------------------------------------

    for _, row in process_events.iterrows():

        pid = safe_int(
            row.get("process_id")
        )

        if pid:
            process_ids.add(pid)

    # -----------------------------------------------------
    # Collect process IDs from relationship observations
    # -----------------------------------------------------

    for _, row in relationship_events.iterrows():

        pid = safe_int(
            row.get("process_id")
        )

        if pid:
            process_ids.add(pid)

    # -----------------------------------------------------
    # Build process-level features
    # -----------------------------------------------------

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

        # -------------------------------------------------
        # Graph structure
        # -------------------------------------------------

        parent_count = graph.in_degree(
            node_id
        )

        child_count = graph.out_degree(
            node_id
        )

        graph_degree = (
            parent_count + child_count
        )

        # -------------------------------------------------
        # Command-line observations
        # -------------------------------------------------

        command_count = len(
            command_events[
                command_events["process_id"]
                .apply(safe_int)
                == pid
            ]
        )

        # -------------------------------------------------
        # Loaded modules
        # -------------------------------------------------

        module_count = len(
            module_events[
                module_events["process_id"]
                .apply(safe_int)
                == pid
            ]
        )

        # -------------------------------------------------
        # Memory regions
        # -------------------------------------------------

        memory_region_count = len(
            memory_events[
                memory_events["process_id"]
                .apply(safe_int)
                == pid
            ]
        )

        # -------------------------------------------------
        # Process creation timestamp
        # -------------------------------------------------

        process_rows = process_events[
            process_events["process_id"]
            .apply(safe_int)
            == pid
        ]

        create_time = pd.NaT

        if not process_rows.empty:

            timestamp = process_rows.iloc[0].get(
                "timestamp"
            )

            parsed_timestamp = pd.to_datetime(
                timestamp,
                errors="coerce",
                utc=True,
            )

            create_time = parsed_timestamp

        # -------------------------------------------------
        # Store row
        # -------------------------------------------------

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
                "graph_degree": graph_degree,
            }
        )

    return pd.DataFrame(rows)


def add_temporal_features(features):
    """
    Add temporal features while preserving the distinction
    between known and unavailable timestamps.

    Missing timestamps are NOT converted into artificial
    temporal values such as -1.
    """

    features = features.copy()

    # -----------------------------------------------------
    # Parse timestamps
    # -----------------------------------------------------

    features["create_time"] = pd.to_datetime(
        features["create_time"],
        errors="coerce",
        utc=True,
    )

    # -----------------------------------------------------
    # Timestamp availability
    # -----------------------------------------------------

    features["timestamp_available"] = (
        features["create_time"]
        .notna()
        .astype(int)
    )

    # -----------------------------------------------------
    # Extract temporal values
    #
    # Missing timestamps remain NaN at this stage.
    # -----------------------------------------------------

    features["hour"] = (
        features["create_time"]
        .dt.hour
    )

    features["day_of_week"] = (
        features["create_time"]
        .dt.dayofweek
    )

    # -----------------------------------------------------
    # After-hours activity
    #
    # Only calculate this when a timestamp exists.
    # Missing timestamps are represented as 0 because
    # there is insufficient evidence to classify them
    # as after-hours activity.
    # -----------------------------------------------------

    features["after_hours"] = 0

    known_timestamp = (
        features["timestamp_available"] == 1
    )

    features.loc[
        known_timestamp,
        "after_hours"
    ] = (
        (features.loc[known_timestamp, "hour"] >= 22)
        |
        (features.loc[known_timestamp, "hour"] < 6)
    ).astype(int)

    # -----------------------------------------------------
    # Model-safe temporal representation
    #
    # For processes without timestamps, use the median
    # observed temporal values rather than an artificial
    # value such as -1.
    #
    # timestamp_available preserves the fact that the
    # timestamp was unavailable.
    # -----------------------------------------------------

    known_hours = features.loc[
        known_timestamp,
        "hour"
    ]

    known_days = features.loc[
        known_timestamp,
        "day_of_week"
    ]

    if not known_hours.empty:
        hour_median = int(
            known_hours.median()
        )
    else:
        hour_median = 12

    if not known_days.empty:
        day_median = int(
            known_days.median()
        )
    else:
        day_median = 3

    features["hour"] = (
        features["hour"]
        .fillna(hour_median)
        .astype(int)
    )

    features["day_of_week"] = (
        features["day_of_week"]
        .fillna(day_median)
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

    # -----------------------------------------------------
    # Load events
    # -----------------------------------------------------

    events = pd.read_csv(
        EVENTS_PATH,
        low_memory=False,
    )

    # -----------------------------------------------------
    # Load graph
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Build process features
    # -----------------------------------------------------

    features = build_process_features(
        events,
        graph,
    )

    # -----------------------------------------------------
    # Add temporal features
    # -----------------------------------------------------

    features = add_temporal_features(
        features
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    save_features(features)

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Timestamp diagnostics
    # -----------------------------------------------------

    timestamp_count = int(
        features["timestamp_available"].sum()
    )

    missing_count = (
        len(features)
        - timestamp_count
    )

    print(
        "\nTimestamp diagnostics:"
    )

    print(
        f"  Available: {timestamp_count}"
    )

    print(
        f"  Unavailable: {missing_count}"
    )

    print(
        "\nTemporal feature ranges:"
    )

    print(
        f"  Hour: "
        f"{features['hour'].min()} - "
        f"{features['hour'].max()}"
    )

    print(
        f"  Day of week: "
        f"{features['day_of_week'].min()} - "
        f"{features['day_of_week'].max()}"
    )


if __name__ == "__main__":
    main()