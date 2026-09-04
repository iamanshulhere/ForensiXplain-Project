from pathlib import Path

import pandas as pd


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

TIMELINE_PATH = (
    BASE_DIR
    / "data"
    / "normalized"
    / "M57-Jean"
    / "timeline.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "features"
    / "M57-Jean"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "temporal_features.csv"
)


# =========================================================
# Configuration
# =========================================================

WINDOWS_SECONDS = [
    10,
    30,
    60,
    300,
]


# =========================================================
# Main
# =========================================================

def main():

    print("=== ForensiXplain Temporal Feature Engineering ===")

    # -----------------------------------------------------
    # Load timeline
    # -----------------------------------------------------

    timeline = pd.read_csv(TIMELINE_PATH)

    print(
        f"Timeline events loaded: {len(timeline)}"
    )

    # -----------------------------------------------------
    # Parse timestamps
    # -----------------------------------------------------

    timeline["timestamp"] = pd.to_datetime(
        timeline["timestamp"],
        errors="coerce",
        utc=True,
    )

    # -----------------------------------------------------
    # Keep only timestamped events
    # -----------------------------------------------------

    temporal = timeline[
        timeline["timestamp"].notna()
    ].copy()

    temporal = temporal.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    print(
        f"Timestamped events: {len(temporal)}"
    )

    if temporal.empty:

        print(
            "[ERROR] No timestamped events found."
        )

        return

    # -----------------------------------------------------
    # Basic temporal features
    # -----------------------------------------------------

    temporal["previous_timestamp"] = (
        temporal["timestamp"].shift(1)
    )

    temporal["time_since_previous_event_seconds"] = (
        temporal["timestamp"]
        - temporal["previous_timestamp"]
    ).dt.total_seconds()

    temporal[
        "time_since_previous_event_seconds"
    ] = temporal[
        "time_since_previous_event_seconds"
    ].fillna(0)

    # -----------------------------------------------------
    # Event sequence information
    # -----------------------------------------------------

    temporal["temporal_sequence"] = (
        range(1, len(temporal) + 1)
    )

    temporal["event_hour"] = (
        temporal["timestamp"].dt.hour
    )

    temporal["event_day_of_week"] = (
        temporal["timestamp"].dt.dayofweek
    )

    temporal["after_hours"] = (
        (
            (temporal["event_hour"] >= 22)
            |
            (temporal["event_hour"] < 6)
        )
        .astype(int)
    )

    # -----------------------------------------------------
    # Local event density
    # -----------------------------------------------------

    timestamps = (
        temporal["timestamp"]
        .astype("int64")
        / 1_000_000_000
    )

    for window in WINDOWS_SECONDS:

        counts = []

        for current_time in timestamps:

            lower_bound = (
                current_time - window
            )

            count = (
                (
                    timestamps >= lower_bound
                )
                &
                (
                    timestamps <= current_time
                )
            ).sum()

            counts.append(int(count))

        temporal[
            f"events_in_{window}s"
        ] = counts

    # -----------------------------------------------------
    # Process transition features
    # -----------------------------------------------------

    temporal["previous_process_id"] = (
        temporal["process_id"].shift(1)
    )

    temporal["previous_process"] = (
        temporal["process"].shift(1)
    )

    temporal["process_transition"] = (
        temporal["previous_process"].fillna(
            "UNKNOWN"
        )
        + " -> "
        + temporal["process"].fillna(
            "UNKNOWN"
        )
    )

    temporal["process_changed"] = (
        (
            temporal["previous_process_id"]
            != temporal["process_id"]
        )
        &
        temporal["previous_process_id"].notna()
    ).astype(int)

    # -----------------------------------------------------
    # Process relationship features
    # -----------------------------------------------------

    temporal["is_process_relationship"] = (
        temporal["event_type"]
        == "process_relationship"
    ).astype(int)

    temporal["is_process_start"] = (
        temporal["event_type"]
        == "process"
    ).astype(int)

    # -----------------------------------------------------
    # Parent-child temporal relationship
    # -----------------------------------------------------

    temporal["parent_process_id"] = pd.to_numeric(
        temporal["parent_process_id"],
        errors="coerce",
    )

    temporal["process_id_numeric"] = pd.to_numeric(
        temporal["process_id"],
        errors="coerce",
    )

    # -----------------------------------------------------
    # Sequence gap categories
    # -----------------------------------------------------

    temporal["rapid_event"] = (
        (
            temporal[
                "time_since_previous_event_seconds"
            ] <= 5
        )
        &
        (
            temporal["temporal_sequence"] > 1
        )
    ).astype(int)

    temporal["short_event_gap"] = (
        (
            temporal[
                "time_since_previous_event_seconds"
            ] <= 30
        )
        &
        (
            temporal["temporal_sequence"] > 1
        )
    ).astype(int)

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporal.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nTemporal features saved to:"
    )

    print(OUTPUT_PATH)

    print(
        f"Rows: {len(temporal)}"
    )

    print(
        f"Columns: {len(temporal.columns)}"
    )

    # -----------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------

    print(
        "\nTime-gap statistics:"
    )

    print(
        temporal[
            "time_since_previous_event_seconds"
        ].describe()
    )

    print(
        "\nTemporal event-density features:"
    )

    for window in WINDOWS_SECONDS:

        column = f"events_in_{window}s"

        print(
            f"{column}: "
            f"min={temporal[column].min()}, "
            f"max={temporal[column].max()}, "
            f"mean={temporal[column].mean():.2f}"
        )

    print(
        "\nProcess transitions:"
    )

    print(
        temporal[
            [
                "timestamp",
                "previous_process",
                "process",
                "process_transition",
                "time_since_previous_event_seconds",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()