from pathlib import Path

import numpy as np
import pandas as pd


CASE_ID = "M57-Jean"

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    BASE_DIR
    / "data"
    / "normalized"
    / CASE_ID
    / "logical_timeline.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "features"
    / CASE_ID
)

OUTPUT_PATH = OUTPUT_DIR / "temporal_features.csv"


def count_events_within_window(
    timestamps_ns,
    window_seconds,
    direction="previous",
):
    """
    Count events in a temporal window around each event.

    previous:
        Events from [current_time - window, current_time]

    next:
        Events from [current_time, current_time + window]
    """

    window_ns = window_seconds * 1_000_000_000

    counts = []

    for current_time in timestamps_ns:

        if direction == "previous":
            lower = current_time - window_ns
            upper = current_time

        elif direction == "next":
            lower = current_time
            upper = current_time + window_ns

        else:
            raise ValueError(
                "direction must be 'previous' or 'next'"
            )

        left = np.searchsorted(
            timestamps_ns,
            lower,
            side="left",
        )

        right = np.searchsorted(
            timestamps_ns,
            upper,
            side="right",
        )

        counts.append(right - left)

    return counts


def main():

    print("=== ForensiXplain Temporal Feature Engineering ===")

    # ---------------------------------------------------------
    # Load logical timeline
    # ---------------------------------------------------------

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Logical timeline not found:\n{INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    print(
        f"Logical timeline events loaded: {len(df)}"
    )

    # ---------------------------------------------------------
    # Timestamp preparation
    # ---------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True,
    )

    df = df[
        df["timestamp"].notna()
    ].copy()

    df = df.sort_values(
        ["timestamp", "process_id"],
        kind="stable",
    ).reset_index(drop=True)

    print(
        f"Timestamped events: {len(df)}"
    )

    # ---------------------------------------------------------
    # Temporal sequence
    # ---------------------------------------------------------

    df["temporal_sequence"] = (
        np.arange(1, len(df) + 1)
    )

    # ---------------------------------------------------------
    # Previous event
    # ---------------------------------------------------------

    df["previous_timestamp"] = (
        df["timestamp"].shift(1)
    )

    df["is_first_event"] = (
        df["previous_timestamp"].isna()
    ).astype(int)

    # ---------------------------------------------------------
    # Time gap
    # ---------------------------------------------------------

    df["time_since_previous_event_seconds"] = (
        df["timestamp"]
        - df["previous_timestamp"]
    ).dt.total_seconds()

    # IMPORTANT:
    # The first event has no previous event.
    # Therefore its temporal gap remains NaN.

    # ---------------------------------------------------------
    # Log-transformed time gap
    #
    # This reduces the extreme effect of very large gaps.
    # Example:
    # 2 seconds -> log1p(2)
    # 228024 sec -> log1p(228024)
    # ---------------------------------------------------------

    df["gap_log_seconds"] = np.log1p(
        df["time_since_previous_event_seconds"]
    )

    # ---------------------------------------------------------
    # Calendar/time features
    # ---------------------------------------------------------

    df["event_hour"] = (
        df["timestamp"].dt.hour
    )

    df["event_day_of_week"] = (
        df["timestamp"].dt.dayofweek
    )

    # ---------------------------------------------------------
    # After-hours
    #
    # 00:00-06:59
    # 22:00-23:59
    # ---------------------------------------------------------

    df["after_hours"] = (
        (df["event_hour"] < 7)
        | (df["event_hour"] >= 22)
    ).astype(int)

    # ---------------------------------------------------------
    # Temporal density
    # ---------------------------------------------------------

    timestamps_ns = (
        df["timestamp"]
        .astype("int64")
        .to_numpy()
    )

    for window in [10, 30, 60]:

        df[
            f"events_prev_{window}s"
        ] = count_events_within_window(
            timestamps_ns,
            window,
            direction="previous",
        )

        df[
            f"events_next_{window}s"
        ] = count_events_within_window(
            timestamps_ns,
            window,
            direction="next",
        )

        # Events around the current event.
        #
        # Subtract one so the current event itself
        # is not counted.

        df[
            f"local_density_{window}s"
        ] = (
            df[f"events_prev_{window}s"]
            + df[f"events_next_{window}s"]
            - 1
        )

    # ---------------------------------------------------------
    # Previous process context
    # ---------------------------------------------------------

    df["previous_process_id"] = (
        pd.to_numeric(
            df["process_id"],
            errors="coerce",
        ).shift(1)
    )

    df["previous_process"] = (
        df["process"].shift(1)
    )

    # ---------------------------------------------------------
    # Process transition
    #
    # IMPORTANT:
    # This represents chronological adjacency only.
    # It does NOT establish causality.
    # ---------------------------------------------------------

    df["process_changed"] = (
        df["process"]
        != df["previous_process"]
    ).astype(int)

    df["process_transition"] = (
        df["previous_process"]
        .fillna("START")
        .astype(str)
        + " -> "
        + df["process"].astype(str)
    )

    # ---------------------------------------------------------
    # Process identifiers
    # ---------------------------------------------------------

    df["process_id"] = pd.to_numeric(
        df["process_id"],
        errors="coerce",
    )

    # ---------------------------------------------------------
    # Parent process
    # ---------------------------------------------------------

    df["parent_process_id"] = (
        df["parent_process_ids"]
        .fillna("")
        .astype(str)
        .str.split(";")
        .str[0]
    )

    df["parent_process_id"] = pd.to_numeric(
        df["parent_process_id"],
        errors="coerce",
    )

    # ---------------------------------------------------------
    # Temporal indicators
    # ---------------------------------------------------------

    # Do NOT classify the first event as rapid.
    df["rapid_event"] = (
        (
            df[
                "time_since_previous_event_seconds"
            ] <= 5
        )
        & (
            df["is_first_event"] == 0
        )
    ).astype(int)

    df["short_event_gap"] = (
        (
            df[
                "time_since_previous_event_seconds"
            ] <= 30
        )
        & (
            df["is_first_event"] == 0
        )
    ).astype(int)

    df["medium_event_gap"] = (
        (
            df[
                "time_since_previous_event_seconds"
            ] <= 300
        )
        & (
            df["is_first_event"] == 0
        )
    ).astype(int)

    df["long_event_gap"] = (
        (
            df[
                "time_since_previous_event_seconds"
            ] > 300
        )
        & (
            df["is_first_event"] == 0
        )
    ).astype(int)

    # ---------------------------------------------------------
    # Evidence support
    # ---------------------------------------------------------

    df["source_observation_count"] = (
        pd.to_numeric(
            df["source_observation_count"],
            errors="coerce",
        ).fillna(0)
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # ---------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------

    print()
    print(
        "=== Temporal Feature Engineering Complete ==="
    )

    print(
        f"Feature rows: {len(df)}"
    )

    print(
        f"Feature columns: {len(df.columns)}"
    )

    print(
        f"Output: {OUTPUT_PATH}"
    )

    print()
    print("Time-gap statistics:")

    print(
        df[
            "time_since_previous_event_seconds"
        ].describe()
    )

    print()
    print("Temporal density statistics:")

    print(
        df[
            [
                "local_density_10s",
                "local_density_30s",
                "local_density_60s",
            ]
        ].describe()
    )

    print()
    print("Temporal indicators:")

    print(
        "First event:",
        int(df["is_first_event"].sum()),
    )

    print(
        "Rapid events:",
        int(df["rapid_event"].sum()),
    )

    print(
        "Short-gap events:",
        int(df["short_event_gap"].sum()),
    )

    print(
        "Medium-gap events:",
        int(df["medium_event_gap"].sum()),
    )

    print(
        "Long-gap events:",
        int(df["long_event_gap"].sum()),
    )


if __name__ == "__main__":
    main()