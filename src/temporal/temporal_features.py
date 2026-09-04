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


def main():
    print("=== ForensiXplain Temporal Feature Engineering ===")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Logical timeline not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    print(f"Logical timeline events loaded: {len(df)}")

    # ---------------------------------------------------------
    # Timestamp preparation
    # ---------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True,
    )

    df = df[df["timestamp"].notna()].copy()

    df = df.sort_values(
        ["timestamp", "process_id"],
        kind="stable",
    ).reset_index(drop=True)

    print(f"Timestamped events: {len(df)}")

    # ---------------------------------------------------------
    # Basic temporal features
    # ---------------------------------------------------------

    df["temporal_sequence"] = np.arange(1, len(df) + 1)

    df["previous_timestamp"] = df["timestamp"].shift(1)

    df["time_since_previous_event_seconds"] = (
        df["timestamp"] - df["previous_timestamp"]
    ).dt.total_seconds()

    df["time_since_previous_event_seconds"] = (
        df["time_since_previous_event_seconds"]
        .fillna(0)
        .clip(lower=0)
    )

    df["event_hour"] = df["timestamp"].dt.hour

    df["event_day_of_week"] = df["timestamp"].dt.dayofweek

    # ---------------------------------------------------------
    # After-hours feature
    #
    # 00:00-06:59 and 22:00-23:59
    # ---------------------------------------------------------

    df["after_hours"] = (
        (df["event_hour"] < 7)
        | (df["event_hour"] >= 22)
    ).astype(int)

    # ---------------------------------------------------------
    # Local temporal density
    #
    # Number of events occurring within the previous
    # N-second window, including the current event.
    # ---------------------------------------------------------

    timestamps = df["timestamp"]

    def events_within_window(seconds):
        values = timestamps.astype("int64").to_numpy()
        window_ns = seconds * 1_000_000_000

        counts = []

        for i, current_time in enumerate(values):
            lower_bound = current_time - window_ns

            start = np.searchsorted(
                values,
                lower_bound,
                side="left",
            )

            counts.append(i - start + 1)

        return counts

    df["events_in_10s"] = events_within_window(10)

    df["events_in_30s"] = events_within_window(30)

    df["events_in_60s"] = events_within_window(60)

    df["events_in_300s"] = events_within_window(300)

    # ---------------------------------------------------------
    # Previous process context
    # ---------------------------------------------------------

    df["previous_process_id"] = (
        pd.to_numeric(
            df["process_id"],
            errors="coerce",
        )
        .shift(1)
    )

    df["previous_process"] = df["process"].shift(1)

    df["process_changed"] = (
        df["process"] != df["previous_process"]
    ).astype(int)

    # ---------------------------------------------------------
    # Chronological process transition
    #
    # This represents temporal adjacency only.
    # It must NOT be interpreted as causality.
    # ---------------------------------------------------------

    df["process_transition"] = (
        df["previous_process"]
        .fillna("START")
        .astype(str)
        + " -> "
        + df["process"].astype(str)
    )

    # ---------------------------------------------------------
    # Process relationship information
    # ---------------------------------------------------------

    df["process_id"] = pd.to_numeric(
        df["process_id"],
        errors="coerce",
    )

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

    df["rapid_event"] = (
        df["time_since_previous_event_seconds"] <= 5
    ).astype(int)

    df["short_event_gap"] = (
        df["time_since_previous_event_seconds"] <= 30
    ).astype(int)

    df["medium_event_gap"] = (
        df["time_since_previous_event_seconds"] <= 300
    ).astype(int)

    df["long_event_gap"] = (
        df["time_since_previous_event_seconds"] > 300
    ).astype(int)

    # ---------------------------------------------------------
    # Evidence support
    # ---------------------------------------------------------

    df["source_observation_count"] = pd.to_numeric(
        df["source_observation_count"],
        errors="coerce",
    ).fillna(0)

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
    print("=== Temporal Feature Engineering Complete ===")
    print(f"Feature rows: {len(df)}")
    print(f"Feature columns: {len(df.columns)}")
    print(f"Output: {OUTPUT_PATH}")

    print()
    print("Time-gap statistics:")

    print(
        df["time_since_previous_event_seconds"].describe()
    )

    print()
    print("Temporal density statistics:")

    print(
        df[
            [
                "events_in_10s",
                "events_in_30s",
                "events_in_60s",
                "events_in_300s",
            ]
        ].describe()
    )

    print()
    print("Rapid/short/medium/long events:")

    print(
        "rapid_event:",
        int(df["rapid_event"].sum()),
    )

    print(
        "short_event_gap:",
        int(df["short_event_gap"].sum()),
    )

    print(
        "medium_event_gap:",
        int(df["medium_event_gap"].sum()),
    )

    print(
        "long_event_gap:",
        int(df["long_event_gap"].sum()),
    )


if __name__ == "__main__":
    main()