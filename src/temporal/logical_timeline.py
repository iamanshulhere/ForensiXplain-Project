from pathlib import Path

import pandas as pd


CASE_ID = "M57-Jean"

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    BASE_DIR
    / "data"
    / "normalized"
    / CASE_ID
    / "timeline.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "normalized"
    / CASE_ID
)

OUTPUT_PATH = OUTPUT_DIR / "logical_timeline.csv"


def build_logical_timeline(timeline, case_id):
    """
    Build logical process-start events from timeline observations.

    The input dataframe is not modified in-place.
    """

    timeline = timeline.copy()

    # Parse timestamps
    timeline["timestamp"] = pd.to_datetime(
        timeline["timestamp"],
        errors="coerce",
        utc=True,
    )

    timeline = timeline[
        timeline["timestamp"].notna()
    ].copy()

    # Normalize process IDs
    timeline["process_id"] = pd.to_numeric(
        timeline["process_id"],
        errors="coerce",
    )

    timeline["parent_process_id"] = pd.to_numeric(
        timeline["parent_process_id"],
        errors="coerce",
    )

    # Sort chronologically
    timeline = timeline.sort_values(
        ["timestamp", "process_id"],
        kind="stable",
    ).reset_index(drop=True)

    # Separate observations
    process_starts = timeline[
        timeline["event_type"] == "process"
    ].copy()

    relationships = timeline[
        timeline["event_type"] == "process_relationship"
    ].copy()

    logical_events = []

    for _, process_event in process_starts.iterrows():

        pid = process_event["process_id"]
        timestamp = process_event["timestamp"]
        if pd.isna(pid):
            continue

        matching_relationship = relationships[
            (relationships["process_id"] == pid)
            & (relationships["timestamp"] == timestamp)
        ]

        # Parent PID
        parent_ids = (
            matching_relationship["parent_process_id"]
            .dropna()
            .astype(int)
            .astype(str)
            .unique()
            .tolist()
        )

        # Evidence
        evidence_ids = [
            str(process_event["evidence_id"])
        ]

        if not matching_relationship.empty:
            evidence_ids.extend(
                matching_relationship["evidence_id"]
                .dropna()
                .astype(str)
                .tolist()
            )

        evidence_ids = list(
            dict.fromkeys(evidence_ids)
        )

        # Provenance
        provenance_values = [
            str(value)
            for value in matching_relationship[
                "provenance"
            ].dropna().tolist()
        ]

        provenance_values.append(
            str(process_event["provenance"])
        )

        provenance_values = list(
            dict.fromkeys(provenance_values)
        )

        # Logical event
        logical_events.append(
            {
                "case_id": case_id,

                "logical_event_id": (
                    f"LEVT-{case_id}-"
                    f"PROCESS-{int(pid)}-"
                    f"{timestamp.strftime('%Y%m%d%H%M%S')}"
                ),

                "timestamp": timestamp,

                "timestamp_confidence": "observed",

                "logical_event_type": "process_start",

                "action": "process_start",

                "process_id": int(pid),

                "process": process_event["process"],

                "parent_process_ids": (
                    ";".join(parent_ids)
                ),

                "evidence_ids": (
                    ";".join(evidence_ids)
                ),

                "source_observation_count": (
                    1 + len(matching_relationship)
                ),

                "provenance": (
                    " | ".join(provenance_values)
                ),
            }
        )

    logical = pd.DataFrame(logical_events)

    # Empty input handling
    if logical.empty:
        logical["temporal_sequence"] = pd.Series(
            dtype="int64"
        )

        logical["previous_timestamp"] = pd.Series(
            dtype="datetime64[ns, UTC]"
        )

        logical[
            "time_since_previous_event_seconds"
        ] = pd.Series(dtype="float64")

        return logical

    # Sort
    logical = logical.sort_values(
        ["timestamp", "process_id"],
        kind="stable",
    ).reset_index(drop=True)

    # Temporal sequence
    logical["temporal_sequence"] = range(
        1,
        len(logical) + 1,
    )

    # Previous timestamp
    logical["previous_timestamp"] = (
        logical["timestamp"].shift(1)
    )

    # Time gap
    logical[
        "time_since_previous_event_seconds"
    ] = (
        logical["timestamp"]
        - logical["previous_timestamp"]
    ).dt.total_seconds().fillna(0)

    return logical


def main():

    print("=== ForensiXplain Logical Timeline ===")

    timeline = pd.read_csv(INPUT_PATH)

    print(
        f"Raw timeline observations: {len(timeline)}"
    )

    logical = build_logical_timeline(
        timeline,
        CASE_ID,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    logical.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\n=== Logical Timeline Complete ==="
    )

    print(
        f"Logical events: {len(logical)}"
    )

    print(
        f"Output: {OUTPUT_PATH}"
    )

    if logical.empty:
        return

    print("\nLogical event types:")

    print(
        logical[
            "logical_event_type"
        ].value_counts()
    )

    print("\nTime-gap statistics:")

    print(
        logical[
            "time_since_previous_event_seconds"
        ].describe()
    )

    print("\nFirst 20 logical events:")

    print(
        logical[
            [
                "temporal_sequence",
                "timestamp",
                "process_id",
                "process",
                "parent_process_ids",
                "source_observation_count",
                "time_since_previous_event_seconds",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()