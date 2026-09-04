from pathlib import Path

import pandas as pd


# =========================================================
# Configuration
# =========================================================

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

OUTPUT_PATH = (
    OUTPUT_DIR
    / "logical_timeline.csv"
)


# =========================================================
# Main
# =========================================================

def main():

    print("=== ForensiXplain Logical Timeline ===")

    # -----------------------------------------------------
    # Load timeline
    # -----------------------------------------------------

    timeline = pd.read_csv(INPUT_PATH)

    print(
        f"Raw timeline observations: {len(timeline)}"
    )

    # -----------------------------------------------------
    # Parse timestamps
    # -----------------------------------------------------

    timeline["timestamp"] = pd.to_datetime(
        timeline["timestamp"],
        errors="coerce",
        utc=True,
    )

    timeline = timeline[
        timeline["timestamp"].notna()
    ].copy()

    # -----------------------------------------------------
    # Normalize process IDs
    # -----------------------------------------------------

    timeline["process_id"] = pd.to_numeric(
        timeline["process_id"],
        errors="coerce",
    )

    timeline["parent_process_id"] = pd.to_numeric(
        timeline["parent_process_id"],
        errors="coerce",
    )

    # -----------------------------------------------------
    # Sort chronologically
    # -----------------------------------------------------

    timeline = timeline.sort_values(
        [
            "timestamp",
            "process_id",
        ],
        kind="stable",
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # Separate process observations
    # -----------------------------------------------------

    process_starts = timeline[
        timeline["event_type"] == "process"
    ].copy()

    relationships = timeline[
        timeline["event_type"]
        == "process_relationship"
    ].copy()

    print(
        f"Process-start observations: "
        f"{len(process_starts)}"
    )

    print(
        f"Relationship observations: "
        f"{len(relationships)}"
    )

    # -----------------------------------------------------
    # Create logical process-start events
    # -----------------------------------------------------

    logical_events = []

    for _, process_event in process_starts.iterrows():

        pid = process_event["process_id"]

        timestamp = process_event["timestamp"]

        # Find corresponding parent relationship
        matching_relationship = relationships[
            (
                relationships["process_id"]
                == pid
            )
            &
            (
                relationships["timestamp"]
                == timestamp
            )
        ]

        # -------------------------------------------------
        # Parent PID
        # -------------------------------------------------

        parent_ids = (
            matching_relationship[
                "parent_process_id"
            ]
            .dropna()
            .astype(int)
            .astype(str)
            .unique()
            .tolist()
        )

        # -------------------------------------------------
        # Supporting evidence
        # -------------------------------------------------

        evidence_ids = [
            str(process_event["evidence_id"])
        ]

        if not matching_relationship.empty:

            evidence_ids.extend(
                matching_relationship[
                    "evidence_id"
                ]
                .dropna()
                .astype(str)
                .tolist()
            )

        # Remove duplicates while preserving order

        evidence_ids = list(
            dict.fromkeys(evidence_ids)
        )

        # -------------------------------------------------
        # Provenance
        # -------------------------------------------------

        provenance_values = [
            str(value)
            for value in (
                matching_relationship[
                    "provenance"
                ]
                .dropna()
                .tolist()
            )
        ]

        provenance_values.append(
            str(process_event["provenance"])
        )

        provenance_values = list(
            dict.fromkeys(
                provenance_values
            )
        )

        # -------------------------------------------------
        # Create logical event
        # -------------------------------------------------

        logical_events.append(
            {
                "case_id": CASE_ID,

                "logical_event_id": (
                    f"LEVT-{CASE_ID}-"
                    f"PROCESS-{int(pid)}-"
                    f"{timestamp.strftime('%Y%m%d%H%M%S')}"
                ),

                "timestamp": timestamp,

                "timestamp_confidence": (
                    "observed"
                ),

                "logical_event_type": (
                    "process_start"
                ),

                "action": "process_start",

                "process_id": int(pid),

                "process": process_event[
                    "process"
                ],

                "parent_process_ids": (
                    ";".join(parent_ids)
                ),

                "evidence_ids": (
                    ";".join(evidence_ids)
                ),

                "source_observation_count": (
                    1 + len(
                        matching_relationship
                    )
                ),

                "provenance": (
                    " | ".join(
                        provenance_values
                    )
                ),
            }
        )

    # -----------------------------------------------------
    # Add logical relationship events
    # -----------------------------------------------------

    #
    # Important:
    #
    # Process relationships are already represented by
    # the parent_process_ids field of the logical process
    # start event.
    #
    # Therefore we do NOT create another logical event
    # for the same PSTree observation.
    #

    logical = pd.DataFrame(
        logical_events
    )

    # -----------------------------------------------------
    # Sort
    # -----------------------------------------------------

    logical = logical.sort_values(
        [
            "timestamp",
            "process_id",
        ],
        kind="stable",
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # Sequence number
    # -----------------------------------------------------

    logical["temporal_sequence"] = (
        range(
            1,
            len(logical) + 1,
        )
    )

    # -----------------------------------------------------
    # Time gap
    # -----------------------------------------------------

    logical["previous_timestamp"] = (
        logical["timestamp"].shift(1)
    )

    logical[
        "time_since_previous_event_seconds"
    ] = (
        logical["timestamp"]
        - logical["previous_timestamp"]
    ).dt.total_seconds()

    logical[
        "time_since_previous_event_seconds"
    ] = logical[
        "time_since_previous_event_seconds"
    ].fillna(0)

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    logical.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # -----------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------

    print(
        "\n=== Logical Timeline Complete ==="
    )

    print(
        f"Logical events: {len(logical)}"
    )

    print(
        f"Output: {OUTPUT_PATH}"
    )

    print(
        "\nLogical event types:"
    )

    print(
        logical[
            "logical_event_type"
        ].value_counts()
    )

    print(
        "\nTime-gap statistics:"
    )

    print(
        logical[
            "time_since_previous_event_seconds"
        ].describe()
    )

    print(
        "\nFirst 20 logical events:"
    )

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