from pathlib import Path

import pandas as pd


CASE_ID = "M57-Jean"

EVENTS_PATH = Path(
    f"data/normalized/{CASE_ID}/events.csv"
)

RESULTS_PATH = Path(
    f"results/{CASE_ID}/isolation_forest_results.csv"
)

OUTPUT_PATH = Path(
    f"results/{CASE_ID}/evidence_linked_anomalies.csv"
)


def safe_pid(value):
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return -1


def main():

    print("=== ForensiXplain Evidence-Linked Anomaly Analysis ===")

    events = pd.read_csv(
        EVENTS_PATH,
        low_memory=False,
    )

    results = pd.read_csv(
        RESULTS_PATH,
        low_memory=False,
    )

    anomalies = results[
        results["predicted_anomaly"] == 1
    ].copy()

    print(
        f"Anomalous processes: {len(anomalies)}"
    )

    rows = []

    for _, anomaly in anomalies.iterrows():

        pid = safe_pid(
            anomaly["process_id"]
        )

        process_name = anomaly["process_name"]

        process_events = events[
            events["process_id"]
            .apply(safe_pid)
            == pid
        ].copy()

        # Count evidence observations by artifact.
        artifact_counts = (
            process_events["artifact_type"]
            .fillna("unknown")
            .value_counts()
            .to_dict()
        )

        # Collect evidence IDs.
        evidence_ids = (
            process_events["evidence_id"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        # Collect event types.
        event_types = (
            process_events["event_type"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        # Command-line observations.
        command_lines = (
            process_events[
                process_events["event_type"]
                == "command_line"
            ]["command_line"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        # Parent relationships.
        parent_events = events[
            (events["event_type"] == "process_relationship")
            & (
                events["process_id"]
                .apply(safe_pid)
                == pid
            )
        ]

        parents = (
            parent_events["parent_process_id"]
            .dropna()
            .apply(safe_pid)
            .unique()
            .tolist()
        )

        # Child relationships.
        child_events = events[
            (events["event_type"] == "process_relationship")
            & (
                events["parent_process_id"]
                .apply(safe_pid)
                == pid
            )
        ]

        children = (
            child_events["process_id"]
            .dropna()
            .apply(safe_pid)
            .unique()
            .tolist()
        )

        rows.append(
            {
                "case_id": CASE_ID,
                "process_id": pid,
                "process_name": process_name,
                "anomaly_score": anomaly[
                    "anomaly_score"
                ],
                "predicted_anomaly": anomaly[
                    "predicted_anomaly"
                ],
                "parent_count": len(parents),
                "child_count": len(children),
                "command_line_count": len(command_lines),
                "module_count": artifact_counts.get(
                    "dlllist",
                    0,
                ),
                "memory_region_count": artifact_counts.get(
                    "malfind",
                    0,
                ),
                "event_types": ";".join(
                    event_types
                ),
                "parent_process_ids": ",".join(
                    map(str, parents)
                ),
                "child_process_ids": ",".join(
                    map(str, children)
                ),
                "command_lines": " | ".join(
                    command_lines
                ),
                "evidence_ids": ";".join(
                    evidence_ids
                ),
            }
        )

    output = pd.DataFrame(rows)

    output = output.sort_values(
        "anomaly_score",
        ascending=False,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nSaved: {OUTPUT_PATH}"
    )

    print(
        f"Rows: {len(output)}"
    )

    print("\nEvidence-linked anomalies:")

    print(
        output[
            [
                "process_id",
                "process_name",
                "anomaly_score",
                "module_count",
                "memory_region_count",
                "parent_count",
                "child_count",
                "evidence_ids",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()