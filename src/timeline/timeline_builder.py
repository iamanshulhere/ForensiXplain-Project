from pathlib import Path
import pandas as pd


INPUT_FILE = Path("data/normalized/M57-Jean/events.csv")
OUTPUT_DIR = Path("data/normalized/M57-Jean")


def load_events():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Events file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce"
    )

    return df


def build_timeline(df):
    """
    Build a chronological timeline using only
    events with observed timestamps.

    Events without timestamps are preserved separately
    and are NOT assigned artificial timestamps.
    """

    observed = df[
        df["timestamp"].notna()
    ].copy()

    unknown = df[
        df["timestamp"].isna()
    ].copy()

    # Sort chronologically.
    observed = observed.sort_values(
        by=["timestamp", "event_id"]
    ).reset_index(drop=True)

    # Assign timeline sequence numbers.
    observed.insert(
        0,
        "timeline_sequence",
        range(1, len(observed) + 1)
    )

    return observed, unknown


def save_outputs(observed, unknown):

    timeline_path = OUTPUT_DIR / "timeline.csv"
    unknown_path = OUTPUT_DIR / "untimestamped_events.csv"

    observed.to_csv(
        timeline_path,
        index=False
    )

    unknown.to_csv(
        unknown_path,
        index=False
    )

    print(f"\nTimeline saved: {timeline_path}")
    print(f"Untimestamped events saved: {unknown_path}")


def print_summary(observed, unknown):

    print("\n=== ForensiXplain Timeline Reconstruction ===")

    print(f"Total normalized events: {len(observed) + len(unknown)}")
    print(f"Timestamped events:       {len(observed)}")
    print(f"Untimestamped events:     {len(unknown)}")

    if not observed.empty:

        print(
            "\nTimeline start:",
            observed["timestamp"].min()
        )

        print(
            "Timeline end:  ",
            observed["timestamp"].max()
        )

        print("\nEvents by artifact:")
        print(
            observed["artifact_type"]
            .value_counts()
        )

        print("\nFirst 10 timeline events:")

        columns = [
            "timeline_sequence",
            "timestamp",
            "artifact_type",
            "event_type",
            "action",
            "process",
            "process_id",
            "parent_process_id"
        ]

        print(
            observed[columns]
            .head(10)
            .to_string(index=False)
        )


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = load_events()

    observed, unknown = build_timeline(df)

    save_outputs(
        observed,
        unknown
    )

    print_summary(
        observed,
        unknown
    )


if __name__ == "__main__":
    main()