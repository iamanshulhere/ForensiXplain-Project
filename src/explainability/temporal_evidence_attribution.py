from pathlib import Path
import pandas as pd


# ============================================================
# ForensiXplain
# Temporal Evidence Attribution
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results" / "M57-Jean"
NORMALIZED_DIR = PROJECT_ROOT / "data" / "normalized" / "M57-Jean"

SHAP_FILE = RESULTS_DIR / "temporal_shap_explanations.csv"
TIMELINE_FILE = NORMALIZED_DIR / "logical_timeline.csv"
FEATURES_DIR = PROJECT_ROOT / "data" / "features" / "M57-Jean"

TEMPORAL_FEATURES_FILE = FEATURES_DIR / "temporal_features.csv"
EVENTS_FILE = NORMALIZED_DIR / "events.csv"

OUTPUT_FILE = RESULTS_DIR / "temporal_evidence_attribution.csv"


# ============================================================
# Utility functions
# ============================================================

def normalize_pid(value):
    """Convert a PID value to int when possible."""
    if pd.isna(value):
        return None

    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


def split_values(value):
    """
    Safely split semicolon-separated values.

    Example:
        A;B;C -> ['A', 'B', 'C']
    """
    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    return [
        item.strip()
        for item in text.split(";")
        if item.strip()
    ]


def clean_values(values):
    """Remove empty values and duplicates."""
    cleaned = set()

    for value in values:

        if value is None:
            continue

        text = str(value).strip()

        if text:
            cleaned.add(text)

    return sorted(cleaned)


def safe_int(value):
    """Convert numeric value to int for display."""
    if pd.isna(value):
        return None

    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


# ============================================================
# Main
# ============================================================

def main():

    print("=== ForensiXplain Temporal Evidence Attribution ===")

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    required_files = [
        SHAP_FILE,
        TIMELINE_FILE,
        TEMPORAL_FEATURES_FILE,
        EVENTS_FILE,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    # --------------------------------------------------------
    # Load files
    # --------------------------------------------------------

    shap_df = pd.read_csv(SHAP_FILE)

    timeline_df = pd.read_csv(
        TIMELINE_FILE
    )

    temporal_df = pd.read_csv(
        TEMPORAL_FEATURES_FILE
    )

    events_df = pd.read_csv(
        EVENTS_FILE
    )

    print(f"SHAP rows: {len(shap_df)}")
    print(f"Logical timeline rows: {len(timeline_df)}")
    print(f"Temporal feature rows: {len(temporal_df)}")
    print(f"Raw event rows: {len(events_df)}")

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    shap_required = [
        "logical_event_id",
        "temporal_anomaly_rank",
        "temporal_anomaly_score",
        "temporal_predicted_anomaly",
        "temporal_sequence",
        "timestamp",
        "process_id",
        "process",
    ]

    timeline_required = [
        "logical_event_id",
        "parent_process_ids",
        "evidence_ids",
        "source_observation_count",
        "provenance",
    ]

    temporal_required = [
        "logical_event_id",
        "previous_timestamp",
        "previous_process_id",
        "previous_process",
        "time_since_previous_event_seconds",
        "events_prev_10s",
        "events_next_10s",
        "local_density_10s",
        "events_prev_30s",
        "events_next_30s",
        "local_density_30s",
        "events_prev_60s",
        "events_next_60s",
        "local_density_60s",
        "process_changed",
        "process_transition",
    ]

    events_required = [
        "process_id",
        "parent_process_id",
        "artifact_type",
        "source",
        "evidence_id",
    ]

    for column in shap_required:

        if column not in shap_df.columns:

            raise ValueError(
                f"Missing SHAP column: {column}"
            )

    for column in timeline_required:

        if column not in timeline_df.columns:

            raise ValueError(
                f"Missing logical timeline column: {column}"
            )

    for column in temporal_required:

        if column not in temporal_df.columns:

            raise ValueError(
                f"Missing temporal feature column: {column}"
            )

    for column in events_required:

        if column not in events_df.columns:

            raise ValueError(
                f"Missing events column: {column}"
            )

    # --------------------------------------------------------
    # Select anomalies
    # --------------------------------------------------------

    anomalies = shap_df[
        shap_df["temporal_predicted_anomaly"].astype(bool)
    ].copy()

    print(
        f"Temporal anomalies: {len(anomalies)}"
    )

    if anomalies.empty:

        print(
            "\nNo temporal anomalies found."
        )

        return

    # --------------------------------------------------------
    # Model information
    # --------------------------------------------------------

    model_columns = [
        "logical_event_id",
        "temporal_anomaly_rank",
        "temporal_anomaly_score",
        "temporal_predicted_anomaly",
        "temporal_sequence",
        "timestamp",
        "process_id",
        "process",

        "shap_gap_log_seconds",
        "shap_local_density_10s",
        "shap_local_density_30s",
        "shap_local_density_60s",
        "shap_process_changed",
    ]

    result = anomalies[
        model_columns
    ].copy()

    result["process_id"] = (
        result["process_id"]
        .apply(normalize_pid)
    )

    # --------------------------------------------------------
    # Add logical timeline context
    # --------------------------------------------------------

    timeline_columns = [
        "logical_event_id",
        "parent_process_ids",
        "evidence_ids",
        "source_observation_count",
        "provenance",
    ]

    timeline_context = timeline_df[
        timeline_columns
    ].copy()

    # Ensure unique logical event IDs
    if timeline_context[
        "logical_event_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate logical_event_id values "
            "found in logical_timeline.csv"
        )

    result = result.merge(
        timeline_context,
        on="logical_event_id",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Add temporal feature context
    # --------------------------------------------------------

    temporal_columns = [
        "logical_event_id",
        "previous_timestamp",
        "previous_process_id",
        "previous_process",
        "time_since_previous_event_seconds",
        "events_prev_10s",
        "events_next_10s",
        "local_density_10s",
        "events_prev_30s",
        "events_next_30s",
        "local_density_30s",
        "events_prev_60s",
        "events_next_60s",
        "local_density_60s",
        "process_changed",
        "process_transition",
    ]

    temporal_context = temporal_df[
        temporal_columns
    ].copy()

    temporal_context[
        "previous_process_id"
    ] = (
        temporal_context[
            "previous_process_id"
        ]
        .apply(normalize_pid)
    )

    if temporal_context[
        "logical_event_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate logical_event_id values "
            "found in temporal_features.csv"
        )

    result = result.merge(
        temporal_context,
        on="logical_event_id",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Prepare raw event PID columns
    # --------------------------------------------------------

    events_df["_process_pid"] = (
        events_df["process_id"]
        .apply(normalize_pid)
    )

    events_df["_parent_pid"] = (
        events_df["parent_process_id"]
        .apply(normalize_pid)
    )

    # --------------------------------------------------------
    # Evidence attribution
    # --------------------------------------------------------

    evidence_rows = []

    for _, row in result.iterrows():

        pid = row["process_id"]

        # ----------------------------------------------------
        # Raw events belonging to this process
        # ----------------------------------------------------

        related_events = events_df[
            events_df["_process_pid"] == pid
        ].copy()

        # ----------------------------------------------------
        # Relationship events involving this process
        # ----------------------------------------------------

        relationship_events = events_df[
            (
                events_df["_process_pid"] == pid
            )
            |
            (
                events_df["_parent_pid"] == pid
            )
        ].copy()

        # ----------------------------------------------------
        # Artifact counts
        # ----------------------------------------------------

        artifact_counts = {}

        if (
            not related_events.empty
            and "artifact_type" in related_events.columns
        ):

            artifact_counts = (
                related_events[
                    "artifact_type"
                ]
                .fillna("")
                .astype(str)
                .value_counts()
                .to_dict()
            )

        # ----------------------------------------------------
        # Source counts
        # ----------------------------------------------------

        source_counts = {}

        if (
            not related_events.empty
            and "source" in related_events.columns
        ):

            source_counts = (
                related_events[
                    "source"
                ]
                .fillna("")
                .astype(str)
                .value_counts()
                .to_dict()
            )

        # ----------------------------------------------------
        # Evidence IDs
        # ----------------------------------------------------

        evidence_ids = set()

        # IDs from logical timeline
        evidence_ids.update(
            split_values(
                row.get(
                    "evidence_ids",
                    ""
                )
            )
        )

        # IDs from raw events
        if (
            not related_events.empty
            and "evidence_id" in related_events.columns
        ):

            for evidence_id in (
                related_events[
                    "evidence_id"
                ]
                .dropna()
                .astype(str)
                .str.strip()
            ):

                if evidence_id:

                    evidence_ids.add(
                        evidence_id
                    )

        evidence_ids = clean_values(
            evidence_ids
        )

        # ----------------------------------------------------
        # Parent processes
        # ----------------------------------------------------

        parent_ids = set()

        # Parent IDs from logical timeline
        parent_ids.update(
            split_values(
                row.get(
                    "parent_process_ids",
                    ""
                )
            )
        )

        # Parent IDs from raw relationship evidence
        for _, rel in relationship_events.iterrows():

            child_pid = normalize_pid(
                rel.get("process_id")
            )

            parent_pid = normalize_pid(
                rel.get("parent_process_id")
            )

            if (
                child_pid == pid
                and parent_pid is not None
            ):

                parent_ids.add(
                    str(parent_pid)
                )

        parent_ids = clean_values(
            parent_ids
        )

        # ----------------------------------------------------
        # Child processes
        # ----------------------------------------------------

        child_ids = set()

        for _, rel in relationship_events.iterrows():

            child_pid = normalize_pid(
                rel.get("process_id")
            )

            parent_pid = normalize_pid(
                rel.get("parent_process_id")
            )

            if (
                parent_pid == pid
                and child_pid is not None
            ):

                child_ids.add(
                    str(child_pid)
                )

        child_ids = clean_values(
            child_ids
        )

        # ----------------------------------------------------
        # Artifact-specific evidence IDs
        # ----------------------------------------------------

        artifact_evidence = {}

        if (
            not related_events.empty
            and "artifact_type" in related_events.columns
            and "evidence_id" in related_events.columns
        ):

            for artifact_type, group in (
                related_events.groupby(
                    "artifact_type"
                )
            ):

                ids = (
                    group["evidence_id"]
                    .dropna()
                    .astype(str)
                    .str.strip()
                )

                ids = sorted(
                    {
                        item
                        for item in ids
                        if item
                    }
                )

                artifact_evidence[
                    str(artifact_type)
                ] = ids

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        evidence_rows.append({

            "raw_event_count":
                int(len(related_events)),

            "relationship_event_count":
                int(len(relationship_events)),

            "pslist_count":
                int(
                    artifact_counts.get(
                        "pslist",
                        0
                    )
                ),

            "pstree_count":
                int(
                    artifact_counts.get(
                        "pstree",
                        0
                    )
                ),

            "cmdline_count":
                int(
                    artifact_counts.get(
                        "cmdline",
                        0
                    )
                ),

            "dlllist_count":
                int(
                    artifact_counts.get(
                        "dlllist",
                        0
                    )
                ),

            "malfind_count":
                int(
                    artifact_counts.get(
                        "malfind",
                        0
                    )
                ),

            "source_counts":
                str(source_counts),

            "parent_process_ids_attributed":
                ";".join(parent_ids),

            "child_process_ids":
                ";".join(child_ids),

            "evidence_id_count":
                int(len(evidence_ids)),

            "evidence_ids_attributed":
                ";".join(evidence_ids),

            "artifact_evidence_ids":
                str(artifact_evidence),
        })

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    evidence_df = pd.DataFrame(
        evidence_rows
    )

    result = pd.concat(
        [
            result.reset_index(drop=True),
            evidence_df.reset_index(drop=True),
        ],
        axis=1,
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    result = result.sort_values(
        "temporal_anomaly_rank"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Validate no duplicate columns
    # --------------------------------------------------------

    duplicate_columns = (
        result.columns[
            result.columns.duplicated()
        ]
        .tolist()
    )

    if duplicate_columns:

        raise ValueError(
            "Duplicate columns detected in "
            f"final result: {duplicate_columns}"
        )

    # --------------------------------------------------------
    # Validate evidence IDs
    # --------------------------------------------------------

    for _, row in result.iterrows():

        evidence_text = str(
            row["evidence_ids_attributed"]
        )

        if evidence_text:

            ids = split_values(
                evidence_text
            )

            for evidence_id in ids:

                if (
                    evidence_id == "-"
                    or len(evidence_id) <= 1
                ):

                    raise ValueError(
                        "Invalid evidence ID detected: "
                        f"{evidence_id!r}"
                    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print(
        "\n=== Temporal Evidence Attribution Complete ==="
    )

    print(
        f"Anomalies attributed: {len(result)}"
    )

    print(
        f"Output columns: {len(result.columns)}"
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 75)

    for _, row in result.iterrows():

        rank = safe_int(
            row["temporal_anomaly_rank"]
        )

        pid = safe_int(
            row["process_id"]
        )

        process = row["process"]

        score = float(
            row["temporal_anomaly_score"]
        )

        print(
            f"Rank {rank}: "
            f"{process} "
            f"(PID {pid})"
        )

        print(
            f"  Timestamp: "
            f"{row['timestamp']}"
        )

        print(
            f"  Anomaly score: "
            f"{score:.6f}"
        )

        previous_pid = safe_int(
            row["previous_process_id"]
        )

        previous_process = row[
            "previous_process"
        ]

        if (
            pd.notna(previous_process)
            and str(previous_process).strip()
        ):

            print(
                f"  Previous process: "
                f"{previous_process} "
                f"(PID {previous_pid})"
            )

        else:

            print(
                "  Previous process: N/A"
            )

        print(
            f"  Time since previous: "
            f"{row['time_since_previous_event_seconds']}"
        )

        print(
            f"  Process transition: "
            f"{row['process_transition']}"
        )

        print(
            "  Parent PID(s): "
            + (
                row["parent_process_ids_attributed"]
                if row["parent_process_ids_attributed"]
                else "None"
            )
        )

        print(
            "  Child PID(s): "
            + (
                row["child_process_ids"]
                if row["child_process_ids"]
                else "None"
            )
        )

        print(
            f"  Raw events: "
            f"{int(row['raw_event_count'])}"
        )

        print(
            f"  Evidence IDs: "
            f"{int(row['evidence_id_count'])}"
        )

        print(
            "  Artifact counts: "
            f"pslist={int(row['pslist_count'])}, "
            f"pstree={int(row['pstree_count'])}, "
            f"cmdline={int(row['cmdline_count'])}, "
            f"dlllist={int(row['dlllist_count'])}, "
            f"malfind={int(row['malfind_count'])}"
        )

        print(
            f"  Evidence IDs: "
            f"{row['evidence_ids_attributed']}"
        )

        print("-" * 75)


if __name__ == "__main__":
    main()