"""
ForensiXplain - Graph-Only Evidence Attribution

Purpose:
    Connect graph-only anomaly explanations to the underlying
    forensic evidence and process relationships.

Inputs:
    data/features/M57-Jean/graph_features.csv
    data/normalized/M57-Jean/events.csv
    data/normalized/M57-Jean/logical_timeline.csv
    results/M57-Jean/graph_only_shap_explanations.csv
    results/M57-Jean/graph_only_anomalies.csv

Output:
    results/M57-Jean/graph_only_evidence_attribution.csv

Important:
    This module uses only graph-derived anomaly features.

    It does NOT decide whether a process is malicious.

    It connects:
        graph-only anomaly score
            ->
        graph-only SHAP explanation
            ->
        process
            ->
        graph relationships
            ->
        forensic evidence
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

CASE_ID = "M57-Jean"


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GRAPH_FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / CASE_ID
    / "graph_features.csv"
)

RAW_EVENT_FILE = (
    PROJECT_ROOT
    / "data"
    / "normalized"
    / CASE_ID
    / "events.csv"
)

LOGICAL_TIMELINE_FILE = (
    PROJECT_ROOT
    / "data"
    / "normalized"
    / CASE_ID
    / "logical_timeline.csv"
)

SHAP_FILE = (
    PROJECT_ROOT
    / "results"
    / CASE_ID
    / "graph_only_shap_explanations.csv"
)

ANOMALY_FILE = (
    PROJECT_ROOT
    / "results"
    / CASE_ID
    / "graph_only_anomalies.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / CASE_ID
)

OUTPUT_FILE = (
    RESULTS_DIR
    / "graph_only_evidence_attribution.csv"
)


# ============================================================
# Graph-only feature pairs
# ============================================================

GRAPH_FEATURE_PAIRS = [
    (
        "parent_count",
        "shap_parent_count",
    ),
    (
        "child_count",
        "shap_child_count",
    ),
    (
        "graph_degree",
        "shap_graph_degree",
    ),
    (
        "in_degree",
        "shap_in_degree",
    ),
    (
        "out_degree",
        "shap_out_degree",
    ),
    (
        "command_line_count",
        "shap_command_line_count",
    ),
    (
        "module_count",
        "shap_module_count",
    ),
    (
        "memory_region_count",
        "shap_memory_region_count",
    ),
    (
        "relationship_type_count",
        "shap_relationship_type_count",
    ),
]


# ============================================================
# Utility functions
# ============================================================

def normalize_pid(value):
    """
    Convert process identifiers into integers where possible.
    """
    if pd.isna(value):
        return None

    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def safe_float(value):
    """
    Convert a value safely to float.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def unique_join(values):
    """
    Join unique non-empty values using semicolon separators.
    """
    cleaned = []

    for value in values:

        if pd.isna(value):
            continue

        value = str(value).strip()

        if not value:
            continue

        if value not in cleaned:
            cleaned.append(value)

    return ";".join(cleaned)


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=== ForensiXplain Graph-Only Evidence Attribution ==="
    )

    # --------------------------------------------------------
    # Validate files
    # --------------------------------------------------------

    required_files = [
        GRAPH_FEATURE_FILE,
        RAW_EVENT_FILE,
        LOGICAL_TIMELINE_FILE,
        SHAP_FILE,
        ANOMALY_FILE,
    ]

    for file_path in required_files:

        if not file_path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n{file_path}"
            )

    # --------------------------------------------------------
    # Load files
    # --------------------------------------------------------

    graph_df = pd.read_csv(
        GRAPH_FEATURE_FILE
    )

    raw_events_df = pd.read_csv(
        RAW_EVENT_FILE
    )

    logical_df = pd.read_csv(
        LOGICAL_TIMELINE_FILE
    )

    shap_df = pd.read_csv(
        SHAP_FILE
    )

    anomaly_df = pd.read_csv(
        ANOMALY_FILE
    )

    print(
        f"Graph feature rows: {len(graph_df)}"
    )

    print(
        f"Raw forensic events: {len(raw_events_df)}"
    )

    print(
        f"Logical timeline events: {len(logical_df)}"
    )

    print(
        f"Graph-only SHAP rows: {len(shap_df)}"
    )

    print(
        f"Graph-only anomaly rows: {len(anomaly_df)}"
    )

    # --------------------------------------------------------
    # Normalize process IDs
    # --------------------------------------------------------

    for df in [
        graph_df,
        raw_events_df,
        logical_df,
        shap_df,
        anomaly_df,
    ]:

        if "process_id" in df.columns:

            df["process_id_normalized"] = (
                df["process_id"]
                .apply(normalize_pid)
            )

    if "parent_process_id" in raw_events_df.columns:

        raw_events_df[
            "parent_process_id_normalized"
        ] = (
            raw_events_df["parent_process_id"]
            .apply(normalize_pid)
        )

    else:

        raw_events_df[
            "parent_process_id_normalized"
        ] = None

    # --------------------------------------------------------
    # Validate anomaly / SHAP relationship
    # --------------------------------------------------------

    required_anomaly_columns = [
        "logical_event_id",
        "graph_only_anomaly_score",
        "graph_only_predicted_anomaly",
        "graph_only_anomaly_rank",
    ]

    missing_anomaly_columns = [
        column
        for column in required_anomaly_columns
        if column not in anomaly_df.columns
    ]

    if missing_anomaly_columns:

        raise ValueError(
            "Missing required graph-only anomaly columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing_anomaly_columns
            )
        )

    shap_check = shap_df[
        ["logical_event_id"]
    ].merge(
        anomaly_df[
            required_anomaly_columns
        ],
        on="logical_event_id",
        how="left",
        validate="one_to_one",
    )

    if shap_check[
        "graph_only_anomaly_score"
    ].isna().any():

        raise ValueError(
            "Some graph-only SHAP events do not have "
            "matching graph-only anomaly records."
        )

    # --------------------------------------------------------
    # Select anomalous events only
    # --------------------------------------------------------

    anomalous_shap = shap_df[
        shap_df[
            "graph_only_predicted_anomaly"
        ] == True
    ].copy()

    print(
        f"Graph-only anomalies to attribute: "
        f"{len(anomalous_shap)}"
    )

    # ========================================================
    # Build evidence attribution records
    # ========================================================

    records = []

    for _, anomaly in anomalous_shap.iterrows():

        pid = normalize_pid(
            anomaly.get("process_id")
        )

        process_name = (
            str(anomaly["process"])
            if (
                "process" in anomaly
                and not pd.isna(anomaly["process"])
            )
            else "unknown"
        )

        logical_event_id = (
            anomaly["logical_event_id"]
        )

        # ----------------------------------------------------
        # Graph-only SHAP contributors
        # ----------------------------------------------------

        contributions = []

        for feature, shap_column in GRAPH_FEATURE_PAIRS:

            if shap_column not in anomaly:

                continue

            shap_value = safe_float(
                anomaly[shap_column]
            )

            if np.isnan(shap_value):

                continue

            value_column = (
                f"value_{feature}"
            )

            if value_column in anomaly:

                feature_value = safe_float(
                    anomaly[value_column]
                )

            else:

                feature_value = np.nan

            contributions.append(
                {
                    "feature": feature,
                    "value": feature_value,
                    "shap": shap_value,
                    "abs_shap": abs(shap_value),
                }
            )

        contributions.sort(
            key=lambda item: item["abs_shap"],
            reverse=True
        )

        top_contributions = contributions[:3]

        top_features = unique_join(
            [
                item["feature"]
                for item in top_contributions
            ]
        )

        top_shap_values = unique_join(
            [
                f"{item['shap']:+.6f}"
                for item in top_contributions
            ]
        )

        top_feature_values = unique_join(
            [
                (
                    f"{item['feature']}="
                    f"{item['value']:.6f}"
                )
                for item in top_contributions
                if not np.isnan(item["value"])
            ]
        )

        # ----------------------------------------------------
        # Graph feature record
        # ----------------------------------------------------

        graph_row = graph_df[
            graph_df[
                "process_id_normalized"
            ] == pid
        ]

        if len(graph_row) > 1:

            graph_event_match = graph_row[
                graph_row[
                    "logical_event_id"
                ] == logical_event_id
            ]

            if len(graph_event_match) > 0:

                graph_row = graph_event_match

        if len(graph_row) > 0:

            graph_row = graph_row.iloc[0]

        else:

            graph_row = None

        def graph_value(column):

            if graph_row is None:
                return np.nan

            if column not in graph_row:
                return np.nan

            return safe_float(
                graph_row[column]
            )

        parent_count = graph_value(
            "parent_count"
        )

        child_count = graph_value(
            "child_count"
        )

        graph_degree = graph_value(
            "graph_degree"
        )

        in_degree = graph_value(
            "in_degree"
        )

        out_degree = graph_value(
            "out_degree"
        )

        command_line_count = graph_value(
            "command_line_count"
        )

        module_count = graph_value(
            "module_count"
        )

        memory_region_count = graph_value(
            "memory_region_count"
        )

        relationship_type_count = graph_value(
            "relationship_type_count"
        )

        # ----------------------------------------------------
        # Logical timeline
        # ----------------------------------------------------

        logical_match = logical_df[
            logical_df[
                "logical_event_id"
            ] == logical_event_id
        ]

        if len(logical_match) > 0:

            logical_row = logical_match.iloc[0]

        else:

            logical_row = None

        if logical_row is not None:

            timestamp = logical_row.get(
                "timestamp",
                ""
            )

            previous_process_id = normalize_pid(
                logical_row.get(
                    "previous_process_id",
                    None
                )
            )

            previous_process = (
                str(
                    logical_row.get(
                        "previous_process",
                        ""
                    )
                )
                if not pd.isna(
                    logical_row.get(
                        "previous_process",
                        ""
                    )
                )
                else ""
            )

            parent_process_ids = (
                str(
                    logical_row.get(
                        "parent_process_ids",
                        ""
                    )
                )
                if not pd.isna(
                    logical_row.get(
                        "parent_process_ids",
                        ""
                    )
                )
                else ""
            )

            timeline_evidence_ids = (
                str(
                    logical_row.get(
                        "evidence_ids",
                        ""
                    )
                )
                if not pd.isna(
                    logical_row.get(
                        "evidence_ids",
                        ""
                    )
                )
                else ""
            )

        else:

            timestamp = ""
            previous_process_id = None
            previous_process = ""
            parent_process_ids = ""
            timeline_evidence_ids = ""

        # ----------------------------------------------------
        # Parent relationships
        # ----------------------------------------------------

        parent_relationships = raw_events_df[
            (
                raw_events_df[
                    "process_id_normalized"
                ] == pid
            )
            &
            (
                raw_events_df[
                    "relationship"
                ]
                .astype(str)
                .str.lower()
                == "parent_of"
            )
        ]

        parent_ids = []

        if "parent_process_id_normalized" in raw_events_df:

            for value in parent_relationships[
                "parent_process_id_normalized"
            ].dropna().unique():

                parent_ids.append(
                    int(value)
                )

        # ----------------------------------------------------
        # Child relationships
        # ----------------------------------------------------

        child_relationships = raw_events_df[
            (
                raw_events_df[
                    "parent_process_id_normalized"
                ] == pid
            )
            &
            (
                raw_events_df[
                    "relationship"
                ]
                .astype(str)
                .str.lower()
                == "parent_of"
            )
        ]

        child_ids = []

        for value in child_relationships[
            "process_id_normalized"
        ].dropna().unique():

            child_ids.append(
                int(value)
            )

        # ----------------------------------------------------
        # Process evidence
        # ----------------------------------------------------

        process_events = raw_events_df[
            raw_events_df[
                "process_id_normalized"
            ] == pid
        ].copy()

        raw_event_count = len(
            process_events
        )

        # ----------------------------------------------------
        # Artifact counts
        # ----------------------------------------------------

        artifact_counts = {}

        if "artifact_type" in process_events.columns:

            artifact_counts = (
                process_events[
                    "artifact_type"
                ]
                .fillna("unknown")
                .astype(str)
                .value_counts()
                .to_dict()
            )

        # ----------------------------------------------------
        # Event type counts
        # ----------------------------------------------------

        event_type_counts = {}

        if "event_type" in process_events.columns:

            event_type_counts = (
                process_events[
                    "event_type"
                ]
                .fillna("unknown")
                .astype(str)
                .value_counts()
                .to_dict()
            )

        # ----------------------------------------------------
        # Evidence IDs
        # ----------------------------------------------------

        evidence_ids = []

        if "evidence_id" in process_events.columns:

            evidence_ids = [
                str(value)
                for value in process_events[
                    "evidence_id"
                ].dropna().unique()
            ]

        # ----------------------------------------------------
        # Provenance
        # ----------------------------------------------------

        provenance_values = []

        if "provenance" in process_events.columns:

            provenance_values = [
                str(value)
                for value in process_events[
                    "provenance"
                ].dropna().unique()
            ]

        # ----------------------------------------------------
        # Command lines
        # ----------------------------------------------------

        command_lines = []

        if "command_line" in process_events.columns:

            command_lines = [
                str(value)
                for value in process_events[
                    "command_line"
                ].dropna().unique()
                if str(value).strip()
                and str(value).strip() != "-"
            ]

        # ----------------------------------------------------
        # Evidence by artifact
        # ----------------------------------------------------

        evidence_by_artifact = []

        if (
            "artifact_type" in process_events.columns
            and "evidence_id" in process_events.columns
        ):

            grouped = (
                process_events[
                    [
                        "artifact_type",
                        "evidence_id",
                    ]
                ]
                .dropna()
                .groupby(
                    "artifact_type"
                )["evidence_id"]
                .apply(
                    lambda values: unique_join(values)
                )
            )

            for artifact_type, ids in grouped.items():

                evidence_by_artifact.append(
                    f"{artifact_type}: {ids}"
                )

        # ----------------------------------------------------
        # Create record
        # ----------------------------------------------------

        records.append(
            {
                "case_id":
                    anomaly.get(
                        "case_id",
                        CASE_ID
                    ),

                "logical_event_id":
                    logical_event_id,

                "temporal_sequence":
                    anomaly.get(
                        "temporal_sequence",
                        np.nan
                    ),

                "timestamp":
                    timestamp,

                "process_id":
                    pid,

                "process":
                    process_name,

                "graph_only_anomaly_score":
                    safe_float(
                        anomaly[
                            "graph_only_anomaly_score"
                        ]
                    ),

                "graph_only_anomaly_rank":
                    anomaly.get(
                        "graph_only_anomaly_rank",
                        np.nan
                    ),

                "top_graph_shap_features":
                    top_features,

                "top_graph_shap_values":
                    top_shap_values,

                "top_graph_feature_values":
                    top_feature_values,

                "parent_count":
                    parent_count,

                "child_count":
                    child_count,

                "graph_degree":
                    graph_degree,

                "in_degree":
                    in_degree,

                "out_degree":
                    out_degree,

                "command_line_count":
                    command_line_count,

                "module_count":
                    module_count,

                "memory_region_count":
                    memory_region_count,

                "relationship_type_count":
                    relationship_type_count,

                "parent_process_ids":
                    unique_join(
                        parent_ids
                    ),

                "child_process_ids":
                    unique_join(
                        child_ids
                    ),

                "previous_process_id":
                    previous_process_id,

                "previous_process":
                    previous_process,

                "raw_event_count":
                    raw_event_count,

                "artifact_type_counts":
                    "; ".join(
                        f"{key}={value}"
                        for key, value
                        in artifact_counts.items()
                    ),

                "event_type_counts":
                    "; ".join(
                        f"{key}={value}"
                        for key, value
                        in event_type_counts.items()
                    ),

                "evidence_by_artifact":
                    "; ".join(
                        evidence_by_artifact
                    ),

                "evidence_ids":
                    unique_join(
                        evidence_ids
                    ),

                "provenance":
                    unique_join(
                        provenance_values
                    ),

                "command_lines":
                    " || ".join(
                        command_lines
                    ),

                "timeline_evidence_ids":
                    timeline_evidence_ids,

                "timeline_parent_process_ids":
                    parent_process_ids,

                "assessment":
                    (
                        "Graph-only anomaly candidate "
                        "with SHAP-based graph feature "
                        "attribution linked to underlying "
                        "forensic evidence."
                    ),

                "limitation":
                    (
                        "Anomaly score and SHAP attribution "
                        "do not establish malicious activity. "
                        "Investigator review of the underlying "
                        "forensic evidence is required."
                    ),
            }
        )

    # ========================================================
    # Build output
    # ========================================================

    result_df = pd.DataFrame(
        records
    )

    result_df = result_df.sort_values(
        by="graph_only_anomaly_rank",
        ascending=True,
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    # ========================================================
    # Summary
    # ========================================================

    print("")
    print(
        "=== Graph-Only Evidence Attribution Complete ==="
    )

    print(
        f"Graph-only anomalies attributed: "
        f"{len(result_df)}"
    )

    print(
        f"Output columns: "
        f"{len(result_df.columns)}"
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )

    print("")
    print(
        "Top graph-only evidence summaries:"
    )

    for _, row in result_df.iterrows():

        print("")

        print(
            f"Rank {int(row['graph_only_anomaly_rank'])}: "
            f"PID {int(row['process_id'])} "
            f"{row['process']} "
            f"score="
            f"{row['graph_only_anomaly_score']:.6f}"
        )

        print(
            f"  Top graph SHAP: "
            f"{row['top_graph_shap_features']}"
        )

        print(
            f"  Parent processes: "
            f"{row['parent_process_ids'] or 'none'}"
        )

        print(
            f"  Child processes: "
            f"{row['child_process_ids'] or 'none'}"
        )

        print(
            f"  Raw evidence events: "
            f"{int(row['raw_event_count'])}"
        )

        print(
            f"  Evidence by artifact: "
            f"{row['artifact_type_counts'] or 'none'}"
        )


if __name__ == "__main__":
    main()