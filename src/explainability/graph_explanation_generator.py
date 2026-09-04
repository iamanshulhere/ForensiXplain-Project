"""
ForensiXplain - Graph Investigator Explanation Generator

Purpose:
    Convert graph evidence attribution records into
    investigator-readable explanations.

Input:
    results/M57-Jean/graph_evidence_attribution.csv

Outputs:
    results/M57-Jean/graph_investigator_explanations.csv
    results/M57-Jean/graph_investigator_report.txt

Important:
    This module explains anomaly candidates.
    It does NOT determine whether a process is malicious.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "M57-Jean"
    / "graph_evidence_attribution.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "M57-Jean"
)

CSV_OUTPUT = (
    RESULTS_DIR
    / "graph_investigator_explanations.csv"
)

REPORT_OUTPUT = (
    RESULTS_DIR
    / "graph_investigator_report.txt"
)


# ============================================================
# Feature descriptions
# ============================================================

FEATURE_DESCRIPTIONS = {

    "gap_log_seconds":
        "log-transformed time gap from the previous logical event",

    "local_density_10s":
        "number of nearby events within the 10-second window",

    "local_density_30s":
        "number of nearby events within the 30-second window",

    "local_density_60s":
        "number of nearby events within the 60-second window",

    "process_changed":
        "whether the process differed from the previous logical event",

    "parent_count":
        "number of parent relationships represented in the graph",

    "child_count":
        "number of child-process relationships represented in the graph",

    "graph_degree":
        "total graph connectivity of the process node",

    "in_degree":
        "number of incoming graph relationships",

    "out_degree":
        "number of outgoing graph relationships",

    "command_line_count":
        "number of command-line observations linked to the process",

    "module_count":
        "number of loaded-module observations linked to the process",

    "memory_region_count":
        "number of memory-region observations linked to the process",

    "relationship_type_count":
        "number of distinct relationship observations linked to the process",
}


# ============================================================
# Helper functions
# ============================================================

def format_number(value, decimals=3):

    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "unknown"


def clean_text(value):

    if pd.isna(value):
        return ""

    return str(value).strip()


def format_list(value, empty_text="none observed"):

    value = clean_text(value)

    if not value:
        return empty_text

    return value


def feature_explanation(
    feature,
    value,
    shap_value,
):
    """
    Create a careful explanation of a SHAP contributor.

    Important:
        SHAP direction is described as contribution to the
        model output, not as proof that a feature caused
        malicious activity.
    """

    description = FEATURE_DESCRIPTIONS.get(
        feature,
        feature.replace("_", " ")
    )

    try:
        numeric_value = float(value)
        numeric_text = f"{numeric_value:.3f}"
    except (TypeError, ValueError):
        numeric_text = "unknown"

    try:
        numeric_shap = float(shap_value)
    except (TypeError, ValueError):
        numeric_shap = 0.0

    if numeric_shap >= 0:

        direction = (
            "positive contribution to the "
            "tree-model output"
        )

    else:

        direction = (
            "negative contribution to the "
            "tree-model output"
        )

    return (
        f"{feature}={numeric_text} "
        f"({direction}, SHAP={numeric_shap:+.6f}; "
        f"{description})"
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=== ForensiXplain Graph Investigator "
        "Explanation Generator ==="
    )

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    # --------------------------------------------------------
    # Load attribution data
    # --------------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Evidence attribution rows: {len(df)}"
    )

    if len(df) == 0:

        raise ValueError(
            "No graph evidence attribution records found."
        )

    # --------------------------------------------------------
    # Ensure anomaly ranking exists
    # --------------------------------------------------------

    if "graph_anomaly_rank" in df.columns:

        df = df.sort_values(
            by="graph_anomaly_rank"
        ).reset_index(
            drop=True
        )

    # --------------------------------------------------------
    # Build explanation records
    # --------------------------------------------------------

    explanation_records = []

    for _, row in df.iterrows():

        process_id = clean_text(
            row.get("process_id", "")
        )

        process_name = clean_text(
            row.get("process", "unknown")
        )

        score = format_number(
            row.get(
                "graph_anomaly_score",
                None
            ),
            6
        )

        rank = clean_text(
            row.get(
                "graph_anomaly_rank",
                ""
            )
        )

        timestamp = clean_text(
            row.get(
                "timestamp",
                ""
            )
        )

        top_features = format_list(
            row.get(
                "top_shap_features",
                ""
            )
        )

        top_shap_values = format_list(
            row.get(
                "top_shap_values",
                ""
            )
        )

        top_feature_values = format_list(
            row.get(
                "top_feature_values",
                ""
            )
        )

        parent_processes = format_list(
            row.get(
                "parent_process_ids",
                ""
            )
        )

        child_processes = format_list(
            row.get(
                "child_process_ids",
                ""
            )
        )

        previous_process_id = clean_text(
            row.get(
                "previous_process_id",
                ""
            )
        )

        previous_process = clean_text(
            row.get(
                "previous_process",
                ""
            )
        )

        time_gap = row.get(
            "time_since_previous_event_seconds",
            None
        )

        raw_event_count = clean_text(
            row.get(
                "raw_event_count",
                ""
            )
        )

        artifact_counts = format_list(
            row.get(
                "artifact_type_counts",
                ""
            )
        )

        event_type_counts = format_list(
            row.get(
                "event_type_counts",
                ""
            )
        )

        evidence_by_artifact = format_list(
            row.get(
                "evidence_by_artifact",
                ""
            )
        )

        evidence_ids = format_list(
            row.get(
                "evidence_ids",
                ""
            )
        )

        timeline_evidence_ids = format_list(
            row.get(
                "timeline_evidence_ids",
                ""
            )
        )

        provenance = format_list(
            row.get(
                "provenance",
                ""
            )
        )

        command_lines = format_list(
            row.get(
                "command_lines",
                ""
            )
        )

        # ----------------------------------------------------
        # Build feature-level explanation text
        # ----------------------------------------------------

        feature_lines = []

        feature_pairs = [
            (
                "gap_log_seconds",
                "shap_gap_log_seconds",
            ),
            (
                "local_density_10s",
                "shap_local_density_10s",
            ),
            (
                "local_density_30s",
                "shap_local_density_30s",
            ),
            (
                "local_density_60s",
                "shap_local_density_60s",
            ),
            (
                "process_changed",
                "shap_process_changed",
            ),
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

        contributions = []

        for feature, shap_column in feature_pairs:

            value_column = (
                f"value_{feature}"
            )

            if (
                value_column not in row
                or shap_column not in row
            ):
                continue

            try:

                value = float(
                    row[value_column]
                )

                shap_value = float(
                    row[shap_column]
                )

            except (
                TypeError,
                ValueError
            ):

                continue

            contributions.append(
                (
                    feature,
                    value,
                    shap_value,
                )
            )

        contributions.sort(
            key=lambda item: abs(item[2]),
            reverse=True
        )

        for feature, value, shap_value in (
            contributions[:3]
        ):

            feature_lines.append(
                feature_explanation(
                    feature,
                    value,
                    shap_value
                )
            )

        feature_summary = (
            " | ".join(feature_lines)
            if feature_lines
            else "No SHAP feature attribution available."
        )

        # ----------------------------------------------------
        # Time-gap explanation
        # ----------------------------------------------------

        if pd.isna(time_gap):

            time_gap_text = (
                "No previous logical event available."
            )

        else:

            try:

                gap_seconds = float(
                    time_gap
                )

                time_gap_text = (
                    f"{gap_seconds:.3f} seconds"
                )

            except (
                TypeError,
                ValueError
            ):

                time_gap_text = (
                    "Unknown"
                )

        # ----------------------------------------------------
        # Assessment
        # ----------------------------------------------------

        assessment = (
            "This process is an anomaly candidate under "
            "the combined temporal/graph Isolation Forest "
            "baseline. The SHAP results identify features "
            "that contributed to the model output, while "
            "the evidence fields connect the candidate to "
            "underlying forensic observations."
        )

        limitation = (
            "The anomaly score and SHAP attribution do not "
            "establish malicious activity, compromise, or "
            "intent. Investigator review of the original "
            "forensic evidence is required."
        )

        # ----------------------------------------------------
        # Human-readable explanation
        # ----------------------------------------------------

        explanation_text = (
            f"Process {process_name} (PID {process_id}) "
            f"was ranked #{rank} with a graph-aware anomaly "
            f"score of {score}. The strongest model "
            f"contributors were: {feature_summary}. "
            f"The event occurred at {timestamp}. "
            f"The previous logical process was "
            f"{previous_process or 'unknown'} "
            f"(PID {previous_process_id or 'unknown'}), "
            f"with a time gap of {time_gap_text}. "
            f"The graph records parent process(es) "
            f"{parent_processes} and child process(es) "
            f"{child_processes}. "
            f"The process has {raw_event_count} raw forensic "
            f"event observations. "
            f"Artifact observations were "
            f"{artifact_counts}. "
            f"Evidence identifiers include "
            f"{evidence_ids}. "
            f"Assessment: {assessment} "
            f"Limitation: {limitation}"
        )

        # ----------------------------------------------------
        # Output record
        # ----------------------------------------------------

        explanation_records.append(
            {
                "case_id":
                    clean_text(
                        row.get(
                            "case_id",
                            ""
                        )
                    ),

                "logical_event_id":
                    clean_text(
                        row.get(
                            "logical_event_id",
                            ""
                        )
                    ),

                "graph_anomaly_rank":
                    rank,

                "process_id":
                    process_id,

                "process":
                    process_name,

                "timestamp":
                    timestamp,

                "graph_anomaly_score":
                    score,

                "top_shap_features":
                    top_features,

                "top_shap_values":
                    top_shap_values,

                "top_feature_values":
                    top_feature_values,

                "parent_process_ids":
                    parent_processes,

                "child_process_ids":
                    child_processes,

                "previous_process_id":
                    previous_process_id,

                "previous_process":
                    previous_process,

                "time_since_previous_event_seconds":
                    time_gap,

                "raw_event_count":
                    raw_event_count,

                "artifact_type_counts":
                    artifact_counts,

                "event_type_counts":
                    event_type_counts,

                "evidence_by_artifact":
                    evidence_by_artifact,

                "evidence_ids":
                    evidence_ids,

                "timeline_evidence_ids":
                    timeline_evidence_ids,

                "provenance":
                    provenance,

                "command_lines":
                    command_lines,

                "assessment":
                    assessment,

                "limitation":
                    limitation,

                "investigator_explanation":
                    explanation_text,
            }
        )

    # ========================================================
    # Create DataFrame
    # ========================================================

    explanation_df = pd.DataFrame(
        explanation_records
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    explanation_df.to_csv(
        CSV_OUTPUT,
        index=False,
        encoding="utf-8",
    )

    # ========================================================
    # Build text report
    # ========================================================

    report_lines = []

    report_lines.append(
        "FORENSIXPLAIN"
    )

    report_lines.append(
        "Graph Investigator Explanation Report"
    )

    report_lines.append(
        "=" * 70
    )

    report_lines.append("")

    report_lines.append(
        "Case: M57-Jean"
    )

    report_lines.append(
        f"Graph anomaly candidates: "
        f"{len(explanation_df)}"
    )

    report_lines.append("")

    report_lines.append(
        "IMPORTANT INTERPRETATION NOTE"
    )

    report_lines.append(
        "-" * 70
    )

    report_lines.append(
        "The graph-aware Isolation Forest identifies "
        "statistical anomaly candidates relative to "
        "the feature baseline."
    )

    report_lines.append(
        "SHAP values describe feature contributions to "
        "the tree-model output."
    )

    report_lines.append(
        "Neither anomaly scores nor SHAP values establish "
        "malicious activity."
    )

    report_lines.append(
        "All candidates require investigator review of "
        "the underlying forensic evidence."
    )

    report_lines.append("")

    # ========================================================
    # Candidate sections
    # ========================================================

    for _, row in explanation_df.iterrows():

        rank = row[
            "graph_anomaly_rank"
        ]

        process_id = row[
            "process_id"
        ]

        process = row[
            "process"
        ]

        score = row[
            "graph_anomaly_score"
        ]

        timestamp = row[
            "timestamp"
        ]

        report_lines.append(
            "=" * 70
        )

        report_lines.append(
            f"GRAPH ANOMALY CANDIDATE #{rank}"
        )

        report_lines.append(
            "=" * 70
        )

        report_lines.append("")

        report_lines.append(
            f"Process       : {process}"
        )

        report_lines.append(
            f"PID           : {process_id}"
        )

        report_lines.append(
            f"Timestamp     : {timestamp}"
        )

        report_lines.append(
            f"Anomaly score : {score}"
        )

        report_lines.append("")

        report_lines.append(
            "TOP SHAP CONTRIBUTORS"
        )

        report_lines.append(
            "-" * 40
        )

        report_lines.append(
            row[
                "top_shap_features"
            ]
        )

        report_lines.append(
            f"SHAP values: "
            f"{row['top_shap_values']}"
        )

        report_lines.append(
            f"Feature values: "
            f"{row['top_feature_values']}"
        )

        report_lines.append("")

        report_lines.append(
            "TEMPORAL CONTEXT"
        )

        report_lines.append(
            "-" * 40
        )

        report_lines.append(
            f"Previous process : "
            f"{row['previous_process'] or 'unknown'}"
        )

        report_lines.append(
            f"Previous PID     : "
            f"{row['previous_process_id'] or 'unknown'}"
        )

        report_lines.append(
            f"Time gap         : "
            f"{row['time_since_previous_event_seconds']}"
            f" seconds"
            if not pd.isna(
                row[
                    "time_since_previous_event_seconds"
                ]
            )
            else
            "Time gap         : unknown"
        )

        report_lines.append("")

        report_lines.append(
            "GRAPH CONTEXT"
        )

        report_lines.append(
            "-" * 40
        )

        report_lines.append(
            f"Parent process(es): "
            f"{row['parent_process_ids'] or 'none observed'}"
        )

        report_lines.append(
            f"Child process(es) : "
            f"{row['child_process_ids'] or 'none observed'}"
        )

        report_lines.append("")

        report_lines.append(
            "FORENSIC EVIDENCE"
        )

        report_lines.append(
            "-" * 40
        )

        report_lines.append(
            f"Raw event observations: "
            f"{row['raw_event_count']}"
        )

        report_lines.append(
            f"Artifact observations: "
            f"{row['artifact_type_counts']}"
        )

        report_lines.append(
            f"Event types: "
            f"{row['event_type_counts']}"
        )

        report_lines.append(
            f"Evidence by artifact: "
            f"{row['evidence_by_artifact']}"
        )

        report_lines.append("")

        report_lines.append(
            f"Evidence IDs: "
            f"{row['evidence_ids']}"
        )

        report_lines.append(
            f"Timeline evidence IDs: "
            f"{row['timeline_evidence_ids']}"
        )

        report_lines.append("")

        if row["command_lines"]:

            report_lines.append(
                "COMMAND-LINE OBSERVATIONS"
            )

            report_lines.append(
                "-" * 40
            )

            report_lines.append(
                row["command_lines"]
            )

            report_lines.append("")

        report_lines.append(
            "ASSESSMENT"
        )

        report_lines.append(
            "-" * 40
        )

        report_lines.append(
            row["assessment"]
        )

        report_lines.append("")

        report_lines.append(
            "LIMITATION"
        )

        report_lines.append(
            "-" * 40
        )

        report_lines.append(
            row["limitation"]
        )

        report_lines.append("")

    # ========================================================
    # Save text report
    # ========================================================

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    # ========================================================
    # Final summary
    # ========================================================

    print("")
    print(
        "=== Graph Investigator Explanation "
        "Complete ==="
    )

    print(
        f"Explanation records: "
        f"{len(explanation_df)}"
    )

    print(
        f"CSV output: {CSV_OUTPUT}"
    )

    print(
        f"Text report: {REPORT_OUTPUT}"
    )


if __name__ == "__main__":
    main()