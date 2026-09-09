"""
ForensiXplain - Temporal Investigator Explanation Generator

Purpose:
    Convert temporal anomaly + SHAP + evidence attribution results
    into investigator-readable explanations.

Inputs:
    results/M57-Jean/temporal_evidence_attribution.csv
    results/M57-Jean/temporal_shap_explanations.csv

Outputs:
    results/M57-Jean/temporal_investigator_explanations.csv
    results/M57-Jean/temporal_investigator_report.txt

Important:
    - Anomaly does NOT mean malicious activity.
    - Explanations are grounded in observed forensic evidence.
    - Chronological adjacency is not treated as causality.
    - Full evidence IDs are preserved in the CSV output.
"""

from pathlib import Path
import pandas as pd


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results" / "M57-Jean"

ATTRIBUTION_FILE = RESULTS_DIR / "temporal_evidence_attribution.csv"
SHAP_FILE = RESULTS_DIR / "temporal_shap_explanations.csv"

OUTPUT_CSV = RESULTS_DIR / "temporal_investigator_explanations.csv"
OUTPUT_REPORT = RESULTS_DIR / "temporal_investigator_report.txt"


# ============================================================
# Utility functions
# ============================================================

def safe_float(value, default=0.0):
    """Safely convert a value to float."""
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default=0):
    """Safely convert a value to integer."""
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def clean_text(value, default=""):
    """Safely convert a value to clean text."""
    if pd.isna(value):
        return default
    return str(value).strip()


# ============================================================
# SHAP feature extraction
# ============================================================

def get_top_shap_features(row):
    """
    Return the top three SHAP contributors.

    The actual feature value is read from the direct feature column
    when available, otherwise from the corresponding value_* column.
    """

    feature_pairs = [
        ("gap_log_seconds", "shap_gap_log_seconds"),
        ("local_density_10s", "shap_local_density_10s"),
        ("local_density_30s", "shap_local_density_30s"),
        ("local_density_60s", "shap_local_density_60s"),
        ("process_changed", "shap_process_changed"),
    ]

    contributors = []

    for feature_name, shap_column in feature_pairs:
        if shap_column not in row.index:
            continue

        shap_value = safe_float(row[shap_column])

        if feature_name in row.index:
            feature_value = safe_float(row[feature_name])
        elif f"value_{feature_name}" in row.index:
            feature_value = safe_float(row[f"value_{feature_name}"])
        else:
            feature_value = 0.0

        contributors.append(
            (
                feature_name,
                feature_value,
                shap_value,
            )
        )

    contributors.sort(
        key=lambda item: abs(item[2]),
        reverse=True,
    )

    return contributors[:3]


# ============================================================
# Feature descriptions
# ============================================================

def feature_description(feature, value, shap_value):
    """Convert a SHAP contribution into investigator-readable text."""

    feature_name = feature.replace("shap_", "")

    if shap_value > 0:
        direction = "increased"
    elif shap_value < 0:
        direction = "decreased"
    else:
        direction = "had no measurable effect on"

    if feature_name == "gap_log_seconds":
        return (
            f"gap_log_seconds had value {value:.4f} and "
            f"{direction} the anomaly score "
            f"(SHAP={shap_value:+.6f})."
        )

    if feature_name == "local_density_10s":
        return (
            f"local_density_10s had value {value:.0f} and "
            f"{direction} the anomaly score "
            f"(SHAP={shap_value:+.6f})."
        )

    if feature_name == "local_density_30s":
        return (
            f"local_density_30s had value {value:.0f} and "
            f"{direction} the anomaly score "
            f"(SHAP={shap_value:+.6f})."
        )

    if feature_name == "local_density_60s":
        return (
            f"local_density_60s had value {value:.0f} and "
            f"{direction} the anomaly score "
            f"(SHAP={shap_value:+.6f})."
        )

    if feature_name == "process_changed":
        return (
            f"process_changed had value {value:.0f} and "
            f"{direction} the anomaly score "
            f"(SHAP={shap_value:+.6f})."
        )

    return (
        f"{feature_name} had value {value:.4f} and "
        f"{direction} the anomaly score "
        f"(SHAP={shap_value:+.6f})."
    )


# ============================================================
# Evidence helpers
# ============================================================

def build_evidence_summary(row):
    """Build a concise evidence summary from the attribution record."""

    evidence_parts = []

    artifact_columns = [
        ("pslist_count", "PSList"),
        ("pstree_count", "PSTree"),
        ("cmdline_count", "Command line"),
        ("dlllist_count", "DLLList"),
        ("malfind_count", "Malfind"),
    ]

    for column, label in artifact_columns:
        if column not in row.index:
            continue

        count = safe_int(row[column])

        if count > 0:
            evidence_parts.append(f"{label}={count}")

    if not evidence_parts:
        return "No artifact-specific observations were recorded."

    return ", ".join(evidence_parts)


def build_evidence_ids(row):
    """Return complete evidence IDs from the attribution record."""

    if "evidence_ids" not in row.index:
        return ""

    return clean_text(row["evidence_ids"])


# ============================================================
# Process relationship summary
# ============================================================

def build_relationship_summary(row):
    """
    Build parent/child relationship text.

    parent_process_ids is preferred because it is the column used
    by the temporal evidence attribution output.
    """

    parent_id = clean_text(
        row.get(
            "parent_process_ids",
            row.get("parent_process_id", ""),
        ),
        default="",
    )

    child_ids = clean_text(
        row.get("child_process_ids", ""),
        default="",
    )

    parts = []

    if parent_id:
        parts.append(f"Parent process: PID {parent_id}")

    if child_ids:
        parts.append(f"Child processes: {child_ids}")
    else:
        parts.append("Child processes: none observed")

    return "; ".join(parts)


# ============================================================
# Main explanation generation
# ============================================================

def generate_explanations():
    print(
        "=== ForensiXplain Temporal Investigator "
        "Explanation Generator ==="
    )

    # --------------------------------------------------------
    # Validate input files
    # --------------------------------------------------------

    if not ATTRIBUTION_FILE.exists():
        raise FileNotFoundError(
            f"Attribution file not found:\n{ATTRIBUTION_FILE}"
        )

    if not SHAP_FILE.exists():
        raise FileNotFoundError(
            f"SHAP file not found:\n{SHAP_FILE}"
        )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    attribution_df = pd.read_csv(ATTRIBUTION_FILE)
    shap_df = pd.read_csv(SHAP_FILE)

    print(f"Attribution rows: {len(attribution_df)}")
    print(f"SHAP rows: {len(shap_df)}")

    # --------------------------------------------------------
    # Validate required identifier
    # --------------------------------------------------------

    if "logical_event_id" not in attribution_df.columns:
        raise ValueError(
            "temporal_evidence_attribution.csv must contain "
            "'logical_event_id'."
        )

    if "logical_event_id" not in shap_df.columns:
        raise ValueError(
            "temporal_shap_explanations.csv must contain "
            "'logical_event_id'."
        )

    # --------------------------------------------------------
    # Select SHAP columns
    # --------------------------------------------------------

    shap_columns = [
        "logical_event_id",
        "temporal_sequence",
        "timestamp",
        "process_id",
        "process",
        "time_since_previous_event_seconds",
        "temporal_anomaly_score",
        "temporal_predicted_anomaly",
        "temporal_anomaly_rank",
        "gap_log_seconds",
        "local_density_10s",
        "local_density_30s",
        "local_density_60s",
        "process_changed",
        "shap_gap_log_seconds",
        "shap_local_density_10s",
        "shap_local_density_30s",
        "shap_local_density_60s",
        "shap_process_changed",
        "value_gap_log_seconds",
        "value_local_density_10s",
        "value_local_density_30s",
        "value_local_density_60s",
        "value_process_changed",
    ]

    available_shap_columns = [
        column
        for column in shap_columns
        if column in shap_df.columns
    ]

    shap_subset = shap_df[available_shap_columns].copy()

    # Prevent duplicate columns during merge.
    duplicate_columns = [
        column
        for column in shap_subset.columns
        if column != "logical_event_id"
        and column in attribution_df.columns
    ]

    if duplicate_columns:
        shap_subset = shap_subset.drop(columns=duplicate_columns)

    # --------------------------------------------------------
    # Merge attribution + SHAP
    # --------------------------------------------------------

    merged_df = attribution_df.merge(
        shap_subset,
        on="logical_event_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_shap"),
    )

    if "temporal_anomaly_score" not in merged_df.columns:
        raise ValueError(
            "Merged data does not contain temporal_anomaly_score."
        )

    merged_df = merged_df[
        merged_df["temporal_anomaly_score"].notna()
    ].copy()

    merged_df = merged_df.sort_values(
        by="temporal_anomaly_score",
        ascending=False,
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Generate explanation records
    # --------------------------------------------------------

    explanation_records = []

    for _, row in merged_df.iterrows():
        process_id = clean_text(
            row.get("process_id", "")
        )

        process_name = clean_text(
            row.get("process", "unknown"),
            default="unknown",
        )

        anomaly_score = safe_float(
            row.get("temporal_anomaly_score", 0)
        )

        anomaly_rank = safe_int(
            row.get("temporal_anomaly_rank", 0)
        )

        timestamp = clean_text(
            row.get("timestamp", "")
        )

        temporal_sequence = safe_int(
            row.get("temporal_sequence", 0)
        )

        # ----------------------------------------------------
        # SHAP explanation
        # ----------------------------------------------------

        contributors = get_top_shap_features(row)

        contributor_text = []

        for feature_name, value, shap_value in contributors:
            contributor_text.append(
                feature_description(
                    feature_name,
                    value,
                    shap_value,
                )
            )

        shap_explanation = " ".join(contributor_text)

        if not shap_explanation:
            shap_explanation = "No SHAP explanation available."

        # ----------------------------------------------------
        # Temporal context
        # ----------------------------------------------------

        previous_process = clean_text(
            row.get("previous_process", ""),
            default="unknown",
        )

        previous_process_id = clean_text(
            row.get("previous_process_id", ""),
            default="",
        )

        process_transition = clean_text(
            row.get("process_transition", ""),
            default="not available",
        )

        time_gap = safe_float(
            row.get(
                "time_since_previous_event_seconds",
                0,
            )
        )

        local_density_10s = safe_int(
            row.get("local_density_10s", 0)
        )

        local_density_30s = safe_int(
            row.get("local_density_30s", 0)
        )

        local_density_60s = safe_int(
            row.get("local_density_60s", 0)
        )

        process_changed = safe_int(
            row.get("process_changed", 0)
        )

        # ----------------------------------------------------
        # Evidence
        # ----------------------------------------------------

        evidence_summary = build_evidence_summary(row)
        evidence_ids = build_evidence_ids(row)
        relationship_summary = build_relationship_summary(row)

        # ----------------------------------------------------
        # Assessment and limitation
        # ----------------------------------------------------

        assessment = (
            "The event was ranked as temporally anomalous relative "
            "to the Isolation Forest baseline. The anomaly indication "
            "should be reviewed against the underlying forensic "
            "evidence and surrounding timeline context."
        )

        limitation = (
            "An anomaly score does not establish maliciousness, "
            "attack activity, or causality. Chronological adjacency "
            "is not treated as causal evidence. The result should "
            "be interpreted together with the linked forensic "
            "artifacts."
        )

        # ----------------------------------------------------
        # Machine-readable record
        # ----------------------------------------------------

        explanation_records.append(
            {
                "logical_event_id": clean_text(
                    row.get("logical_event_id", "")
                ),
                "temporal_anomaly_rank": anomaly_rank,
                "temporal_anomaly_score": anomaly_score,
                "temporal_sequence": temporal_sequence,
                "timestamp": timestamp,
                "process_id": process_id,
                "process": process_name,
                "previous_process_id": previous_process_id,
                "previous_process": previous_process,
                "process_transition": process_transition,
                "time_since_previous_event_seconds": time_gap,
                "local_density_10s": local_density_10s,
                "local_density_30s": local_density_30s,
                "local_density_60s": local_density_60s,
                "process_changed": process_changed,
                "shap_explanation": shap_explanation,
                "evidence_summary": evidence_summary,
                "evidence_ids": evidence_ids,
                "relationship_summary": relationship_summary,
                "assessment": assessment,
                "limitation": limitation,
            }
        )

    explanations_df = pd.DataFrame(explanation_records)

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    explanations_df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Generate investigator report
    # --------------------------------------------------------

    report_lines = [
        "ForensiXplain - Temporal Investigator Report",
        "=" * 60,
        "",
        "Purpose:",
        (
            "Identify temporally anomalous process-start events "
            "and provide evidence-grounded explanations."
        ),
        "",
        "Important interpretation:",
        "Temporal anomaly does not mean malicious activity.",
        "Chronological adjacency is not treated as causality.",
        (
            "Investigators should validate each candidate against "
            "the underlying forensic artifacts."
        ),
        "",
        f"Total temporal anomalies: {len(explanations_df)}",
        "",
    ]

    for _, row in explanations_df.iterrows():
        report_lines.extend(
            [
                "-" * 60,
                (
                    "Temporal Anomaly Rank: "
                    f"{safe_int(row['temporal_anomaly_rank'])}"
                ),
                (
                    "Anomaly Score: "
                    f"{safe_float(row['temporal_anomaly_score']):.6f}"
                ),
                (
                    "Logical Event: "
                    f"{clean_text(row['logical_event_id'])}"
                ),
                (
                    "Timeline Sequence: "
                    f"{safe_int(row['temporal_sequence'])}"
                ),
                (
                    "Timestamp: "
                    f"{clean_text(row['timestamp'])}"
                ),
                (
                    "Process: "
                    f"{clean_text(row['process'])} "
                    f"(PID {clean_text(row['process_id'])})"
                ),
                "",
                "Temporal Context:",
                (
                    "Previous process: "
                    f"{clean_text(row['previous_process'])} "
                    f"(PID {clean_text(row['previous_process_id'])})"
                ),
                (
                    "Process transition: "
                    f"{clean_text(row['process_transition'], 'not available')}"
                ),
                (
                    "Time since previous event: "
                    f"{safe_float(row['time_since_previous_event_seconds']):.2f} seconds"
                ),
                (
                    "Local density 10s: "
                    f"{safe_int(row['local_density_10s'])}"
                ),
                (
                    "Local density 30s: "
                    f"{safe_int(row['local_density_30s'])}"
                ),
                (
                    "Local density 60s: "
                    f"{safe_int(row['local_density_60s'])}"
                ),
                (
                    "Process changed: "
                    f"{safe_int(row['process_changed'])}"
                ),
                "",
                "SHAP Explanation:",
                clean_text(
                    row["shap_explanation"],
                    default="No SHAP explanation available.",
                ),
                "",
                "Evidence Summary:",
                clean_text(
                    row["evidence_summary"],
                    default="No artifact-specific observations recorded.",
                ),
                "",
                "Process Relationships:",
                clean_text(
                    row["relationship_summary"],
                    default="No process relationship information.",
                ),
                "",
                "Evidence IDs:",
                clean_text(
                    row["evidence_ids"],
                    default="No evidence IDs recorded.",
                ),
                "",
                "Assessment:",
                clean_text(row["assessment"]),
                "",
                "Limitation:",
                clean_text(row["limitation"]),
                "",
            ]
        )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    OUTPUT_REPORT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print("")
    print("=== Temporal Investigator Explanation Complete ===")
    print(f"Explanation rows: {len(explanations_df)}")
    print(f"CSV output: {OUTPUT_CSV}")
    print(f"Report output: {OUTPUT_REPORT}")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    generate_explanations()
