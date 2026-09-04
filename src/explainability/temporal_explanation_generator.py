from pathlib import Path
import pandas as pd


# ============================================================
# ForensiXplain
# Temporal Investigator Explanation Generator
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results" / "M57-Jean"

ATTRIBUTION_FILE = (
    RESULTS_DIR / "temporal_evidence_attribution.csv"
)

SHAP_FILE = (
    RESULTS_DIR / "temporal_shap_explanations.csv"
)

OUTPUT_CSV = (
    RESULTS_DIR / "temporal_investigator_explanations.csv"
)

OUTPUT_TXT = (
    RESULTS_DIR / "temporal_investigator_report.txt"
)


# ============================================================
# Helpers
# ============================================================

def safe_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def safe_int(value):
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


def format_seconds(value):
    if pd.isna(value):
        return "unknown"

    seconds = safe_float(value)

    if seconds < 60:
        return f"{seconds:.0f} seconds"

    if seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"

    hours = seconds / 3600
    return f"{hours:.2f} hours"


def feature_description(feature, value, shap_value):
    """
    Convert temporal SHAP features into investigator-readable
    explanations.
    """

    direction = (
        "increased"
        if shap_value > 0
        else "decreased"
    )

    descriptions = {

        "shap_gap_log_seconds":
            f"The logarithmic time gap ({value:.4f}) "
            f"{direction} the temporal anomaly score.",

        "shap_local_density_10s":
            f"The 10-second local event density ({value:.0f}) "
            f"{direction} the temporal anomaly score.",

        "shap_local_density_30s":
            f"The 30-second local event density ({value:.0f}) "
            f"{direction} the temporal anomaly score.",

        "shap_local_density_60s":
            f"The 60-second local event density ({value:.0f}) "
            f"{direction} the temporal anomaly score.",

        "shap_process_changed":
            f"The process-transition indicator ({value:.0f}) "
            f"{direction} the temporal anomaly score.",
    }

    return descriptions.get(
        feature,
        f"{feature} "
        f"{direction} the temporal anomaly score."
    )


def get_top_shap_features(row):
    """
    Extract the three strongest SHAP contributors.
    """

    feature_pairs = [
        (
            "shap_gap_log_seconds",
            "gap_log_seconds"
        ),
        (
            "shap_local_density_10s",
            "local_density_10s"
        ),
        (
            "shap_local_density_30s",
            "local_density_30s"
        ),
        (
            "shap_local_density_60s",
            "local_density_60s"
        ),
        (
            "shap_process_changed",
            "process_changed"
        ),
    ]

    contributors = []

    for shap_column, value_column in feature_pairs:

        if shap_column not in row.index:
            continue

        shap_value = safe_float(
            row[shap_column]
        )

        value = safe_float(
            row.get(value_column, 0)
        )

        contributors.append(
            (
                shap_column,
                value_column,
                value,
                shap_value
            )
        )

    contributors.sort(
        key=lambda item: abs(item[3]),
        reverse=True
    )

    return contributors[:3]


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=== ForensiXplain Temporal Investigator "
        "Explanation Generator ==="
    )

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not ATTRIBUTION_FILE.exists():

        raise FileNotFoundError(
            f"Missing attribution file:\n"
            f"{ATTRIBUTION_FILE}"
        )

    if not SHAP_FILE.exists():

        raise FileNotFoundError(
            f"Missing SHAP file:\n"
            f"{SHAP_FILE}"
        )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    attribution_df = pd.read_csv(
        ATTRIBUTION_FILE
    )

    shap_df = pd.read_csv(
        SHAP_FILE
    )

    print(
        f"Attribution rows: "
        f"{len(attribution_df)}"
    )

    print(
        f"SHAP rows: "
        f"{len(shap_df)}"
    )

    # --------------------------------------------------------
    # Merge SHAP values
    # --------------------------------------------------------

    shap_columns = [
        "logical_event_id",
        "shap_gap_log_seconds",
        "shap_local_density_10s",
        "shap_local_density_30s",
        "shap_local_density_60s",
        "shap_process_changed",

        "gap_log_seconds",
        "local_density_10s",
        "local_density_30s",
        "local_density_60s",
        "process_changed",
    ]

    shap_columns = [
        column
        for column in shap_columns
        if column in shap_df.columns
    ]

    shap_context = shap_df[
        shap_columns
    ].copy()

    # --------------------------------------------------------
    # Prevent duplicate SHAP rows
    # --------------------------------------------------------

    if shap_context[
        "logical_event_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate logical_event_id values "
            "found in temporal SHAP output."
        )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    df = attribution_df.merge(
        shap_context,
        on="logical_event_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_shap")
    )

    # --------------------------------------------------------
    # Generate explanation records
    # --------------------------------------------------------

    explanation_rows = []

    report_sections = []

    for _, row in df.iterrows():

        rank = safe_int(
            row["temporal_anomaly_rank"]
        )

        pid = safe_int(
            row["process_id"]
        )

        process = str(
            row["process"]
        )

        timestamp = str(
            row["timestamp"]
        )

        score = safe_float(
            row["temporal_anomaly_score"]
        )

        previous_process = row.get(
            "previous_process"
        )

        previous_pid = safe_int(
            row.get("previous_process_id")
        )

        time_gap = row.get(
            "time_since_previous_event_seconds"
        )

        transition = row.get(
            "process_transition"
        )

        parent_ids = str(
            row.get(
                "parent_process_ids_attributed",
                ""
            )
        )

        child_ids = str(
            row.get(
                "child_process_ids",
                ""
            )
        )

        evidence_ids = str(
            row.get(
                "evidence_ids_attributed",
                ""
            )
        )

        evidence_count = safe_int(
            row.get(
                "evidence_id_count",
                0
            )
        )

        raw_event_count = safe_int(
            row.get(
                "raw_event_count",
                0
            )
        )

        pslist_count = safe_int(
            row.get(
                "pslist_count",
                0
            )
        )

        pstree_count = safe_int(
            row.get(
                "pstree_count",
                0
            )
        )

        cmdline_count = safe_int(
            row.get(
                "cmdline_count",
                0
            )
        )

        dlllist_count = safe_int(
            row.get(
                "dlllist_count",
                0
            )
        )

        malfind_count = safe_int(
            row.get(
                "malfind_count",
                0
            )
        )

        # ----------------------------------------------------
        # SHAP contributors
        # ----------------------------------------------------

        contributors = get_top_shap_features(
            row
        )

        contributor_text = []

        for (
            shap_column,
            value_column,
            value,
            shap_value
        ) in contributors:

            contributor_text.append(
                feature_description(
                    shap_column,
                    value,
                    shap_value
                )
            )

        # ----------------------------------------------------
        # Previous process
        # ----------------------------------------------------

        if (
            pd.notna(previous_process)
            and str(previous_process).strip()
        ):

            previous_text = (
                f"{previous_process} "
                f"(PID {previous_pid})"
            )

        else:

            previous_text = "Unavailable"

        # ----------------------------------------------------
        # Parent
        # ----------------------------------------------------

        parent_text = (
            parent_ids
            if parent_ids
            and parent_ids != "nan"
            else "None identified"
        )

        # ----------------------------------------------------
        # Children
        # ----------------------------------------------------

        child_text = (
            child_ids
            if child_ids
            and child_ids != "nan"
            else "None identified"
        )

        # ----------------------------------------------------
        # Evidence
        # ----------------------------------------------------

        evidence_text = (
            evidence_ids
            if evidence_ids
            and evidence_ids != "nan"
            else "None identified"
        )

        # ----------------------------------------------------
        # Assessment
        # ----------------------------------------------------

        assessment = (
            "Temporal anomaly candidate requiring "
            "investigator review. The model identifies "
            "a deviation from the learned temporal "
            "baseline; this finding does not by itself "
            "establish malicious activity."
        )

        # ----------------------------------------------------
        # Explanation record
        # ----------------------------------------------------

        explanation_rows.append({

            "temporal_anomaly_rank":
                rank,

            "logical_event_id":
                row["logical_event_id"],

            "temporal_sequence":
                row["temporal_sequence"],

            "timestamp":
                timestamp,

            "process_id":
                pid,

            "process":
                process,

            "temporal_anomaly_score":
                score,

            "previous_process_id":
                previous_pid,

            "previous_process":
                previous_process,

            "time_since_previous_event_seconds":
                time_gap,

            "process_transition":
                transition,

            "parent_process_ids":
                parent_text,

            "child_process_ids":
                child_text,

            "top_shap_feature_1":
                contributors[0][0]
                if len(contributors) > 0
                else "",

            "top_shap_value_1":
                contributors[0][3]
                if len(contributors) > 0
                else 0.0,

            "top_shap_feature_2":
                contributors[1][0]
                if len(contributors) > 1
                else "",

            "top_shap_value_2":
                contributors[1][3]
                if len(contributors) > 1
                else 0.0,

            "top_shap_feature_3":
                contributors[2][0]
                if len(contributors) > 2
                else "",

            "top_shap_value_3":
                contributors[2][3]
                if len(contributors) > 2
                else 0.0,

            "raw_event_count":
                raw_event_count,

            "evidence_id_count":
                evidence_count,

            "pslist_count":
                pslist_count,

            "pstree_count":
                pstree_count,

            "cmdline_count":
                cmdline_count,

            "dlllist_count":
                dlllist_count,

            "malfind_count":
                malfind_count,

            "evidence_ids":
                evidence_text,

            "assessment":
                assessment,
        })

        # ----------------------------------------------------
        # Human-readable report
        # ----------------------------------------------------

        section = []

        section.append(
            f"TEMPORAL ANOMALY #{rank}"
        )

        section.append(
            "=" * 70
        )

        section.append(
            f"Process: {process} (PID {pid})"
        )

        section.append(
            f"Timestamp: {timestamp}"
        )

        section.append(
            f"Logical timeline sequence: "
            f"{row['temporal_sequence']}"
        )

        section.append("")

        section.append(
            "MODEL FINDING"
        )

        section.append(
            f"Temporal anomaly score: "
            f"{score:.6f}"
        )

        section.append(
            "Interpretation: The process was identified "
            "as a temporal anomaly candidate relative "
            "to the learned baseline."
        )

        section.append("")

        section.append(
            "TEMPORAL CONTEXT"
        )

        section.append(
            f"Previous process: "
            f"{previous_text}"
        )

        section.append(
            f"Time since previous event: "
            f"{format_seconds(time_gap)}"
        )

        section.append(
            f"Process transition: "
            f"{transition}"
        )

        section.append(
            f"Parent PID(s): "
            f"{parent_text}"
        )

        section.append(
            f"Child PID(s): "
            f"{child_text}"
        )

        section.append("")

        section.append(
            "SHAP EXPLANATION"
        )

        if contributor_text:

            for index, text in enumerate(
                contributor_text,
                start=1
            ):

                section.append(
                    f"{index}. {text}"
                )

        else:

            section.append(
                "No SHAP contributors available."
            )

        section.append("")

        section.append(
            "FORENSIC EVIDENCE"
        )

        section.append(
            f"Raw process observations: "
            f"{raw_event_count}"
        )

        section.append(
            f"Evidence IDs: "
            f"{evidence_count}"
        )

        section.append(
            f"PSList observations: "
            f"{pslist_count}"
        )

        section.append(
            f"PSTree observations: "
            f"{pstree_count}"
        )

        section.append(
            f"Command-line observations: "
            f"{cmdline_count}"
        )

        section.append(
            f"DLLList observations: "
            f"{dlllist_count}"
        )

        section.append(
            f"Malfind observations: "
            f"{malfind_count}"
        )

        section.append(
            f"Evidence IDs: "
            f"{evidence_text}"
        )

        section.append("")

        section.append(
            "ASSESSMENT"
        )

        section.append(
            assessment
        )

        section.append("")

        section.append(
            "LIMITATION"
        )

        section.append(
            "The temporal anomaly score is an "
            "unsupervised model output. It should "
            "not be interpreted as proof of malware, "
            "attack activity, or user intent."
        )

        section.append("")

        report_sections.append(
            "\n".join(section)
        )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    explanation_df = pd.DataFrame(
        explanation_rows
    )

    explanation_df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Save TXT report
    # --------------------------------------------------------

    report_header = []

    report_header.append(
        "ForensiXplain"
    )

    report_header.append(
        "Temporal Investigator Report"
    )

    report_header.append(
        "=" * 70
    )

    report_header.append("")

    report_header.append(
        "This report summarizes temporal anomaly "
        "candidates generated from the M57-Jean "
        "forensic memory analysis."
    )

    report_header.append("")

    report_header.append(
        "IMPORTANT: Temporal anomaly scores indicate "
        "deviation from the learned temporal baseline. "
        "They do not establish malicious activity."
    )

    report_header.append("")

    report_header.extend(
        report_sections
    )

    OUTPUT_TXT.write_text(
        "\n".join(report_header),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print(
        "\n=== Temporal Investigator Explanation "
        "Complete ==="
    )

    print(
        f"Explanation rows: "
        f"{len(explanation_df)}"
    )

    print(
        f"CSV output: "
        f"{OUTPUT_CSV}"
    )

    print(
        f"Report output: "
        f"{OUTPUT_TXT}"
    )


if __name__ == "__main__":
    main()