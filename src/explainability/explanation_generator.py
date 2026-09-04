from pathlib import Path

import pandas as pd


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

EVIDENCE_PATH = (
    BASE_DIR
    / "results"
    / "M57-Jean"
    / "evidence_attribution.csv"
)

EVENTS_PATH = (
    BASE_DIR
    / "data"
    / "normalized"
    / "M57-Jean"
    / "events.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "M57-Jean"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "investigator_explanations.csv"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "investigator_report.txt"
)


# =========================================================
# Feature explanation templates
# =========================================================

FEATURE_EXPLANATIONS = {

    "graph_degree": (
        "The process has an unusual graph connectivity profile "
        "relative to other observed processes."
    ),

    "child_count": (
        "The process has an unusual number of child-process "
        "relationships."
    ),

    "command_line_count": (
        "The process has an unusual command-line observation profile."
    ),

    "module_count": (
        "The process has an unusual number of loaded-module "
        "observations."
    ),

    "memory_region_count": (
        "The process has an unusual number of memory-region "
        "observations from the memory-forensics analysis."
    ),

    "hour": (
        "The process creation time contributes to its unusual "
        "temporal profile."
    ),

    "day_of_week": (
        "The process creation day contributes to its unusual "
        "temporal profile."
    ),

    "after_hours": (
        "The process creation time falls into the configured "
        "after-hours period and contributes to the anomaly score."
    ),
}


# =========================================================
# Helpers
# =========================================================

def format_list(values):
    """Convert a semicolon-separated value into readable text."""

    if pd.isna(values) or str(values).strip() == "":
        return "None observed"

    items = [
        item.strip()
        for item in str(values).split(";")
        if item.strip()
    ]

    if not items:
        return "None observed"

    return ", ".join(items)


def format_shap(feature, value):
    """Create a readable SHAP explanation."""

    direction = (
        "increases"
        if value > 0
        else "decreases"
    )

    description = FEATURE_EXPLANATIONS.get(
        feature,
        f"The {feature} feature contributes to the anomaly score."
    )

    return (
        f"{description} "
        f"SHAP contribution = {value:+.6f}, "
        f"which {direction} the anomaly score."
    )


def build_explanation(row):
    """Build a structured investigator explanation."""

    process_id = int(row["process_id"])
    process_name = row["process_name"]

    anomaly_score = float(
        row["anomaly_score"]
    )

    # -----------------------------------------------------
    # SHAP factors
    # -----------------------------------------------------

    shap_explanations = []

    for index in range(1, 4):

        feature_column = (
            f"top_feature_{index}"
        )

        shap_column = (
            f"top_feature_{index}_shap"
        )

        feature = row.get(
            feature_column,
            ""
        )

        value = row.get(
            shap_column,
            0
        )

        if (
            pd.isna(feature)
            or str(feature).strip() == ""
        ):
            continue

        shap_explanations.append(
            format_shap(
                str(feature),
                float(value),
            )
        )

    # -----------------------------------------------------
    # Evidence
    # -----------------------------------------------------

    evidence_summary = (
        f"Process observations: "
        f"{int(row['process_event_count'])}; "
        f"PSList: {int(row['pslist_count'])}; "
        f"PSTree: {int(row['pstree_count'])}; "
        f"Command line: {int(row['cmdline_count'])}; "
        f"DLLList: {int(row['dlllist_count'])}; "
        f"Malfind: {int(row['malfind_count'])}."
    )

    # -----------------------------------------------------
    # Relationships
    # -----------------------------------------------------

    parent_ids = format_list(
        row["parent_process_ids"]
    )

    child_ids = format_list(
        row["child_process_ids"]
    )

    # -----------------------------------------------------
    # Commands
    # -----------------------------------------------------

    commands = format_list(
        row["command_lines"]
    )

    # -----------------------------------------------------
    # Evidence IDs
    # -----------------------------------------------------

    evidence_ids = format_list(
        row["evidence_ids"]
    )

    # -----------------------------------------------------
    # Assessment
    # -----------------------------------------------------

    assessment = (
        "The process exhibits an anomalous feature profile "
        "relative to the processes evaluated by the Isolation "
        "Forest baseline. The anomaly indication should be "
        "reviewed against the underlying forensic evidence."
    )

    limitation = (
        "This anomaly score does not by itself establish "
        "malicious activity. The result represents an "
        "evidence-linked anomaly indication."
    )

    # -----------------------------------------------------
    # Full explanation
    # -----------------------------------------------------

    explanation = f"""
PROCESS
-------
Process name: {process_name}
Process ID: {process_id}
Anomaly score: {anomaly_score:.6f}

WHY WAS IT FLAGGED?
-------------------
"""

    if shap_explanations:

        for number, item in enumerate(
            shap_explanations,
            start=1
        ):
            explanation += (
                f"{number}. {item}\n"
            )

    else:

        explanation += (
            "No SHAP feature explanation available.\n"
        )

    explanation += f"""
FORENSIC EVIDENCE
-----------------
{evidence_summary}

PARENT PROCESS IDS
------------------
{parent_ids}

CHILD PROCESS IDS
-----------------
{child_ids}

OBSERVED COMMAND LINES
----------------------
{commands}

EVIDENCE IDS
------------
{evidence_ids}

ASSESSMENT
----------
{assessment}

LIMITATION
----------
{limitation}
"""

    return explanation.strip()


# =========================================================
# Main
# =========================================================

def main():

    print(
        "=== ForensiXplain Explanation Generator ==="
    )

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------

    evidence_df = pd.read_csv(
        EVIDENCE_PATH
    )

    events_df = pd.read_csv(
        EVENTS_PATH
    )

    print(
        f"Evidence records: {len(evidence_df)}"
    )

    print(
        f"Events available: {len(events_df)}"
    )

    # -----------------------------------------------------
    # Generate explanations
    # -----------------------------------------------------

    explanation_records = []

    for _, row in evidence_df.iterrows():

        explanation = build_explanation(
            row
        )

        explanation_records.append(
            {
                "case_id": row["case_id"],
                "process_id": int(
                    row["process_id"]
                ),
                "process_name": row[
                    "process_name"
                ],
                "anomaly_score": float(
                    row["anomaly_score"]
                ),
                "explanation": explanation,
            }
        )

    output = pd.DataFrame(
        explanation_records
    )

    # -----------------------------------------------------
    # Save CSV
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"Saved: {OUTPUT_PATH}"
    )

    print(
        f"Explanation rows: {len(output)}"
    )

    # -----------------------------------------------------
    # Save human-readable report
    # -----------------------------------------------------

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "FORENSIXPLAIN INVESTIGATOR REPORT\n"
        )

        file.write(
            "=================================\n\n"
        )

        file.write(
            f"Case: M57-Jean\n"
        )

        file.write(
            f"Anomalous processes: {len(output)}\n\n"
        )

        for index, row in output.iterrows():

            file.write(
                f"\n{'=' * 70}\n"
            )

            file.write(
                f"ANOMALY {index + 1}\n"
            )

            file.write(
                f"{'=' * 70}\n\n"
            )

            file.write(
                row["explanation"]
            )

            file.write("\n\n")

    print(
        f"Saved: {REPORT_PATH}"
    )

    # -----------------------------------------------------
    # Display top explanation
    # -----------------------------------------------------

    if not output.empty:

        print(
            "\n=== Top Investigator Explanation ===\n"
        )

        print(
            output.iloc[0]["explanation"]
        )


if __name__ == "__main__":
    main()