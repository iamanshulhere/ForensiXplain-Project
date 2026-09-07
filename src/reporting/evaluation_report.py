"""
ForensiXplain - Final Evaluation Report

Purpose:
    Consolidate temporal, graph-aware, and graph-only anomaly
    results into a final evaluation summary.

Input:
    results/M57-Jean/temporal_anomalies.csv
    results/M57-Jean/graph_anomalies.csv
    results/M57-Jean/graph_only_anomalies.csv
    results/M57-Jean/model_comparison.csv
    results/M57-Jean/temporal_investigator_explanations.csv
    results/M57-Jean/graph_investigator_explanations.csv
    results/M57-Jean/graph_only_evidence_attribution.csv

Outputs:
    results/M57-Jean/evaluation_summary.csv
    results/M57-Jean/evaluation_report.txt

Important:
    This report evaluates and consolidates anomaly candidates.
    It does NOT determine whether any process is malicious.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# Configuration
# ============================================================

CASE_ID = "M57-Jean"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results" / CASE_ID

TEMPORAL_FILE = RESULTS_DIR / "temporal_anomalies.csv"
GRAPH_FILE = RESULTS_DIR / "graph_anomalies.csv"
GRAPH_ONLY_FILE = RESULTS_DIR / "graph_only_anomalies.csv"
COMPARISON_FILE = RESULTS_DIR / "model_comparison.csv"

TEMPORAL_EXPLANATION_FILE = (
    RESULTS_DIR / "temporal_investigator_explanations.csv"
)

GRAPH_EXPLANATION_FILE = (
    RESULTS_DIR / "graph_investigator_explanations.csv"
)

GRAPH_ONLY_EVIDENCE_FILE = (
    RESULTS_DIR / "graph_only_evidence_attribution.csv"
)

SUMMARY_FILE = RESULTS_DIR / "evaluation_summary.csv"
REPORT_FILE = RESULTS_DIR / "evaluation_report.txt"


# ============================================================
# Helpers
# ============================================================

def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    return pd.read_csv(path)


def anomaly_pids(df, flag_column):
    values = df.loc[df[flag_column].astype(bool), "process_id"]
    return set(values.dropna().astype(int))


def sorted_list(values):
    return sorted(int(x) for x in values)


def join_values(values):
    if not values:
        return "None"
    return ", ".join(str(x) for x in sorted_list(values))


# ============================================================
# Main evaluation
# ============================================================

def main():

    print("=== ForensiXplain Final Evaluation ===")

    temporal = load_csv(TEMPORAL_FILE)
    graph = load_csv(GRAPH_FILE)
    graph_only = load_csv(GRAPH_ONLY_FILE)
    comparison = load_csv(COMPARISON_FILE)

    temporal_exp = load_csv(TEMPORAL_EXPLANATION_FILE)
    graph_exp = load_csv(GRAPH_EXPLANATION_FILE)
    graph_only_exp = load_csv(GRAPH_ONLY_EVIDENCE_FILE)

    temporal_set = anomaly_pids(
        temporal,
        "temporal_predicted_anomaly"
    )

    graph_set = anomaly_pids(
        graph,
        "graph_predicted_anomaly"
    )

    graph_only_set = anomaly_pids(
        graph_only,
        "graph_only_predicted_anomaly"
    )

    temporal_only = (
        temporal_set - graph_set - graph_only_set
    )

    graph_only_model = (
        graph_only_set - temporal_set - graph_set
    )

    temporal_graph = (
        temporal_set & graph_set
    )

    graph_graph_only = (
        graph_set & graph_only_set
    )

    all_three = (
        temporal_set & graph_set & graph_only_set
    )

    print(f"Temporal rows: {len(temporal)}")
    print(f"Graph-aware rows: {len(graph)}")
    print(f"Graph-only rows: {len(graph_only)}")

    print()
    print("=== Detection Summary ===")
    print(f"Temporal anomalies: {len(temporal_set)}")
    print(f"Graph-aware anomalies: {len(graph_set)}")
    print(f"Graph-only anomalies: {len(graph_only_set)}")

    print()
    print("Temporal-only:", sorted_list(temporal_only))
    print("Temporal + Graph-aware:", sorted_list(temporal_graph))
    print(
        "Graph-aware + Graph-only:",
        sorted_list(graph_graph_only)
    )
    print("Graph-only model only:", sorted_list(graph_only_model))
    print("All three:", sorted_list(all_three))

    # --------------------------------------------------------
    # Evaluation summary
    # --------------------------------------------------------

    summary_rows = [
        {
            "case_id": CASE_ID,
            "category": "temporal_only",
            "process_count": len(temporal_only),
            "process_ids": join_values(temporal_only),
        },
        {
            "case_id": CASE_ID,
            "category": "temporal_and_graph",
            "process_count": len(temporal_graph),
            "process_ids": join_values(temporal_graph),
        },
        {
            "case_id": CASE_ID,
            "category": "graph_and_graph_only",
            "process_count": len(graph_graph_only),
            "process_ids": join_values(graph_graph_only),
        },
        {
            "case_id": CASE_ID,
            "category": "graph_only_model_only",
            "process_count": len(graph_only_model),
            "process_ids": join_values(graph_only_model),
        },
        {
            "case_id": CASE_ID,
            "category": "all_three_models",
            "process_count": len(all_three),
            "process_ids": join_values(all_three),
        },
    ]

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_FILE, index=False)

    # --------------------------------------------------------
    # Candidate-level information
    # --------------------------------------------------------

    comparison_index = comparison.set_index("process_id")

    candidate_ids = sorted(
        temporal_set | graph_set | graph_only_set
    )

    candidate_rows = []

    for pid in candidate_ids:

        row = {
            "case_id": CASE_ID,
            "process_id": pid,
            "process": "unknown",
            "temporal_anomaly": pid in temporal_set,
            "graph_anomaly": pid in graph_set,
            "graph_only_anomaly": pid in graph_only_set,
        }

        if pid in comparison_index.index:
            comparison_row = comparison_index.loc[pid]

            row["process"] = comparison_row.get(
                "process",
                "unknown"
            )

            row["agreement_category"] = comparison_row.get(
                "agreement_category",
                "unknown"
            )

            row["detection_count"] = comparison_row.get(
                "detection_count",
                0
            )

        else:
            row["agreement_category"] = "unknown"
            row["detection_count"] = sum(
                [
                    row["temporal_anomaly"],
                    row["graph_anomaly"],
                    row["graph_only_anomaly"],
                ]
            )

        # Temporal explanation
        temporal_match = temporal_exp[
            temporal_exp["process_id"] == pid
        ]

        if not temporal_match.empty:
            t = temporal_match.iloc[0]

            row["temporal_score"] = t.get(
                "temporal_anomaly_score",
                ""
            )

            row["temporal_shap"] = t.get(
                "shap_explanation",
                ""
            )

        else:
            row["temporal_score"] = ""
            row["temporal_shap"] = ""

        # Graph explanation
        graph_match = graph_exp[
            graph_exp["process_id"] == pid
        ]

        if not graph_match.empty:
            g = graph_match.iloc[0]

            row["graph_score"] = g.get(
                "graph_anomaly_score",
                ""
            )

            row["graph_shap"] = g.get(
                "top_shap_features",
                ""
            )

            row["graph_evidence"] = g.get(
                "evidence_by_artifact",
                ""
            )

        else:
            row["graph_score"] = ""
            row["graph_shap"] = ""
            row["graph_evidence"] = ""

        # Graph-only explanation
        graph_only_match = graph_only_exp[
            graph_only_exp["process_id"] == pid
        ]

        if not graph_only_match.empty:
            go = graph_only_match.iloc[0]

            row["graph_only_score"] = go.get(
                "graph_only_anomaly_score",
                ""
            )

            row["graph_only_shap"] = go.get(
                "top_graph_shap_features",
                ""
            )

            row["graph_only_evidence"] = go.get(
                "evidence_by_artifact",
                ""
            )

        else:
            row["graph_only_score"] = ""
            row["graph_only_shap"] = ""
            row["graph_only_evidence"] = ""

        candidate_rows.append(row)

    candidate_df = pd.DataFrame(candidate_rows)

    # Add candidate details below the category summary.
    candidate_output = RESULTS_DIR / "evaluation_candidates.csv"
    candidate_df.to_csv(candidate_output, index=False)

    # --------------------------------------------------------
    # Text report
    # --------------------------------------------------------

    lines = []

    lines.append("ForensiXplain Final Evaluation Report")
    lines.append("=" * 45)
    lines.append("")
    lines.append(f"Case: {CASE_ID}")
    lines.append("")

    lines.append("Dataset / pipeline size:")
    lines.append(f"  Temporal model rows: {len(temporal)}")
    lines.append(f"  Graph-aware model rows: {len(graph)}")
    lines.append(f"  Graph-only model rows: {len(graph_only)}")
    lines.append("")

    lines.append("Anomaly counts:")
    lines.append(
        f"  Temporal model: {len(temporal_set)}"
    )
    lines.append(
        f"  Graph-aware model: {len(graph_set)}"
    )
    lines.append(
        f"  Graph-only model: {len(graph_only_set)}"
    )
    lines.append("")

    lines.append("Detection relationships:")
    lines.append(
        f"  Temporal-only: {join_values(temporal_only)}"
    )
    lines.append(
        f"  Temporal AND Graph-aware: "
        f"{join_values(temporal_graph)}"
    )
    lines.append(
        f"  Graph-aware AND Graph-only: "
        f"{join_values(graph_graph_only)}"
    )
    lines.append(
        f"  Graph-only model only: "
        f"{join_values(graph_only_model)}"
    )
    lines.append(
        f"  All three models: {join_values(all_three)}"
    )
    lines.append("")

    lines.append("Candidate details:")
    lines.append("")

    for _, row in candidate_df.iterrows():

        lines.append(
            f"PID {int(row['process_id'])} "
            f"{row['process']}"
        )

        lines.append(
            f"  Models detecting candidate: "
            f"{row['detection_count']}"
        )

        lines.append(
            f"  Agreement category: "
            f"{row['agreement_category']}"
        )

        if row["temporal_anomaly"]:
            lines.append(
                f"  Temporal score: "
                f"{row['temporal_score']}"
            )

        if row["graph_anomaly"]:
            lines.append(
                f"  Graph score: "
                f"{row['graph_score']}"
            )

        if row["graph_only_anomaly"]:
            lines.append(
                f"  Graph-only score: "
                f"{row['graph_only_score']}"
            )

        if row["temporal_shap"]:
            lines.append(
                f"  Temporal SHAP: "
                f"{row['temporal_shap']}"
            )

        if row["graph_shap"]:
            lines.append(
                f"  Graph SHAP: "
                f"{row['graph_shap']}"
            )

        if row["graph_only_shap"]:
            lines.append(
                f"  Graph-only SHAP: "
                f"{row['graph_only_shap']}"
            )

        if row["temporal_anomaly"]:
            lines.append(
                "  Temporal evidence: "
                "available in temporal investigator explanation."
            )

        if row["graph_anomaly"]:
            lines.append(
                "  Graph evidence: "
                f"{row['graph_evidence']}"
            )

        if row["graph_only_anomaly"]:
            lines.append(
                "  Graph-only evidence: "
                f"{row['graph_only_evidence']}"
            )

        lines.append("")

    lines.append("Interpretation:")
    lines.append(
        "  Model disagreement indicates that the models are "
        "responding to different feature representations."
    )
    lines.append(
        "  Overlap between models identifies candidates that "
        "are unusual under multiple representations."
    )
    lines.append(
        "  Model-specific candidates remain important for "
        "investigator review because they may reflect "
        "patterns captured by only one representation."
    )
    lines.append("")

    lines.append("Important limitation:")
    lines.append(
        "  An anomaly indicates an unusual feature profile "
        "under the corresponding model."
    )
    lines.append(
        "  An anomaly score or SHAP attribution does not "
        "establish malicious activity."
    )
    lines.append(
        "  Final interpretation requires review of the "
        "underlying forensic evidence."
    )

    REPORT_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print()
    print("=== Final Evaluation Complete ===")
    print(f"Summary: {SUMMARY_FILE}")
    print(f"Candidates: {candidate_output}")
    print(f"Report: {REPORT_FILE}")


if __name__ == "__main__":
    main()