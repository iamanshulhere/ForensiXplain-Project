from pathlib import Path
import pandas as pd


DATASET = "M57-Jean"

GRAPH_ANOMALIES = Path("results") / DATASET / "graph_anomalies.csv"
TEMPORAL_ANOMALIES = Path("results") / DATASET / "temporal_anomalies.csv"

OUTPUT_CSV = Path("results") / DATASET / "graph_anomaly_results.csv"
OUTPUT_REPORT = Path("results") / DATASET / "graph_anomaly_results_report.txt"


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file and raise a clear error if it does not exist."""
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    return pd.read_csv(path)


def get_anomaly_flag(
    df: pd.DataFrame,
    preferred_columns: list[str],
) -> pd.Series:
    """
    Return a boolean anomaly mask from the first available anomaly column.

    Handles boolean values stored as strings in CSV files.
    """
    for column in preferred_columns:
        if column in df.columns:
            values = df[column]

            if values.dtype == bool:
                return values

            return (
                values.astype(str)
                .str.strip()
                .str.lower()
                .isin(["true", "1", "yes"])
            )

    raise ValueError(
        "Could not find an anomaly flag column. "
        f"Available columns: {list(df.columns)}"
    )


def main():
    print("=== ForensiXplain Graph Anomaly Results Evaluation ===")

    # ---------------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------------
    graph = load_csv(GRAPH_ANOMALIES)
    temporal = load_csv(TEMPORAL_ANOMALIES)

    # ---------------------------------------------------------
    # 1. TOTAL EVENTS
    # ---------------------------------------------------------
    total_events = len(graph)

    # ---------------------------------------------------------
    # 2. GRAPH ANOMALIES
    # ---------------------------------------------------------
    graph_flag = get_anomaly_flag(
        graph,
        [
            "graph_predicted_anomaly",
            "graph_anomaly",
            "prediction",
            "is_anomaly",
        ],
    )

    graph_anomaly_rows = graph.loc[graph_flag].copy()

    graph_anomaly_count = len(graph_anomaly_rows)

    graph_anomaly_rate = (
        graph_anomaly_count / total_events * 100
        if total_events > 0
        else 0.0
    )

    # ---------------------------------------------------------
    # 3. FIND COMMON IDENTIFIER
    # ---------------------------------------------------------
    possible_ids = [
        "logical_event_id",
        "event_id",
        "process_id",
        "pid",
    ]

    common_id = None

    for column in possible_ids:
        if column in graph.columns and column in temporal.columns:
            common_id = column
            break

    # ---------------------------------------------------------
    # 4. TEMPORAL ANOMALIES + OVERLAP
    # ---------------------------------------------------------
    temporal_anomaly_count = None
    overlap_count = None
    graph_only_count = None

    if common_id is not None:

        try:
            temporal_flag = get_anomaly_flag(
                temporal,
                [
                    "temporal_predicted_anomaly",
                    "temporal_anomaly",
                    "prediction",
                    "is_anomaly",
                    "anomaly",
                ],
            )

            temporal_ids = set(
                temporal.loc[temporal_flag, common_id]
                .dropna()
                .astype(str)
            )

            graph_ids = set(
                graph_anomaly_rows[common_id]
                .dropna()
                .astype(str)
            )

            temporal_anomaly_count = len(temporal_ids)

            overlap_count = len(
                graph_ids.intersection(temporal_ids)
            )

            graph_only_count = len(
                graph_ids.difference(temporal_ids)
            )

        except ValueError:
            temporal_anomaly_count = None
            overlap_count = None
            graph_only_count = None

    # ---------------------------------------------------------
    # 5. DETECTION METHOD
    # ---------------------------------------------------------
    detection_method = "Graph Isolation Forest"

    # ---------------------------------------------------------
    # 6. CREATE RESULT TABLE
    # ---------------------------------------------------------
    results = {
        "dataset": DATASET,
        "total_events": total_events,
        "graph_anomalies_flagged": graph_anomaly_count,
        "graph_anomaly_rate_percent": graph_anomaly_rate,
        "detection_method": detection_method,
        "comparison_identifier": (
            common_id if common_id is not None else "unavailable"
        ),
        "temporal_anomalies": temporal_anomaly_count,
        "temporal_graph_overlap": overlap_count,
        "graph_only_anomalies": graph_only_count,
    }

    result_df = pd.DataFrame([results])

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # ---------------------------------------------------------
    # 7. HUMAN-READABLE REPORT
    # ---------------------------------------------------------
    report = []

    report.append(
        "ForensiXplain Graph Anomaly Detection Results"
    )
    report.append("=" * 55)
    report.append("")

    report.append(f"Dataset: {DATASET}")
    report.append(f"Total events: {total_events}")
    report.append(
        f"Graph anomalies flagged: {graph_anomaly_count}"
    )
    report.append(
        f"Graph anomaly rate: {graph_anomaly_rate:.2f}%"
    )
    report.append(
        f"Detection method: {detection_method}"
    )
    report.append("")

    # ---------------------------------------------------------
    # TEMPORAL / GRAPH COMPARISON
    # ---------------------------------------------------------
    if common_id is not None:

        report.append(
            f"Comparison identifier: {common_id}"
        )

        # Convert values to display strings first.
        temporal_text = (
            str(temporal_anomaly_count)
            if temporal_anomaly_count is not None
            else "N/A"
        )

        overlap_text = (
            str(overlap_count)
            if overlap_count is not None
            else "N/A"
        )

        graph_only_text = (
            str(graph_only_count)
            if graph_only_count is not None
            else "N/A"
        )

        report.append(
            f"Temporal anomalies: {temporal_text}"
        )

        report.append(
            f"Temporal/graph overlap: {overlap_text}"
        )

        report.append(
            f"Graph-only anomalies: {graph_only_text}"
        )

    else:

        report.append(
            "Temporal/graph overlap could not be calculated "
            "because no common event identifier was found."
        )

    # ---------------------------------------------------------
    # TOP GRAPH ANOMALY CANDIDATES
    # ---------------------------------------------------------
    report.append("")
    report.append("Top graph anomaly candidates")
    report.append("-" * 55)

    display_columns = [
        column
        for column in [
            "graph_anomaly_rank",
            "process_id",
            "process_name",
            "process",
            "graph_anomaly_score",
            "graph_predicted_anomaly",
        ]
        if column in graph_anomaly_rows.columns
    ]

    if display_columns:

        report.append(
            graph_anomaly_rows[
                display_columns
            ]
            .head(10)
            .to_string(index=False)
        )

    else:

        report.append(
            graph_anomaly_rows
            .head(10)
            .to_string(index=False)
        )

    # ---------------------------------------------------------
    # SAVE REPORT
    # ---------------------------------------------------------
    OUTPUT_REPORT.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # TERMINAL OUTPUT
    # ---------------------------------------------------------
    print("")
    print("=== Graph Anomaly Results Complete ===")
    print(f"Dataset: {DATASET}")
    print(f"Total events: {total_events}")
    print(f"Graph anomalies: {graph_anomaly_count}")
    print(
        f"Graph anomaly rate: {graph_anomaly_rate:.2f}%"
    )
    print(
        f"Detection method: {detection_method}"
    )

    if common_id is not None:

        print(
            f"Comparison identifier: {common_id}"
        )

        print(
            f"Temporal anomalies: "
            f"{temporal_anomaly_count}"
        )

        print(
            f"Temporal/graph overlap: "
            f"{overlap_count}"
        )

        print(
            f"Graph-only anomalies: "
            f"{graph_only_count}"
        )

    print("")
    print(f"CSV output: {OUTPUT_CSV}")
    print(f"Report output: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()