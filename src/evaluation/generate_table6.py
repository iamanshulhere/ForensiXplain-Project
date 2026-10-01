"""Generate the Temporal/Graph fusion proxy evaluation for M57-Jean."""

from pathlib import Path

import pandas as pd

from src.evaluation.temporal_graph_fusion import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_RANDOM_TRIALS,
    DEFAULT_TOP_K,
    build_fused_frame,
    calculate_overlaps,
    random_baseline,
    select_top_k,
)


TEMPORAL_PATH = Path("results/M57-Jean/temporal_anomalies.csv")
GRAPH_PATH = Path("results/M57-Jean/graph_anomalies.csv")
OUTPUT_PATH = Path("results/M57-Jean/table6_temporal_graph_fusion.csv")

FUSION_WEIGHT = 0.5


def main() -> None:
    frame = build_fused_frame(
        str(TEMPORAL_PATH),
        str(GRAPH_PATH),
        temporal_weight=FUSION_WEIGHT,
    )

    temporal_events = set(
        frame.loc[
            frame["temporal_predicted_anomaly"],
            "logical_event_id",
        ]
    )

    graph_events = set(
        frame.loc[
            frame["graph_predicted_anomaly"],
            "logical_event_id",
        ]
    )

    fused_events = select_top_k(
        frame,
        "fused_score",
        k=DEFAULT_TOP_K,
    )

    overlap_stats = calculate_overlaps(
        temporal_events,
        graph_events,
        fused_events,
    )

    random_stats = random_baseline(
        frame["logical_event_id"],
        temporal_events,
        graph_events,
        fused_events,
        k=DEFAULT_TOP_K,
        trials=DEFAULT_RANDOM_TRIALS,
        random_state=DEFAULT_RANDOM_STATE,
    )

    rows = [
        {
            "method": "Temporal-only",
            "selected_anomalies": overlap_stats["temporal_count"],
            "temporal_overlap": "-",
            "graph_overlap": overlap_stats["temporal_graph_overlap"],
            "fused_overlap": overlap_stats["fused_temporal_overlap"],
            "unique_findings": (
                overlap_stats["temporal_count"]
                - overlap_stats["temporal_graph_overlap"]
            ),
            "evaluation_type": "Unsupervised proxy",
        },
        {
            "method": "Graph-only",
            "selected_anomalies": overlap_stats["graph_count"],
            "temporal_overlap": overlap_stats["temporal_graph_overlap"],
            "graph_overlap": "-",
            "fused_overlap": overlap_stats["fused_graph_overlap"],
            "unique_findings": (
                overlap_stats["graph_count"]
                - overlap_stats["temporal_graph_overlap"]
            ),
            "evaluation_type": "Unsupervised proxy",
        },
        {
            "method": "Temporal + Graph fused",
            "selected_anomalies": overlap_stats["fused_count"],
            "temporal_overlap": overlap_stats["fused_temporal_overlap"],
            "graph_overlap": overlap_stats["fused_graph_overlap"],
            "fused_overlap": "-",
            "unique_findings": overlap_stats["fused_unique_findings"],
            "evaluation_type": "Unsupervised proxy",
        },
        {
            "method": "Random baseline",
            "selected_anomalies": DEFAULT_TOP_K,
            "temporal_overlap": (
                f'{random_stats["random_temporal_overlap_mean"]:.3f}'
                " +/- "
                f'{random_stats["random_temporal_overlap_std"]:.3f}'
            ),
            "graph_overlap": (
                f'{random_stats["random_graph_overlap_mean"]:.3f}'
                " +/- "
                f'{random_stats["random_graph_overlap_std"]:.3f}'
            ),
            "fused_overlap": (
                f'{random_stats["random_fused_overlap_mean"]:.3f}'
                " +/- "
                f'{random_stats["random_fused_overlap_std"]:.3f}'
            ),
            "unique_findings": "N/A",
            "evaluation_type": "Random reference",
        },
    ]

    table = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_PATH, index=False)

    print("=== Table 6 Temporal-Graph Fusion Evaluation ===")
    print(f"Events evaluated: {len(frame)}")
    print(f"Fusion weights: temporal={FUSION_WEIGHT:.1f}, graph={1.0 - FUSION_WEIGHT:.1f}")
    print(f"Top-k: {DEFAULT_TOP_K}")
    print(f"Random trials: {DEFAULT_RANDOM_TRIALS}")
    print(f"Random seed: {DEFAULT_RANDOM_STATE}")
    print()
    print(table.to_string(index=False))
    print()
    print("Top fused events:")
    print(
        frame.head(DEFAULT_TOP_K)[
            [
                "process_id",
                "process",
                "temporal_anomaly_score",
                "graph_anomaly_score",
                "fused_score",
            ]
        ].to_string(index=False)
    )
    print()
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()


