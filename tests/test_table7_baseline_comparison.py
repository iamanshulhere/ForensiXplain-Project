import numpy as np
import pandas as pd

from src.evaluation.generate_table7 import (
    BASELINE_PATH,
    FEATURES_PATH,
    GRAPH_PATH,
    OUTPUT_PATH,
    REPORT_PATH,
    TEMPORAL_PATH,
    TOP_K,
    build_table7,
    select_top_k_events,
    write_table7_outputs,
)


EXPECTED_METHODS = [
    "Isolation Forest baseline",
    "Temporal Isolation Forest",
    "Graph-aware Isolation Forest",
    "Temporal + Graph fused",
]


def test_table7_inputs_exist_and_event_cohort_is_present():
    for path in [BASELINE_PATH, TEMPORAL_PATH, GRAPH_PATH, FEATURES_PATH]:
        assert path.exists()

    table = build_table7()
    assert table["events_evaluated"].tolist() == [46] * 4
    for event_list in table["top_k_event_ids"]:
        ids = event_list.split(";")
        assert len(ids) == TOP_K
        assert len(ids) == len(set(ids))
        assert all(event_id.startswith("LEVT-M57-Jean-") for event_id in ids)


def test_top_k_selection_is_reproducible_and_breaks_ties_by_event_id():
    frame = pd.DataFrame(
        {
            "logical_event_id": ["C", "A", "B"],
            "score": [0.5, 0.5, 0.9],
        }
    )

    first = select_top_k_events(frame, "score", k=2)
    second = select_top_k_events(frame, "score", k=2)

    assert first == second == ["B", "A"]


def test_table7_is_deterministic_and_has_expected_method_rows():
    first = build_table7()
    second = build_table7()

    pd.testing.assert_frame_equal(first, second)
    assert first["method"].tolist() == EXPECTED_METHODS
    assert first["top_k"].tolist() == [TOP_K] * 4
    assert first["evaluation_type"].eq(
        "Unsupervised proxy (rank overlap; no ground truth)"
    ).all()
    assert first.loc[
        first["method"] == "Temporal + Graph fused", "native_flagged_anomalies"
    ].item() == "N/A (rank selection only)"


def test_top_k_overlap_and_unique_selection_counts_are_valid():
    table = build_table7()
    overlap_columns = [
        "top_k_overlap_isolation_forest",
        "top_k_overlap_temporal",
        "top_k_overlap_graph",
        "top_k_overlap_fused",
    ]

    assert table[overlap_columns].ge(0).all().all()
    assert table[overlap_columns].le(TOP_K).all().all()
    assert table["events_selected_only_by_method_in_top_k"].between(0, TOP_K).all()
    assert table["top_k_overlap_isolation_forest"].iloc[0] == TOP_K
    assert table["top_k_overlap_temporal"].iloc[1] == TOP_K
    assert table["top_k_overlap_graph"].iloc[2] == TOP_K
    assert table["top_k_overlap_fused"].iloc[3] == TOP_K


def test_output_files_are_written_without_unexpected_nan_or_infinity():
    table = build_table7()

    write_table7_outputs(table, OUTPUT_PATH, REPORT_PATH)

    assert OUTPUT_PATH.exists()
    assert REPORT_PATH.exists()
    written = pd.read_csv(OUTPUT_PATH)
    assert written.shape == table.shape
    assert not written.isna().any().any()
    numeric_columns = [
        "events_evaluated",
        "top_k",
        "events_selected_only_by_method_in_top_k",
        "top_k_overlap_isolation_forest",
        "top_k_overlap_temporal",
        "top_k_overlap_graph",
        "top_k_overlap_fused",
    ]
    for column in numeric_columns:
        assert np.isfinite(pd.to_numeric(written[column]).to_numpy()).all()

    report = REPORT_PATH.read_text(encoding="utf-8")
    assert "does not measure detection accuracy" in report
    assert "unnormalized score-scale behavior" in report

