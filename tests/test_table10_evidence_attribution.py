from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results" / "M57-Jean"


def evidence_id_count(series):
    return series.fillna("").apply(
        lambda value: 0 if not value else len(str(value).split(";"))
    )


def test_table10_evidence_attribution():
    fused = pd.read_csv(
        RESULTS_DIR / "evidence_attribution.csv"
    )
    graph = pd.read_csv(
        RESULTS_DIR / "graph_evidence_attribution.csv"
    )
    temporal = pd.read_csv(
        RESULTS_DIR / "temporal_evidence_attribution.csv"
    )

    fused_ids = evidence_id_count(fused["evidence_ids"])
    graph_ids = evidence_id_count(graph["evidence_ids"])

    assert len(fused) == 16
    assert len(graph) == 5
    assert len(temporal) == 5

    assert (fused_ids > 0).sum() == 16
    assert (graph_ids > 0).sum() == 5
    assert (temporal["evidence_id_count"] > 0).sum() == 5

    assert fused_ids.mean() == 54.6875
    assert graph_ids.mean() == 118.8
    assert temporal["evidence_id_count"].mean() == 10.8

    assert fused_ids.median() == 22
    assert graph_ids.median() == 115
    assert temporal["evidence_id_count"].median() == 2

    assert (fused["pslist_count"] > 0).sum() == 16
    assert (fused["pstree_count"] > 0).sum() == 16
    assert (fused["cmdline_count"] > 0).sum() == 8
    assert (fused["dlllist_count"] > 0).sum() == 9
    assert (fused["malfind_count"] > 0).sum() == 7

    graph_artifacts = graph["evidence_by_artifact"].fillna("").str.lower()

    assert graph_artifacts.str.contains(
        r"\bpslist\s*:", regex=True
    ).sum() == 5
    assert graph_artifacts.str.contains(
        r"\bpstree\s*:", regex=True
    ).sum() == 5
    assert graph_artifacts.str.contains(
        r"\bcmdline\s*:", regex=True
    ).sum() == 4
    assert graph_artifacts.str.contains(
        r"\bdlllist\s*:", regex=True
    ).sum() == 4
    assert graph_artifacts.str.contains(
        r"\bmalfind\s*:", regex=True
    ).sum() == 4

    assert (temporal["pslist_count"] > 0).sum() == 5
    assert (temporal["pstree_count"] > 0).sum() == 5
    assert (temporal["cmdline_count"] > 0).sum() == 2
    assert (temporal["dlllist_count"] > 0).sum() == 2
    assert (temporal["malfind_count"] > 0).sum() == 1
