import networkx as nx
import pandas as pd

from app import data_loader


def test_load_csv_reads_valid_file(tmp_path):
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame({"event_id": [1, 2], "event_type": ["start", "stop"]}).to_csv(
        csv_path, index=False
    )

    result = data_loader.load_csv(csv_path)

    assert len(result) == 2
    assert list(result.columns) == ["event_id", "event_type"]


def test_load_csv_returns_empty_dataframe_for_missing_file(tmp_path):
    result = data_loader.load_csv(tmp_path / "missing.csv")

    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_load_csv_returns_empty_dataframe_for_empty_file(tmp_path):
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("", encoding="utf-8")

    result = data_loader.load_csv(csv_path)

    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_load_m57_results_loads_existing_outputs_and_handles_missing_files(
    tmp_path, monkeypatch
):
    results_dir = tmp_path / "results"
    normalized_dir = tmp_path / "normalized"
    results_dir.mkdir()
    normalized_dir.mkdir()

    pd.DataFrame({"event_id": ["E1"], "score": [0.9]}).to_csv(
        results_dir / "graph_anomalies.csv", index=False
    )
    pd.DataFrame({"event_id": ["E1"], "timestamp": ["2026-01-01"]}).to_csv(
        normalized_dir / "logical_timeline.csv", index=False
    )

    monkeypatch.setattr(data_loader, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(data_loader, "NORMALIZED_DIR", normalized_dir)

    result = data_loader.load_m57_results()

    assert result["graph_anomalies"].shape[0] == 1
    assert result["timeline"].shape[0] == 1
    assert result["temporal_anomalies"].empty
    assert result["events"].empty


def test_load_process_graph_returns_none_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(data_loader, "NORMALIZED_DIR", tmp_path)

    assert data_loader.load_process_graph() is None


def test_load_process_graph_reads_saved_graphml(tmp_path, monkeypatch):
    graph = nx.MultiDiGraph()
    graph.add_node("process-1", type="process")
    graph.add_node("process-2", type="process")
    graph.add_edge("process-1", "process-2", type="parent_of")

    graph_path = tmp_path / "temporal_graph.graphml"
    nx.write_graphml(graph, graph_path)
    monkeypatch.setattr(data_loader, "NORMALIZED_DIR", tmp_path)

    result = data_loader.load_process_graph()

    assert result is not None
    assert result.number_of_nodes() == 2
    assert result.number_of_edges() == 1
