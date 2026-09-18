import pandas as pd
import networkx as nx

from src.graph import temporal_graph as tg


def make_row(**overrides):
    """Create a synthetic forensic-event row with safe defaults."""
    row = {
        "process_id": "100",
        "parent_process_id": "1",
        "process": "test.exe",
        "timestamp": "2009-11-23 12:00:00+00:00",
        "timestamp_confidence": "high",
        "evidence_id": "E1",
        "source": "test",
        "artifact_type": "pslist",
        "provenance": "synthetic",
        "command_line": "test.exe -arg",
        "file": "test.dll",
        "file_path": r"C:\test\test.dll",
        "event_type": "process",
    }
    row.update(overrides)
    return row


def test_clean_value():
    assert tg.clean_value(None) == ""
    assert tg.clean_value(float("nan")) == ""
    assert tg.clean_value("  test.exe  ") == "  test.exe  "


def test_add_process_node_creates_normalized_process_node():
    graph = nx.MultiDiGraph()
    row = make_row(process_id="812.0", process="notepad.exe")

    node = tg.add_process_node(graph, row)

    assert node == "process:812"
    assert node in graph
    assert graph.nodes[node]["node_type"] == "process"
    assert graph.nodes[node]["pid"] == "812"
    assert graph.nodes[node]["name"] == "notepad.exe"


def test_add_process_node_preserves_process_metadata():
    graph = nx.MultiDiGraph()
    row = make_row(
        process_id="100",
        process="test.exe",
        timestamp="2009-11-23 12:30:00+00:00",
        timestamp_confidence="high",
        evidence_id="PROC-1",
    )

    tg.add_process_node(graph, row)

    data = graph.nodes["process:100"]

    assert data["name"] == "test.exe"
    assert data["create_time"] == "2009-11-23 12:30:00+00:00"
    assert data["create_time_confidence"] == "high"
    assert data["process_evidence_id"] == "PROC-1"


def test_add_process_node_ignores_missing_pid():
    graph = nx.MultiDiGraph()
    row = make_row(process_id="")

    result = tg.add_process_node(graph, row)

    assert result is None
    assert graph.number_of_nodes() == 0


def test_add_process_relationship_creates_parent_child_edge():
    graph = nx.MultiDiGraph()
    row = make_row(
        process_id="100",
        parent_process_id="50",
        process="child.exe",
        event_type="process_relationship",
        evidence_id="REL-1",
    )

    tg.add_process_relationship(graph, row)

    assert "process:50" in graph
    assert "process:100" in graph

    edges = list(graph.edges(data=True))
    assert len(edges) == 1

    source, target, data = edges[0]

    assert source == "process:50"
    assert target == "process:100"
    assert data["relationship"] == "parent_of"
    assert data["evidence_id"] == "REL-1"


def test_add_process_relationship_ignores_missing_or_invalid_pid():
    graph = nx.MultiDiGraph()

    tg.add_process_relationship(
        graph,
        make_row(process_id="", parent_process_id="50")
    )

    tg.add_process_relationship(
        graph,
        make_row(process_id="abc", parent_process_id="50")
    )

    tg.add_process_relationship(
        graph,
        make_row(process_id="100", parent_process_id="xyz")
    )

    assert graph.number_of_nodes() == 0
    assert graph.number_of_edges() == 0


def test_add_command_line_creates_command_node_and_edge():
    graph = nx.MultiDiGraph()
    row = make_row(
        process_id="100",
        process="cmd.exe",
        command_line="cmd.exe /c whoami",
        evidence_id="CMD-1",
        event_type="command_line",
        artifact_type="cmdline",
    )

    tg.add_command_line(graph, row)

    assert "process:100" in graph
    assert "command:CMD-1" in graph

    command_data = graph.nodes["command:CMD-1"]

    assert command_data["node_type"] == "command_line"
    assert command_data["command_line"] == "cmd.exe /c whoami"
    assert command_data["evidence_id"] == "CMD-1"

    edge_data = graph["process:100"]["command:CMD-1"][0]

    assert edge_data["relationship"] == "has_command_line"


def test_add_module_creates_module_node_and_edge():
    graph = nx.MultiDiGraph()
    row = make_row(
        process_id="100",
        process="test.exe",
        file="kernel32.dll",
        file_path=r"C:\Windows\System32\kernel32.dll",
        evidence_id="DLL-1",
        event_type="module",
        artifact_type="dlllist",
    )

    tg.add_module(graph, row)

    assert "process:100" in graph
    assert "module:DLL-1" in graph

    module_data = graph.nodes["module:DLL-1"]

    assert module_data["node_type"] == "module"
    assert module_data["name"] == "kernel32.dll"
    assert module_data["path"] == r"C:\Windows\System32\kernel32.dll"

    edge_data = graph["process:100"]["module:DLL-1"][0]

    assert edge_data["relationship"] == "loaded_module"


def test_add_memory_region_creates_memory_node_and_edge():
    graph = nx.MultiDiGraph()
    row = make_row(
        process_id="100",
        process="test.exe",
        evidence_id="MAL-1",
        event_type="memory_region",
        artifact_type="malfind",
    )

    tg.add_memory_region(graph, row)

    assert "process:100" in graph
    assert "memory_region:MAL-1" in graph

    memory_data = graph.nodes["memory_region:MAL-1"]

    assert memory_data["node_type"] == "memory_region"
    assert memory_data["evidence_id"] == "MAL-1"

    edge_data = graph["process:100"]["memory_region:MAL-1"][0]

    assert edge_data["relationship"] == "has_memory_region"


def test_build_graph_dispatches_all_supported_event_types():
    events = pd.DataFrame(
        [
            make_row(
                event_type="process",
                process_id="100",
                parent_process_id="1",
                evidence_id="P-1",
            ),
            make_row(
                event_type="process_relationship",
                process_id="100",
                parent_process_id="1",
                evidence_id="R-1",
            ),
            make_row(
                event_type="command_line",
                process_id="100",
                evidence_id="C-1",
                command_line="test.exe -x",
            ),
            make_row(
                event_type="module",
                process_id="100",
                evidence_id="M-1",
                file="test.dll",
                file_path=r"C:\test\test.dll",
            ),
            make_row(
                event_type="memory_region",
                process_id="100",
                evidence_id="MEM-1",
            ),
            make_row(
                event_type="unknown",
                process_id="100",
                evidence_id="U-1",
            ),
        ]
    )

    graph = tg.build_graph(events)

    assert isinstance(graph, nx.MultiDiGraph)

    assert "process:1" in graph
    assert "process:100" in graph
    assert "command:C-1" in graph
    assert "module:M-1" in graph
    assert "memory_region:MEM-1" in graph

    assert graph.number_of_edges() == 4

    relationships = {
        data["relationship"]
        for _, _, data in graph.edges(data=True)
    }

    assert relationships == {
        "parent_of",
        "has_command_line",
        "loaded_module",
        "has_memory_region",
    }


def test_build_graph_preserves_multiple_relationships_in_multidigraph():
    events = pd.DataFrame(
        [
            make_row(
                event_type="process_relationship",
                process_id="100",
                parent_process_id="50",
                evidence_id="REL-1",
            ),
            make_row(
                event_type="process_relationship",
                process_id="100",
                parent_process_id="50",
                evidence_id="REL-2",
            ),
        ]
    )

    graph = tg.build_graph(events)

    assert isinstance(graph, nx.MultiDiGraph)
    assert graph.number_of_edges() == 2
    assert graph.number_of_edges("process:50", "process:100") == 2


def test_make_graphml_safe_replaces_none_attributes():
    graph = nx.MultiDiGraph()

    graph.add_node(
        "process:100",
        node_type="process",
        pid="100",
        optional_value=None,
    )

    graph.add_node("process:50", node_type="process", pid="50")

    graph.add_edge(
        "process:50",
        "process:100",
        relationship="parent_of",
        optional_value=None,
    )

    tg.make_graphml_safe(graph)

    assert graph.nodes["process:100"]["optional_value"] == ""

    edge_data = graph["process:50"]["process:100"][0]
    assert edge_data["optional_value"] == ""


def test_save_graph_writes_reloadable_graphml(tmp_path, monkeypatch):
    graph = nx.MultiDiGraph(case_id="M57-Jean")

    graph.add_node(
        "process:100",
        node_type="process",
        pid="100",
        name="test.exe",
    )

    graph.add_node(
        "process:50",
        node_type="process",
        pid="50",
        name="parent.exe",
    )

    graph.add_edge(
        "process:50",
        "process:100",
        relationship="parent_of",
        evidence_id="REL-1",
    )

    output_file = tmp_path / "test_graph.graphml"
    monkeypatch.setattr(tg, "OUTPUT_PATH", output_file)

    tg.save_graph(graph)

    assert output_file.exists()

    loaded = nx.read_graphml(output_file)

    assert "process:100" in loaded
    assert "process:50" in loaded
    assert loaded.number_of_nodes() == 2
    assert loaded.number_of_edges() == 1

    edge_data = list(loaded.edges(data=True))[0][2]
    assert edge_data["relationship"] == "parent_of"
    assert edge_data["evidence_id"] == "REL-1"