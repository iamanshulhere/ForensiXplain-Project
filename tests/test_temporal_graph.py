import pandas as pd

from src.graph.temporal_graph import (
    build_graph,
    clean_value,
)


def make_event(**overrides):
    row = {
        "process_id": "100",
        "parent_process_id": "",
        "process": "test.exe",
        "event_type": "process",
        "timestamp": "2026-01-01T10:00:00Z",
        "timestamp_confidence": "observed",
        "evidence_id": "E001",
        "source": "memory",
        "artifact_type": "test",
        "provenance": "test_provenance",
        "command_line": "",
        "file": "",
        "file_path": "",
    }
    row.update(overrides)
    return row


def test_clean_value_converts_nan_to_empty_string():
    assert clean_value(float("nan")) == ""


def test_clean_value_converts_value_to_string():
    assert clean_value(812.0) == "812.0"


def test_process_event_creates_process_node():
    events = pd.DataFrame([
        make_event(
            process_id="812",
            process="explorer.exe",
            event_type="process",
        )
    ])

    graph = build_graph(events)

    assert "process:812" in graph
    assert graph.nodes["process:812"]["node_type"] == "process"
    assert graph.nodes["process:812"]["pid"] == "812"
    assert graph.nodes["process:812"]["name"] == "explorer.exe"


def test_process_relationship_creates_parent_child_edge():
    events = pd.DataFrame([
        make_event(
            process_id="812",
            parent_process_id="400",
            process="child.exe",
            event_type="process_relationship",
            evidence_id="REL001",
        )
    ])

    graph = build_graph(events)

    assert "process:400" in graph
    assert "process:812" in graph

    assert graph.has_edge("process:400", "process:812")

    edge_data = next(iter(graph["process:400"]["process:812"].values()))

    assert edge_data["relationship"] == "parent_of"
    assert edge_data["evidence_id"] == "REL001"


def test_command_line_event_creates_command_node_and_edge():
    events = pd.DataFrame([
        make_event(
            process_id="812",
            process="cmd.exe",
            event_type="command_line",
            command_line="cmd.exe /c whoami",
            evidence_id="CMD001",
        )
    ])

    graph = build_graph(events)

    assert "process:812" in graph
    assert "command:CMD001" in graph

    assert graph.has_edge("process:812", "command:CMD001")

    edge_data = next(iter(graph["process:812"]["command:CMD001"].values()))

    assert edge_data["relationship"] == "has_command_line"


def test_module_event_creates_module_node_and_edge():
    events = pd.DataFrame([
        make_event(
            process_id="812",
            process="test.exe",
            event_type="module",
            file="kernel32.dll",
            file_path="C:\\Windows\\System32\\kernel32.dll",
            evidence_id="MOD001",
        )
    ])

    graph = build_graph(events)

    assert "process:812" in graph
    assert "module:MOD001" in graph

    assert graph.nodes["module:MOD001"]["node_type"] == "module"
    assert graph.nodes["module:MOD001"]["name"] == "kernel32.dll"

    assert graph.has_edge("process:812", "module:MOD001")

    edge_data = next(iter(graph["process:812"]["module:MOD001"].values()))

    assert edge_data["relationship"] == "loaded_module"


def test_memory_region_event_creates_memory_node_and_edge():
    events = pd.DataFrame([
        make_event(
            process_id="812",
            process="test.exe",
            event_type="memory_region",
            evidence_id="MEM001",
        )
    ])

    graph = build_graph(events)

    assert "process:812" in graph
    assert "memory_region:MEM001" in graph

    assert graph.nodes["memory_region:MEM001"]["node_type"] == "memory_region"

    assert graph.has_edge("process:812", "memory_region:MEM001")

    edge_data = next(iter(graph["process:812"]["memory_region:MEM001"].values()))

    assert edge_data["relationship"] == "has_memory_region"


def test_invalid_process_id_does_not_create_process_node():
    events = pd.DataFrame([
        make_event(
            process_id="",
            event_type="process",
        )
    ])

    graph = build_graph(events)

    assert graph.number_of_nodes() == 0
    assert graph.number_of_edges() == 0


def test_process_id_float_string_is_normalized():
    events = pd.DataFrame([
        make_event(
            process_id="812.0",
            process="explorer.exe",
            event_type="process",
        )
    ])

    graph = build_graph(events)

    assert "process:812" in graph
    assert graph.nodes["process:812"]["pid"] == "812"