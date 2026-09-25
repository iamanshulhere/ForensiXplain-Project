import networkx as nx
import pandas as pd

from src.anomaly.graph_features import (
    safe_int,
    get_process_nodes,
    extract_process_graph_features,
    combine_with_temporal_features,
)


def test_safe_int_converts_valid_value():
    assert safe_int("812") == 812
    assert safe_int(812.0) == 812


def test_safe_int_returns_zero_for_invalid_value():
    assert safe_int(None) == 0
    assert safe_int("invalid") == 0


def test_get_process_nodes_returns_only_process_nodes():
    graph = nx.MultiDiGraph()

    graph.add_node(
        "process:100",
        node_type="process",
        pid="100",
        name="test.exe",
    )

    graph.add_node(
        "command:CMD001",
        node_type="command_line",
        command_line="whoami",
    )

    graph.add_node(
        "module:MOD001",
        node_type="module",
        name="kernel32.dll",
    )

    assert get_process_nodes(graph) == ["process:100"]


def test_extract_process_graph_features_counts_relationships():
    graph = nx.MultiDiGraph()

    graph.add_node(
        "process:100",
        node_type="process",
        pid="100",
        name="parent.exe",
    )

    graph.add_node(
        "process:200",
        node_type="process",
        pid="200",
        name="child.exe",
    )

    graph.add_node(
        "command:CMD001",
        node_type="command_line",
        command_line="child.exe /c test",
    )

    graph.add_node(
        "module:MOD001",
        node_type="module",
        name="test.dll",
    )

    graph.add_node(
        "memory_region:MEM001",
        node_type="memory_region",
    )

    graph.add_edge(
        "process:100",
        "process:200",
        relationship="parent_of",
    )

    graph.add_edge(
        "process:200",
        "command:CMD001",
        relationship="has_command_line",
    )

    graph.add_edge(
        "process:200",
        "module:MOD001",
        relationship="loaded_module",
    )

    graph.add_edge(
        "process:200",
        "memory_region:MEM001",
        relationship="has_memory_region",
    )

    features = extract_process_graph_features(graph)

    row = features.loc[
        features["process_node"] == "process:200"
    ].iloc[0]

    assert row["process_id"] == 200
    assert row["process"] == "child.exe"

    assert row["parent_count"] == 1
    assert row["child_count"] == 0

    assert row["command_line_count"] == 1
    assert row["module_count"] == 1
    assert row["memory_region_count"] == 1

    assert row["in_degree"] == 1
    assert row["out_degree"] == 3
    assert row["graph_degree"] == 4

    assert row["relationship_type_count"] == 4


def test_extract_process_graph_features_handles_isolated_process():
    graph = nx.MultiDiGraph()

    graph.add_node(
        "process:100",
        node_type="process",
        pid="100",
        name="isolated.exe",
    )

    features = extract_process_graph_features(graph)

    assert len(features) == 1

    row = features.iloc[0]

    assert row["process_id"] == 100
    assert row["parent_count"] == 0
    assert row["child_count"] == 0
    assert row["graph_degree"] == 0
    assert row["in_degree"] == 0
    assert row["out_degree"] == 0
    assert row["command_line_count"] == 0
    assert row["module_count"] == 0
    assert row["memory_region_count"] == 0
    assert row["relationship_type_count"] == 0


def test_extract_process_graph_features_ignores_non_process_nodes():
    graph = nx.MultiDiGraph()

    graph.add_node(
        "command:CMD001",
        node_type="command_line",
        command_line="whoami",
    )

    graph.add_node(
        "module:MOD001",
        node_type="module",
        name="test.dll",
    )

    features = extract_process_graph_features(graph)

    assert features.empty


def test_combine_with_temporal_features_matches_by_process_id():
    graph_features = pd.DataFrame([
        {
            "process_node": "process:100",
            "process_id": 100,
            "process": "test.exe",
            "parent_count": 1,
            "child_count": 2,
            "graph_degree": 3,
            "in_degree": 1,
            "out_degree": 2,
            "command_line_count": 1,
            "module_count": 1,
            "memory_region_count": 0,
            "relationship_type_count": 3,
        }
    ])

    temporal_features = pd.DataFrame([
        {
            "process_id": 100,
            "process": "test.exe",
            "temporal_sequence": 1,
        }
    ])

    combined = combine_with_temporal_features(
        graph_features,
        temporal_features,
    )

    row = combined.iloc[0]

    assert row["process_id"] == 100
    assert row["parent_count"] == 1
    assert row["child_count"] == 2
    assert row["graph_degree"] == 3
    assert row["graph_node_available"] == 1


def test_combine_with_temporal_features_fills_missing_graph_counts():
    graph_features = pd.DataFrame([
        {
            "process_node": "process:100",
            "process_id": 100,
            "process": "test.exe",
            "parent_count": 1,
            "child_count": 2,
            "graph_degree": 3,
            "in_degree": 1,
            "out_degree": 2,
            "command_line_count": 1,
            "module_count": 1,
            "memory_region_count": 0,
            "relationship_type_count": 3,
        }
    ])

    temporal_features = pd.DataFrame([
        {
            "process_id": 999,
            "process": "missing.exe",
            "temporal_sequence": 2,
        }
    ])

    combined = combine_with_temporal_features(
        graph_features,
        temporal_features,
    )

    row = combined.iloc[0]

    assert row["process_id"] == 999

    assert row["parent_count"] == 0
    assert row["child_count"] == 0
    assert row["graph_degree"] == 0
    assert row["in_degree"] == 0
    assert row["out_degree"] == 0
    assert row["command_line_count"] == 0
    assert row["module_count"] == 0
    assert row["memory_region_count"] == 0
    assert row["relationship_type_count"] == 0

    assert row["graph_node_available"] == 0


def test_combine_with_temporal_features_normalizes_process_ids():
    graph_features = pd.DataFrame([
        {
            "process_node": "process:100",
            "process_id": "100",
            "process": "test.exe",
            "parent_count": 1,
            "child_count": 0,
            "graph_degree": 1,
            "in_degree": 0,
            "out_degree": 1,
            "command_line_count": 1,
            "module_count": 0,
            "memory_region_count": 0,
            "relationship_type_count": 1,
        }
    ])

    temporal_features = pd.DataFrame([
        {
            "process_id": "100.0",
            "process": "test.exe",
            "temporal_sequence": 1,
        }
    ])

    combined = combine_with_temporal_features(
        graph_features,
        temporal_features,
    )

    assert combined.iloc[0]["graph_degree"] == 1
    assert combined.iloc[0]["graph_node_available"] == 1