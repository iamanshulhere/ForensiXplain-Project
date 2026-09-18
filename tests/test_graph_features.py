import networkx as nx
import pandas as pd

from src.anomaly import graph_features as gf


def make_process_graph():
    graph = nx.MultiDiGraph()

    graph.add_node(
        "process:100",
        node_type="process",
        pid="100",
        name="child.exe",
    )

    graph.add_node(
        "process:50",
        node_type="process",
        pid="50",
        name="parent.exe",
    )

    graph.add_node(
        "process:75",
        node_type="process",
        pid="75",
        name="parent2.exe",
    )

    graph.add_node(
        "command:C1",
        node_type="command_line",
    )

    graph.add_node(
        "module:M1",
        node_type="module",
    )

    graph.add_node(
        "module:M2",
        node_type="module",
    )

    graph.add_node(
        "memory_region:R1",
        node_type="memory_region",
    )

    graph.add_edge(
        "process:50",
        "process:100",
        relationship="parent_of",
    )

    graph.add_edge(
        "process:75",
        "process:100",
        relationship="parent_of",
    )

    graph.add_edge(
        "process:100",
        "process:200",
        relationship="parent_of",
    )

    graph.add_edge(
        "process:100",
        "command:C1",
        relationship="has_command_line",
    )

    graph.add_edge(
        "process:100",
        "module:M1",
        relationship="loaded_module",
    )

    graph.add_edge(
        "process:100",
        "module:M2",
        relationship="loaded_module",
    )

    graph.add_edge(
        "process:100",
        "memory_region:R1",
        relationship="has_memory_region",
    )

    return graph


def test_safe_int_valid_integer():
    assert gf.safe_int(10) == 10


def test_safe_int_float_value():
    assert gf.safe_int(812.0) == 812


def test_safe_int_invalid_value():
    assert gf.safe_int("abc") == 0


def test_safe_int_missing_value():
    assert gf.safe_int(None) == 0


def test_get_process_nodes_returns_only_process_nodes():
    graph = make_process_graph()

    nodes = gf.get_process_nodes(graph)

    assert set(nodes) == {
        "process:50",
        "process:75",
        "process:100",
    }


def test_extract_process_graph_features_creates_process_rows():
    graph = make_process_graph()

    result = gf.extract_process_graph_features(graph)

    assert len(result) == 3
    assert set(result["process_id"]) == {50, 75, 100}


def test_parent_count():
    graph = make_process_graph()

    result = gf.extract_process_graph_features(graph)
    row = result[result["process_id"] == 100].iloc[0]

    assert row["parent_count"] == 2


def test_child_count():
    graph = make_process_graph()

    result = gf.extract_process_graph_features(graph)
    row = result[result["process_id"] == 100].iloc[0]

    assert row["child_count"] == 1


def test_in_and_out_degree():
    graph = make_process_graph()

    result = gf.extract_process_graph_features(graph)
    row = result[result["process_id"] == 100].iloc[0]

    assert row["in_degree"] == 2
    assert row["out_degree"] == 5


def test_total_graph_degree():
    graph = make_process_graph()

    result = gf.extract_process_graph_features(graph)
    row = result[result["process_id"] == 100].iloc[0]

    assert row["graph_degree"] == 7


def test_command_line_count():
    graph = make_process_graph()

    result = gf.extract_process_graph_features(graph)
    row = result[result["process_id"] == 100].iloc[0]

    assert row["command_line_count"] == 1


def test_module_count():
    graph = make_process_graph()

    result = gf.extract_process_graph_features(graph)
    row = result[result["process_id"] == 100].iloc[0]

    assert row["module_count"] == 2


def test_memory_region_count():
    graph = make_process_graph()

    result = gf.extract_process_graph_features(graph)
    row = result[result["process_id"] == 100].iloc[0]

    assert row["memory_region_count"] == 1


def test_relationship_type_count():
    graph = make_process_graph()

    result = gf.extract_process_graph_features(graph)
    row = result[result["process_id"] == 100].iloc[0]

    assert row["relationship_type_count"] == 4


def test_relationship_type_count_is_distinct():
    graph = nx.MultiDiGraph()

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

    graph.add_node(
        "module:M1",
        node_type="module",
    )

    graph.add_node(
        "module:M2",
        node_type="module",
    )

    graph.add_edge(
        "process:50",
        "process:100",
        relationship="parent_of",
    )

    graph.add_edge(
        "process:100",
        "module:M1",
        relationship="loaded_module",
    )

    graph.add_edge(
        "process:100",
        "module:M2",
        relationship="loaded_module",
    )

    result = gf.extract_process_graph_features(graph)
    row = result[result["process_id"] == 100].iloc[0]

    assert row["relationship_type_count"] == 2


def test_isolated_process_has_zero_graph_counts():
    graph = nx.MultiDiGraph()

    graph.add_node(
        "process:100",
        node_type="process",
        pid="100",
        name="isolated.exe",
    )

    result = gf.extract_process_graph_features(graph)
    row = result[result["process_id"] == 100].iloc[0]

    assert row["parent_count"] == 0
    assert row["child_count"] == 0
    assert row["graph_degree"] == 0
    assert row["in_degree"] == 0
    assert row["out_degree"] == 0
    assert row["command_line_count"] == 0
    assert row["module_count"] == 0
    assert row["memory_region_count"] == 0
    assert row["relationship_type_count"] == 0


def test_combine_with_temporal_features_joins_on_process_id():
    graph_features = pd.DataFrame(
        [
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
                "memory_region_count": 1,
                "relationship_type_count": 3,
            }
        ]
    )

    temporal_features = pd.DataFrame(
        [
            {
                "logical_event_id": "LEVT-1",
                "process_id": 100,
                "gap_log_seconds": 2.0,
            }
        ]
    )

    result = gf.combine_with_temporal_features(
        graph_features,
        temporal_features,
    )

    assert len(result) == 1
    assert result.iloc[0]["parent_count"] == 1
    assert result.iloc[0]["graph_degree"] == 3
    assert result.iloc[0]["graph_node_available"] == 1


def test_combine_preserves_temporal_rows():
    graph_features = pd.DataFrame(
        [
            {
                "process_node": "process:100",
                "process_id": 100,
                "process": "test.exe",
                "parent_count": 1,
                "child_count": 0,
                "graph_degree": 1,
                "in_degree": 1,
                "out_degree": 0,
                "command_line_count": 0,
                "module_count": 0,
                "memory_region_count": 0,
                "relationship_type_count": 1,
            }
        ]
    )

    temporal_features = pd.DataFrame(
        [
            {
                "logical_event_id": "LEVT-1",
                "process_id": 100,
                "gap_log_seconds": 2.0,
            },
            {
                "logical_event_id": "LEVT-2",
                "process_id": 999,
                "gap_log_seconds": 3.0,
            },
        ]
    )

    result = gf.combine_with_temporal_features(
        graph_features,
        temporal_features,
    )

    assert len(result) == 2
    assert set(result["logical_event_id"]) == {"LEVT-1", "LEVT-2"}


def test_missing_graph_match_gets_zero_counts():
    graph_features = pd.DataFrame(
        [
            {
                "process_node": "process:100",
                "process_id": 100,
                "process": "test.exe",
                "parent_count": 1,
                "child_count": 0,
                "graph_degree": 1,
                "in_degree": 1,
                "out_degree": 0,
                "command_line_count": 0,
                "module_count": 0,
                "memory_region_count": 0,
                "relationship_type_count": 1,
            }
        ]
    )

    temporal_features = pd.DataFrame(
        [
            {
                "logical_event_id": "LEVT-999",
                "process_id": 999,
                "gap_log_seconds": 3.0,
            }
        ]
    )

    result = gf.combine_with_temporal_features(
        graph_features,
        temporal_features,
    )

    row = result.iloc[0]

    assert row["graph_degree"] == 0
    assert row["parent_count"] == 0
    assert row["child_count"] == 0
    assert row["graph_node_available"] == 0