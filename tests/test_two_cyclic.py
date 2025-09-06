import os
import pathlib
import pytest
import networkx as nx
from project import t1_graph_utils as t1

TMP_DATASET_DIR = pathlib.Path(__file__).parent / "datasets/tmp"


@pytest.fixture
def tmp_dataset_dir():
    try:
        TMP_DATASET_DIR.mkdir()
    except OSError:
        print("Creation of the directory %s failed" % TMP_DATASET_DIR)
    else:
        print("Successfully created the directory %s " % TMP_DATASET_DIR)


def test_save_nx_graph_to_dot_creates_file():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(1, 2, label="b")

    filename = TMP_DATASET_DIR / "graph.dot"
    t1.save_nx_graph_to_dot(graph, str(filename))

    assert filename.exists()
    content = filename.read_text()
    assert "digraph" in content or "graph" in content

    loaded_graph = t1.read_graph_from_dot(filename)
    assert loaded_graph.number_of_nodes() == 3
    assert loaded_graph.number_of_edges() == 2
    edge_labels = [d.get("label") for _, _, d in loaded_graph.edges(data=True)]
    assert "a" in edge_labels
    assert "b" in edge_labels


def test_create_and_save_two_cyclic_graph():
    filename = TMP_DATASET_DIR / "two_cycles.dot"
    cycle_sizes = (3, 3)
    labels = ("x", "y")

    t1.create_and_save_two_cyclic_graph(cycle_sizes, labels, str(filename))

    assert filename.exists()

    loaded_graph = t1.read_graph_from_dot(filename)
    total_nodes = sum(cycle_sizes)
    assert loaded_graph.number_of_nodes() == total_nodes + 1
    edge_labels = [d.get("label") for _, _, d in loaded_graph.edges(data=True)]
    for label in labels:
        assert label in edge_labels
