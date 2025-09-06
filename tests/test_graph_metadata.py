import math
import pytest
import csv
import pathlib
import networkx as nx
from project import t1_graph_utils as t1

DATASET_DIR = pathlib.Path(__file__).parent / "datasets"


def helper_get_meta_directly_from_csv(csv_path):
    vertices_set = set()
    edges_count = 0
    with open(csv_path, newline="") as f:
        reader = csv.reader(f, delimiter=" ")
        for row in reader:
            source = int(row[0])
            target = int(row[1])
            vertices_set.update([source, target])
            edges_count += 1

    expected_vertices = len(vertices_set)
    expected_edges = edges_count

    return expected_edges, expected_vertices


def test_empty_graph():
    meta = t1.get_graph_md_from_loc_csv(str(DATASET_DIR / "empty_graph.csv"))

    assert meta.vertices_num == 0
    assert meta.edges_num == 0
    assert meta.labels == []


@pytest.mark.parametrize("filename",
                         ["regular_graph_1.csv",
                          "regular_graph_2.csv",
                          "regular_graph_3.csv"])
def test_regular_graph(filename):
    filepath = str(DATASET_DIR / filename)
    direct_meta = helper_get_meta_directly_from_csv(filepath)
    meta = t1.get_graph_md_from_loc_csv(str(DATASET_DIR / filename))

    assert meta.edges_num == direct_meta[0]
    assert meta.vertices_num == direct_meta[1]
    assert any(meta.labels)


def test_unknown_graph_name():
    with pytest.raises(FileNotFoundError):
        t1.get_graph_md_from_loc_csv("not_existing.csv")


def test_no_labels_graph():
    filepath = str(DATASET_DIR / "no_labels_graph.csv")
    meta = t1.get_graph_md_from_loc_csv(filepath)

    assert meta.vertices_num > 0
    assert meta.edges_num > 0
    assert all(math.isnan(list(label.values())[0]) for label in meta.labels)


@pytest.mark.parametrize("graph_name", [
    "skos",
    "wc",
    "travel",
    "atom",
    "biomedical"
])
def test_get_cfpq_graph_by_name(graph_name):
    graph = t1.get_cfpq_graph_by_name(graph_name)

    assert isinstance(graph, nx.MultiDiGraph)

    assert graph.number_of_nodes() > 0
    assert graph.number_of_edges() > 0


def test_get_cfpq_graph_by_name_invalid_name():
    with pytest.raises(FileNotFoundError):
        t1.get_cfpq_graph_by_name("non_existing_graph_xyz")
