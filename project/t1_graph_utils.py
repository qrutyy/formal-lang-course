import cfpq_data as cd
import networkx as nx
import dataclasses
from typing import Tuple


@dataclasses.dataclass
class GraphMetadata:
    vertices_num: int
    edges_num: int
    labels: list


def get_graph_metadata(graph: nx.MultiDiGraph):
    labels = [d for _, _, d in graph.edges.data()]
    return GraphMetadata(graph.number_of_nodes, graph.number_of_edges, labels)


def get_graph_by_name(graph_name):
    path = cd.download(graph_name)
    return cd.graph_from_csv(path)


def get_graph_metadata_by_name(graph_name):
    graph = get_graph_by_name(graph_name)
    return get_graph_metadata(graph)


def get_graph_md_from_loc_csv(graph_name):
    graph = cd.graph_from_csv(graph_name)
    return get_graph_metadata(graph)


def save_nx_graph_to_dot(nx_graph: nx.MultiDiGraph, filename):
    dot_graph = nx.drawing.nx_pydot.to_pydot(nx_graph).create_dot()
    dot_graph.write_dot(filename)


def create_and_save_two_cyclic_graph(
        cycle_sizes: Tuple[int, int],
        labels: Tuple[str, str],
        filename
):
    graph = cd.labeled_two_cycles_graph(cycle_sizes[0], cycle_sizes[1],
                                        0, labels)
    save_nx_graph_to_dot(graph, filename)
