from project import t2_fa_utils as t2
from project import t3_graph_fa as t3
from networkx import MultiDiGraph
import numpy as np


def ms_bfs_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    reg_dfa = t2.regex_to_dfa(regex)
    graph_nfa = t2.graph_to_nfa(graph, start_nodes, final_nodes)
    reg_am = t3.AdjacencyMatrixFA(reg_dfa)
    graph_am = t3.AdjacencyMatrixFA(graph_nfa)

    if reg_am.n_states == 0 or graph_am.n_states == 0:
        return set()

    intersection = t3.intersect_automata(reg_am, graph_am)
    if intersection.n_states == 0:
        return set()

    tc = intersection.get_trans_closure()

    start_indices = np.where(intersection.start_states)[0]
    final_indices = np.where(intersection.final_states)[0]
    result = set()

    for start_idx in start_indices:
        reachable_from_start = tc[start_idx, :].nonzero()[1]
        reachable_final_states = np.intersect1d(reachable_from_start, final_indices)

        for final_idx in reachable_final_states:
            start_state_pair = intersection.state_of_index[start_idx]
            final_state_pair = intersection.state_of_index[final_idx]

            _, graph_start_node = start_state_pair.value
            _, graph_final_node = final_state_pair.value

            result.add((graph_start_node, graph_final_node))

    return result
