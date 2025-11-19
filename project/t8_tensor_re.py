import networkx as nx
from pyformlang.cfg import CFG
from pyformlang.rsa import RecursiveAutomaton
from pyformlang.finite_automaton import State, NondeterministicFiniteAutomaton
from project.t2_fa_utils import graph_to_nfa
from project.t3_graph_fa import AdjacencyMatrixFA, intersect_automata
import scipy.sparse as scsp
import numpy as np


def cfg_to_rsm(cfg: CFG) -> RecursiveAutomaton:
    """Converts a CFG to a Recursive State Machine."""
    return RecursiveAutomaton.from_text(cfg.to_text())


def ebnf_to_rsm(ebnf: str) -> RecursiveAutomaton:
    """Converts an EBNF string to a Recursive State Machine."""
    return RecursiveAutomaton.from_text(ebnf)


def _build_rsm_fa(rsm: RecursiveAutomaton) -> NondeterministicFiniteAutomaton:
    """Builds a single NFA representation from an RSM."""
    rsm_nfa = NondeterministicFiniteAutomaton()
    for sym, box in rsm.boxes.items():
        # add all transitions, renaming states to be unique (Symbol, State) tuples
        for s_from, trans in box.dfa.to_dict().items():
            for label, s_to in trans.items():
                rsm_nfa.add_transition(
                    State((sym, s_from.value)), label, State((sym, s_to.value))
                )

        for start_state in box.start_state:
            rsm_nfa.add_start_state(State((sym, start_state.value)))
        for final_state in box.final_states:
            rsm_nfa.add_final_state(State((sym, final_state.value)))

    return rsm_nfa


def _define_start_states(
    intersection: AdjacencyMatrixFA, adj_rsm: AdjacencyMatrixFA
) -> set[State]:
    """Finds the start states in the intersection based on the RSM's start states."""
    rsm_starts = {adj_rsm.state_of_index[i] for i in np.where(adj_rsm.start_states)[0]}
    res = set()
    for st in intersection.states:
        # intersection state value is a tuple: (graph_state_val, rsm_state_val)
        _, rsm_st_val = st.value
        if State(rsm_st_val) in rsm_starts:
            res.add(st)
    return res


def _ms_bfs_with_paths(intersection: AdjacencyMatrixFA, adj_rsm: AdjacencyMatrixFA):
    """
    Performs a matrix-based Breadth-First Search to find reachability
    from the RSM's start states within the intersection automaton.
    """
    n = intersection.n_states
    if n == 0:
        return scsp.csr_matrix((0, 0), dtype=bool)

    reachability = scsp.lil_matrix((n, n), dtype=bool)
    start_states = _define_start_states(intersection, adj_rsm)
    for s in start_states:
        i = intersection.index_of_state[s]
        reachability[i, i] = True
    reachability = reachability.tocsr()

    prev_nnz = -1
    while reachability.nnz != prev_nnz:
        prev_nnz = reachability.nnz
        for mat in intersection.transitions.values():
            reachability += reachability @ mat

    return reachability


def _add_nonterms(
    reachability: scsp.spmatrix,
    adj_graph: AdjacencyMatrixFA,
    adj_rsm: AdjacencyMatrixFA,
    intersection: AdjacencyMatrixFA,
) -> bool:
    """Adds new edges to the graph automaton for each completed non-terminal path."""
    new_info_added = False

    rsm_starts = {adj_rsm.state_of_index[i] for i in np.where(adj_rsm.start_states)[0]}
    rsm_finals = {adj_rsm.state_of_index[i] for i in np.where(adj_rsm.final_states)[0]}

    reach_coo = reachability.tocoo()
    for i, j in zip(reach_coo.row, reach_coo.col):
        inter_start = intersection.state_of_index[i]
        inter_end = intersection.state_of_index[j]

        gr_start_val, rsm_start_val = inter_start.value
        gr_end_val, rsm_end_val = inter_end.value

        rsm_start_state = State(rsm_start_val)
        rsm_end_state = State(rsm_end_val)

        # check if the path corresponds to a full non-terminal production (S ->* w)
        if rsm_start_state in rsm_starts and rsm_end_state in rsm_finals:
            start_symbol, _ = rsm_start_val
            end_symbol, _ = rsm_end_val

            if start_symbol == end_symbol:
                non_terminal = start_symbol

                i_graph = adj_graph.index_of_state[State(gr_start_val)]
                j_graph = adj_graph.index_of_state[State(gr_end_val)]

                if non_terminal not in adj_graph.transitions:
                    adj_graph.transitions[non_terminal] = scsp.lil_matrix(
                        (adj_graph.n_states, adj_graph.n_states), dtype=bool
                    )
                    adj_graph.alphabet.add(non_terminal)

                if not adj_graph.transitions[non_terminal][i_graph, j_graph]:
                    adj_graph.transitions[non_terminal][i_graph, j_graph] = True
                    new_info_added = True

    return new_info_added


def tensor_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
    matrix_format="csr",
) -> set[tuple[int, int]]:
    """
    Performs context-free path querying using the tensor-based algorithm.
    """
    graph_nfa = graph_to_nfa(graph, start_nodes, final_nodes)
    adj_graph = AdjacencyMatrixFA(graph_nfa, matrix_format=matrix_format)
    rsm_fa = _build_rsm_fa(rsm)
    adj_rsm = AdjacencyMatrixFA(rsm_fa, matrix_format=matrix_format)

    info_added = True
    while info_added:
        # convert LIL matrices to CSR for efficient multiplication in the next intersection
        for label in adj_graph.transitions:
            adj_graph.transitions[label] = adj_graph.transitions[label].tocsr()

        intersection = intersect_automata(adj_graph, adj_rsm)
        if intersection.n_states == 0:
            break

        reachability = _ms_bfs_with_paths(intersection, adj_rsm)
        info_added = _add_nonterms(reachability, adj_graph, adj_rsm, intersection)

    result = set()
    initial_symbol = rsm.initial_label
    if initial_symbol in adj_graph.transitions:
        mat = adj_graph.transitions[initial_symbol].tocoo()
        for i, j in zip(mat.row, mat.col):
            s = adj_graph.state_of_index[i]
            f = adj_graph.state_of_index[j]

            is_start_node = not start_nodes or s.value in start_nodes
            is_final_node = not final_nodes or f.value in final_nodes

            if is_start_node and is_final_node:
                result.add((s.value, f.value))

    return result
