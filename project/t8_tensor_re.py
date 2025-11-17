import networkx as nx
import scipy.sparse as sp
from pyformlang.cfg import CFG
from pyformlang.rsa import RecursiveAutomaton, Box
from pyformlang.finite_automaton import State, NondeterministicFiniteAutomaton
from project.task2 import graph_to_nfa
from project.task3 import AdjacencyMatrixFA, intersect_automata


# -----------------------------
# Grammar-to-RSM transformations
# -----------------------------
def cfg_to_rsm(cfg: CFG) -> RecursiveAutomaton:
    """
    Convert a CFG to an RSM using textual representation.
    """
    return RecursiveAutomaton.from_text(cfg.to_text())


def ebnf_to_rsm(ebnf: str) -> RecursiveAutomaton:
    """
    Convert an EBNF string to an RSM.
    """
    return RecursiveAutomaton.from_text(ebnf)


def build_rsm_as_nfa(rsm: RecursiveAutomaton) -> NondeterministicFiniteAutomaton:
    """
    Unfold all RSM boxes into a single NFA with unique states.
    """
    transitions = []
    start_states = set()
    final_states = set()

    for sym in rsm.boxes:
        box: Box = rsm.get_box(sym)
        nx_graph = box.dfa.to_networkx()
        for u, v, data in nx_graph.edges(data=True):
            lbl = data.get("label")
            transitions.append((State((sym, u)), lbl, State((sym, v))))
        for s in box.start_state:
            start_states.add(State((sym, s.value)))
        for f in box.final_states:
            final_states.add(State((sym, f.value)))

    nfa = NondeterministicFiniteAutomaton(start_states, final_states)
    nfa.add_transitions(transitions)
    return nfa


# -----------------------------
# Core tensor-based CFPQ algorithm
# -----------------------------
def tensor_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
    matrix_format: str = "csr",
) -> set[tuple[int, int]]:
    """
    Compute all pairs reachability for a graph with context-free constraints using
    the tensor (Kronecker) product algorithm.
    """
    # Convert graph to NFA
    fa_graph = graph_to_nfa(graph, start_nodes, final_nodes)
    adj_graph = AdjacencyMatrixFA(fa_graph, matrix_format=matrix_format)

    # Convert RSM to NFA and adjacency-matrix representation
    fa_rsm = build_rsm_as_nfa(rsm)
    adj_rsm = AdjacencyMatrixFA(fa_rsm, matrix_format=matrix_format)

    updated = True
    while updated:
        updated = False

        # Compute intersection of graph NFA and RSM NFA
        intersection = intersect_automata(adj_graph, adj_rsm)

        # Compute reachability using boolean matrix multiplication
        reach_matrix = _compute_reachability(intersection, adj_rsm)

        # Propagate new nonterminals back to the graph
        adj_graph, added = _propagate_nonterminals(reach_matrix, adj_graph, adj_rsm, intersection)
        updated = updated or added

    # Extract result pairs from the graph's adjacency matrices
    result = set()
    if rsm.initial_label in adj_graph.boolean_decompress:
        mat = adj_graph.boolean_decompress[rsm.initial_label]
        rows, cols = mat.nonzero()
        for i, j in zip(rows, cols):
            src = adj_graph.state_of_index[i]
            tgt = adj_graph.state_of_index[j]
            if (not start_nodes or src in start_nodes) and (not final_nodes or tgt in final_nodes):
                result.add((src, tgt))
    return result


# -----------------------------
# Helper functions
# -----------------------------
def _compute_reachability(intersection: AdjacencyMatrixFA, adj_rsm: AdjacencyMatrixFA):
    """
    Perform boolean matrix multiplication to compute all reachable pairs
    in the intersection automaton.
    """
    n = len(intersection.states)
    mat_type = getattr(sp, f"{intersection.matrix_format}_matrix", sp.csr_matrix)
    reach = mat_type((n, n), dtype=bool)

    # Identify initial states corresponding to RSM starts
    starts = [s for s in intersection.states if s.value[1] in adj_rsm.start_states]
    for st in starts:
        idx = intersection.index_of_state[st]
        reach[idx, idx] = True

    # Iteratively update reachability until no new pairs are added
    changed = True
    while changed:
        changed = False
        for sym, mat in intersection.boolean_decompress.items():
            new_reach = reach @ mat
            delta = new_reach > reach
            if delta.count_nonzero() > 0:
                reach += new_reach
                changed = True
    return reach


def _propagate_nonterminals(reach: sp.spmatrix, adj_graph: AdjacencyMatrixFA, adj_rsm: AdjacencyMatrixFA, intersection: AdjacencyMatrixFA):
    """
    Update the graph's adjacency matrices with new nonterminal edges
    based on the reachability matrix.
    """
    updated = False
    rows, cols = reach.nonzero()

    for i, j in zip(rows, cols):
        gr_s, rsm_s = intersection.state_of_index[i].value
        gr_f, rsm_f = intersection.state_of_index[j].value

        box_start, _ = rsm_s.value
        box_end, _ = rsm_f.value

        # Only propagate edges if start and end of RSM box match and they are start/final states
        if box_start == box_end and rsm_s in adj_rsm.start_states and rsm_f in adj_rsm.final_states:
            if box_start not in adj_graph.boolean_decompress:
                n = len(adj_graph.states)
                mat_type = getattr(sp, f"{adj_graph.matrix_format}_matrix", sp.csr_matrix)
                new_mat = mat_type((n, n), dtype=bool)
                new_mat[adj_graph.index_of_state[gr_s], adj_graph.index_of_state[gr_f]] = True
                adj_graph.boolean_decompress[box_start] = new_mat
                adj_graph.labels.add(box_start)
                updated = True
            else:
                mat = adj_graph.boolean_decompress[box_start]
                if not mat[adj_graph.index_of_state[gr_s], adj_graph.index_of_state[gr_f]]:
                    mat[adj_graph.index_of_state[gr_s], adj_graph.index_of_state[gr_f]] = True
                    updated = True
    return adj_graph, updated
:with expression as target:
    pass
