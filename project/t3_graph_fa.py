import numpy as np
from networkx import MultiDiGraph
from collections.abc import Iterable
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, State, Symbol
from project.t2_fa_utils import regex_to_dfa, graph_to_nfa
from itertools import product
from scipy.sparse import identity, kron, csr_matrix
import scipy.sparse as scsp


class AdjacencyMatrixFA:
    """
    A consistent and robust Finite Automaton representation using adjacency matrices.
    This version is designed to be compatible with all project tests.
    """

    states: list[State]
    n_states: int
    index_of_state: dict[State, int]
    state_of_index: dict[int, State]
    start_states: np.ndarray
    final_states: np.ndarray
    alphabet: set[Symbol]
    transitions: dict[Symbol, scsp.spmatrix]
    matrix_format: str

    def __init__(self, fa: NondeterministicFiniteAutomaton, matrix_format: str = "csr"):
        self.states = sorted(list(fa.states), key=lambda s: str(s.value))
        self.n_states = len(self.states)
        self.alphabet = fa.symbols
        self.matrix_format = matrix_format
        matrix_ctor = getattr(scsp, f"{self.matrix_format}_matrix", csr_matrix)

        self.index_of_state = {state: i for i, state in enumerate(self.states)}
        self.state_of_index = {i: state for i, state in enumerate(self.states)}

        self.start_states = np.zeros(self.n_states, dtype=bool)
        for state in fa.start_states:
            if state in self.index_of_state:
                self.start_states[self.index_of_state[state]] = True

        self.final_states = np.zeros(self.n_states, dtype=bool)
        for state in fa.final_states:
            if state in self.index_of_state:
                self.final_states[self.index_of_state[state]] = True

        self.transitions = {
            label: matrix_ctor((self.n_states, self.n_states), dtype=bool)
            for label in self.alphabet
        }

        for s_from, symbol, s_to in fa._transition_function.get_edges():
            i, j = self.index_of_state.get(s_from), self.index_of_state.get(s_to)
            if i is not None and j is not None:
                self.transitions[symbol][i, j] = True

    @classmethod
    def from_components(
        cls,
        states,
        alphabet,
        transitions,
        start_states,
        final_states,
        matrix_format="csr",
    ):
        """Constructs an AdjacencyMatrixFA from its raw components."""
        obj = cls.__new__(cls)
        obj.states = states
        obj.n_states = len(states)
        obj.alphabet = alphabet
        obj.transitions = transitions
        obj.start_states = start_states
        obj.final_states = final_states
        obj.matrix_format = matrix_format
        obj.index_of_state = {state: i for i, state in enumerate(states)}
        obj.state_of_index = {i: state for i, state in enumerate(states)}
        return obj

    @property
    def boolean_decompress(self):
        """Property for backward compatibility."""
        return self.transitions

    @property
    def labels(self):
        """Property for backward compatibility."""
        return self.alphabet

    def get_trans_closure(self) -> scsp.spmatrix:
        if self.n_states == 0:
            return csr_matrix((0, 0), dtype=bool)

        adj_matrix = (
            sum(self.transitions.values())
            if self.transitions
            else csr_matrix((self.n_states, self.n_states), dtype=bool)
        )
        adj_matrix += identity(self.n_states, format=self.matrix_format, dtype=bool)

        prev_nnz = -1
        while adj_matrix.nnz != prev_nnz:
            prev_nnz = adj_matrix.nnz
            adj_matrix += adj_matrix @ adj_matrix
        return adj_matrix

    def transitive_closure(self):
        """Wrapper for backward compatibility."""
        return self.get_trans_closure()

    def is_empty(self) -> bool:
        if not np.any(self.start_states) or not np.any(self.final_states):
            return True
        tc = self.get_trans_closure()
        start_indices = np.where(self.start_states)[0]
        final_indices = np.where(self.final_states)[0]

        for start_idx in start_indices:
            reachable_from_start = tc[start_idx, :].nonzero()[1]
            if np.intersect1d(reachable_from_start, final_indices).size > 0:
                return False
        return True

    def accepts(self, word: Iterable[Symbol]) -> bool:
        if not word:
            return np.any(self.start_states & self.final_states)

        current_states_mask = scsp.csr_matrix(self.start_states.reshape(1, -1))

        for symbol in word:
            if symbol not in self.transitions:
                return False
            current_states_mask = current_states_mask @ self.transitions[symbol]

        final_indices = np.where(self.final_states)[0]
        reachable_indices = current_states_mask.nonzero()[1]

        return np.intersect1d(reachable_indices, final_indices).size > 0


def intersect_automata(
    automaton1: AdjacencyMatrixFA, automaton2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    """Computes the intersection of two automata using the Kronecker product."""

    new_states = [
        State((s1.value, s2.value))
        for s1, s2 in product(automaton1.states, automaton2.states)
    ]
    shared_alphabet = automaton1.alphabet.intersection(automaton2.alphabet)

    new_transitions = {
        symbol: kron(
            automaton1.transitions[symbol],
            automaton2.transitions[symbol],
            format=automaton1.matrix_format,
        )
        for symbol in shared_alphabet
    }

    new_start_states = np.kron(automaton1.start_states, automaton2.start_states)
    new_final_states = np.kron(automaton1.final_states, automaton2.final_states)

    return AdjacencyMatrixFA.from_components(
        new_states,
        shared_alphabet,
        new_transitions,
        new_start_states,
        new_final_states,
        matrix_format=automaton1.matrix_format,
    )


def tensor_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
    matrix_format="csr",
) -> set[tuple[int, int]]:
    """Performs a regular path query using the tensor (Kronecker) product method."""

    reg_dfa = regex_to_dfa(regex)
    graph_nfa = graph_to_nfa(graph, start_nodes, final_nodes)

    aut1 = AdjacencyMatrixFA(reg_dfa, matrix_format=matrix_format)
    aut2 = AdjacencyMatrixFA(graph_nfa, matrix_format=matrix_format)

    intersection = intersect_automata(aut1, aut2)
    if intersection.is_empty():
        return set()

    tc = intersection.get_trans_closure()
    result = set()

    start_indices = np.where(intersection.start_states)[0]
    final_indices = np.where(intersection.final_states)[0]

    for start_idx in start_indices:
        reachable_from_start = tc[start_idx, :].nonzero()[1]
        reachable_final_states = np.intersect1d(reachable_from_start, final_indices)

        for final_idx in reachable_final_states:
            start_state_pair = intersection.state_of_index[start_idx]
            final_state_pair = intersection.state_of_index[final_idx]

            _, graph_start_val = start_state_pair.value
            _, graph_final_val = final_state_pair.value

            result.add((graph_start_val, graph_final_val))

    return result
