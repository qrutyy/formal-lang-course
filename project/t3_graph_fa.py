import numpy as np
import scipy.sparse as sp
import functools
from collections.abc import Iterable
from pyformlang.finite_automaton import (
    NondeterministicFiniteAutomaton,
    Symbol
)


class AdjacencyMatrixFA:
    def __init__(self,
                 n_states: int,
                 alphabet,
                 transitions: dict,
                 start_states: np.ndarray,
                 final_states: np.ndarray):
        """
        n_states: number of states
        alphabet: set of symbols
        transitions: dict[Symbol, sp.csr_matrix] — boolean adjacency matrices
        start_states: np.ndarray[bool] of length n_states
        final_states: np.ndarray[bool] of length n_states
        """
        self.n_states = n_states
        self.alphabet = list(alphabet)
        self.transitions = transitions
        self.start_states = start_states
        self.final_states = final_states

    @classmethod
    def from_nfa(cls, nfa: NondeterministicFiniteAutomaton):
        states = list(nfa.states)
        index = {s: i for i, s in enumerate(states)}
        n = len(states)

        transitions = {a: [] for a in nfa.symbols}

        for s_from, symb, s_to in nfa._transition_function.get_edges():
            i = index[s_from]
            j = index[s_to]
            transitions[symb].append((i, j))

        matrices = {}
        for a, edges in transitions.items():
            if edges:
                rows, cols = zip(*edges)
                data = np.ones(len(edges), dtype=bool)
                mat = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
            else:
                mat = sp.csr_matrix((n, n), dtype=bool)
            matrices[a] = mat

        start_states = np.zeros(n, dtype=bool)
        for s in nfa.start_states:
            start_states[index[s]] = True

        final_states = np.zeros(n, dtype=bool)
        for s in nfa.final_states:
            final_states[index[s]] = True

        return cls(n, nfa.symbols, matrices, start_states, final_states)

    def transitive_closure(self) -> sp.csc_matrix:
        """
        Compute the transitive closure of the adjacency matrix of the automaton
        Returns boolean matrix T where T[i, j] = True if j is reachable from i.
        """
        n = self.n_states

        matrices = list(self.transitions.values())
        if matrices:
            adj_matrix = functools.reduce(lambda x, y: x + y, matrices)
            adj_matrix.data = np.ones_like(adj_matrix.data, dtype=bool)
        else:
            adj_matrix = sp.csr_matrix((n, n), dtype=bool)

        adj_matrix = adj_matrix + sp.identity(n, dtype=bool, format='csr')

        result = adj_matrix.copy()
        for _ in range(n - 1):
            result = (result @ adj_matrix).astype(bool)

        return result

    def accepts(self, word: Iterable[Symbol]) -> bool:
        current_states = set(np.where(self.start_states)[0])

        for symbol in word:
            if symbol not in self.transitions:
                return False

            next_states = set()
            mat = self.transitions[symbol].tocoo()
            for i, j in zip(mat.row, mat.col):
                if i in current_states:
                    next_states.add(j)

            current_states = next_states
            if not current_states:
                return False

        final_indices = set(np.where(self.final_states)[0])
        return bool(current_states & final_indices)

    def is_empty(self) -> bool:
        """
        Check if the language of the automaton is empty.
        Returns True if no final state is reachable from any start state.
        """
        tr_cl = self.transitive_closure()

        start_indices = np.where(self.start_states)[0]
        final_indices = np.where(self.final_states)[0]
        print(start_indices, final_indices)
        print(tr_cl)

        for i in start_indices:
            for j in final_indices:
                if tr_cl[i, j]:
                    print(i, j)
                    return False

        return True
