import networkx as nx
from typing import Set, Tuple
from scipy.sparse import dok_matrix, csr_matrix
import pyformlang.cfg
from pyformlang.cfg import Terminal
from project import t6_cfg_actions as t6


def matrix_based_cfpq(
    cfg: pyformlang.cfg.CFG,
    graph: nx.DiGraph,
    start_nodes: Set[int] = None,
    final_nodes: Set[int] = None,
) -> Set[Tuple[int, int]]:
    """
    Implementation of matrix CFPQ based on Azimov's algorithm.
    Returns (u, v) pairs if there exists a path from u to v within graph G.
    """

    node_to_idx = {node: idx for idx, node in enumerate(graph.nodes)}
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}
    n = len(node_to_idx)
    if n == 0:
        return set()

    wcnf = t6.cfg_to_weak_normal_form(cfg)
    eps_prods = {p.head for p in wcnf.productions if len(p.body) == 0}
    term_prods = {}
    var_prods = []

    for p in wcnf.productions:
        if len(p.body) == 1 and isinstance(p.body[0], Terminal):
            a = p.body[0].value
            term_prods.setdefault(a, set()).add(p.head)
        elif len(p.body) == 2:
            b, c = p.body
            var_prods.append((p.head, b, c))

    variables = list(wcnf.variables)
    matrices = {var: dok_matrix((n, n), dtype=bool) for var in variables}

    for edge in graph.edges(data=True):
        if len(edge) == 3:
            u, v, data = edge
        else:
            u, v, _k, data = edge
        label = data.get("label")
        if label in term_prods:
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            for A in term_prods[label]:
                matrices[A][u_idx, v_idx] = True

    for A in eps_prods:
        mat = matrices[A]
        for i in range(n):
            mat[i, i] = True

    var_prods_filtered = [(A, B, C) for (A, B, C) in var_prods if A in matrices and B in matrices and C in matrices]
    changed = True
    while changed:
        changed = False
        for A, B, C in var_prods_filtered:
            MB = matrices[B].tocsr()
            MC = matrices[C].tocsr()
            if MB.nnz == 0 or MC.nnz == 0:
                continue
            prod = (MB @ MC).astype(bool).tocsr()
            MA = matrices[A].tocsr()
            before = MA.nnz
            MA = (MA + prod).astype(bool).tocsr()
            after = MA.nnz
            if after > before:
                changed = True
                matrices[A] = MA.todok()  # save for next iteration

    # prepare start and final nodes
    if start_nodes is None or len(start_nodes) == 0:
        start_nodes = set(graph.nodes)
    if final_nodes is None or len(final_nodes) == 0:
        final_nodes = set(graph.nodes)
    start_indices = {node_to_idx[u] for u in start_nodes if u in node_to_idx}
    final_indices = {node_to_idx[v] for v in final_nodes if v in node_to_idx}

    result = set()
    S = wcnf.start_symbol
    if S not in matrices:
        return result

    MS = matrices[S].tocsr()
    for u_idx in start_indices:
        row = MS.getrow(u_idx).nonzero()[1]
        for v_idx in row:
            if v_idx in final_indices:
                result.add((idx_to_node[u_idx], idx_to_node[v_idx]))

    return result
