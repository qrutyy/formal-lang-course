from typing import Set, Dict, Tuple
import networkx as nx
from pyformlang.cfg import Symbol
from pyformlang.rsa import RecursiveAutomaton
from pyformlang.finite_automaton import DeterministicFiniteAutomaton

LABEL_NAME = "label"

class GSSNode:
    def __init__(self, state: Tuple[Symbol, str], node: int):
        self.state = state
        self.node = node
        self.edges: Dict[Tuple[Symbol, str], Set["GSSNode"]] = {}
        self.pop_set: Set[int] = set()

    def pop(self, cur_node: int):
        res_set = set()
        if cur_node not in self.pop_set:
            for ret_state, gss_nodes in self.edges.items():
                for gssn in gss_nodes:
                    res_set.add((gssn, ret_state, cur_node))
            self.pop_set.add(cur_node)
        return res_set

    def add_edge(self, ret_state: Tuple[Symbol, str], ptr: "GSSNode"):
        res_set = set()
        if ret_state not in self.edges:
            self.edges[ret_state] = set()
        if ptr not in self.edges[ret_state]:
            self.edges[ret_state].add(ptr)
            for cur_node in self.pop_set:
                res_set.add((ptr, ret_state, cur_node))
        return res_set


class GSStack:
    def __init__(self):
        self.body: Dict[Tuple[Tuple[Symbol, str], int], GSSNode] = {}

    def get_node(self, rsm_state: Tuple[Symbol, str], node: int):
        key = (rsm_state, node)
        if key not in self.body:
            self.body[key] = GSSNode(rsm_state, node)
        return self.body[key]


def init_graph_edges(graph: nx.DiGraph):
    nodes2edges: Dict[int, Dict[Symbol, Set[int]]] = {}
    for n in graph.nodes:
        nodes2edges[n] = {}
    for u, v, data in graph.edges(data=LABEL_NAME):
        sym = data
        if sym is not None:
            if sym not in nodes2edges[u]:
                nodes2edges[u][sym] = set()
            nodes2edges[u][sym].add(v)
    return nodes2edges


def init_rsm_data(rsm: RecursiveAutomaton):
    rsmstate2data: Dict[Symbol, Dict[str, Dict]] = {}

    def is_terminal(sym):
        return sym not in rsm.boxes

    for var, box in rsm.boxes.items():
        rsmstate2data[var] = {}
        fa: DeterministicFiniteAutomaton = box.dfa
        nx_g = fa.to_networkx()
        for st in nx_g.nodes:
            rsmstate2data[var][st] = {"term_edges": {}, "var_edges": {}, "is_final": st in fa.final_states}
        for from_st, to_st, sym in nx_g.edges(data=LABEL_NAME):
            if sym is not None:
                if is_terminal(sym):
                    rsmstate2data[var][from_st]["term_edges"][sym] = (var, to_st)
                else:
                    sub_fa = rsm.boxes[Symbol(sym)].dfa
                    start_sub = sub_fa.start_state.value
                    rsmstate2data[var][from_st]["var_edges"][sym] = ((Symbol(sym), start_sub), (var, to_st))
    start_symb = rsm.initial_label
    start_state = rsm.boxes[start_symb].dfa.start_state.value
    start_rstate = (start_symb, start_state)
    return rsmstate2data, start_rstate


def add_sppf_nodes(unprocessed, added, nodes):
    nodes.difference_update(added)
    added.update(nodes)
    unprocessed.update(nodes)


def gll_step(sppfnode, nodes2edges, rsmstate2data, gss):
    gssn, rsm_st, gnode = sppfnode
    rsm_dat = rsmstate2data[rsm_st[0]][rsm_st[1]]
    reach_set = set()

    for term, rsm_new_st in rsm_dat["term_edges"].items():
        if term in nodes2edges[gnode]:
            for gn in nodes2edges[gnode][term]:
                unprocessed.add((gssn, rsm_new_st, gn))

    for var, (var_start_rsm_st, ret_rsm_st) in rsm_dat["var_edges"].items():
        inner_gss_node = gss.get_node(var_start_rsm_st, gnode)
        post_pop_nodes = inner_gss_node.add_edge(ret_rsm_st, gssn)

        for pp_node in post_pop_nodes:
            if pp_node[0] is accept_gssnode:
                reach_set.add((sppfnode[2], pp_node[2]))

        unprocessed.add((inner_gss_node, var_start_rsm_st, gnode))

    if rsm_dat["is_final"]:
        for pop_node in gssn.pop(gnode):
            gssn_pop, ret_st, node = pop_node
            if gssn_pop is accept_gssnode:
                reach_set.add((sppfnode[2], node))
            else:
                unprocessed.add((gssn_pop, ret_st, node))

    return reach_set


def gll_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: Set[int] = None,
    final_nodes: Set[int] = None,
) -> Set[Tuple[int, int]]:

    if start_nodes is None:
        start_nodes = set(graph.nodes)
    if final_nodes is None:
        final_nodes = set(graph.nodes)

    nodes2edges = init_graph_edges(graph)
    rsmstate2data, start_rstate = init_rsm_data(rsm)

    gss = GSStack()
    global accept_gssnode
    accept_gssnode = gss.get_node(("$", "fin"), -1)

    unprocessed: Set[Tuple[GSSNode, Tuple[Symbol, str], int]] = set()
    added: Set[Tuple[GSSNode, Tuple[Symbol, str], int]] = set()
    reach_set: Set[Tuple[int, int]] = set()

    for sn in start_nodes:
        gssn = gss.get_node(start_rstate, sn)
        gssn.add_edge(("$", "fin"), accept_gssnode)
        unprocessed.add((gssn, start_rstate, sn))

    while unprocessed:
        node = unprocessed.pop()
        reach_set.update(gll_step(node, nodes2edges, rsmstate2data, gss))

    return {(s, f) for (s, f) in reach_set if f in final_nodes}
