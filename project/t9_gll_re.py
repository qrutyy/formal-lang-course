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
            self.pop_set.add(cur_node)
            for ret_state, gss_nodes in self.edges.items():
                for gssn in gss_nodes:
                    res_set.add((gssn, ret_state, cur_node))
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


def init_graph_edges(graph: nx.DiGraph) -> Dict[int, Dict[Symbol, Set[int]]]:
    nodes2edges = {n: {} for n in graph.nodes}
    for u, v, data in graph.edges(data=True):
        sym = data.get(LABEL_NAME)
        if sym is not None:
            if sym not in nodes2edges[u]:
                nodes2edges[u][sym] = set()
            nodes2edges[u][sym].add(v)
    return nodes2edges


def init_rsm_data(rsm: RecursiveAutomaton):
    rsmstate2data: Dict[Symbol, Dict[str, Dict]] = {}

    for var, box in rsm.boxes.items():
        rsmstate2data[var] = {}
        fa: DeterministicFiniteAutomaton = box.dfa

        for state in fa.states:
            rsmstate2data[var][state.value] = {
                "term_edges": {},
                "var_edges": {},
                "is_final": state in fa.final_states
            }

        for st_from, transitions in fa.to_dict().items():
            state_key = st_from.value
            for sym, st_to in transitions.items():
                if sym not in rsm.boxes:
                    rsmstate2data[var][state_key]["term_edges"][sym] = (var, st_to.value)
                else:
                    start_sub = rsm.boxes[sym].dfa.start_state.value
                    rsmstate2data[var][state_key]["var_edges"][sym] = ((sym, start_sub), (var, st_to.value))

    start_symb = rsm.initial_label
    start_state = rsm.boxes[start_symb].dfa.start_state.value
    start_rstate = (start_symb, start_state)
    return rsmstate2data, start_rstate


def gll_step(sppf_node, nodes2edges, rsmstate2data, gss, accept_gssnode):
    gss_node, rsm_state, graph_node = sppf_node
    rsm_data = rsmstate2data[rsm_state[0]][rsm_state[1]]

    new_reach = set()
    new_unprocessed = set()

    for term, new_rsm_state in rsm_data["term_edges"].items():
        if term in nodes2edges.get(graph_node, {}):
            for next_graph_node in nodes2edges[graph_node][term]:
                new_unprocessed.add((gss_node, new_rsm_state, next_graph_node))

    for var, (var_start_rsm_state, ret_rsm_state) in rsm_data["var_edges"].items():
        new_gss_node = gss.get_node(var_start_rsm_state, graph_node)
        new_unprocessed.add((new_gss_node, var_start_rsm_state, graph_node))

        post_pop_nodes = new_gss_node.add_edge(ret_rsm_state, gss_node)
        for pp_node in post_pop_nodes:
             new_unprocessed.add(pp_node)

    if rsm_data["is_final"]:
        start_node_of_this_path = gss_node.node
        for pop_node in gss_node.pop(graph_node):
            gssn_pop, ret_state, ret_graph_node = pop_node

            if gssn_pop is accept_gssnode:
                new_reach.add((start_node_of_this_path, ret_graph_node))
            else:
                new_unprocessed.add((gssn_pop, ret_state, ret_graph_node))

    return new_reach, new_unprocessed


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

    accept_gssnode = gss.get_node(("$", "fin"), -1)

    unprocessed = set()
    added = set()
    reach_set = set()

    for sn in start_nodes:
        start_gss_node = gss.get_node(start_rstate, sn)
        start_gss_node.add_edge(("$", "fin"), accept_gssnode)

        initial_node = (start_gss_node, start_rstate, sn)
        unprocessed.add(initial_node)
        added.add(initial_node)

    while unprocessed:
        node_to_process = unprocessed.pop()

        new_reach, new_unprocessed = gll_step(node_to_process, nodes2edges, rsmstate2data, gss, accept_gssnode)

        reach_set.update(new_reach)

        for item in new_unprocessed:
            if item not in added:
                unprocessed.add(item)
                added.add(item)

    return {(s, f) for s, f in reach_set if s in start_nodes and f in final_nodes}
