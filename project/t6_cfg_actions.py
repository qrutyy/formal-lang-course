from pyformlang.cfg import CFG, Variable, Terminal, Production
from collections import defaultdict, deque
import networkx as nx
import itertools
import pyformlang


def refactor_long_productions(cfg):
    new_productions = set()
    counter = 0

    for p in cfg.productions:
        body = list(p.body)

        if len(body) <= 2:
            new_productions.add(p)
            continue

        head = p.head

        while len(body) > 2:
            counter += 1
            new_var = Variable(f"X_{counter}")
            new_productions.add(Production(head, [body[0], new_var]))
            head = new_var
            body = body[1:]

        new_productions.add(Production(head, body))

    return CFG(start_symbol=cfg.start_symbol, productions=new_productions)
def remove_e_productions(cfg: CFG) -> CFG:
    """
    Eliminates epsilon-productions from a CFG.
    It finds nullable variables and then adds all necessary compensatory productions.
    """
    nullable = set()
    changed = True
    while changed:
        changed = False
        for p in cfg.productions:
            if all(symbol in nullable for symbol in p.body):
                if p.head not in nullable:
                    nullable.add(p.head)
                    changed = True

    new_productions = set()
    for p in cfg.productions:
        options = []
        for symbol in p.body:
            if symbol in nullable:
                options.append([symbol, None])
            else:
                options.append([symbol])

        for body_tuple in itertools.product(*options):
            new_body = [symbol for symbol in body_tuple if symbol is not None]

            if new_body:
                new_productions.add(Production(p.head, new_body))
            else:
                # add explicit epsilon production
                new_productions.add(Production(p.head, []))

    return CFG(start_symbol=cfg.start_symbol, productions=new_productions)

def get_terminal_mapping(cfg: CFG):
    """Find variables that deterministically reduce to a single terminal."""
    mapping = {}
    changed = True

    while changed:
        changed = False
        for p in cfg.productions:
            head = p.head
            body = list(p.body)

            # Rule of form A -> a
            if len(body) == 1 and isinstance(body[0], Terminal):
                if head not in mapping:
                    mapping[head] = body[0]
                    changed = True
                    continue

            # Rule of form A -> B (and B already maps to terminal)
            if len(body) == 1 and isinstance(body[0], Variable):
                if body[0] in mapping:
                    if head not in mapping:
                        mapping[head] = mapping[body[0]]
                        changed = True

    return mapping

def remove_unit_productions(cfg: CFG) -> CFG:
    """
    Eliminates unit productions (A -> B) from a CFG.
    """
    # find all pairs (A, B) such that A ->* B via unit productions.
    unit_pairs = {(p.head, p.body[0]) for p in cfg.productions if len(p.body) == 1 and isinstance(p.body[0], Variable)}

    while True:
        new_pairs = {(A, C) for A, B1 in unit_pairs for B2, C in unit_pairs if B1 == B2}
        if new_pairs.issubset(unit_pairs):
            break
        unit_pairs.update(new_pairs)

    for var in cfg.variables:
        unit_pairs.add((var, var))

    # new set of productions.
    new_productions = set()
    for A, B in unit_pairs:
        # for each pair (A, B) and each original non-unit production B -> body...
        for p in cfg.productions:
            if p.head == B and not (len(p.body) == 1 and isinstance(p.body[0], Variable)):
                # ...add a new production A -> body.
                new_productions.add(Production(A, p.body))

    return CFG(start_symbol=cfg.start_symbol, productions=new_productions)

def remove_nonproductive(cfg: CFG) -> CFG:
    productive = set()

    changed = True
    while changed:
        changed = False
        for p in cfg.productions:
            if all(isinstance(s, Terminal) or s in productive for s in p.body):
                if p.head not in productive:
                    productive.add(p.head)
                    changed = True

    new_productions = {
        p
        for p in cfg.productions
        if p.head in productive
        and all(isinstance(s, Terminal) or s in productive for s in p.body)
    }

    return CFG(start_symbol=cfg.start_symbol, productions=new_productions)


def remove_unreachable(cfg: CFG) -> CFG:
    reachable = {cfg.start_symbol}
    changed = True

    while changed:
        changed = False
        for p in cfg.productions:
            if p.head in reachable:
                for s in p.body:
                    if isinstance(s, Variable) and s not in reachable:
                        reachable.add(s)
                        changed = True

    new_productions = {
        p
        for p in cfg.productions
        if p.head in reachable
        and all(not isinstance(s, Variable) or s in reachable for s in p.body)
    }

    return CFG(start_symbol=cfg.start_symbol, productions=new_productions)


def clear_grammatics(cfg: CFG) -> CFG:
    cfg = remove_nonproductive(cfg)
    cfg = remove_unreachable(cfg)
    return cfg


def replace_terminals(cfg: CFG) -> CFG:
    new_productions = set()
    terminal_vars = {}  # Terminal -> Variable
    counter = 0

    for p in cfg.productions:
        new_body = []

        for s in p.body:
            if isinstance(s, Terminal) and len(p.body) > 1:
                if s not in terminal_vars:
                    counter += 1
                    new_var = Variable(f"F_{counter}")
                    terminal_vars[s] = new_var
                    new_productions.add(Production(new_var, [s]))
                new_body.append(terminal_vars[s])
            else:
                new_body.append(s)

        new_productions.add(Production(p.head, new_body))

    return CFG(start_symbol=cfg.start_symbol, productions=new_productions)


def cfg_to_weak_normal_form(cfg: pyformlang.cfg.CFG) -> pyformlang.cfg.CFG:
    cfg = refactor_long_productions(cfg)
    cfg = remove_e_productions(cfg)
    cfg = remove_unit_productions(cfg)
    cfg = clear_grammatics(cfg)
    cfg = replace_terminals(cfg)
    return cfg


def hellings_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    """
    This algorithm finds all pairs of nodes (u, v) in a graph such that there
    is a path from u to v whose edge labels form a word in the language
    generated by the given Context-Free Grammar.
    """
    wcnf = cfg_to_weak_normal_form(cfg)
    eps_prods = {p.head for p in wcnf.productions if not p.body}
    term_prods = defaultdict(set)
    var_prods = defaultdict(set)

    for p in wcnf.productions:
        if len(p.body) == 1 and isinstance(p.body[0], Terminal):
            term_prods[p.body[0].value].add(p.head)
        elif len(p.body) == 2:
            var1, var2 = p.body
            var_prods[(var1, var2)].add(p.head)

    r = set()
    new = deque()

    r_by_end = defaultdict(set)
    r_by_start = defaultdict(set)

    def add_triple(head, u, v):
        if (head, u, v) not in r:
            r.add((head, u, v))
            new.append((head, u, v))
            r_by_end[v].add((head, u))
            r_by_start[u].add((head, v))

    for head in eps_prods:
        for node in graph.nodes:
            add_triple(head, node, node)

    for u, v, edge_data in graph.edges(data=True):
        label = edge_data.get("label")
        if label in term_prods:
            for head in term_prods[label]:
                add_triple(head, u, v)

    while new:
        N, n, m = new.popleft()

        if n in r_by_end:
            for M, n_prime in list(r_by_end[n]):
                if (M, N) in var_prods:
                    for P in var_prods[(M, N)]:
                        add_triple(P, n_prime, m)

        if m in r_by_start:
            for M, m_prime in list(r_by_start[m]):
                if (N, M) in var_prods:
                    for P in var_prods[(N, M)]:
                        add_triple(P, n, m_prime)

    if start_nodes is None:
        start_nodes = set(graph.nodes)
    if final_nodes is None:
        final_nodes = set(graph.nodes)

    result = set()
    for var, start_node, final_node in r:
        if (var == cfg.start_symbol and
            start_node in start_nodes and
            final_node in final_nodes):
            result.add((start_node, final_node))

    return result
