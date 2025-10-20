import pytest
import cfpq_data as cd
import pathlib
from pyformlang.cfg import CFG

from project.t6_cfg_actions import hellings_based_cfpq

DATASET_DIR = pathlib.Path(__file__).parent.parent / "tests/datasets"


@pytest.mark.parametrize(
    "filename, cfg_text, start_nodes, final_nodes, exp_result",
    [
        # Test 1: Simple sequential grammar (equivalent to regex "a.b.c")
        # Graph: 0 --a--> 1 --b--> 0 --c--> 0
        (
            "regular_graph4.csv",
            "S -> abc",
            {0},
            {0},
            set()
        ),
        (
            "regular_graph4.csv",
            "S -> a b c",
            {0},
            {2}, # Node 2 does not exist
            set()
        ),

        # Test 2: Simple terminal grammar (equivalent to regex "a")
        (
            "regular_graph4.csv",
            "S -> a",
            {0},
            {2}, # No path from 0 to 2
            set()
        ),
        (
            "regular_graph4.csv",
            "S -> a",
            {0},
            {1},
            set()
        ),

        # Test 3: equivalent to regex "(x|y|z)*"
        # The grammar S -> x S | y S | z S | epsilon generates zero or more x, y, or z
        # Graph: 0--x-->3, 1--y-->3, 2--z-->3, 3--x-->3
        (
            "regular_graph5.csv",
            "S -> x S | y S | z S |", # The trailing '|' means epsilon
            {0, 1, 2},
            {3},
            {(0, 3), (1, 3), (2, 3)}
        ),
        # A simple path from 0 to 3 via x is found.
        # A simple path from 3 to 3 via x is found.
        # Thus, a path 0 -> 3 -> 3 should be found for grammar "S -> x S", S->x
        (
            "regular_graph5.csv",
            "S -> x S | x",
            {0, 3},
            {3},
            {(0, 3), (3, 3)}
        ),

        # Test 4: A true Context-Free Grammar (Dyck language a^n b^n)
        # This cannot be expressed with a regular expression.
        # Graph: 0--a-->1--a-->2--b-->3--b-->4
        (
            "regular_graph6.csv",
            "S -> a S b |", # Recognizes a^n b^n
            {0},
            {4},
            # Path 0->1->2->3->4 gives "aabb", which is in the language
            {(0, 4)}
        ),
        (
            "regular_graph6.csv",
            "S -> a S b |",
            {1},
            {3},
            # Path 1->2->3 gives "ab", which is in the language
            {(1, 3)}
        ),
        (
            "regular_graph6.csv",
            "S -> a S b |",
            {0},
            {3},
            # Path 0->1->2->3 gives "aab", which is NOT in the language
            set()
        ),
        (
            "regular_graph6.csv",
            "S -> a S b |",
            {0, 1, 2, 3},
            {0, 1, 2, 3},
            # Expected:
            # S -> epsilon gives (0,0), (1,1), (2,2), (3,3), (4,4)
            # S -> a S b with S->epsilon gives (1,3) for path "ab"
            # S -> a S b with S->a S b->ab gives (0,4) for path "aabb"
            {
                (0, 0), (1, 1), (2, 2), (3, 3), # From epsilon productions
                (1, 3), # For path "ab"
            }
        ),
    ],
)
def test_cfpq_on_graphs(filename, cfg_text, start_nodes, final_nodes, exp_result):
    filepath = str(DATASET_DIR / filename)
    graph = cd.graph_from_csv(filepath)
    cfg = CFG.from_text(cfg_text)

    result = hellings_based_cfpq(cfg, graph, start_nodes, final_nodes)

    assert result == exp_result
