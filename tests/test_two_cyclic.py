import pathlib
import pytest
from project import t1_graph_utils as t1

TMP_DATASET_DIR = pathlib.Path(__file__).parent / "datasets/tmp"


@pytest.fixture
def tmp_dataset_dir():
    try:
        TMP_DATASET_DIR.mkdir(parents=True, exist_ok=True)
        print("created")
    except OSError:
        print("Creation of the directory %s failed" % TMP_DATASET_DIR)
    else:
        print("Successfully created the directory %s " % TMP_DATASET_DIR)

    yield TMP_DATASET_DIR
    for file in TMP_DATASET_DIR.iterdir():
        if file.is_file():
            file.unlink()


def test_create_and_save_two_cyclic_graph(tmp_dataset_dir):
    filename = TMP_DATASET_DIR / "two_cycles.dot"
    cycle_sizes = (3, 3)
    labels = ("x", "y")
    t1.create_and_save_two_cyclic_graph(cycle_sizes, labels, str(filename))

    assert filename.exists()

    loaded_graph = t1.read_graph_from_dot(filename)
    total_nodes = sum(cycle_sizes)
    assert loaded_graph.number_of_nodes() == total_nodes + 1
    edge_labels = [d.get("label") for _, _, d in loaded_graph.edges(data=True)]
    for label in labels:
        assert label in edge_labels
