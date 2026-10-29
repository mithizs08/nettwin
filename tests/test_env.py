import networkx as nx
import pytest
import topohub

EXPECTED = {
    "topozoo/Aarnet": (19, 24),
    "topozoo/Geant2012": (37, 58),
    "sndlib/germany50": (50, 88),
}


@pytest.mark.parametrize("name,size", EXPECTED.items())
def test_test_topologies_load(name, size):
    g = nx.node_link_graph(topohub.get(name, use_names=True), edges="edges")
    assert (g.number_of_nodes(), g.number_of_edges()) == size
    assert nx.is_connected(g)


def test_torch_geometric_imports():
    import torch
    from torch_geometric.nn import GraphSAGE

    x = torch.randn(4, 3)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    assert GraphSAGE(3, 8, num_layers=2)(x, edge_index).shape == (4, 8)


def test_nettwin_package_imports():
    import nettwin  # noqa: F401
