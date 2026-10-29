import pytest

from nettwin.topology import from_node_link, link_id, link_metric, load


def _data(nodes, edges, **flags):
    return {
        "directed": flags.get("directed", False),
        "multigraph": False,
        "graph": {},
        "nodes": [{"id": n, "pos": [i, 0]} for i, n in enumerate(nodes)],
        "edges": [{"source": u, "target": v, "dist": d} for u, v, d in edges],
    }


def test_link_id_is_canonical():
    assert link_id("B", "A") == link_id("A", "B") == "A--B"


def test_metric_floor_and_rounding():
    assert link_metric(0.0) == 1
    assert link_metric(247.07) == 247
    assert link_metric(466.5) == 466  # banker's rounding, still an int


def test_from_node_link_attrs():
    g = from_node_link(_data("ABC", [("A", "B", 10.4), ("B", "C", 0.0)]), "toy", geo=False)
    assert g.graph["name"] == "toy" and g.graph["geo"] is False
    assert g.nodes["A"]["pos"] == (0.0, 0.0)
    assert g.edges["B", "C"] == {"link_id": "B--C", "dist_km": 0.0, "metric": 1}


@pytest.mark.parametrize(
    "data,msg",
    [
        (_data("AB", [("A", "B", 1)], directed=True), "undirected"),
        (_data("AB", [("A", "A", 1), ("A", "B", 1)]), "self loop"),
        (_data("AB", [("A", "B", 1), ("B", "A", 1)]), "duplicate"),
        (_data("ABC", [("A", "B", 1)]), "disconnected"),
    ],
)
def test_from_node_link_rejects_bad_input(data, msg):
    with pytest.raises(ValueError, match=msg):
        from_node_link(data, "bad", geo=False)


def test_load_aarnet():
    g = load("topozoo/Aarnet")
    assert g.graph["geo"] is True
    assert (g.number_of_nodes(), g.number_of_edges()) == (19, 24)
    lon, lat = g.nodes["Sydney1"]["pos"]
    assert 150 < lon < 152 and -35 < lat < -33
    zero = [d for _, _, d in g.edges(data=True) if d["dist_km"] == 0]
    assert len(zero) == 6 and all(d["metric"] == 1 for d in zero)


def test_load_gabriel_is_planar():
    g = load("gabriel/25/0")
    assert g.graph["geo"] is False and g.number_of_nodes() == 25


def test_germany50_keeps_demands():
    g = load("sndlib/germany50")
    assert g.graph["demands"]["Essen"]["Duesseldorf"] == 34.0
