"""Topology loading from topohub (PLAN.md section 3).

`load(name)` returns an undirected `nx.Graph` with:
- graph attrs: `name`, `geo` (True if `pos` is lon/lat), `demands` (real demand matrix, germany50 only)
- node attr `pos`: (lon, lat) for Topology Zoo / SNDlib, planar (x, y) for synthetic Gabriel graphs
- edge attrs: `link_id`, `dist_km` (as given by topohub), `metric` (integer routing weight, at least 1)
"""
import networkx as nx
import topohub

GEO_PREFIXES = ("topozoo/", "sndlib/")
SEP = "--"


def link_id(u, v):
    """Canonical undirected link id, endpoints sorted: 'A--B'."""
    a, b = sorted((u, v))
    return f"{a}{SEP}{b}"


def link_metric(dist_km):
    """Integer IGP-style metric: km rounded, floored at 1 so co-located PoP links are not free."""
    return max(1, round(dist_km))


def from_node_link(data, name, geo):
    """Build a validated topology graph from topohub's node-link dict."""
    if data.get("directed") or data.get("multigraph"):
        raise ValueError(f"{name}: expected an undirected simple graph")
    g = nx.Graph(name=name, geo=geo, demands=data.get("graph", {}).get("demands", {}))
    for n in data["nodes"]:
        if SEP in str(n["id"]):
            raise ValueError(f"{name}: node id {n['id']!r} contains {SEP!r}")
        x, y = n["pos"]
        g.add_node(n["id"], pos=(float(x), float(y)))
    for e in data["edges"]:
        u, v = e["source"], e["target"]
        if u == v:
            raise ValueError(f"{name}: self loop at {u}")
        if g.has_edge(u, v):
            raise ValueError(f"{name}: duplicate link {link_id(u, v)}")
        dist = float(e["dist"])
        g.add_edge(u, v, link_id=link_id(u, v), dist_km=dist, metric=link_metric(dist))
    if g.number_of_nodes() == 0 or not nx.is_connected(g):
        raise ValueError(f"{name}: topology is empty or disconnected")
    return g


def load(name, use_names=True):
    """Load a topohub topology, e.g. 'topozoo/Aarnet', 'sndlib/germany50', 'gabriel/25/0'."""
    data = topohub.get(name, use_names=use_names)
    return from_node_link(data, name, geo=name.startswith(GEO_PREFIXES))
