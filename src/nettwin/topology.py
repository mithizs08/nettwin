"""Topology loading from topohub (PLAN.md section 3).

`load(name)` returns an undirected `nx.Graph` with:
- graph attrs: `name`, `geo` (True if `pos` is lon/lat), `demands` (real demand matrix, germany50 only)
- node attr `pos`: (lon, lat) for Topology Zoo / SNDlib, planar (x, y) km for synthetic graphs
- node attr `lon` (synthetic graphs only): pseudo-longitude for local time zones
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
        g.add_node(n["id"], pos=(float(x), float(y)), **({"lon": float(n["lon"])} if "lon" in n else {}))
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


def load(name, use_names=True, cfg=None):
    """Load a topology: topohub keys ('topozoo/Aarnet', 'sndlib/germany50', 'gabriel/25/0') or a generated,
    augmented graph 'gabaug/<n>/<seed>' (needs cfg for the reference statistics)."""
    if name.startswith("gabaug/"):
        from nettwin.synth import generate

        _, n, seed = name.split("/")
        return generate(int(n), int(seed), cfg)
    data = topohub.get(name, use_names=use_names)
    return from_node_link(data, name, geo=name.startswith(GEO_PREFIXES))


def local_longitude(topo, node):
    """Longitude used for local time: real lon for geo topologies, pseudo-longitude for generated ones, else None."""
    d = topo.nodes[node]
    if "lon" in d:
        return d["lon"]
    return d["pos"][0] if topo.graph["geo"] else None
