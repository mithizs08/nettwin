"""Tests for the knowledge graph and every query function.

Toy topology (planar, metrics in brackets):

    A --[10]-- B
    |          |
   [10]       [10]
    |          |
    C --[10]-- D --[5]-- E

A..D and B..C each have two equal-cost paths.
"""
import itertools

import pandas as pd
import pytest
import yaml

from nettwin.kg import FAULT_TYPES, KINDS, SCHEMA, NetTwinKG, generate_services, link_node, path_node, pop_node
from nettwin.topology import from_node_link, load

AB, AC, BD, CD, DE = (link_node(*p) for p in ("AB", "AC", "BD", "CD", "DE"))
A, B, C, D, E = (pop_node(n) for n in "ABCDE")

SERVICES = [
    {"id": "svc:AE", "src": "E", "dst": "A", "cls": "gold", "sla_latency_ms": 1.0, "sla_loss_pct": 0.1},
    {"id": "svc:AB", "src": "A", "dst": "B", "cls": "bronze", "sla_latency_ms": 1.0, "sla_loss_pct": 1.0},
    {"id": "svc:CD", "src": "C", "dst": "D", "cls": "silver", "sla_latency_ms": 1.0, "sla_loss_pct": 0.5},
]


def toy_topo():
    pos = {"A": (0, 1), "B": (1, 1), "C": (0, 0), "D": (1, 0), "E": (2, 0)}
    edges = [("A", "B", 10), ("A", "C", 10), ("B", "D", 10), ("C", "D", 10), ("D", "E", 5)]
    data = {
        "directed": False,
        "multigraph": False,
        "graph": {},
        "nodes": [{"id": n, "pos": list(p)} for n, p in pos.items()],
        "edges": [{"source": u, "target": v, "dist": d} for u, v, d in edges],
    }
    return from_node_link(data, "toy", geo=False)


@pytest.fixture
def kg():
    return NetTwinKG(toy_topo(), services=SERVICES)


@pytest.fixture(scope="module")
def cfg():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def brute_paths_through(kg, element, t):
    """Reference implementation: scan every active path's PoPs / links."""
    out = []
    for p in kg.nodes_of_kind("RoutingPath"):
        if p not in kg.paths_between(pop_node(kg.G.nodes[p]["src"]), pop_node(kg.G.nodes[p]["dst"]), t):
            continue
        on = element in kg.links_of_path(p) or element in [pop_node(n) for n in kg.G.nodes[p]["pops"]]
        if on:
            out.append(p)
    return sorted(out)


# ------------------------------------------------------------------ construction and schema


def test_layers_and_counts(kg):
    s = kg.summary()
    assert s["nodes_by_kind"]["PoP"] == 5
    assert s["nodes_by_kind"]["Link"] == 5
    assert s["nodes_by_kind"]["Service"] == 3
    assert s["nodes_by_kind"]["FaultType"] == len(FAULT_TYPES)
    assert s["edges_by_rel"]["CONNECTS"] == 10
    assert s["pop_pairs"] == s["reachable_pairs"] == 10
    assert s["ecmp_pairs"] == 3  # A-D, B-C, A-E
    assert kg.validate()


def test_ecmp_paths_and_order(kg):
    assert kg.paths_between(A, D) == ["path:A>B>D", "path:A>C>D"]
    assert kg.paths_between(E, A) == ["path:A>B>D>E", "path:A>C>D>E"]  # order of arguments does not matter
    assert kg.links_of_path("path:A>B>D>E") == [AB, BD, DE]
    attrs = kg.G.nodes["path:A>B>D>E"]
    assert attrs["hops"] == 3 and attrs["dist_km"] == 25 and attrs["prop_latency_ms"] == 0.125


def test_ecmp_cap():
    kg = NetTwinKG(toy_topo(), ecmp_max_paths=1)
    assert len(kg.paths_between(A, D)) == 1


def test_schema_rejects_bad_edges(kg):
    with pytest.raises(ValueError, match="ON not allowed"):
        kg.add_alert("x", A, 0, metric="loss")  # alerts attach to Link or Service, not PoP
    with pytest.raises(ValueError, match="AFFECTS not allowed"):
        kg.add_maintenance("m", ["svc:AB"], 0, 1)


def test_validate_catches_corruption(kg):
    kg.G.add_node("weird", kind="Banana")
    with pytest.raises(ValueError, match="unknown kind"):
        kg.validate()


# ------------------------------------------------------------------ paths_through


def test_paths_through_link(kg):
    got = kg.paths_through(DE)
    assert got == ["path:A>B>D>E", "path:A>C>D>E", "path:B>D>E", "path:C>D>E", "path:D>E"]
    for element in (AB, AC, BD, CD, DE):
        assert kg.paths_through(element) == brute_paths_through(kg, element, 0)


def test_paths_through_pop_includes_endpoints(kg):
    assert kg.paths_through(E) == kg.paths_through(DE)  # E is a leaf, every path to E uses D--E
    for element in (A, B, C, D, E):
        assert kg.paths_through(element) == brute_paths_through(kg, element, 0)


def test_paths_through_rejects_wrong_kind(kg):
    with pytest.raises(ValueError):
        kg.paths_through("svc:AB")
    with pytest.raises(KeyError):
        kg.paths_through("link:X--Y")


def test_paths_through_respects_time(kg):
    kg.apply_change(5, links_down=[AB])
    assert kg.paths_through(AB, t=5) == []
    assert kg.paths_through(AB, t=4) == brute_paths_through(kg, AB, 4) != []
    assert kg.paths_through(AB) == []  # t=None means current time (5)
    assert "path:A>C>D>B" in kg.paths_through(CD, t=5)
    assert "path:A>C>D>B" not in kg.paths_through(CD, t=4)


# ------------------------------------------------------------------ services_on


def test_services_on_link_pop_and_path(kg):
    assert kg.services_on(DE) == ["svc:AE"]
    assert kg.services_on(AB) == ["svc:AB", "svc:AE"]
    assert kg.services_on(CD) == ["svc:AE", "svc:CD"]
    assert kg.services_on(B) == ["svc:AB", "svc:AE"]
    assert kg.services_on("path:A>C>D>E") == ["svc:AE"]
    assert kg.services_on("path:B>D") == []


def test_services_on_after_failure(kg):
    kg.apply_change(5, links_down=[AB])
    assert kg.services_on(AB) == []
    assert kg.services_on(BD) == ["svc:AB"]  # rerouted A>C>D>B
    assert kg.services_on(AB, t=0) == ["svc:AB", "svc:AE"]
    assert kg.services_on("path:A>B", t=5) == []


def test_service_paths(kg):
    assert kg.service_paths("svc:AE") == ["path:A>B>D>E", "path:A>C>D>E"]
    kg.apply_change(3, links_down=[DE])
    assert kg.service_paths("svc:AE") == []
    assert kg.service_paths("svc:AE", t=2) == ["path:A>B>D>E", "path:A>C>D>E"]


# ------------------------------------------------------------------ neighbours


def test_neighbours_pop(kg):
    assert kg.neighbours(A) == [B, C]
    assert kg.neighbours(A, hops=2) == [B, C, D]
    assert kg.neighbours(A, hops=3) == [B, C, D, E]
    assert kg.neighbours(A, hops=0) == []


def test_neighbours_link(kg):
    assert kg.neighbours(AB) == [AC, BD]
    assert kg.neighbours(AB, hops=2) == [AC, BD, CD, DE]
    assert kg.neighbours(DE) == [BD, CD]


def test_neighbours_ignore_failures(kg):
    kg.apply_change(1, links_down=[AB], pops_down=[B])
    assert kg.neighbours(A) == [B, C]
    with pytest.raises(ValueError):
        kg.neighbours("svc:AB")


# ------------------------------------------------------------------ shared_elements


def test_shared_elements(kg):
    got = kg.shared_elements(["path:A>B>D", "path:A>C>D"])
    assert got == [(A, 2), (D, 2), (AB, 1), (AC, 1), (BD, 1), (CD, 1), (B, 1), (C, 1)]


def test_shared_elements_kinds_and_empty(kg):
    assert kg.shared_elements(["path:A>B>D>E", "path:C>D>E"], kinds=("Link",)) == [(DE, 2), (AB, 1), (BD, 1), (CD, 1)]
    assert kg.shared_elements([]) == []
    with pytest.raises(ValueError):
        kg.shared_elements([AB])


def test_shared_elements_finds_common_link(kg):
    # every path to E crosses D--E; it should rank first among links
    paths = kg.paths_through(E)
    assert kg.shared_elements(paths, kinds=("Link",))[0] == (DE, len(paths))


# ------------------------------------------------------------------ changes_between and dynamics


def test_changes_between(kg):
    assert kg.changes_between(0, 100) == []
    kg.apply_change(5, links_down=[AB], reason="cut")
    kg.apply_change(9, links_up=[AB])
    ev = kg.changes_between(0, 5)
    assert ev[0] == {"t": 5, "type": "link_down", "element": AB, "reason": "cut"}
    reroutes = [e for e in ev if e["type"] == "reroute"]
    ab = next(e for e in reroutes if e["pair"] == ["A", "B"])
    assert ab["removed"] == ["path:A>B"] and ab["added"] == ["path:A>C>D>B"] and not ab["unreachable"]
    assert all(e["t"] == 5 for e in ev)
    assert [e["type"] for e in kg.changes_between(9, 9)][0] == "link_up"
    assert kg.changes_between(6, 8) == []


def test_restore_reopens_same_path_node(kg):
    kg.apply_change(5, links_down=[AB])
    kg.apply_change(9, links_up=[AB])
    assert kg.G.nodes["path:A>B"]["intervals"] == [[0, 5], [9, None]]
    assert kg.paths_between(A, B, t=7) == ["path:A>C>D>B"]
    assert kg.paths_between(A, B, t=9) == ["path:A>B"]
    assert kg.is_up(AB, t=7) is False and kg.is_up(AB, t=9) is True


def test_unreachable_and_pop_down(kg):
    kg.apply_change(2, pops_down=[D])
    ev = kg.changes_between(2, 2)
    assert {"t": 2, "type": "pop_down", "element": D, "reason": ""} in ev
    unreachable = sorted(tuple(e["pair"]) for e in ev if e["type"] == "reroute" and e["unreachable"])
    assert unreachable == [("A", "D"), ("A", "E"), ("B", "D"), ("B", "E"), ("C", "D"), ("C", "E"), ("D", "E")]
    assert kg.paths_through(D) == []
    assert kg.paths_between(B, C) == ["path:B>A>C"]


def test_metric_override(kg):
    kg.apply_change(4, metrics={AC: 100})
    assert kg.paths_between(A, D) == ["path:A>B>D"]
    assert kg.changes_between(4, 4)[0] == {"t": 4, "type": "metric_change", "element": AC, "old": 10, "new": 100, "reason": ""}
    kg.apply_change(6, metrics={AC: None})
    assert kg.paths_between(A, D) == ["path:A>B>D", "path:A>C>D"]


def test_time_cannot_go_backwards(kg):
    kg.apply_change(5)
    with pytest.raises(ValueError, match="before current time"):
        kg.apply_change(4)


def test_repeated_down_is_idempotent(kg):
    kg.apply_change(1, links_down=[AB])
    kg.apply_change(2, links_down=[AB])
    assert kg.G.nodes[AB]["intervals_down"] == [[1, None]]
    assert [e for e in kg.changes_between(2, 2) if e["type"] == "link_down"] == []


# ------------------------------------------------------------------ operational stubs


def test_operational_stubs(kg):
    aid = kg.add_alert("a1", DE, 7, metric="loss", severity="major", value=0.02)
    assert kg.G.nodes[aid]["kind"] == "Alert" and kg.G.has_edge(aid, DE, key="ON")
    kg.add_alert("a2", "svc:AE", 7, metric="latency")
    mid = kg.add_maintenance("m1", [AB, C], 10, 20, "fibre work")
    assert kg.G.has_edge(mid, AB, key="AFFECTS") and kg.G.has_edge(mid, C, key="AFFECTS")
    assert [e["type"] for e in kg.changes_between(10, 20)] == ["maintenance_start", "maintenance_end"]
    rid = kg.add_runbook("r1", "Link down", ["link_down", "link_flapping"])
    assert sorted(v for _, v in kg.G.out_edges(rid)) == ["fault:link_down", "fault:link_flapping"]
    with pytest.raises(KeyError):
        kg.add_runbook("r2", "x", ["not_a_fault"])
    assert kg.validate()


# ------------------------------------------------------------------ export


def test_export_csv(kg, tmp_path):
    kg.apply_change(5, links_down=[AB])
    files = kg.export_csv(tmp_path)
    assert "nodes_PoP.csv" in files and "rels_TRAVERSES.csv" in files
    for kind in ("PoP", "Link", "RoutingPath", "Service", "FaultType"):
        df = pd.read_csv(tmp_path / f"nodes_{kind}.csv")
        assert list(df.columns)[0] == "id:ID" and list(df.columns)[-1] == ":LABEL"
        assert len(df) == len(kg.nodes_of_kind(kind)) and set(df[":LABEL"]) == {kind}
    for rel in ("CONNECTS", "TRAVERSES", "USES"):
        df = pd.read_csv(tmp_path / f"rels_{rel}.csv")
        assert list(df.columns)[:3] == [":START_ID", ":END_ID", ":TYPE"]
        assert len(df) == sum(1 for *_, k in kg.G.edges(keys=True) if k == rel)
    trav = pd.read_csv(tmp_path / "rels_TRAVERSES.csv")
    row = trav[trav[":START_ID"] == "path:A>B>D>E"]
    assert list(row[":END_ID"]) == [AB, BD, DE] and list(row["order:long"]) == [0, 1, 2]
    paths = pd.read_csv(tmp_path / "nodes_RoutingPath.csv").set_index("id:ID")
    assert str(paths.loc["path:A>B", "active_to:long[]"]) == "5"
    assert str(paths.loc["path:A>C>D>B", "active_to:long[]"]) == "-1"
    links = pd.read_csv(tmp_path / "nodes_Link.csv").set_index("id:ID")
    assert "metric:long" in links.columns and "dist_km:float" in links.columns


# ------------------------------------------------------------------ real topologies


def test_generate_services_deterministic(cfg):
    topo = load("topozoo/Aarnet")
    args = (topo, 38, cfg["kg"]["service_classes"], cfg["seeds"]["kg_services"], cfg["kg"]["fibre_km_per_ms"])
    s1, s2 = generate_services(*args), generate_services(*args)
    assert s1 == s2 and len(s1) == 38
    assert len({(s["src"], s["dst"]) for s in s1}) == 38
    assert all(s["src"] < s["dst"] and s["sla_latency_ms"] > 0 for s in s1)


@pytest.mark.parametrize("name", ["topozoo/Aarnet", "topozoo/Aconet", "topozoo/BeyondTheNetwork", "topozoo/Quest"])  # dev + validation, never test
def test_real_topologies(cfg, name):
    topo = load(name)
    kg = NetTwinKG.from_config(topo, cfg)
    assert kg.validate()
    s = kg.summary()
    n = topo.number_of_nodes()
    assert s["nodes_by_kind"]["PoP"] == n and s["nodes_by_kind"]["Link"] == topo.number_of_edges()
    assert s["reachable_pairs"] == n * (n - 1) // 2
    assert s["nodes_by_kind"]["Service"] == cfg["kg"]["services_per_pop"] * n
    for svc in kg.nodes_of_kind("Service"):
        assert kg.service_paths(svc)
    for a, b in itertools.islice(itertools.combinations(sorted(topo.nodes), 2), 50):
        assert kg.paths_between(pop_node(a), pop_node(b))


def test_aarnet_failure_reroutes_services(cfg):
    kg = NetTwinKG.from_config(load("topozoo/Aarnet"), cfg)
    busiest = max(kg.nodes_of_kind("Link"), key=lambda l: (len(kg.paths_through(l)), l))
    before = kg.services_on(busiest)
    kg.apply_change(1, links_down=[busiest])
    assert kg.paths_through(busiest) == [] and kg.services_on(busiest) == []
    assert all(kg.service_paths(s) for s in before)  # the busiest link is not a bridge, so every service reroutes
    assert kg.validate()


def test_schema_constants():
    assert set(KINDS) >= {"PoP", "Link", "RoutingPath", "Service", "Alert", "Maintenance", "Runbook"}
    assert SCHEMA["TRAVERSES"] == ({"RoutingPath"}, {"Link"})
    assert path_node(["A", "B"]) == "path:A>B"
