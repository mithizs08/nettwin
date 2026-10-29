"""Typed multi-layer knowledge graph and query API (PLAN.md section 5).

The KG is a `nx.MultiDiGraph`. Every node has `kind`; every edge has `rel` (also used as the edge key).

Node ids:  pop:<name>  link:<a>--<b>  path:<a>><...>><b>  svc:<id>  alert:<id>  mnt:<id>  rb:<id>  fault:<type>

Layers built now (Phase 1): PoP, Link, RoutingPath, Service. Alert, Maintenance and Runbook exist as schema
plus `add_*` methods; they are populated from Phase 2 (alerts, maintenance) and Phase 6 (runbooks).

Time is an integer step index (5-minute steps from the simulator). Routing is recomputed on every
`apply_change`; RoutingPath nodes are never deleted, they carry `intervals` of [start, end) during which
they were active (end None means still active). This is how the KG answers "what did the path look like at t".
Paths are kept per unordered PoP pair, written from the lexicographically smaller PoP to the larger.
"""
import itertools
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from nettwin.topology import link_id

KINDS = ("PoP", "Link", "RoutingPath", "Service", "Alert", "Maintenance", "Runbook", "FaultType")
SCHEMA = {
    "CONNECTS": ({"Link"}, {"PoP"}),
    "TRAVERSES": ({"RoutingPath"}, {"Link"}),
    "USES": ({"Service"}, {"RoutingPath"}),
    "ON": ({"Alert"}, {"Link", "Service"}),
    "AFFECTS": ({"Maintenance"}, {"Link", "PoP"}),
    "APPLIES_TO": ({"Runbook"}, {"FaultType"}),
}
FAULT_TYPES = (
    "link_down",
    "node_down",
    "link_degradation",
    "link_flapping",
    "demand_surge",
    "routing_misconfig",
)


def pop_node(name):
    return f"pop:{name}"


def link_node(u, v):
    return f"link:{link_id(u, v)}"


def path_node(pops):
    return "path:" + ">".join(pops)


def _active(intervals, t):
    return any(s <= t and (e is None or t < e) for s, e in intervals)


def generate_services(topo, n, classes, seed, km_per_ms):
    """Sample n synthetic customer services between distinct PoP pairs, with SLAs from propagation latency."""
    pairs = list(itertools.combinations(sorted(topo.nodes), 2))
    rng = np.random.default_rng([seed, zlib.crc32(topo.graph["name"].encode())])
    chosen = sorted(rng.choice(len(pairs), size=min(n, len(pairs)), replace=False))
    names = list(classes)
    shares = np.array([classes[c]["share"] for c in names], dtype=float)
    services = []
    for i, j in enumerate(chosen):
        src, dst = pairs[j]
        cls = names[rng.choice(len(names), p=shares / shares.sum())]
        spec = classes[cls]
        hops = nx.shortest_path(topo, src, dst, weight="metric")
        latency = sum(topo.edges[a, b]["dist_km"] for a, b in zip(hops, hops[1:])) / km_per_ms
        services.append(
            {
                "id": f"svc:S{i:03d}",
                "src": src,
                "dst": dst,
                "cls": cls,
                "sla_latency_ms": round(latency * spec["latency_factor"] + spec["latency_margin_ms"], 3),
                "sla_loss_pct": spec["loss_pct"],
            }
        )
    return services


class NetTwinKG:
    def __init__(self, topo, *, ecmp_max_paths=4, fibre_km_per_ms=200.0, services=()):
        self.topo = topo
        self.name = topo.graph["name"]
        self.ecmp_max_paths = ecmp_max_paths
        self.fibre_km_per_ms = fibre_km_per_ms
        self.G = nx.MultiDiGraph(name=self.name)
        self.t = 0
        self.events = []
        self._down_links, self._down_pops, self._metric = set(), set(), {}
        self._pairs = list(itertools.combinations(sorted(topo.nodes), 2))
        self._active_paths = {}
        self._pair_paths = defaultdict(list)
        self._pair_services = defaultdict(list)
        self._build_physical()
        for ft in FAULT_TYPES:
            self.G.add_node(f"fault:{ft}", kind="FaultType", name=ft)
        for s in services:
            self._add_service(s)
        self._sync_paths(0, initial=True)

    @classmethod
    def from_config(cls, topo, cfg):
        kg = cfg["kg"]
        services = generate_services(
            topo,
            kg["services_per_pop"] * topo.number_of_nodes(),
            kg["service_classes"],
            cfg["seeds"]["kg_services"],
            kg["fibre_km_per_ms"],
        )
        return cls(
            topo,
            ecmp_max_paths=kg["ecmp_max_paths"],
            fibre_km_per_ms=kg["fibre_km_per_ms"],
            services=services,
        )

    # ---------------------------------------------------------------- construction

    def _edge(self, src, dst, rel, **attrs):
        allowed_src, allowed_dst = SCHEMA[rel]
        ks, kd = self.G.nodes[src]["kind"], self.G.nodes[dst]["kind"]
        if ks not in allowed_src or kd not in allowed_dst:
            raise ValueError(f"{rel} not allowed from {ks} to {kd}")
        self.G.add_edge(src, dst, key=rel, rel=rel, **attrs)

    def _build_physical(self):
        for n, d in sorted(self.topo.nodes(data=True)):
            x, y = d["pos"]
            self.G.add_node(pop_node(n), kind="PoP", name=n, x=x, y=y, intervals_down=[])
        for u, v, d in sorted(self.topo.edges(data=True), key=lambda e: e[2]["link_id"]):
            a, b = sorted((u, v))
            lid = link_node(a, b)
            self.G.add_node(lid, kind="Link", u=a, v=b, dist_km=d["dist_km"], metric=d["metric"], intervals_down=[])
            self._edge(lid, pop_node(a), "CONNECTS")
            self._edge(lid, pop_node(b), "CONNECTS")

    def _add_service(self, s):
        src, dst = sorted((s["src"], s["dst"]))
        attrs = {k: v for k, v in s.items() if k not in ("id", "src", "dst")}
        self.G.add_node(s["id"], kind="Service", src=src, dst=dst, **attrs)
        self._pair_services[(src, dst)].append(s["id"])

    def _add_path(self, pops):
        pid = path_node(pops)
        if pid in self.G:
            return pid
        dist = sum(self.topo.edges[a, b]["dist_km"] for a, b in zip(pops, pops[1:]))
        self.G.add_node(
            pid,
            kind="RoutingPath",
            src=pops[0],
            dst=pops[-1],
            pops=list(pops),
            hops=len(pops) - 1,
            dist_km=round(dist, 3),
            prop_latency_ms=round(dist / self.fibre_km_per_ms, 4),
            intervals=[],
        )
        for i, (a, b) in enumerate(zip(pops, pops[1:])):
            self._edge(pid, link_node(a, b), "TRAVERSES", order=i)
        for sid in self._pair_services[(pops[0], pops[-1])]:
            self._edge(sid, pid, "USES")
        self._pair_paths[(pops[0], pops[-1])].append(pid)
        return pid

    # ---------------------------------------------------------------- routing and dynamics

    def _routing_graph(self):
        h = nx.Graph()
        h.add_nodes_from(n for n in self.topo if pop_node(n) not in self._down_pops)
        for u, v, d in self.topo.edges(data=True):
            lid = link_node(u, v)
            if lid in self._down_links or u not in h or v not in h:
                continue
            h.add_edge(u, v, metric=self._metric.get(lid, d["metric"]))
        return h

    def _enumerate(self, pred, src, dst):
        """Up to ecmp_max_paths equal-cost paths src -> dst from Dijkstra predecessors, deterministic order."""
        out = []

        def back(node, suffix):
            if len(out) >= self.ecmp_max_paths:
                return
            if node == src:
                out.append((src,) + suffix)
                return
            for p in sorted(pred[node]):
                back(p, (node,) + suffix)

        back(dst, ())
        return sorted(out)

    def _compute_routes(self):
        h = self._routing_graph()
        routes = {}
        nodes = sorted(self.topo.nodes)
        for s in nodes:
            if s not in h:
                continue
            pred, dist = nx.dijkstra_predecessor_and_distance(h, s, weight="metric")
            for d in nodes:
                if d > s and d in dist:
                    routes[(s, d)] = self._enumerate(pred, s, d)
        return routes

    def _sync_paths(self, t, initial=False):
        routes = self._compute_routes()
        for pair in self._pairs:
            old = self._active_paths.get(pair, ())
            new = tuple(self._add_path(p) for p in routes.get(pair, []))
            if old == new:
                continue
            for pid in set(old) - set(new):
                self.G.nodes[pid]["intervals"][-1][1] = t
            for pid in set(new) - set(old):
                self.G.nodes[pid]["intervals"].append([t, None])
            self._active_paths[pair] = new
            if not initial:
                self.events.append(
                    {
                        "t": t,
                        "type": "reroute",
                        "pair": list(pair),
                        "removed": sorted(set(old) - set(new)),
                        "added": sorted(set(new) - set(old)),
                        "unreachable": not new,
                    }
                )

    def apply_change(self, t, *, links_down=(), links_up=(), pops_down=(), pops_up=(), metrics=None, reason=""):
        """Change element state at step t and re-derive routing. metrics maps link id to a new metric (None resets)."""
        if t < self.t:
            raise ValueError(f"t={t} is before current time {self.t}")
        self.t = t
        for ids, kind, down_set, going_down, etype in (
            (links_down, "Link", self._down_links, True, "link_down"),
            (links_up, "Link", self._down_links, False, "link_up"),
            (pops_down, "PoP", self._down_pops, True, "pop_down"),
            (pops_up, "PoP", self._down_pops, False, "pop_up"),
        ):
            for e in ids:
                self._require(e, kind)
                if going_down == (e in down_set):
                    continue
                ivs = self.G.nodes[e]["intervals_down"]
                if going_down:
                    down_set.add(e)
                    ivs.append([t, None])
                else:
                    down_set.discard(e)
                    ivs[-1][1] = t
                self.events.append({"t": t, "type": etype, "element": e, "reason": reason})
        for lid, m in (metrics or {}).items():
            self._require(lid, "Link")
            old = self._metric.get(lid, self.G.nodes[lid]["metric"])
            if m is None:
                self._metric.pop(lid, None)
            else:
                self._metric[lid] = int(m)
            new = self._metric.get(lid, self.G.nodes[lid]["metric"])
            self.events.append({"t": t, "type": "metric_change", "element": lid, "old": old, "new": new, "reason": reason})
        self._sync_paths(t)

    # ---------------------------------------------------------------- helpers

    def _t(self, t):
        return self.t if t is None else t

    def kind(self, element):
        if element not in self.G:
            raise KeyError(element)
        return self.G.nodes[element]["kind"]

    def _require(self, element, *kinds):
        k = self.kind(element)
        if k not in kinds:
            raise ValueError(f"{element} is a {k}, expected {' or '.join(kinds)}")

    def _in(self, node, rel):
        return [u for u, _, k in self.G.in_edges(node, keys=True) if k == rel]

    def _out(self, node, rel):
        return [v for _, v, k in self.G.out_edges(node, keys=True) if k == rel]

    def links_of_path(self, path):
        """Links of a RoutingPath in traversal order."""
        self._require(path, "RoutingPath")
        edges = [(d["order"], v) for _, v, k, d in self.G.out_edges(path, keys=True, data=True) if k == "TRAVERSES"]
        return [v for _, v in sorted(edges)]

    def is_up(self, element, t=None):
        self._require(element, "Link", "PoP")
        return not _active(self.G.nodes[element]["intervals_down"], self._t(t))

    def nodes_of_kind(self, kind):
        return sorted(n for n, d in self.G.nodes(data=True) if d["kind"] == kind)

    # ---------------------------------------------------------------- query API

    def paths_through(self, element, t=None):
        """RoutingPaths active at t that traverse a Link, or pass through a PoP (endpoints included)."""
        t = self._t(t)
        self._require(element, "Link", "PoP")
        links = [element] if self.kind(element) == "Link" else self._in(element, "CONNECTS")
        cands = {p for lid in links for p in self._in(lid, "TRAVERSES")}
        return sorted(p for p in cands if _active(self.G.nodes[p]["intervals"], t))

    def services_on(self, element, t=None):
        """Services whose active paths at t use a Link, PoP or RoutingPath."""
        t = self._t(t)
        if self.kind(element) == "RoutingPath":
            paths = [element] if _active(self.G.nodes[element]["intervals"], t) else []
        else:
            paths = self.paths_through(element, t)
        return sorted({s for p in paths for s in self._in(p, "USES")})

    def neighbours(self, element, hops=1):
        """Physical elements of the same kind within `hops` (PoP: adjacent PoPs; Link: links sharing a PoP).

        Uses the static topology, ignoring current up/down state, so failed elements still have neighbours.
        """
        self._require(element, "Link", "PoP")
        if self.kind(element) == "PoP":
            adj = lambda n: [pop_node(m) for m in self.topo.neighbors(self.G.nodes[n]["name"])]
        else:
            adj = lambda n: [lid for pop in self._out(n, "CONNECTS") for lid in self._in(pop, "CONNECTS") if lid != n]
        seen, frontier = {element}, {element}
        for _ in range(hops):
            frontier = {m for n in frontier for m in adj(n)} - seen
            seen |= frontier
        return sorted(seen - {element})

    def shared_elements(self, paths, kinds=("Link", "PoP")):
        """Count how many of the given RoutingPaths contain each Link / PoP. Sorted by count desc, then id."""
        counts = Counter()
        for p in paths:
            self._require(p, "RoutingPath")
            if "Link" in kinds:
                counts.update(self.links_of_path(p))
            if "PoP" in kinds:
                counts.update(pop_node(n) for n in self.G.nodes[p]["pops"])
        return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))

    def changes_between(self, t0, t1):
        """Logged changes (element state, metric, reroutes, maintenance) with t0 <= t <= t1, ordered by t."""
        return sorted((dict(e) for e in self.events if t0 <= e["t"] <= t1), key=lambda e: e["t"])

    def paths_between(self, src, dst, t=None):
        """Active RoutingPaths between two PoPs at t (ECMP gives more than one)."""
        self._require(src, "PoP")
        self._require(dst, "PoP")
        pair = tuple(sorted((self.G.nodes[src]["name"], self.G.nodes[dst]["name"])))
        t = self._t(t)
        return sorted(p for p in self._pair_paths[pair] if _active(self.G.nodes[p]["intervals"], t))

    def service_paths(self, service, t=None):
        """Active RoutingPaths a Service uses at t; empty if its endpoints are disconnected."""
        self._require(service, "Service")
        t = self._t(t)
        return sorted(p for p in self._out(service, "USES") if _active(self.G.nodes[p]["intervals"], t))

    # ---------------------------------------------------------------- operational layer (stubs, filled later)

    def add_alert(self, alert_id, target, t, *, metric, severity="warning", value=None):
        """Alert -ON-> Link|Service. Populated from telemetry in Phase 2."""
        aid = f"alert:{alert_id}"
        self.G.add_node(aid, kind="Alert", t=t, metric=metric, severity=severity, value=value)
        self._edge(aid, target, "ON")
        return aid

    def add_maintenance(self, mnt_id, targets, t_start, t_end, description=""):
        """Maintenance -AFFECTS-> Link|PoP, also written to the change log. Populated in Phase 2."""
        mid = f"mnt:{mnt_id}"
        self.G.add_node(mid, kind="Maintenance", t_start=t_start, t_end=t_end, description=description)
        for e in targets:
            self._edge(mid, e, "AFFECTS")
        for t, etype in ((t_start, "maintenance_start"), (t_end, "maintenance_end")):
            self.events.append({"t": t, "type": etype, "maintenance": mid, "targets": sorted(targets)})
        return mid

    def add_runbook(self, rb_id, title, fault_types, text=""):
        """Runbook -APPLIES_TO-> FaultType. Written in Phase 6."""
        rid = f"rb:{rb_id}"
        self.G.add_node(rid, kind="Runbook", title=title, text=text)
        for ft in fault_types:
            self._edge(rid, f"fault:{ft}", "APPLIES_TO")
        return rid

    # ---------------------------------------------------------------- checks, summary, export

    def validate(self):
        """Raise ValueError if any node or edge breaks the schema, or a path's links are not contiguous."""
        for n, d in self.G.nodes(data=True):
            if d.get("kind") not in KINDS:
                raise ValueError(f"{n}: unknown kind {d.get('kind')}")
        for u, v, k, d in self.G.edges(keys=True, data=True):
            if k not in SCHEMA or d.get("rel") != k:
                raise ValueError(f"{u}->{v}: bad rel {k}")
            src_ok, dst_ok = SCHEMA[k]
            if self.G.nodes[u]["kind"] not in src_ok or self.G.nodes[v]["kind"] not in dst_ok:
                raise ValueError(f"{u}-{k}->{v}: kinds not allowed")
        for p in self.nodes_of_kind("RoutingPath"):
            pops = self.G.nodes[p]["pops"]
            if self.links_of_path(p) != [link_node(a, b) for a, b in zip(pops, pops[1:])]:
                raise ValueError(f"{p}: TRAVERSES edges do not match its PoP sequence")
        return True

    def summary(self):
        kinds = Counter(d["kind"] for _, d in self.G.nodes(data=True))
        rels = Counter(k for _, _, k in self.G.edges(keys=True))
        active = [len(v) for v in self._active_paths.values() if v]
        return {
            "topology": self.name,
            "nodes_by_kind": {k: kinds.get(k, 0) for k in KINDS},
            "edges_by_rel": {r: rels.get(r, 0) for r in SCHEMA},
            "pop_pairs": len(self._pairs),
            "reachable_pairs": len(active),
            "ecmp_pairs": sum(n > 1 for n in active),
            "services_by_class": dict(sorted(Counter(self.G.nodes[s]["cls"] for s in self.nodes_of_kind("Service")).items())),
        }

    def export_csv(self, out_dir):
        """Neo4j-style CSV (neo4j-admin import format): nodes_<Kind>.csv and rels_<REL>.csv.

        Import with e.g.: neo4j-admin database import full --nodes=nodes_PoP.csv ... --relationships=rels_CONNECTS.csv ...
        Intervals are flattened to active_from/active_to (or down_from/down_to) long arrays; -1 marks an open end.
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        written = []
        for kind in KINDS:
            ids = self.nodes_of_kind(kind)
            if not ids:
                continue
            rows = [self._csv_row(n) for n in ids]
            cols = ["id:ID"] + sorted({c for r in rows for c in r} - {"id:ID", ":LABEL"}) + [":LABEL"]
            f = out / f"nodes_{kind}.csv"
            pd.DataFrame(rows, columns=cols).to_csv(f, index=False)
            written.append(f.name)
        for rel in SCHEMA:
            edges = sorted(
                ((u, v, d) for u, v, k, d in self.G.edges(keys=True, data=True) if k == rel),
                key=lambda e: (e[0], e[2].get("order", 0), e[1]),
            )
            if not edges:
                continue
            rows = [{":START_ID": u, ":END_ID": v, ":TYPE": rel, **({"order:long": d["order"]} if "order" in d else {})} for u, v, d in edges]
            f = out / f"rels_{rel}.csv"
            pd.DataFrame(rows).to_csv(f, index=False)
            written.append(f.name)
        return written

    def _csv_row(self, n):
        d = self.G.nodes[n]
        row = {"id:ID": n, ":LABEL": d["kind"]}
        for k, v in d.items():
            if k == "kind":
                continue
            if k in ("intervals", "intervals_down"):
                prefix = "active" if k == "intervals" else "down"
                row[f"{prefix}_from:long[]"] = ";".join(str(s) for s, _ in v)
                row[f"{prefix}_to:long[]"] = ";".join(str(-1 if e is None else e) for _, e in v)
            elif isinstance(v, list):
                row[f"{k}:string[]"] = ";".join(map(str, v))
            elif isinstance(v, bool) or v is None:
                row[k] = "" if v is None else str(v).lower()
            elif isinstance(v, int):
                row[f"{k}:long"] = v
            elif isinstance(v, float):
                row[f"{k}:float"] = v
            else:
                row[k] = v
        return row
