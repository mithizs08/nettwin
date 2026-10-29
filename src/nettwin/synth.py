"""Generated training / validation topologies: topohub's Gabriel generator plus augmentation (audit M4, Q6).

A graph 'gabaug/<n>/<seed>' is built deterministically:
1. Draw one network from the reference pool (non-test, non-dev, non-validation Topology Zoo networks picked by
   scripts/select_topologies.py) and take its structure as targets, each jittered x U(0.85, 1.15):
   average degree, leaf fraction, zero-length (dual-router) link fraction, mean link length in km, longitude span.
2. Generate a Gabriel core with topohub's generator (stdout silenced, global `random` state restored).
3. Thin the core by removing random non-bridge links until the target average degree is reached.
4. Rescale coordinates so the mean link length matches the target km.
5. Add leaf spurs and dual-router PoPs (a twin at the same position, a 0 km link, half the links moved to it).
6. Assign a pseudo-longitude across the target span, so PoPs get local time zones.
"""
import contextlib
import functools
import io
import json
import math
import random
from pathlib import Path

import networkx as nx
import numpy as np

from nettwin.topology import from_node_link, load

ROOT = Path(__file__).resolve().parents[2]


@functools.lru_cache(maxsize=None)
def _reference_stats(pool_file):
    pool = json.loads(Path(pool_file).read_text())["reference_pool"]
    stats = []
    for name in pool:
        g = load(name)
        n, m = g.number_of_nodes(), g.number_of_edges()
        dists = [d["dist_km"] for *_, d in g.edges(data=True)]
        lons = [g.nodes[v]["pos"][0] for v in g]
        stats.append(
            {
                "name": name,
                "nodes": n,
                "avg_degree": 2 * m / n,
                "leaf_frac": sum(1 for v in g if g.degree(v) == 1) / n,
                "zero_frac": sum(1 for x in dists if x == 0) / m,
                "mean_km": float(np.mean([x for x in dists if x > 0])) if any(x > 0 for x in dists) else 0.0,
                "lon_span": min(180.0, max(lons) - min(lons)),
            }
        )
    return tuple(stats)


def reference_stats(cfg):
    return _reference_stats(str(ROOT / cfg["topologies"]["reference_pool_file"]))


def _gabriel_core(n, seed):
    from topohub.providers.gabriel import GabrielGenerator

    state = random.getstate()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            data = GabrielGenerator.generate_topo(n, seed)
    finally:
        random.setstate(state)
    g = nx.Graph()
    for nd in data["nodes"]:
        g.add_node(f"R{nd['id']}", pos=tuple(nd["pos"]))
    for e in data["edges"]:
        g.add_edge(f"R{e['source']}", f"R{e['target']}")
    return g


def generate(n, seed, cfg):
    stats = reference_stats(cfg)
    rng = np.random.default_rng([seed, 31])
    ref = stats[int(rng.integers(len(stats)))]
    jit = lambda: float(rng.uniform(0.85, 1.15))  # noqa: E731
    km_pool = [s["mean_km"] for s in stats if s["mean_km"] > 0]
    deg_t = float(np.clip(ref["avg_degree"] * jit(), 2.1, 4.0))
    leaf_t = float(np.clip(ref["leaf_frac"] * jit(), 0.0, 0.3))
    zero_t = float(np.clip(ref["zero_frac"] * jit(), 0.0, 0.3))
    km_t = (ref["mean_km"] if ref["mean_km"] > 0 else float(np.median(km_pool))) * jit()
    lon_t = max(1.0, ref["lon_span"] * jit())
    n_dual = min(n // 5, round(zero_t * deg_t * n / 2))
    n_leaf = min(n // 4, round(leaf_t * n))
    n_core = max(8, n - n_dual - n_leaf)
    n_leaf = max(0, n - n_core - n_dual)
    n_dual = n - n_core - n_leaf

    g = _gabriel_core(n_core, seed)
    # 3. thin toward the target degree, never disconnecting and never creating a leaf (leaves come from step 5a)
    m_target = max(n_core, round(deg_t * n / 2) - n_leaf - n_dual)
    while g.number_of_edges() > m_target:
        bridges = set(nx.bridges(g))
        cands = sorted(
            e for e in g.edges
            if e not in bridges and (e[1], e[0]) not in bridges and g.degree(e[0]) >= 3 and g.degree(e[1]) >= 3
        )
        if not cands:
            break
        g.remove_edge(*cands[int(rng.integers(len(cands)))])
    # 4. rescale to backbone km
    pos = {v: np.array(g.nodes[v]["pos"], dtype=float) for v in g}
    mean_len = float(np.mean([np.linalg.norm(pos[u] - pos[v]) for u, v in g.edges]))
    s = km_t / mean_len
    pos = {v: p * s for v, p in pos.items()}
    dist = {tuple(sorted(e)): round(float(np.linalg.norm(pos[e[0]] - pos[e[1]])), 2) for e in g.edges}
    core = sorted(g.nodes, key=lambda v: int(v[1:]))
    # 5a. leaf spurs
    for k in range(n_leaf):
        parent = core[int(rng.integers(len(core)))]
        length = km_t * float(rng.lognormal(math.log(0.6), 0.4))
        ang = float(rng.uniform(0, 2 * math.pi))
        leaf = f"L{k}"
        pos[leaf] = pos[parent] + length * np.array([math.cos(ang), math.sin(ang)])
        g.add_edge(parent, leaf)
        dist[tuple(sorted((parent, leaf)))] = round(length, 2)
    # 5b. dual-router PoPs
    hosts = [v for v in core if g.degree(v) >= 2]
    for v in [hosts[i] for i in rng.permutation(len(hosts))[:n_dual]]:
        twin = f"{v}b"
        pos[twin] = pos[v].copy()
        nbrs = sorted(g.neighbors(v))
        moved = [nbrs[i] for i in rng.permutation(len(nbrs))[: max(1, len(nbrs) // 2)]]
        for u in moved:
            d = dist.pop(tuple(sorted((u, v))))
            g.remove_edge(u, v)
            g.add_edge(twin, u)
            dist[tuple(sorted((twin, u)))] = d
        g.add_edge(v, twin)
        dist[tuple(sorted((v, twin)))] = 0.0
    # 6. pseudo-longitude
    xs = np.array([p[0] for p in pos.values()])
    x0, span = xs.min(), max(xs.max() - xs.min(), 1e-9)
    name = f"gabaug/{n}/{seed}"
    data = {
        "directed": False,
        "multigraph": False,
        "graph": {"name": name, "demands": {}},
        "nodes": [{"id": v, "pos": [round(float(pos[v][0]), 2), round(float(pos[v][1]), 2)], "lon": round(float((pos[v][0] - x0) / span * lon_t), 3)} for v in sorted(g.nodes)],
        "edges": [{"source": a, "target": b, "dist": d} for (a, b), d in sorted(dist.items())],
    }
    topo = from_node_link(data, name, geo=False)
    topo.graph["augmentation"] = {"reference": ref["name"], "avg_degree": deg_t, "leaf_frac": leaf_t, "zero_frac": zero_t, "mean_km": km_t, "lon_span": lon_t}
    return topo


def generated_split(cfg):
    """Names of all generated graphs, split by graph and stratified by size into train / val."""
    sc = cfg["topologies"]["synthetic"]
    lo, hi = sc["n_range"]
    sizes = list(range(lo, hi + 1))
    names = [f"{sc['prefix']}/{sizes[i % len(sizes)]}/{sc['seed_base'] + i}" for i in range(sc["count"])]
    rng = np.random.default_rng(sc["split_seed"])
    split = {"train": [], "val": []}
    for n in sizes:
        group = [x for x in names if int(x.split("/")[1]) == n]
        order = rng.permutation(len(group))
        k = round(sc["val_frac"] * len(group))
        for j, i in enumerate(order):
            split["val" if j < k else "train"].append(group[i])
    return {k: sorted(v, key=lambda x: (int(x.split("/")[1]), int(x.split("/")[2]))) for k, v in split.items()}
