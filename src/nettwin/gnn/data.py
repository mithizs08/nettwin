"""Line-graph data for the Phase 5 GNN.

Nodes are links; two links are adjacent when they share a PoP (edge attribute: log degree of that PoP). Inputs per link
per step are the B2 window features (`baselines.b2_features`: relative levels, one-day lags, short rolling stats,
missing-data flags) over the last W steps, so the same banned-feature rules hold: no clock, no time since the start,
no file metadata, and step t sees data up to t only. Static per-link features come from the operator inventory:
link length, and (the "static path features", dropped in one ablation) the number of normal routing paths and
services over the link, plus the mean degree of its end PoPs.

Training samples: per train scenario and variant, K steps (half inside observable fault windows, half anywhere after
day 0, maintenance included), each with the W-step window of every link, per-link detection targets (label >= 1) and
the root elements of the observable faults active at that step (localisation target).
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from nettwin.baselines import DAY, FEATURES, b2_features, index_for
from nettwin.observed import inventory

W = 12
STATIC = ("log_km", "end_degree", "log_paths", "log_services")
PATH_STATIC = ("log_paths", "log_services")


@dataclass
class Graph:
    """Static structure of one topology for the model."""
    n_links: int
    n_pops: int
    edge_index: np.ndarray   # (2, E) line-graph edges, both directions
    edge_attr: np.ndarray    # (E, 1) log degree of the shared PoP
    link_pop: np.ndarray     # (2, 2L) link -> PoP incidence pairs
    static: np.ndarray       # (L, len(STATIC))
    pop_static: np.ndarray   # (P, 1) log degree
    elements: list


@lru_cache(maxsize=None)
def graph_for(topology, cfg_s):
    kg = inventory(topology, cfg_s)
    links, pops = kg.nodes_of_kind("Link"), kg.nodes_of_kind("PoP")
    pidx = {p: i for i, p in enumerate(pops)}
    ends = np.array([(pidx[f"pop:{kg.G.nodes[l]['u']}"], pidx[f"pop:{kg.G.nodes[l]['v']}"]) for l in links])
    deg = np.bincount(ends.ravel(), minlength=len(pops))
    src, dst, attr = [], [], []
    for p in range(len(pops)):
        at = np.flatnonzero((ends == p).any(1))
        for a in at:
            for b in at:
                if a != b:
                    src.append(a); dst.append(b); attr.append(np.log(deg[p]))
    static = np.stack([
        np.log1p([kg.G.nodes[l]["dist_km"] for l in links]),
        np.log(deg[ends]).mean(1),
        np.log1p([len(kg.paths_through(l, t=0)) for l in links]),
        np.log1p([len(kg.services_on(l, t=0)) for l in links]),
    ], axis=1)
    L = len(links)
    return Graph(L, len(pops), np.array([src, dst], dtype=np.int64).reshape(2, -1), np.array(attr, np.float32).reshape(-1, 1),
                 np.array([np.repeat(np.arange(L), 2), ends.ravel()], dtype=np.int64), static.astype(np.float32),
                 np.log(deg).astype(np.float32).reshape(-1, 1), links + pops)


def windows(feat, steps):
    """(len(steps), L, W, F) windows ending at each step (inclusive); steps before W - 1 repeat the first row."""
    T = feat.shape[0]
    idx = np.clip(np.asarray(steps)[:, None] + np.arange(-W + 1, 1)[None, :], 0, T - 1)
    return np.transpose(feat[idx], (0, 2, 1, 3))


def active_roots(entry, t, elements):
    """Element indices of the observable faults whose window contains step t."""
    return [elements.index(f["root_cause"]) for f in entry["faults"]
            if f["impact"]["observable"] and f["t_start"] <= t < f["t_end"]]


def training_samples(obs, entry, lab, cfg_s, k, rng):
    """K training samples from one scenario: windows (K, L, W, F) float16, detection targets (K, L), root lists."""
    feat = b2_features(obs).astype(np.float32)
    T = feat.shape[0]
    in_fault = np.zeros(T, bool)
    for f in entry["faults"]:
        if f["impact"]["observable"]:
            in_fault[max(f["t_start"], DAY):f["t_end"]] = True
    cand_f = np.flatnonzero(in_fault)
    cand_a = np.arange(DAY, T)
    nf = min(k // 2, len(cand_f))
    steps = np.concatenate([rng.choice(cand_f, nf, replace=False) if nf else np.zeros(0, int),
                            rng.choice(cand_a, k - nf, replace=False)]).astype(int)
    g = graph_for(obs.topology, cfg_s)
    return (windows(feat, steps).astype(np.float16), (lab[steps] >= 1).astype(np.float32),
            [active_roots(entry, t, g.elements) for t in steps])


def check_inputs(obs, kg):
    """The model sees the same links, in the same order, as the inventory (and so as the harness)."""
    return index_for(obs, kg)


N_FEATURES = len(FEATURES)
