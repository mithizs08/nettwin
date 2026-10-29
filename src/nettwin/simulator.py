"""Telemetry simulator (PLAN.md section 4; revised after the Phase 0 to 2 audit, docs/audit_phases_0_2.md).

One `Simulator` per topology holds the fixed network design (demand weights, link capacities, synthetic
services). `simulate(spec)` turns a scenario spec (see faults.py) into telemetry, deterministically from
spec["seed"].

What the simulator produces, and who may see it:

  Observable (inputs for detectors and the agent)
    link telemetry per step:    util (interface counters), latency_ms (one-hop probe RTT estimate),
                                loss_pct (interface discard + error counters)
    service telemetry per step: latency_ms, loss_pct from end-to-end probes (binomial on true loss plus
                                background probe loss)
    observed events:            link status and PoP reachability from polling (with lag), planned-maintenance log
  Ground truth (evaluation only, written to a separate directory)
    the fault list, noise-free metrics, true routing, per-fault impact and observability, per-link labels

The twin's routing view (`observed_kg`) is rebuilt from observed events only, using intended link metrics, so a
routing misconfiguration is invisible to it. Gray failures and demand surges never appear as events.

Model summary
  demand      gravity model (degree x lognormal weights; real matrix if the topology has one), diurnal profile in
              each PoP's local time (real or pseudo-longitude), weekend dip, per-PoP day factors, AR(1) log-noise,
              event days ramped over an hour
  routing     Phase 1 KG shortest paths, ECMP split evenly, per distinct network state; a flapping link
              time-shares its up and down states within a step
  carried     per-path fixed point on congested steps: flow reaching hop k is the offered flow times survival
              on hops 1..k-1 (traffic dropped upstream no longer loads downstream links)
  capacity    peak of a noisy fault-free history week / U(0.4, 0.7) (hot links U(0.8, 0.95)), floored; half the
              non-hot links also sized for the worst single-link failure; 0 km links get the largest adjacent
              capacity; fixed per topology from seeds.network
  latency     propagation (dist / 200 km per ms) + per-hop + queueing scale x rho / (1 - rho), capped at buffer
  loss        ~0 below rho 0.9, cubic rise to 0.2% at capacity, plus all excess above capacity; gray loss above
              ~0.5% also cuts goodput (Mathis-style 1/sqrt(p)); hard downs and restores lose traffic for the
              convergence time
"""
import json
import zlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from nettwin.kg import NetTwinKG, link_node, pop_node
from nettwin.topology import local_longitude

STATUS_LAG_MAX = 3  # steps; observability windows extend this far past a fault


def queue_delay_ms(rho, scale_ms, buffer_ms):
    r = np.minimum(rho, 0.999999)
    q = np.where(rho < 1.0, scale_ms * r / (1.0 - r), buffer_ms)
    return np.minimum(q, buffer_ms)


def congestion_loss(rho, knee, at_capacity_pct):
    """Drop probability from congestion: microburst loss above the knee plus all offered load above capacity."""
    x = np.clip((rho - knee) / (1.0 - knee), 0.0, 1.0)
    burst = at_capacity_pct / 100.0 * x**3
    excess = np.where(rho > 1.0, 1.0 - 1.0 / np.maximum(rho, 1.0), 0.0)
    return 1.0 - (1.0 - burst) * (1.0 - excess)


def _crc(s):
    return zlib.crc32(str(s).encode())


@dataclass
class Routing:
    rf: np.ndarray            # (L, P) fraction of directed demand p crossing link l in u->v direction
    rb: np.ndarray            # (L, P) same, v->u
    paths: np.ndarray         # (n_paths, 2L) incidence of service paths on directed links
    weights: np.ndarray       # (S, n_paths) ECMP weight of each path for each service
    reachable: np.ndarray     # (S,) bool
    pair_paths: dict = field(repr=False)
    hop_pair: np.ndarray = field(repr=False, default=None)   # (H,) directed pair of each hop occurrence
    hop_w: np.ndarray = field(repr=False, default=None)      # (H,) ECMP weight of its path
    hop_dl: np.ndarray = field(repr=False, default=None)     # (H,) directed link 2 * l + dir
    hop_first: np.ndarray = field(repr=False, default=None)  # (H,) index of the first hop of the same path
    hop_m: object = field(repr=False, default=None)          # (H, 2L) sparse one-hot of hop_dl


@dataclass
class SimResult:
    spec: dict
    links: list
    services: list
    obs: dict                 # util, latency_ms, loss_pct: (T, L); svc_latency_ms, svc_loss_pct: (T, S)
    truth: dict               # noise-free arrays, down_frac, routing keys per step
    observed_events: list
    impacts: dict = field(default_factory=dict)
    labels: np.ndarray = None  # (T, L) int8: 2 root, 1 symptomatic, 0 unaffected (audit M7)


class Simulator:
    def __init__(self, topo, cfg):
        self.topo, self.cfg = topo, cfg
        self.sc = cfg["simulator"]
        self.name = topo.graph["name"]
        self.kg = NetTwinKG.from_config(topo, cfg)
        self.links = self.kg.nodes_of_kind("Link")
        self.link_idx = {l: i for i, l in enumerate(self.links)}
        self.pops = sorted(topo.nodes)
        self.pop_idx = {p: i for i, p in enumerate(self.pops)}
        self.services = self.kg.nodes_of_kind("Service")
        self.pairs = [(a, b) for a in self.pops for b in self.pops if a != b]
        self.pair_idx = {p: i for i, p in enumerate(self.pairs)}
        self.pair_src = np.array([self.pop_idx[a] for a, _ in self.pairs])
        self.pair_dst = np.array([self.pop_idx[b] for _, b in self.pairs])
        self.step_minutes = self.sc["step_minutes"]
        self.steps_per_day = 24 * 60 // self.step_minutes
        self.n_steps = int(self.sc["days"] * self.steps_per_day)
        self.warmup_steps = int(self.sc["warmup_days"] * self.steps_per_day)
        G = self.kg.G
        self.base_metric = {l: G.nodes[l]["metric"] for l in self.links}
        self.link_ends = np.array([(self.pop_idx[G.nodes[l]["u"]], self.pop_idx[G.nodes[l]["v"]]) for l in self.links])
        self.prop_ms = np.array([G.nodes[l]["dist_km"] for l in self.links]) / cfg["kg"]["fibre_km_per_ms"]
        lon = [local_longitude(topo, p) for p in self.pops]
        if all(x is not None for x in lon):
            lon = np.array(lon, dtype=float)
            self.local_offset_h = (lon - lon.mean()) / 15.0
        else:
            self.local_offset_h = np.zeros(len(self.pops))
        self.svc_src = np.array([self.pop_idx[G.nodes[s]["src"]] for s in self.services], dtype=int)
        self.svc_dst = np.array([self.pop_idx[G.nodes[s]["dst"]] for s in self.services], dtype=int)
        self._cache = {}
        rng = np.random.default_rng([cfg["seeds"]["network"], _crc(self.name)])
        self.base_demand = self._base_demand(rng)
        self.capacity = self._design_capacity(rng)
        self.capacity2 = np.repeat(self.capacity, 2)

    # ------------------------------------------------------------------ network design

    def _base_demand(self, rng):
        dc = self.sc["demand"]
        real = self.topo.graph.get("demands") or {}
        if real:
            unknown = sorted(({a for a in real} | {b for row in real.values() for b in row}) - set(self.pops))
            if unknown:
                raise ValueError(f"{self.name}: demand matrix names not in topology: {unknown}")
            d = np.zeros(len(self.pairs))
            for a, row in real.items():
                for b, v in row.items():
                    if a != b:
                        d[self.pair_idx[(a, b)]] += v
                        d[self.pair_idx[(b, a)]] += v
            if d.sum() <= 0:
                raise ValueError(f"{self.name}: demand matrix is empty")
        else:
            deg = np.array([self.topo.degree(p) for p in self.pops], dtype=float)
            w = deg * rng.lognormal(0.0, dc["node_weight_sigma"], len(self.pops))
            d = w[self.pair_src] * w[self.pair_dst]
        return d / d.sum() * dc["total"]

    def _design_capacity(self, rng):
        """Capacity from the peak of a noisy fault-free 'history' week on intended routing (as an operator plans from
        measured traffic). Uses the network seed, not any scenario seed."""
        L = len(self.links)
        demand = self._demand(np.arange(self.n_steps), rng)
        r = self.routing(((), (), ()))
        peak = np.maximum((demand @ r.rf.T).max(0), (demand @ r.rb.T).max(0))
        target = rng.uniform(*self.sc["target_peak_utilisation"], L)
        hot = (rng.random(L) < self.sc["hot_links"]["frac"]) & (peak > 0)    # audit m8: never a zero-peak hot link
        target = np.where(hot, rng.uniform(*self.sc["hot_links"]["target_utilisation"], L), target)
        n1 = self.sc["n1_planning"]
        planned = (rng.random(L) < n1["planned_frac"]) & ~hot
        n1_target = rng.uniform(*n1["target_utilisation"], L)
        cap = np.where(peak > 0, peak / target, 0.0)
        floor = self.sc["capacity_floor_frac"] * np.median(cap[cap > 0])
        cap = np.where(hot, cap, np.maximum(cap, floor))
        self.hot_links = [l for l, h in zip(self.links, hot) if h]
        if planned.any():
            busiest = np.argsort(demand.sum(1))[-n1["peak_steps"]:]
            dpk = demand[busiest]
            n1_peak = np.zeros(L)
            for li, link in enumerate(self.links):
                rr = self.routing(((link,), (), ()))
                load = np.maximum((dpk @ rr.rf.T).max(0), (dpk @ rr.rb.T).max(0))
                load[li] = 0.0
                n1_peak = np.maximum(n1_peak, load)
            cap = np.where(planned, np.maximum(cap, n1_peak / n1_target), cap)
        # links between co-located routers (0 km) are cheap and sized generously: match the biggest adjacent link
        for li, l in enumerate(self.links):
            if self.kg.G.nodes[l]["dist_km"] == 0:
                ends = set(self.link_ends[li])
                adj = [j for j, (u, v) in enumerate(self.link_ends) if j != li and (u in ends or v in ends)]
                if adj:
                    cap[li] = max(cap[li], cap[adj].max())
        return np.where(cap > 0, cap, floor)

    def _node_profile(self, t):
        dc = self.sc["demand"]
        hours = t[:, None] * self.step_minutes / 60.0 + self.local_offset_h[None, :]
        h = hours % 24.0
        shape = 0.5 * (1 - np.cos(2 * np.pi * (h - dc["trough_hour"]) / 24.0))
        shape += 0.12 * np.sin(4 * np.pi * (h - dc["trough_hour"]) / 24.0)  # pushes the peak toward the evening
        prof = (1 - dc["diurnal_amplitude"]) + dc["diurnal_amplitude"] * np.clip(shape, 0, None)
        weekend = (np.floor(hours / 24.0) % 7) >= 5
        return prof * np.where(weekend, dc["weekend_factor"], 1.0)

    def _pair_profile(self, t):
        prof = self._node_profile(t)
        return np.sqrt(prof[:, self.pair_src] * prof[:, self.pair_dst])

    def event_factor(self, t, event_days):
        """Whole-network multiplier per step, ramped over an hour at day boundaries (audit minor: no hard step)."""
        day_idx = t // self.steps_per_day
        f = np.ones(len(t))
        for ev in event_days:
            f[day_idx == ev["day"]] *= ev["factor"]
        w = max(1, 60 // self.step_minutes)
        pad = np.pad(f, (w // 2, w - 1 - w // 2), mode="edge")
        return np.convolve(pad, np.ones(w) / w, mode="valid")

    def _demand(self, t, rng, event_days=()):
        """(T, P) directed demand: base x diurnal/weekly profile x per-PoP day factor x AR(1) log-noise x event days."""
        dc = self.sc["demand"]
        T, N = len(t), len(self.pops)
        n_days = int(np.ceil(T / self.steps_per_day)) + 1
        day_node = rng.normal(1.0, dc["day_factor_sigma"], (n_days, N))
        eps = rng.normal(0.0, 1.0, (T, len(self.pairs)))
        x = np.empty_like(eps)
        x[0] = eps[0] * dc["ar_sigma"] / np.sqrt(1 - dc["ar_phi"] ** 2)
        for i in range(1, T):
            x[i] = dc["ar_phi"] * x[i - 1] + dc["ar_sigma"] * eps[i]
        day_idx = t // self.steps_per_day
        day_pair = np.sqrt(day_node[day_idx][:, self.pair_src] * day_node[day_idx][:, self.pair_dst])
        return self.base_demand * self._pair_profile(t) * np.exp(x) * day_pair * self.event_factor(t, event_days)[:, None]

    # ------------------------------------------------------------------ routing via the Phase 1 KG

    def routing(self, key):
        """key = (links_down, pops_down, metric_overrides) as sorted tuples. Cached."""
        if key in self._cache:
            return self._cache[key]
        links_down, pops_down, metrics = key
        kg = NetTwinKG(self.topo, ecmp_max_paths=self.cfg["kg"]["ecmp_max_paths"], fibre_km_per_ms=self.cfg["kg"]["fibre_km_per_ms"])
        if links_down or pops_down or metrics:
            kg.apply_change(1, links_down=links_down, pops_down=[pop_node(p) for p in pops_down], metrics=dict(metrics))
        L, P = len(self.links), len(self.pairs)
        rf, rb = np.zeros((L, P)), np.zeros((L, P))
        pair_paths = {}
        hp, hw, hd, hf = [], [], [], []
        for a, b in ((a, b) for a, b in self.pairs if a < b):
            paths = [kg.G.nodes[p]["pops"] for p in kg.paths_between(pop_node(a), pop_node(b))]
            pair_paths[(a, b)] = paths
            if not paths:
                continue
            w = 1.0 / len(paths)
            for pops in paths:
                for seq in (pops, pops[::-1]):
                    d = self.pair_idx[(seq[0], seq[-1])]
                    first = len(hp)
                    for x, y in zip(seq, seq[1:]):
                        li = self.link_idx[link_node(x, y)]
                        fwd = x < y  # crosses the link u->v
                        (rf if fwd else rb)[li, d] += w
                        hp.append(d)
                        hw.append(w)
                        hd.append(2 * li + (0 if fwd else 1))
                        hf.append(first)
        rows, cols = [], []
        reachable = np.zeros(len(self.services), bool)
        for si, s in enumerate(self.services):
            src, dst = self.kg.G.nodes[s]["src"], self.kg.G.nodes[s]["dst"]
            paths = pair_paths[(src, dst)]
            reachable[si] = bool(paths)
            for pops in paths:
                v = np.zeros(2 * L)
                for x, y in zip(pops, pops[1:]):
                    v[2 * self.link_idx[link_node(x, y)] + (0 if x < y else 1)] = 1.0
                rows.append(v)
                cols.append((si, 1.0 / len(paths)))
        weights = np.zeros((len(self.services), len(rows)))
        for j, (si, w) in enumerate(cols):
            weights[si, j] = w
        hd = np.array(hd, dtype=int)
        hop_m = sparse.csr_matrix((np.ones(len(hd)), (np.arange(len(hd)), hd)), shape=(len(hd), 2 * L))
        r = Routing(rf, rb, np.array(rows).reshape(len(rows), 2 * L), weights, reachable, pair_paths,
                    np.array(hp, dtype=int), np.array(hw), hd, np.array(hf, dtype=int), hop_m)
        self._cache[key] = r
        return r

    def planned_peak_util(self, key, t0, t1, event_days=()):
        """Operator's pre-change check: noise-free max link utilisation over [t0, t1) under a routing state, with the
        event-day factor and a planning margin (audit m3)."""
        t = np.arange(t0, t1)
        demand = self.base_demand * self._pair_profile(t) * self.event_factor(t, event_days)[:, None]
        r = self.routing(key)
        margin = self.cfg["distractors"]["maintenance_review_margin"]
        return float((np.maximum(demand @ r.rf.T, demand @ r.rb.T) / self.capacity).max() * margin)

    def routing_changes(self, metrics):
        """True if overriding these link metrics changes any pair's paths."""
        base = self.routing(((), (), ()))
        new = self.routing(((), (), tuple(sorted(metrics.items()))))
        return base.pair_paths != new.pair_paths

    def _carried(self, r, dem, extra_drop):
        """Directed-link arriving load (n, 2L) with drops on earlier hops removed (audit M3). Naive aggregate loads are
        exact when nothing drops; congested or lossy steps run a short per-path fixed point."""
        lc = self.sc["loss"]
        loads = np.empty((dem.shape[0], 2 * len(self.links)))
        loads[:, 0::2] = dem @ r.rf.T
        loads[:, 1::2] = dem @ r.rb.T
        drop = 1 - (1 - congestion_loss(loads / self.capacity2, lc["knee"], lc["at_capacity_pct"])) * (1 - extra_drop)
        need = drop.max(1) > 1e-3
        if need.any() and len(r.hop_dl):
            d, x = dem[need], extra_drop[need]
            ld = loads[need]
            H = len(r.hop_dl)
            for _ in range(self.sc["carried_iterations"]):
                dr = 1 - (1 - congestion_loss(ld / self.capacity2, lc["knee"], lc["at_capacity_pct"])) * (1 - x)
                logs = np.log1p(-np.minimum(dr, 1 - 1e-12))[:, r.hop_dl]
                csp = np.concatenate([np.zeros((len(d), 1)), np.cumsum(logs, axis=1)], axis=1)
                before = csp[:, :H] - csp[:, r.hop_first]
                contrib = d[:, r.hop_pair] * r.hop_w * np.exp(before)
                ld = np.asarray((r.hop_m.T @ contrib.T).T)
            loads[need] = ld
        return loads

    # ------------------------------------------------------------------ simulation

    def simulate(self, spec, counterfactual=True, with_blind=False):
        """Run a scenario. With counterfactual=True, each fault's impact and observability are measured against a run
        with only that fault removed: every other fault and all maintenance stay in (audit re-verify N1). Labels are
        measured against a run with every fault removed. Each fault has its own RNG stream, so removing one leaves the
        others unchanged.

        with_blind=True also returns the status-blind view of the same run (see `blind_view`), with its own
        observability; returns (default, blind)."""
        res = self._run(spec)
        blind = self.blind_view(res) if with_blind else None
        if counterfactual:
            faults = spec["faults"]
            cf_all = self._run({**spec, "faults": []}) if faults else res
            res.labels = self._labels(res, cf_all)
            for f in faults:
                cf = cf_all if len(faults) == 1 else self._run({**spec, "faults": [g for g in faults if g["id"] != f["id"]]})
                res.impacts[f["id"]] = self._impact(res, cf, f)
                if blind is not None:
                    blind.impacts[f["id"]] = self._impact(blind, cf, f)
            if blind is not None:
                blind.labels = res.labels
        return (res, blind) if with_blind else res

    @staticmethod
    def blind_view(res):
        """The status-blind variant of a finished run: down links fully missing and no status or reachability events,
        exactly what `spec["status_blind"] = True` produces (tested), without simulating twice."""
        obs = {k: v.copy() for k, v in res.obs.items()}
        down = res.truth["fully_down"]
        for k in ("util", "latency_ms", "loss_pct"):
            obs[k][down] = np.nan
        return SimResult(spec={**res.spec, "status_blind": True}, links=res.links, services=res.services, obs=obs, truth=res.truth,
                         observed_events=[e for e in res.observed_events if e["type"] == "maintenance"])

    def _rngs(self, seed):
        return {k: np.random.default_rng([seed, i]) for i, k in enumerate(["demand", "measure", "missing", "faults", "status"], 1)}

    def _run(self, spec):
        T, L, S, N = spec["n_steps"], len(self.links), len(self.services), len(self.pops)
        sc, lc, ltc = self.sc, self.sc["latency"], self.sc["loss"]
        rng = self._rngs(spec["seed"])
        t = np.arange(T)
        demand = self._demand(t, rng["demand"], spec.get("event_days", []))

        # ---- fault and maintenance state (each fault has its own RNG stream, so removing one changes no other)
        down_frac = np.zeros((T, L))
        pop_down = np.zeros((T, N), bool)
        gray = np.zeros((T, L))
        gray_visible = np.zeros((T, L))
        conv = np.zeros((T, L))
        overrides = [dict() for _ in range(T)]
        hard_events = []  # (t, links, convergence_s, "down" | "up")
        for f in spec["faults"]:
            s, e, p = f["t_start"], f["t_end"], f["params"]
            ft, root = f["type"], f["root_cause"]
            frng = np.random.default_rng([spec["seed"], 4, _crc(f["id"])])
            n = e - s
            if ft == "link_down":
                down_frac[s:e, self.link_idx[root]] = 1.0
                hard_events += [(s, [self.link_idx[root]], p["convergence_s"], "down"), (e, [self.link_idx[root]], p["convergence_s"], "up")]
            elif ft == "node_down":
                pi = self.pop_idx[root.split(":", 1)[1]]
                pop_down[s:e, pi] = True
                inc = [i for i, (u, v) in enumerate(self.link_ends) if pi in (u, v)]
                hard_events += [(s, inc, p["convergence_s"], "down"), (e, inc, p["convergence_s"], "up")]
            elif ft == "link_degradation":
                li = self.link_idx[root]
                i = np.arange(n)
                ramp = np.minimum(np.minimum(1.0, (i + 1) / max(1, p["ramp_steps"])), np.clip((n - i) / max(1, p.get("ramp_down_steps", 0)), 0, 1))
                burst = np.exp(frng.normal(0, p["burst_sigma"], n) - p["burst_sigma"] ** 2 / 2)
                g = np.clip(p["loss_pct"] / 100.0 * ramp * burst, 0, 0.5)
                gray[s:e, li] = 1 - (1 - gray[s:e, li]) * (1 - g)
                if not p["silent"]:
                    gray_visible[s:e, li] += g                       # audit m1: visibility per fault window
            elif ft == "link_flapping":
                li = self.link_idx[root]
                active = frng.random(n) < p["active_prob"]
                frac = frng.uniform(min(0.02, p["max_down_frac"]), p["max_down_frac"], n)  # audit m2
                flaps = frng.poisson(p["flaps_per_step"], n) + 1
                down_frac[s:e, li] = np.where(active, frac, 0.0)
                conv[s:e, li] = np.where(active, np.minimum(1.0, flaps * p["convergence_s"] / (self.step_minutes * 60)), 0.0)
            elif ft == "demand_surge":
                pi = self.pop_idx[root.split(":", 1)[1]]
                i = np.arange(n)
                ramp = np.minimum(np.minimum(1.0, (i + 1) / max(1, p["ramp_steps"])), np.clip((n - i) / max(1, p.get("ramp_down_steps", 0)), 0, 1))
                touch = (self.pair_src == pi) | (self.pair_dst == pi)
                demand[s:e][:, touch] *= (1.0 + (p["factor"] - 1.0) * ramp)[:, None]
            elif ft == "routing_misconfig":
                for k in range(s, e):
                    overrides[k][root] = p["new_metric"]
        for m in spec.get("maintenance", []):
            s, e = m["t_start"], m["t_end"]
            if m["action"] == "down":            # planned: traffic is drained first, so no convergence loss
                down_frac[s:e, self.link_idx[m["target"]]] = 1.0
            elif m["action"] == "drain":
                for k in range(s, e):
                    overrides[k][m["target"]] = sc["drain_metric"]

        # ---- group steps by network state; a flapping link time-shares its up and down states
        groups = defaultdict(list)
        keys = []
        for i in range(T):
            hard = tuple(self.links[j] for j in np.flatnonzero(down_frac[i] >= 1.0))
            pops = tuple(self.pops[j] for j in np.flatnonzero(pop_down[i]))
            ov = tuple(sorted(overrides[i].items()))
            partial = np.flatnonzero((down_frac[i] > 0) & (down_frac[i] < 1))
            base = (hard, pops, ov)
            keys.append(base)
            if len(partial) == 0:
                groups[base].append((i, 1.0))
            else:
                if len(partial) > 1:
                    raise ValueError("at most one link may be partially down per step")
                f = down_frac[i, partial[0]]
                groups[base].append((i, 1.0 - f))
                groups[(tuple(sorted(hard + (self.links[partial[0]],))), pops, ov)].append((i, f))

        extra = 1 - (1 - np.repeat(gray, 2, axis=1)) * (1 - np.repeat(conv, 2, axis=1))  # (T, 2L)
        knee = sc["gray_goodput_knee_pct"] / 100.0
        rho_acc, util_acc = np.zeros((T, 2 * L)), np.zeros((T, 2 * L))
        q_acc, pc_acc = np.zeros((T, 2 * L)), np.zeros((T, 2 * L))
        svc_loss = np.zeros((T, S))
        svc_lat_acc, svc_lat_w = np.zeros((T, S)), np.zeros((T, S))
        prop2 = np.repeat(self.prop_ms, 2)
        for key, tw in groups.items():
            r = self.routing(key)
            ts = np.array([a for a, _ in tw])
            w = np.array([b for _, b in tw])[:, None]
            dem = demand[ts].copy()
            for li in np.flatnonzero(gray[ts].max(0) > knee):     # loss-limited TCP flows send less (audit minor)
                m = np.minimum(1.0, np.sqrt(knee / np.maximum(gray[ts, li], 1e-12)))
                dem *= 1 - (r.rf[li] + r.rb[li])[None, :] * (1 - m)[:, None]
            loads = self._carried(r, dem, extra[ts])
            rho = loads / self.capacity2
            q = queue_delay_ms(rho, lc["queue_scale_ms"], lc["buffer_ms"])
            pc = congestion_loss(rho, ltc["knee"], ltc["at_capacity_pct"])
            rho_acc[ts] += w * rho
            util_acc[ts] += w * np.minimum(rho, 1.0)
            q_acc[ts] += w * q
            pc_acc[ts] += w * pc
            if r.paths.shape[0]:
                lat_dir = prop2 + lc["per_hop_ms"] + q
                logsurv = np.log1p(-np.minimum(1 - (1 - pc) * (1 - extra[ts]), 1 - 1e-12))
                lat_s = (lat_dir @ r.paths.T) @ r.weights.T
                loss_s = (1 - np.exp(logsurv @ r.paths.T)) @ r.weights.T
            else:
                lat_s = loss_s = np.zeros((len(ts), S))
            reach = r.reachable[None, :]
            svc_loss[ts] += w * np.where(reach, loss_s, 1.0)
            svc_lat_acc[ts] += w * np.where(reach, lat_s, 0.0)
            svc_lat_w[ts] += w * reach
        with np.errstate(invalid="ignore", divide="ignore"):
            svc_lat = np.where(svc_lat_w > 0, svc_lat_acc / svc_lat_w, np.nan)

        # ---- convergence loss when a hard down starts or ends (audit M3 follow-up / minor)
        for tt, lis, conv_s, kind in hard_events:
            if not 0 < tt < T:
                continue
            r = self.routing(keys[tt - 1] if kind == "down" else keys[tt])
            if not r.paths.shape[0]:
                continue
            cols = np.concatenate([[2 * li, 2 * li + 1] for li in lis])
            uses = r.paths[:, cols].max(1) > 0
            share = r.weights @ uses.astype(float)
            frac = min(1.0, conv_s / (self.step_minutes * 60))
            svc_loss[tt] = 1 - (1 - svc_loss[tt]) * (1 - share * frac)

        # ---- link truth
        dead_end = pop_down[:, self.link_ends[:, 0]] | pop_down[:, self.link_ends[:, 1]]
        fully_down = (down_frac >= 1.0) | dead_end
        rho_l = rho_acc.reshape(T, L, 2).max(2)
        util_true = util_acc.reshape(T, L, 2).max(2)
        lat_true = self.prop_ms[None, :] + lc["per_hop_ms"] + q_acc.reshape(T, L, 2).max(2)
        counter_true = pc_acc.reshape(T, L, 2).max(2) + gray_visible

        # ---- measurement (fixed-shape draws, so noise is identical with or without faults)
        mr = rng["measure"]
        util_obs = np.clip(util_true + mr.normal(0, sc["util_noise"], (T, L)), 0, 1)
        lat_obs = lat_true * (1 + mr.normal(0, lc["jitter_rel"], (T, L))) + np.abs(mr.normal(0, lc["jitter_abs_ms"], (T, L)))
        lat_obs += (mr.random((T, L)) < lc["outlier_prob"]) * mr.uniform(1, 20, (T, L))
        bg = (mr.random((T, L)) < ltc["background_prob"]) * mr.exponential(ltc["background_mean_pct"] / 100, (T, L))
        loss_obs = 100 * (np.clip(counter_true * (1 + mr.normal(0, ltc["counter_noise_rel"], (T, L))), 0, 1) + bg)
        svc_lat_obs = svc_lat * (1 + mr.normal(0, lc["jitter_rel"], (T, S))) + np.abs(mr.normal(0, lc["jitter_abs_ms"], (T, S)))
        svc_lat_obs += (mr.random((T, S)) < lc["outlier_prob"]) * mr.uniform(1, 20, (T, S))
        # background end-to-end probe loss (audit C2): per-service base rate plus rare bursty probe-host episodes
        pb = sc["probe_background"]
        base = np.exp(mr.uniform(np.log(pb["base_rate"][0]), np.log(pb["base_rate"][1]), S))
        starts = mr.random((T, S)) < pb["episodes_per_day"] / self.steps_per_day
        lengths = mr.geometric(1.0 / pb["episode_mean_steps"], (T, S))
        levels = mr.uniform(*pb["episode_loss"], (T, S))
        episode = np.zeros((T, S))
        for ti, si in zip(*np.nonzero(starts)):
            episode[ti:ti + lengths[ti, si], si] = np.maximum(episode[ti:ti + lengths[ti, si], si], levels[ti, si])
        svc_bg = 1 - (1 - base[None, :]) * (1 - episode)
        p_probe = 1 - (1 - np.clip(svc_loss, 0, 1)) * (1 - svc_bg)
        svc_loss_obs = 100 * mr.binomial(sc["probes_per_step"], p_probe) / sc["probes_per_step"]
        probe_lost = mr.random((T, L)) < sc["probe_missing_prob"]   # one-hop probe timeouts on healthy links

        # ---- what the poller sees for a down link (audit M1). A down interface on a reachable router reads 0 traffic
        # and 0 discards; the one-hop probe fails. A dead router's links are reported by the neighbour the same way.
        # Status-blind variant: down links are simply missing, like a gap.
        blind = bool(spec.get("status_blind", False))
        if blind:
            for a in (util_obs, lat_obs, loss_obs):
                a[fully_down] = np.nan
        else:
            util_obs[fully_down] = 0.0
            loss_obs[fully_down] = 0.0
            lat_obs[fully_down] = np.nan
        lat_obs[probe_lost] = np.nan

        # ---- missing data: random drop-outs, collector gaps, probe-responder gaps
        mm = rng["missing"]
        link_missing = mm.random((T, L)) < sc["missing_prob"]
        svc_missing = mm.random((T, S)) < sc["missing_prob"]
        svc_dead = np.zeros((T, S), bool)
        for g in spec.get("gaps", []):
            s, e = g["t_start"], g["t_end"]
            if g["scope"] == "link":
                link_missing[s:e, self.link_idx[g["target"]]] = True
            elif g["scope"] == "probe":
                lat_obs[s:e, self.link_idx[g["target"]]] = np.nan
            else:
                pi = self.pop_idx[g["target"].split(":", 1)[1]]
                link_missing[s:e] |= (self.link_ends[:, 0] == pi) | (self.link_ends[:, 1] == pi)
                at_pop = (self.svc_src == pi) | (self.svc_dst == pi)                       # audit m5: either endpoint
                if g.get("responder_down"):
                    svc_dead[s:e] |= at_pop    # probes to a dead responder are lost, as with an unreachable PoP
                else:
                    svc_missing[s:e] |= at_pop
        for a in (util_obs, lat_obs, loss_obs):
            a[link_missing] = np.nan
        svc_loss_obs[svc_dead] = 100.0
        svc_lat_obs[svc_dead] = np.nan
        svc_lat_obs[svc_missing] = np.nan
        svc_loss_obs[svc_missing] = np.nan

        observed_events = self._observed_events(spec, down_frac, fully_down, pop_down, rng["status"])
        if blind:  # no status or reachability events at all; the change log remains
            observed_events = [ev for ev in observed_events if ev["type"] == "maintenance"]
        return SimResult(
            spec=spec,
            links=self.links,
            services=self.services,
            obs={"util": util_obs, "latency_ms": lat_obs, "loss_pct": loss_obs, "svc_latency_ms": svc_lat_obs, "svc_loss_pct": svc_loss_obs},
            truth={
                "util": util_true,
                "rho": rho_l,
                "latency_ms": lat_true,
                "counter_loss_pct": 100 * counter_true,
                "svc_latency_ms": svc_lat,
                "svc_loss_pct": 100 * svc_loss,
                "svc_bg_loss": svc_bg,
                "down_frac": down_frac,
                "fully_down": fully_down,
                "pop_down": pop_down,
                "keys": keys,
            },
            observed_events=observed_events,
        )

    def _observed_events(self, spec, down_frac, fully_down, pop_down, srng):
        """Status polling with lag, plus PoP reachability and the planned-maintenance log. Never contains fault labels,
        gray failures or metric misconfigurations.

        - link_status: polled once per step (a flapping link is seen down with probability = its down fraction); each
          change is reported after a lag of 0 to 2 steps plus an occasional missed poll (audit M6). A down and an up
          that land on the same step cancel out (a missed short outage).
        - pop_status: the NMS loses reachability to a router after 1 to 3 failed polls, and only raises it sometimes.
          Sources: a dead router (node_down) and a PoP management / collector outage (pop-scope gap), so the event type
          alone does not identify node_down (audit M1)."""
        T, L = down_frac.shape
        st = self.sc["status"]
        partial = (down_frac > 0) & (down_frac < 1)
        seen_link = fully_down | (partial & (srng.random((T, L)) < down_frac))
        lag = srng.choice(len(st["lag_probs"]), size=(T, L), p=st["lag_probs"]) + (srng.random((T, L)) < st["missed_poll_prob"])
        events = []
        for m in spec.get("maintenance", []):
            events.append({"t": m["t_start"], "type": "maintenance", **{k: m[k] for k in ("id", "target", "t_start", "t_end", "action", "description")}})
        for j in range(L):
            col = seen_link[:, j]
            changes = np.flatnonzero(col[1:] != col[:-1]) + 1
            if col[0]:
                changes = np.concatenate([[0], changes])
            out = []
            for i in changes:
                te = int(i + lag[i, j])
                if out:
                    te = max(te, out[-1][0])
                    if te == out[-1][0]:
                        out.pop()
                        continue
                out.append((te, "down" if col[i] else "up"))
            events += [{"t": te, "type": "link_status", "element": self.links[j], "status": s} for te, s in out if te < T]
        sources = [(f["id"], f["root_cause"], f["t_start"], f["t_end"]) for f in spec["faults"] if f["type"] == "node_down"]
        sources += [(g["id"], g["target"], g["t_start"], g["t_end"]) for g in spec.get("gaps", []) if g["scope"] == "pop"]
        for sid, pop, s, e in sources:
            prng = np.random.default_rng([spec["seed"], 5, _crc(sid)])
            if prng.random() >= st["pop_status_prob"]:
                continue
            timeout = int(prng.integers(st["pop_timeout_polls"][0], st["pop_timeout_polls"][1] + 1))
            if e - s <= timeout:
                continue
            back = int(e + prng.integers(0, STATUS_LAG_MAX))
            events.append({"t": s + timeout, "type": "pop_status", "element": pop, "status": "unreachable"})
            if back < T:
                events.append({"t": back, "type": "pop_status", "element": pop, "status": "reachable"})
        return sorted(events, key=lambda e: (e["t"], e["type"], e.get("element", e.get("id", ""))))

    # ------------------------------------------------------------------ ground-truth labels and impact

    def _symptoms(self, res, cf):
        ic = self.cfg["impact"]
        du = np.abs(res.truth["util"] - cf.truth["util"]) > ic["util_delta"]
        with np.errstate(invalid="ignore", divide="ignore"):
            dlat = (res.truth["latency_ms"] - cf.truth["latency_ms"]) / cf.truth["latency_ms"] > ic["latency_rel"]
        dctr = (res.truth["counter_loss_pct"] - cf.truth["counter_loss_pct"]) > ic["counter_delta_pct"]
        return du, dlat, dctr

    def _root_links(self, root):
        if root in self.link_idx:
            return [self.link_idx[root]]
        pi = self.pop_idx[root.split(":", 1)[1]]
        return [i for i, (u, v) in enumerate(self.link_ends) if pi in (u, v)]

    def _labels(self, res, cf):
        """Per-link per-step detection labels, as pre-registered (audit M7)."""
        du, dlat, dctr = self._symptoms(res, cf)
        lab = (du | dlat | dctr).astype(np.int8)
        for f in res.spec["faults"]:
            lab[f["t_start"]:f["t_end"], self._root_links(f["root_cause"])] = 2
        return lab

    def _impact(self, res, cf, fault):
        s, e = fault["t_start"], fault["t_end"]
        T = res.spec["n_steps"]
        w = slice(s, min(T, e + STATUS_LAG_MAX))
        ic = self.cfg["impact"]
        probes = self.sc["probes_per_step"]
        lat, lat0 = res.truth["svc_latency_ms"][w], cf.truth["svc_latency_ms"][w]
        unreachable = np.isnan(lat) & ~np.isnan(lat0)
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = np.nan_to_num((lat - lat0) / lat0, nan=0.0)
        dloss = res.truth["svc_loss_pct"][w] - cf.truth["svc_loss_pct"][w]
        dutil = np.abs(res.truth["util"][w] - cf.truth["util"][w])
        root_idx = self._root_links(fault["root_cause"])
        others = np.setdiff1d(np.arange(len(self.links)), root_idx)
        affected = (rel.max(0) > ic["affected_latency_rel"]) | (dloss.max(0) > ic["affected_loss_pct"]) | unreachable.any(0)
        root_links = {self.links[i] for i in root_idx}
        # only status changes this fault caused: events absent from its leave-one-out counterfactual (N1). Planned
        # outages and other faults produce the same events in both runs, so they no longer count.
        key = lambda ev: (ev["t"], ev["type"], ev.get("element"), ev.get("status"))  # noqa: E731
        cf_events = {key(ev) for ev in cf.observed_events}
        status_seen = any(
            ev["type"] in ("link_status", "pop_status") and s <= ev["t"] < w.stop and key(ev) not in cf_events
            and (ev["element"] == fault["root_cause"] or ev["element"] in root_links)
            for ev in res.observed_events
        )
        # observability channels (pre-registered, audit M5). A delta only counts on a cell whose observed value exists,
        # so a channel fully covered by a telemetry gap (or blanked by the status-blind variant) is unobservable (N8).
        du, dlat_l, dctr = (x[w] for x in self._symptoms(res, cf))
        du &= ~np.isnan(res.obs["util"][w])
        dctr &= ~np.isnan(res.obs["loss_pct"][w])
        loss_seen = ~np.isnan(res.obs["svc_loss_pct"][w])
        lat_seen = ~np.isnan(res.obs["svc_latency_ms"][w])
        extra = (dloss / 100 * probes * loss_seen).sum(0)
        bg_exp = (res.truth["svc_bg_loss"][w] * probes * loss_seen).sum(0)
        obs = {
            "obs_status": bool(status_seen),
            "obs_link_util": bool((du.sum(0) >= 2).any()),
            "obs_link_counters": bool((dctr.sum(0) >= 2).any()),
            "obs_e2e": bool((extra >= np.maximum(3, 3 * np.sqrt(bg_exp))).any() or (((rel > ic["latency_rel"]) & lat_seen).sum(0) >= 2).any()
                            or (unreachable & loss_seen).any()),
        }
        out = {
            "max_service_latency_delta_ms": float(np.nan_to_num(lat - lat0, nan=0.0).max()) if lat.size else 0.0,
            "max_service_latency_delta_rel": float(rel.max()) if rel.size else 0.0,
            "max_service_loss_delta_pct": float(dloss.max()) if dloss.size else 0.0,
            "services_unreachable": int(unreachable.any(0).sum()),
            "services_affected": int(affected.sum()),
            "max_root_util_delta": float(dutil[:, root_idx].max()) if root_idx else 0.0,
            "max_other_util_delta": float(dutil[:, others].max()) if len(others) else 0.0,
            "max_util_in_window": float(res.truth["util"][s:e].max()),
            "status_change_observed": bool(status_seen),
            "root_counters_show_loss": bool(fault["type"] == "link_degradation" and not fault["params"]["silent"]),
            "observability": obs,
            "observable": any(obs.values()),
        }
        # impact class: relative latency (audit M4); the root's own drop is not a symptom for hard downs (audit m7)
        util_d = out["max_other_util_delta"] if fault["type"] in ("link_down", "node_down") else max(out["max_root_util_delta"], out["max_other_util_delta"])
        lat_r, loss_d = out["max_service_latency_delta_rel"], out["max_service_loss_delta_pct"]
        if out["services_unreachable"] or loss_d >= 1.0 or lat_r >= ic["obvious_latency_rel"] or util_d >= 0.3:
            out["impact_class"] = "obvious"
        elif loss_d < 0.1 and lat_r < ic["subtle_latency_rel"] and util_d < 0.05:
            out["impact_class"] = "subtle"
        else:
            out["impact_class"] = "moderate"
        return out


# ---------------------------------------------------------------------- twin view and persistence


def observed_kg(sim, res):
    """The digital twin's KG: intended metrics, plus only what the observed events reveal. PoP reachability events
    are logged but do not change routing: an unreachable router may still be forwarding."""
    kg = NetTwinKG.from_config(sim.topo, sim.cfg)
    actions = defaultdict(lambda: {"links_down": [], "links_up": [], "pops_down": [], "pops_up": [], "metrics": {}})
    pop_events = []
    for ev in res.observed_events:
        if ev["type"] == "maintenance":
            kg.add_maintenance(ev["id"], [ev["target"]], ev["t_start"], ev["t_end"], ev["description"])
            if ev["action"] == "drain":
                actions[ev["t_start"]]["metrics"][ev["target"]] = sim.sc["drain_metric"]
                actions[ev["t_end"]]["metrics"][ev["target"]] = None
        elif ev["type"] == "link_status":
            actions[ev["t"]]["links_down" if ev["status"] == "down" else "links_up"].append(ev["element"])
        elif ev["type"] == "pop_status":
            pop_events.append(dict(ev))
    for t in sorted(actions):
        kg.apply_change(t, reason="observed", **actions[t])
    kg.events.extend(pop_events)
    return kg


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def to_frames(res):
    T = res.spec["n_steps"]
    links = pd.DataFrame(
        {
            "t": np.repeat(np.arange(T, dtype=np.int32), len(res.links)),
            "link": pd.Categorical(np.tile(res.links, T), categories=res.links),
            **{k: res.obs[k].astype(np.float32).ravel() for k in ("util", "latency_ms", "loss_pct")},
        }
    )
    services = pd.DataFrame(
        {
            "t": np.repeat(np.arange(T, dtype=np.int32), len(res.services)),
            "service": pd.Categorical(np.tile(res.services, T), categories=res.services),
            "latency_ms": res.obs["svc_latency_ms"].astype(np.float32).ravel(),
            "loss_pct": res.obs["svc_loss_pct"].astype(np.float32).ravel(),
        }
    )
    return links, services


# words that must never appear in an observable path (audit re-verify N7)
LEAKY_WORDS = ("link_down", "node_down", "degradation", "gray", "grey", "flap", "surge", "misconfig", "fault", "mild",
               "severe", "subtle", "moderate", "obvious", "distractor", "incident")


def check_neutral_id(sid):
    low = sid.lower()
    bad = [w for w in LEAKY_WORDS if w in low]
    if bad:
        raise ValueError(f"scenario id {sid!r} leaks ground truth through the observable path: {bad}")


OBSERVABLE_META_KEYS = ("scenario_id", "topology", "n_steps", "step_minutes", "status_blind")


def write_scenario(res, out_dir=None, obs_dir=None, truth_dir=None):
    """Observable files under <out>/observable/<sid>/, ground truth under <out>/truth/<sid>/ (audit minor: keep ground
    truth out of reach of detectors and agent tools); `obs_dir` / `truth_dir` override the two roots. The scenario id
    must be neutral (N7); any description belongs in the ground-truth manifest. The observable `meta.json` carries no
    seed, since the seed plus the config would let anyone re-run the sampler and recover the faults. Returns the
    manifest entry (ground truth)."""
    sid = res.spec["scenario_id"]
    check_neutral_id(sid)
    od = Path(obs_dir or Path(out_dir) / "observable") / sid
    td = Path(truth_dir or Path(out_dir) / "truth") / sid
    od.mkdir(parents=True, exist_ok=True)
    td.mkdir(parents=True, exist_ok=True)
    links, services = to_frames(res)
    links.to_parquet(od / "links.parquet", index=False)
    services.to_parquet(od / "services.parquet", index=False)
    (od / "observed_events.json").write_text(json.dumps(res.observed_events, indent=1, default=_json_default) + "\n")
    meta = {k: res.spec.get(k, False) for k in OBSERVABLE_META_KEYS}
    (od / "meta.json").write_text(json.dumps(meta, indent=1, default=_json_default) + "\n")
    if res.labels is not None:
        T = res.spec["n_steps"]
        pd.DataFrame(
            {
                "t": np.repeat(np.arange(T, dtype=np.int32), len(res.links)),
                "link": pd.Categorical(np.tile(res.links, T), categories=res.links),
                "label": res.labels.ravel(),
            }
        ).to_parquet(td / "labels.parquet", index=False)
    entry = {
        **res.spec,
        "faults": [{**f, "impact": res.impacts.get(f["id"])} for f in res.spec["faults"]],
        "files": {
            "links": f"{sid}/links.parquet",
            "services": f"{sid}/services.parquet",
            "observed_events": f"{sid}/observed_events.json",
            "meta": f"{sid}/meta.json",
            "labels": f"{sid}/labels.parquet",
        },
    }
    return json.loads(json.dumps(entry, default=_json_default))


def write_manifest(entries, out_dir):
    path = Path(out_dir) / "truth" / "scenarios.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=1, default=_json_default) + "\n")
    return path
