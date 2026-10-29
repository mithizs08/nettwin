"""Telemetry simulator (PLAN.md section 4).

One `Simulator` per topology holds the fixed network design (demand weights, link capacities, synthetic
services). `simulate(spec)` turns a scenario spec (see faults.py) into telemetry, deterministically from
spec["seed"].

What the simulator produces, and who may see it:

  Observable (inputs for detectors and the agent)
    link telemetry per step:    util (interface counters), latency_ms (one-hop probe RTT estimate),
                                loss_pct (interface discard + error counters)
    service telemetry per step: latency_ms, loss_pct from end-to-end probes (binomial, probes_per_step)
    observed events:            link / PoP status from 5-minute polling, and the planned-maintenance log
  Ground truth (evaluation only)
    the fault list, noise-free metrics, the true routing, and per-fault impact stats

The digital twin's routing view (`observed_kg`) is rebuilt from observed events only. It uses the intended
link metrics, so a routing misconfiguration is invisible to it: the twin holds the wrong paths for the
duration. Gray failures and demand surges also never appear as events.

Model summary
  demand      gravity model (degree x lognormal weights; germany50 uses its real matrix), diurnal profile in
              each PoP's local time (from longitude), weekend dip, per-PoP day factors, AR(1) log-noise
  routing     Phase 1 KG shortest paths with ECMP split evenly; recomputed per distinct network state
  capacity    peak noise-free baseline load / U(0.4, 0.7), floored, fixed per topology
  latency     propagation (dist / 200 km per ms) + per-hop + queueing scale x rho / (1 - rho), capped at buffer
  loss        ~0 below rho 0.9, cubic rise to 0.2% at capacity, plus all excess above capacity
  utilisation measured on the busier direction, carried load capped at 1.0
"""
import json
import zlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from nettwin.kg import NetTwinKG, link_node, pop_node


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


@dataclass
class Routing:
    rf: np.ndarray            # (L, P) fraction of directed demand p crossing link l in u->v direction
    rb: np.ndarray            # (L, P) same, v->u
    paths: np.ndarray         # (n_paths, 2L) incidence of service paths on directed links
    weights: np.ndarray       # (S, n_paths) ECMP weight of each path for each service
    reachable: np.ndarray     # (S,) bool
    pair_paths: dict = field(repr=False)


@dataclass
class SimResult:
    spec: dict
    links: list
    services: list
    obs: dict                 # util, latency_ms, loss_pct: (T, L); svc_latency_ms, svc_loss_pct: (T, S)
    truth: dict               # noise-free arrays, down_frac, routing keys per step
    observed_events: list
    impacts: dict = field(default_factory=dict)


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
        if topo.graph["geo"]:
            lon = np.array([topo.nodes[p]["pos"][0] for p in self.pops])
            self.local_offset_h = (lon - lon.mean()) / 15.0
        else:
            self.local_offset_h = np.zeros(len(self.pops))
        self._cache = {}
        rng = np.random.default_rng([cfg["seeds"]["network"], zlib.crc32(self.name.encode())])
        self.base_demand = self._base_demand(rng)
        self.capacity = self._design_capacity(rng)

    # ------------------------------------------------------------------ network design

    def _base_demand(self, rng):
        dc = self.sc["demand"]
        real = self.topo.graph.get("demands") or {}
        if real:
            d = np.zeros(len(self.pairs))
            for a, row in real.items():
                for b, v in row.items():
                    d[self.pair_idx[(a, b)]] += v
                    d[self.pair_idx[(b, a)]] += v
        else:
            deg = np.array([self.topo.degree(p) for p in self.pops], dtype=float)
            w = deg * rng.lognormal(0.0, dc["node_weight_sigma"], len(self.pops))
            d = w[self.pair_src] * w[self.pair_dst]
        return d / d.sum() * dc["total"]

    def _design_capacity(self, rng):
        """Capacity from the peak of a noisy fault-free 'history' week on intended routing (as an operator would
        plan from measured traffic), so realised baseline peaks sit in the target band. Uses the network seed,
        not any scenario seed."""
        L = len(self.links)
        demand = self._demand(np.arange(self.n_steps), rng)
        r = self.routing(((), (), ()))
        peak = np.maximum((demand @ r.rf.T).max(0), (demand @ r.rb.T).max(0))
        target = rng.uniform(*self.sc["target_peak_utilisation"], L)
        hot = rng.random(L) < self.sc["hot_links"]["frac"]          # a few links run hot at normal peaks
        target = np.where(hot, rng.uniform(*self.sc["hot_links"]["target_utilisation"], L), target)
        n1 = self.sc["n1_planning"]
        planned = (rng.random(L) < n1["planned_frac"]) & ~hot
        n1_target = rng.uniform(*n1["target_utilisation"], L)
        cap = np.where(peak > 0, peak / target, 0.0)
        floor = self.sc["capacity_floor_frac"] * np.median(cap[cap > 0])
        cap = np.where(hot, cap, np.maximum(cap, floor))
        self.hot_links = [l for l, h in zip(self.links, hot) if h and peak[self.link_idx[l]] > 0]
        if planned.any():
            # worst single-link-failure load at the busiest history steps; only some links were upgraded for it
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
        return cap

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
        event = np.ones(T)
        for ev in event_days:
            event[day_idx == ev["day"]] *= ev["factor"]
        return self.base_demand * self._pair_profile(t) * np.exp(x) * day_pair * event[:, None]

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
        for a, b in ((a, b) for a, b in self.pairs if a < b):
            paths = [kg.G.nodes[p]["pops"] for p in kg.paths_between(pop_node(a), pop_node(b))]
            pair_paths[(a, b)] = paths
            if not paths:
                continue
            w = 1.0 / len(paths)
            fwd, bwd = self.pair_idx[(a, b)], self.pair_idx[(b, a)]
            for pops in paths:
                for x, y in zip(pops, pops[1:]):
                    li = self.link_idx[link_node(x, y)]
                    if x < y:  # a->b crosses this link u->v; b->a crosses it v->u
                        rf[li, fwd] += w
                        rb[li, bwd] += w
                    else:
                        rb[li, fwd] += w
                        rf[li, bwd] += w
        rows, weights, reachable = [], np.zeros((len(self.services), 0)), np.zeros(len(self.services), bool)
        cols = []
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
        r = Routing(rf, rb, np.array(rows).reshape(len(rows), 2 * L), weights, reachable, pair_paths)
        self._cache[key] = r
        return r

    def planned_peak_util(self, key, t0, t1):
        """Noise-free max link utilisation over [t0, t1) under a routing state: an operator's pre-change check."""
        demand = self.base_demand * self._pair_profile(np.arange(t0, t1))
        r = self.routing(key)
        return float((np.maximum(demand @ r.rf.T, demand @ r.rb.T) / self.capacity).max())

    def routing_changes(self, metrics):
        """True if overriding these link metrics changes any pair's paths."""
        base = self.routing(((), (), ()))
        new = self.routing(((), (), tuple(sorted(metrics.items()))))
        return base.pair_paths != new.pair_paths

    # ------------------------------------------------------------------ simulation

    def simulate(self, spec, counterfactual=True):
        res = self._run(spec)
        if counterfactual and spec["faults"]:
            cf = self._run({**spec, "faults": []})
            res.impacts = {f["id"]: self._impact(res, cf, f) for f in spec["faults"]}
        return res

    def _rngs(self, seed):
        return {k: np.random.default_rng([seed, i]) for i, k in enumerate(["demand", "measure", "missing", "faults", "status"], 1)}

    def _run(self, spec):
        T, L, S, N = spec["n_steps"], len(self.links), len(self.services), len(self.pops)
        sc = self.sc
        rng = self._rngs(spec["seed"])
        t = np.arange(T)

        # ---- demand (noise drawn before any fault logic so counterfactual runs share it)
        demand = self._demand(t, rng["demand"], spec.get("event_days", []))

        # ---- fault and maintenance state
        down_frac = np.zeros((T, L))
        pop_down = np.zeros((T, N), bool)
        gray = np.zeros((T, L))
        gray_visible = np.zeros(L, bool)
        conv = np.zeros((T, L))
        overrides = [dict() for _ in range(T)]
        frng = rng["faults"]
        for f in spec["faults"]:
            s, e, p = f["t_start"], f["t_end"], f["params"]
            ft, root = f["type"], f["root_cause"]
            if ft == "link_down":
                down_frac[s:e, self.link_idx[root]] = 1.0
            elif ft == "node_down":
                pop_down[s:e, self.pop_idx[root.split(":", 1)[1]]] = True
            elif ft == "link_degradation":
                li = self.link_idx[root]
                n = e - s
                ramp = np.minimum(1.0, (np.arange(n) + 1) / max(1, p["ramp_steps"]))
                burst = np.exp(frng.normal(0, p["burst_sigma"], n) - p["burst_sigma"] ** 2 / 2)
                gray[s:e, li] = np.clip(p["loss_pct"] / 100.0 * ramp * burst, 0, 0.5)
                gray_visible[li] |= not p["silent"]
            elif ft == "link_flapping":
                li = self.link_idx[root]
                n = e - s
                active = frng.random(n) < p["active_prob"]
                frac = frng.uniform(0.02, p["max_down_frac"], n)
                flaps = frng.poisson(p["flaps_per_step"], n) + 1
                down_frac[s:e, li] = np.where(active, frac, 0.0)
                conv[s:e, li] = np.where(active, np.minimum(1.0, flaps * p["convergence_s"] / (self.step_minutes * 60)), 0.0)
            elif ft == "demand_surge":
                pi = self.pop_idx[root.split(":", 1)[1]]
                n = e - s
                ramp = np.minimum(1.0, (np.arange(n) + 1) / max(1, p["ramp_steps"]))
                mult = 1.0 + (p["factor"] - 1.0) * ramp
                touch = (self.pair_src == pi) | (self.pair_dst == pi)
                demand[s:e][:, touch] *= mult[:, None]
            elif ft == "routing_misconfig":
                for i in range(s, e):
                    overrides[i][root] = p["new_metric"]
        for m in spec.get("maintenance", []):
            s, e = m["t_start"], m["t_end"]
            if m["action"] == "down":
                down_frac[s:e, self.link_idx[m["target"]]] = 1.0
            elif m["action"] == "drain":
                for i in range(s, e):
                    overrides[i][m["target"]] = sc["drain_metric"]

        # ---- group steps by network state; a flapping link mixes its up and down routing by down fraction
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

        off_f, off_b = np.zeros((T, L)), np.zeros((T, L))
        for key, tw in groups.items():
            r = self.routing(key)
            ts = np.array([a for a, _ in tw])
            w = np.array([b for _, b in tw])
            dk = demand[ts] * w[:, None]
            off_f[ts] += dk @ r.rf.T
            off_b[ts] += dk @ r.rb.T

        lc, ltc = sc["latency"], sc["loss"]
        rho = np.stack([off_f, off_b], axis=2) / self.capacity[None, :, None]       # (T, L, 2)
        q = queue_delay_ms(rho, lc["queue_scale_ms"], lc["buffer_ms"])
        pc = congestion_loss(rho, ltc["knee"], ltc["at_capacity_pct"])
        drop = 1 - (1 - pc) * (1 - gray[:, :, None]) * (1 - conv[:, :, None])
        lat_dir = self.prop_ms[None, :, None] + lc["per_hop_ms"] + q

        # ---- service truth (mixture over the step's routing states)
        svc_loss = np.zeros((T, S))
        svc_lat_acc, svc_lat_w = np.zeros((T, S)), np.zeros((T, S))
        lat_flat = lat_dir.reshape(T, 2 * L)
        logsurv_flat = np.log1p(-np.minimum(drop, 1 - 1e-12)).reshape(T, 2 * L)
        for key, tw in groups.items():
            r = self.routing(key)
            ts = np.array([a for a, _ in tw])
            w = np.array([b for _, b in tw])[:, None]
            if r.paths.shape[0]:
                lat_s = (lat_flat[ts] @ r.paths.T) @ r.weights.T
                loss_s = (1 - np.exp(logsurv_flat[ts] @ r.paths.T)) @ r.weights.T
            else:
                lat_s = loss_s = np.zeros((len(ts), S))
            reach = r.reachable[None, :]
            svc_loss[ts] += w * np.where(reach, loss_s, 1.0)
            svc_lat_acc[ts] += w * np.where(reach, lat_s, 0.0)
            svc_lat_w[ts] += w * reach
        with np.errstate(invalid="ignore", divide="ignore"):
            svc_lat = np.where(svc_lat_w > 0, svc_lat_acc / svc_lat_w, np.nan)

        # ---- link truth
        fully_down = (down_frac >= 1.0) | pop_down[:, self.link_ends[:, 0]] | pop_down[:, self.link_ends[:, 1]]
        util_true = np.minimum(rho.max(2), 1.0)
        lat_true = self.prop_ms[None, :] + lc["per_hop_ms"] + q.max(2)
        counter_true = pc.max(2) + gray * gray_visible[None, :]

        # ---- measurement (fixed-shape draws, so noise is identical with or without faults)
        mr = rng["measure"]
        util_obs = np.clip(util_true + mr.normal(0, sc["util_noise"], (T, L)), 0, 1)
        lat_obs = lat_true * (1 + mr.normal(0, lc["jitter_rel"], (T, L))) + np.abs(mr.normal(0, lc["jitter_abs_ms"], (T, L)))
        lat_obs += (mr.random((T, L)) < lc["outlier_prob"]) * mr.uniform(1, 20, (T, L))
        bg = (mr.random((T, L)) < ltc["background_prob"]) * mr.exponential(ltc["background_mean_pct"] / 100, (T, L))
        loss_obs = 100 * (np.clip(counter_true * (1 + mr.normal(0, ltc["counter_noise_rel"], (T, L))), 0, 1) + bg)
        util_obs[fully_down] = 0.0
        lat_obs[fully_down] = np.nan
        loss_obs[fully_down] = np.nan

        svc_lat_obs = svc_lat * (1 + mr.normal(0, lc["jitter_rel"], (T, S))) + np.abs(mr.normal(0, lc["jitter_abs_ms"], (T, S)))
        svc_lat_obs += (mr.random((T, S)) < lc["outlier_prob"]) * mr.uniform(1, 20, (T, S))
        svc_loss_obs = 100 * mr.binomial(sc["probes_per_step"], np.clip(svc_loss, 0, 1)) / sc["probes_per_step"]

        # ---- missing data: random drop-outs plus collector gaps
        mm = rng["missing"]
        link_missing = mm.random((T, L)) < sc["missing_prob"]
        svc_missing = mm.random((T, S)) < sc["missing_prob"]
        svc_src = np.array([self.pop_idx[self.kg.G.nodes[s]["src"]] for s in self.services])
        for g in spec.get("gaps", []):
            s, e = g["t_start"], g["t_end"]
            if g["scope"] == "link":
                link_missing[s:e, self.link_idx[g["target"]]] = True
            else:
                pi = self.pop_idx[g["target"].split(":", 1)[1]]
                link_missing[s:e] |= (self.link_ends[:, 0] == pi) | (self.link_ends[:, 1] == pi)
                svc_missing[s:e] |= svc_src == pi
        for a in (util_obs, lat_obs, loss_obs):
            a[link_missing] = np.nan
        svc_lat_obs[svc_missing] = np.nan
        svc_loss_obs[svc_missing] = np.nan

        observed_events = self._observed_events(spec, down_frac, fully_down, pop_down, rng["status"])
        return SimResult(
            spec=spec,
            links=self.links,
            services=self.services,
            obs={"util": util_obs, "latency_ms": lat_obs, "loss_pct": loss_obs, "svc_latency_ms": svc_lat_obs, "svc_loss_pct": svc_loss_obs},
            truth={
                "util": util_true,
                "rho": rho.max(2),
                "latency_ms": lat_true,
                "counter_loss_pct": 100 * counter_true,
                "svc_latency_ms": svc_lat,
                "svc_loss_pct": 100 * svc_loss,
                "down_frac": down_frac,
                "pop_down": pop_down,
                "keys": keys,
            },
            observed_events=observed_events,
        )

    def _observed_events(self, spec, down_frac, fully_down, pop_down, srng):
        """Status polling once per step (a flapping link is seen down with probability = its down fraction),
        plus the planned-maintenance log. Never contains fault labels, gray failures or metric misconfigurations."""
        T, L = down_frac.shape
        partial = (down_frac > 0) & (down_frac < 1)
        seen_link = (down_frac >= 1.0) | (partial & (srng.random((T, L)) < down_frac))
        events = []
        for m in spec.get("maintenance", []):
            events.append({"t": m["t_start"], "type": "maintenance", **{k: m[k] for k in ("id", "target", "t_start", "t_end", "action", "description")}})
        prev_l, prev_p = np.zeros(L, bool), np.zeros(pop_down.shape[1], bool)
        for i in range(T):
            for j in np.flatnonzero(seen_link[i] != prev_l):
                events.append({"t": i, "type": "link_status", "element": self.links[j], "status": "down" if seen_link[i, j] else "up"})
            for j in np.flatnonzero(pop_down[i] != prev_p):
                events.append({"t": i, "type": "pop_status", "element": pop_node(self.pops[j]), "status": "down" if pop_down[i, j] else "up"})
            prev_l, prev_p = seen_link[i], pop_down[i]
        return sorted(events, key=lambda e: (e["t"], e["type"]))

    # ------------------------------------------------------------------ impact (noise-free, vs counterfactual)

    def _impact(self, res, cf, fault):
        s, e = fault["t_start"], fault["t_end"]
        ic = self.cfg["impact"]
        lat, lat0 = res.truth["svc_latency_ms"][s:e], cf.truth["svc_latency_ms"][s:e]
        unreachable = np.isnan(lat) & ~np.isnan(lat0)
        dlat = np.nan_to_num(lat - lat0, nan=0.0)
        dloss = res.truth["svc_loss_pct"][s:e] - cf.truth["svc_loss_pct"][s:e]
        dutil = np.abs(res.truth["util"][s:e] - cf.truth["util"][s:e])
        affected = (dlat.max(0) > ic["affected_latency_ms"]) | (dloss.max(0) > ic["affected_loss_pct"]) | unreachable.any(0)
        root = fault["root_cause"]
        root_idx = [self.link_idx[root]] if root in self.link_idx else [
            i for i, (u, v) in enumerate(self.link_ends) if self.pops[u] == root[4:] or self.pops[v] == root[4:]
        ]
        others = np.setdiff1d(np.arange(len(self.links)), root_idx)
        status_seen = any(
            ev["type"] in ("link_status", "pop_status") and s <= ev["t"] < e and (ev["element"] == root or ev["element"] in [self.links[i] for i in root_idx])
            for ev in res.observed_events
        )
        out = {
            "max_service_latency_delta_ms": float(dlat.max()) if dlat.size else 0.0,
            "max_service_loss_delta_pct": float(dloss.max()) if dloss.size else 0.0,
            "services_unreachable": int(unreachable.any(0).sum()),
            "services_affected": int(affected.sum()),
            "max_root_util_delta": float(dutil[:, root_idx].max()) if root_idx else 0.0,
            "max_other_util_delta": float(dutil[:, others].max()) if len(others) else 0.0,
            "max_util_in_window": float(res.truth["util"][s:e].max()),
            "status_change_observed": bool(status_seen),
            "root_counters_show_loss": bool(fault["type"] == "link_degradation" and not fault["params"]["silent"]),
        }
        lat_d, loss_d = out["max_service_latency_delta_ms"], out["max_service_loss_delta_pct"]
        util_d = max(out["max_root_util_delta"], out["max_other_util_delta"])
        if out["services_unreachable"] or loss_d >= 1.0 or lat_d >= 5.0 or util_d >= 0.3:
            out["impact_class"] = "obvious"
        elif loss_d < 0.1 and lat_d < 0.5 and util_d < 0.05:
            out["impact_class"] = "subtle"
        else:
            out["impact_class"] = "moderate"
        return out


# ---------------------------------------------------------------------- twin view and persistence


def observed_kg(sim, res):
    """The digital twin's KG: intended metrics, plus only what the observed events reveal."""
    kg = NetTwinKG.from_config(sim.topo, sim.cfg)
    actions = defaultdict(lambda: {"links_down": [], "links_up": [], "pops_down": [], "pops_up": [], "metrics": {}})
    for ev in res.observed_events:
        if ev["type"] == "maintenance":
            kg.add_maintenance(ev["id"], [ev["target"]], ev["t_start"], ev["t_end"], ev["description"])
            if ev["action"] == "drain":
                actions[ev["t_start"]]["metrics"][ev["target"]] = sim.sc["drain_metric"]
                actions[ev["t_end"]]["metrics"][ev["target"]] = None
        elif ev["type"] == "link_status":
            actions[ev["t"]]["links_down" if ev["status"] == "down" else "links_up"].append(ev["element"])
        elif ev["type"] == "pop_status":
            actions[ev["t"]]["pops_down" if ev["status"] == "down" else "pops_up"].append(ev["element"])
    for t in sorted(actions):
        kg.apply_change(t, reason="observed", **actions[t])
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


def write_scenario(res, out_dir):
    """Write link / service Parquet and observed events; return the manifest entry (ground truth included)."""
    sid = res.spec["scenario_id"]
    d = Path(out_dir) / sid
    d.mkdir(parents=True, exist_ok=True)
    links, services = to_frames(res)
    links.to_parquet(d / "links.parquet", index=False)
    services.to_parquet(d / "services.parquet", index=False)
    (d / "observed_events.json").write_text(json.dumps(res.observed_events, indent=1, default=_json_default) + "\n")
    entry = {
        **res.spec,
        "faults": [{**f, "impact": res.impacts.get(f["id"])} for f in res.spec["faults"]],
        "files": {k: f"{sid}/{k}" for k in ("links.parquet", "services.parquet", "observed_events.json")},
    }
    return json.loads(json.dumps(entry, default=_json_default))


def write_manifest(entries, out_dir):
    path = Path(out_dir) / "scenarios.json"
    path.write_text(json.dumps(entries, indent=1, default=_json_default) + "\n")
    return path
