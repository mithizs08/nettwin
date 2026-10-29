"""Fault and distractor specs (PLAN.md section 4).

Specs are plain dicts so a scenario serialises straight to JSON. The simulator turns them into effects.

Fault:       {id, type, root_cause, t_start, t_end, params, placement, linked_maintenance, caused_by_maintenance}
             root_cause is a KG id (link:... or pop:...); placement is isolated | maintenance_linked | concurrent
Maintenance: {id, target, t_start, t_end, action, description} action in drain | down | noop (planned, logged)
Gap:         {id, scope, target, t_start, t_end[, responder_down]} scope pop | link | probe (one-hop probe only)
Event day:   {day, factor}                                       whole-network traffic uplift (a normal peak)

Ground truth (faults) and observable inputs (maintenance log, telemetry) are kept apart: see simulator.py.
"""
import math

import numpy as np

from nettwin.kg import FAULT_TYPES, pop_node

LINK_FAULTS = ("link_down", "link_degradation", "link_flapping", "routing_misconfig")
POP_FAULTS = ("node_down", "demand_surge")


def _uniform(rng, lo_hi):
    return float(rng.uniform(*lo_hi))


def _int(rng, lo_hi):
    return int(rng.integers(lo_hi[0], lo_hi[1] + 1))


def _loguniform(rng, lo_hi):
    lo, hi = lo_hi
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def make_fault(sim, ftype, root, t_start, duration, rng, fault_id="F0", **overrides):
    """Build one fault spec with severity sampled from config; `overrides` pin any param."""
    if ftype not in FAULT_TYPES:
        raise ValueError(f"unknown fault type {ftype}")
    sim.kg._require(root, "Link" if ftype in LINK_FAULTS else "PoP")
    fc = sim.cfg["faults"][ftype]
    if ftype == "link_degradation":
        p = {
            "loss_pct": _loguniform(rng, fc["loss_pct"]),
            "silent": bool(rng.random() < fc["silent_prob"]),
            "ramp_steps": _int(rng, fc["ramp_steps"]),
            "ramp_down_steps": _int(rng, fc["ramp_down_steps"]),
            "burst_sigma": fc["burst_sigma"],
        }
    elif ftype == "link_flapping":
        p = {
            "active_prob": _uniform(rng, fc["active_prob"]),
            "max_down_frac": _uniform(rng, fc["max_down_frac"]),
            "flaps_per_step": _uniform(rng, fc["flaps_per_step"]),
            "convergence_s": _uniform(rng, fc["convergence_s"]),
        }
    elif ftype == "demand_surge":
        p = {"factor": _loguniform(rng, fc["factor"]), "ramp_steps": _int(rng, fc["ramp_steps"]), "ramp_down_steps": _int(rng, fc["ramp_down_steps"])}
    elif ftype == "routing_misconfig":
        p = _misconfig_params(sim, root, rng, fc, overrides.get("new_metric"))
    else:  # link_down, node_down: traffic is lost for the convergence time at the start and at the restore
        p = {"convergence_s": _uniform(rng, fc["convergence_s"])}
    p.update(overrides)
    return {"id": fault_id, "type": ftype, "root_cause": root, "t_start": int(t_start), "t_end": int(t_start + duration), "params": p,
            "placement": "isolated", "linked_maintenance": None, "caused_by_maintenance": False}


def _misconfig_params(sim, link, rng, fc, new_metric=None):
    """Metric pushed up (traffic leaves the link) or down (traffic is pulled onto it). Must change routing."""
    old = sim.base_metric[link]
    minimal = False
    if new_metric is not None:
        candidates = [("up" if new_metric > old else "down", int(new_metric))]
    elif rng.random() < fc["minimal_prob"]:
        # near-miss typo: the smallest metric step (x or / 1.1^k) that shifts at least one PoP pair
        minimal = True
        up = bool(rng.random() < 0.5)
        hi = fc["metric_factor"][1]
        candidates, k = [], 1
        while 1.1**k <= hi:
            candidates.append(("up" if up else "down", max(1, round(old * 1.1**k) if up else round(old / 1.1**k))))
            k += 1
    else:
        candidates = []
        for _ in range(50):
            f = _loguniform(rng, fc["metric_factor"])
            up = bool(rng.random() < 0.5)
            candidates.append(("up" if up else "down", max(1, round(old * f) if up else round(old / f))))
    for direction, m in candidates:
        if m != old and sim.routing_changes({link: m}):
            return {"old_metric": int(old), "new_metric": int(m), "direction": direction, "minimal": minimal}
    raise ValueError(f"no metric change on {link} alters routing")


def _clear(windows, s, e, gap):
    return all(e + gap <= a or s >= b + gap for a, b in windows)


def _place(rng, windows, lo, hi, duration, gap, tries=200):
    """Random start in [lo, hi - duration) not within `gap` steps of any existing window."""
    if hi - duration <= lo:
        return None
    for _ in range(tries):
        s = int(rng.integers(lo, hi - duration))
        if _clear(windows, s, s + duration, gap):
            return s
    return None


def _pick_root(sim, ftype, rng):
    if ftype in LINK_FAULTS:
        return sim.links[rng.integers(len(sim.links))]
    return pop_node(sim.pops[rng.integers(len(sim.pops))])


def check_fresh_seed(cfg, seed):
    """Raise if a scenario seed was already viewed during development or diagnostics (audit m12)."""
    for lo, hi in cfg["seeds"]["reserved_ranges"]:
        if lo <= seed <= hi:
            raise ValueError(f"seed {seed} is reserved (viewed range {lo}..{hi})")


def _sample_maintenance(sim, rng, event_days, windows):
    cfg, dc = sim.cfg, sim.cfg["distractors"]
    T, warm, gap, spd = sim.n_steps, sim.warmup_steps, cfg["faults"]["min_gap_steps"], sim.steps_per_day
    actions = list(dc["maintenance_actions"])
    probs = np.array([dc["maintenance_actions"][a] for a in actions], dtype=float)
    out = []
    for i in range(_int(rng, dc["maintenance_per_scenario"])):
        dur = _int(rng, dc["maintenance_duration_steps"])
        action = actions[rng.choice(len(actions), p=probs / probs.sum())]
        daytime = rng.random() < dc["maintenance_daytime_prob"]   # emergency / daytime changes exist too (audit M2)
        hours = dc["maintenance_daytime_hours"] if daytime else dc["maintenance_start_hours"]
        for _ in range(30):
            day = int(rng.integers(warm // spd, T // spd))
            start = day * spd + int(rng.integers(hours[0] * 60, hours[1] * 60) // sim.step_minutes)
            target = sim.links[rng.integers(len(sim.links))]
            if start < warm or start + dur > T or not _clear(windows, start, start + dur, gap):
                continue
            key = ((target,), (), ()) if action == "down" else ((), (), ((target, sim.sc["drain_metric"]),))
            if action == "noop" or sim.planned_peak_util(key, start, start + dur, event_days) <= dc["maintenance_max_util"]:
                break
        else:
            continue
        desc = {"drain": "planned traffic drain", "down": "planned link outage", "noop": "planned config check"}[action]
        out.append({"id": f"M{i}", "target": target, "t_start": start, "t_end": start + dur, "action": action, "description": desc})
        windows.append((start, start + dur))
    return out


def sample_scenario(sim, scenario_id, seed, n_faults=None, fault_types=None, distractors=True):
    """Random scenario. Maintenance is placed first; then each fault is isolated, maintenance-linked (overlapping or
    right after a change window, sometimes on the maintained element) or concurrent with another fault (audit M2).
    The ground truth is always the fault, never the maintenance."""
    cfg, dc = sim.cfg, sim.cfg["distractors"]
    rng = np.random.default_rng([seed, 11])
    T, warm, gap = sim.n_steps, sim.warmup_steps, cfg["faults"]["min_gap_steps"]
    types = list(fault_types or FAULT_TYPES)
    if n_faults is None:
        n_faults = int(rng.integers(1, cfg["faults"]["max_per_scenario"] + 1))
    event_days, maintenance, gaps = [], [], []
    m_windows = []
    if distractors:
        if rng.random() < dc["event_day_prob"]:
            event_days.append({"day": int(rng.integers(0, T // sim.steps_per_day)), "factor": _uniform(rng, dc["event_day_factor"])})
        maintenance = _sample_maintenance(sim, rng, event_days, m_windows)

    faults, f_windows = [], []
    for i in range(n_faults):
        ftype = types[rng.integers(len(types))]
        dur = _int(rng, cfg["faults"][ftype]["duration_steps"])
        u = rng.random()
        mode = "isolated"
        if maintenance and u < dc["overlap_maintenance_prob"]:
            mode = "maintenance_linked"
        elif faults and u < dc["overlap_maintenance_prob"] + dc["concurrent_prob"]:
            mode = "concurrent"
        linked, caused, start, root = None, False, None, None
        flapping_windows = [(f["t_start"], f["t_end"]) for f in faults if f["type"] == "link_flapping"]
        for _ in range(30):
            if mode == "maintenance_linked":
                m = maintenance[int(rng.integers(len(maintenance)))]
                start = m["t_start"] + int(rng.integers(0, m["t_end"] - m["t_start"] + 12))
                caused = bool(rng.random() < dc["caused_by_maintenance_prob"])
                if caused:
                    a, b = m["target"].split(":", 1)[1].split("--")
                    root = m["target"] if ftype in LINK_FAULTS else pop_node(a if rng.random() < 0.5 else b)
                ok = start + dur <= T and _clear(f_windows, start, start + dur, gap)
                linked = m["id"]
            elif mode == "concurrent":
                other = faults[int(rng.integers(len(faults)))]
                start = int(rng.integers(other["t_start"], other["t_end"]))
                ok = start + dur <= T and start >= warm
            else:
                start = _place(rng, f_windows + m_windows, warm, T, dur, gap)
                ok = start is not None
            if ok and ftype == "link_flapping" and not _clear(flapping_windows, start, start + dur, 0):
                ok = False          # the simulator allows one partially-down link per step
            if ok:
                break
        else:
            continue
        for _ in range(20):  # a misconfig on a bridge link cannot move traffic; redraw the root
            r = root or _pick_root(sim, ftype, rng)
            if mode == "concurrent" and any(f["root_cause"] == r for f in faults):
                root = None
                continue
            try:
                f = make_fault(sim, ftype, r, start, dur, rng, fault_id=f"F{i}")
                break
            except ValueError:
                root = None
                continue
        else:
            continue
        f.update(placement=mode, linked_maintenance=linked, caused_by_maintenance=caused and f["root_cause"] == root)
        faults.append(f)
        f_windows.append((start, start + dur))

    if distractors:
        for i in range(_int(rng, dc["gaps_per_scenario"])):
            dur = _int(rng, dc["gap_duration_steps"])
            start = int(rng.integers(0, T - dur))  # gaps may overlap faults, as in real collectors
            if rng.random() < dc["gap_pop_scope_prob"]:
                gaps.append({"id": f"G{i}", "scope": "pop", "target": pop_node(sim.pops[rng.integers(len(sim.pops))]), "t_start": start,
                             "t_end": start + dur, "responder_down": bool(rng.random() < dc["gap_responder_down_prob"])})
            else:
                gaps.append({"id": f"G{i}", "scope": "link", "target": sim.links[rng.integers(len(sim.links))], "t_start": start, "t_end": start + dur})
        for i in range(_int(rng, dc["probe_gaps_per_scenario"])):
            dur = _int(rng, dc["gap_duration_steps"])
            start = int(rng.integers(0, T - dur))
            gaps.append({"id": f"P{i}", "scope": "probe", "target": sim.links[rng.integers(len(sim.links))], "t_start": start, "t_end": start + dur})

    return {
        "scenario_id": scenario_id,
        "topology": sim.name,
        "seed": int(seed),
        "n_steps": T,
        "step_minutes": sim.step_minutes,
        "faults": sorted(faults, key=lambda f: (f["t_start"], f["id"])),
        "maintenance": sorted(maintenance, key=lambda m: m["t_start"]),
        "gaps": gaps,
        "event_days": event_days,
    }
