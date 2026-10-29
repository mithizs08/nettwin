"""Fault and distractor specs (PLAN.md section 4).

Specs are plain dicts so a scenario serialises straight to JSON. The simulator turns them into effects.

Fault:       {id, type, root_cause, t_start, t_end, params}   root_cause is a KG id (link:... or pop:...)
Maintenance: {id, target, t_start, t_end, action, description} action in drain | down | noop (planned, logged)
Gap:         {id, scope, target, t_start, t_end}                 telemetry missing; scope pop | link
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
        p = {"factor": _uniform(rng, fc["factor"]), "ramp_steps": _int(rng, fc["ramp_steps"])}
    elif ftype == "routing_misconfig":
        p = _misconfig_params(sim, root, rng, fc, overrides.get("new_metric"))
    else:
        p = {}
    p.update(overrides)
    return {"id": fault_id, "type": ftype, "root_cause": root, "t_start": int(t_start), "t_end": int(t_start + duration), "params": p}


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


def _place(rng, windows, lo, hi, duration, gap, tries=200):
    """Random start in [lo, hi - duration) not within `gap` steps of any existing window."""
    if hi - duration <= lo:
        return None
    for _ in range(tries):
        s = int(rng.integers(lo, hi - duration))
        if all(s + duration + gap <= a or s >= b + gap for a, b in windows):
            return s
    return None


def _pick_root(sim, ftype, rng):
    if ftype in LINK_FAULTS:
        return sim.links[rng.integers(len(sim.links))]
    return pop_node(sim.pops[rng.integers(len(sim.pops))])


def sample_scenario(sim, scenario_id, seed, n_faults=None, fault_types=None, distractors=True):
    """Random scenario: up to faults.max_per_scenario non-overlapping faults after warmup, plus distractors."""
    cfg = sim.cfg
    rng = np.random.default_rng([seed, 11])
    T, warm, gap = sim.n_steps, sim.warmup_steps, cfg["faults"]["min_gap_steps"]
    types = list(fault_types or FAULT_TYPES)
    if n_faults is None:
        n_faults = int(rng.integers(1, cfg["faults"]["max_per_scenario"] + 1))
    windows, faults = [], []
    for i in range(n_faults):
        ftype = types[rng.integers(len(types))]
        dur = _int(rng, cfg["faults"][ftype]["duration_steps"])
        start = _place(rng, windows, warm, T, dur, gap)
        if start is None:
            continue
        for _ in range(20):  # a misconfig on a bridge link cannot move traffic; redraw the root
            try:
                f = make_fault(sim, ftype, _pick_root(sim, ftype, rng), start, dur, rng, fault_id=f"F{i}")
                break
            except ValueError:
                continue
        else:
            continue
        faults.append(f)
        windows.append((start, start + dur))

    maintenance, gaps, event_days = [], [], []
    if distractors:
        dc = cfg["distractors"]
        actions = list(dc["maintenance_actions"])
        probs = np.array([dc["maintenance_actions"][a] for a in actions], dtype=float)
        spd = sim.steps_per_day
        night = dc["maintenance_start_hours"]
        for i in range(_int(rng, dc["maintenance_per_scenario"])):
            dur = _int(rng, dc["maintenance_duration_steps"])
            action = actions[rng.choice(len(actions), p=probs / probs.sum())]
            # overnight window, clear of faults, and (like a real change review) not congesting anything
            for _ in range(30):
                day = int(rng.integers(warm // spd, T // spd))
                start = day * spd + int(rng.integers(night[0] * 60, night[1] * 60) // sim.step_minutes)
                target = sim.links[rng.integers(len(sim.links))]
                if start + dur > T or not all(start + dur + gap <= a or start >= b + gap for a, b in windows):
                    continue
                key = ((target,), (), ()) if action == "down" else ((), (), ((target, sim.sc["drain_metric"]),))
                if action == "noop" or sim.planned_peak_util(key, start, start + dur) <= dc["maintenance_max_util"]:
                    break
            else:
                continue
            maintenance.append(
                {
                    "id": f"M{i}",
                    "target": target,
                    "t_start": start,
                    "t_end": start + dur,
                    "action": action,
                    "description": {"drain": "planned traffic drain", "down": "planned link outage", "noop": "planned config check"}[action],
                }
            )
            windows.append((start, start + dur))
        for i in range(_int(rng, dc["gaps_per_scenario"])):
            dur = _int(rng, dc["gap_duration_steps"])
            start = int(rng.integers(0, T - dur))  # gaps may overlap faults, as in real collectors
            if rng.random() < dc["gap_pop_scope_prob"]:
                scope, target = "pop", pop_node(sim.pops[rng.integers(len(sim.pops))])
            else:
                scope, target = "link", sim.links[rng.integers(len(sim.links))]
            gaps.append({"id": f"G{i}", "scope": scope, "target": target, "t_start": start, "t_end": start + dur})
        if rng.random() < dc["event_day_prob"]:
            event_days.append({"day": int(rng.integers(0, T // sim.steps_per_day)), "factor": _uniform(rng, dc["event_day_factor"])})

    return {
        "scenario_id": scenario_id,
        "topology": sim.name,
        "seed": int(seed),
        "n_steps": T,
        "step_minutes": sim.step_minutes,
        "faults": sorted(faults, key=lambda f: f["t_start"]),
        "maintenance": sorted(maintenance, key=lambda m: m["t_start"]),
        "gaps": gaps,
        "event_days": event_days,
    }
