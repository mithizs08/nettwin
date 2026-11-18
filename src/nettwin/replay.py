"""Real-time replay without an oracle incident window (Phase 12).

A scenario is replayed step by step from the end of the one-day warm-up, as an operator would live it:
- The live KG starts as the operator inventory and changes only when an observed status event arrives (link up/down,
  PoP unreachable/reachable), at the step it arrives, with the simulator's polling lag already in that step.
- The frozen GNN (GAT-h32 seed 0, the checkpoint the Phase 6 agent tool uses) scores every step. A step alarms when
  some link's detection probability reaches the frozen validation threshold of that checkpoint. Scoring the whole
  scenario in one pass gives the same numbers as scoring each prefix (checked by `check_causal`), so it is causal.
- Each alarm is attributed to the GNN's top-1 root element at that step. Repeated alarms on the same element merge
  into one alarm episode while they come within MERGE_GAP steps of the episode's last alarm.

Ground truth is read only by `score_scenario`, after every decision has been made.
"""
import copy

import numpy as np

from nettwin.agent.eval import localisation_credit
from nettwin.baselines import b2_features, window_score
from nettwin.evaluation import LAG, MAINT_TAIL, WARMUP
from nettwin.gnn.data import graph_for
from nettwin.gnn.models import predict

START = WARMUP            # replay starts after the one-day warm-up, like every other phase
MERGE_GAP = 12            # steps (1 hour): a new alarm on an element whose last alarm is at most this old extends its episode
RECHECK = 6               # steps (30 minutes): a "wait" decision is re-checked once, this many steps later
DETECT_TAIL = MAINT_TAIL  # a page counts for a fault from its start until 1 hour after its end
DAY = 288
CHANGE_ACTIONS = ("drain", "down")


# ------------------------------------------------------------------ GNN stream and episodes

def gnn_stream(obs, model, cfg_s):
    """(graph, det (T, L), elem (T, L + P)) for every step, from observable telemetry only."""
    g = graph_for(obs.topology, cfg_s)
    det, elem = predict(model, b2_features(obs).astype(np.float32), g)
    return g, det, elem


def check_causal(obs, model, cfg_s, steps):
    """Largest absolute difference between one-pass scores and scores from the prefix [0, t] at each given step."""
    g, det, elem = gnn_stream(obs, model, cfg_s)
    feat = b2_features(obs).astype(np.float32)
    worst = 0.0
    for t in steps:
        d, e = predict(model, feat[:t + 1], g)
        worst = max(worst, float(np.abs(d[t] - det[t]).max()), float(np.abs(e[t] - elem[t]).max()))
    return worst


def episodes(det, elem, elements, threshold, start=START):
    """Alarm episodes: [{element, open, last, end, n_alarms, alarm_steps}] in order of opening. `end` is the step after
    which the episode can no longer extend (last alarm + MERGE_GAP), capped at the scenario end."""
    T = det.shape[0]
    alarm = det.max(1) >= threshold
    top = elem.argmax(1)
    open_, out = {}, []
    for t in range(start, T):
        if not alarm[t]:
            continue
        el = elements[int(top[t])]
        ep = open_.get(el)
        if ep is not None and t - ep["last"] <= MERGE_GAP:
            ep["last"] = t
            ep["n_alarms"] += 1
            ep["alarm_steps"].append(t)
        else:
            ep = {"element": el, "open": t, "last": t, "n_alarms": 1, "alarm_steps": [t],
                  "alarm_links": [elements[j] for j in np.flatnonzero(det[t] >= threshold)][:5]}
            open_[el] = ep
            out.append(ep)
    for k, ep in enumerate(out):
        ep["episode_id"] = f"E{k:03d}"
        ep["end"] = min(ep["last"] + MERGE_GAP + 1, T)
    return out


# ------------------------------------------------------------------ live KG from status events

def status_events(obs):
    return [e for e in obs.events if e.get("type") in ("link_status", "pop_status")]


def kg_at(inventory_kg, obs, now):
    """The operator KG as known at step `now`: the inventory with every observed status event that arrived at or before
    `now` applied at its arrival step. A fresh copy each call; the shared inventory is never changed."""
    kg = copy.deepcopy(inventory_kg)
    for e in status_events(obs):
        if e["t"] > now:
            break
        el = e["element"] if e["type"] == "link_status" or e["element"].startswith("pop:") else f"pop:{e['element']}"
        if e["type"] == "link_status":
            kg.apply_change(e["t"], **({"links_down": [el]} if e["status"] == "down" else {"links_up": [el]}), reason="status event")
        else:
            kg.apply_change(e["t"], **({"pops_down": [el]} if e["status"] == "unreachable" else {"pops_up": [el]}), reason="status event")
    return kg


def logged_changes(obs):
    return [e for e in obs.events if e.get("type") == "maintenance"]


def change_active(obs, t):
    """A logged drain or planned outage that has started by step t and ended less than MAINT_TAIL steps before it."""
    return any(c["action"] in CHANGE_ACTIONS and c["t_start"] <= t < c["t_end"] + MAINT_TAIL for c in logged_changes(obs))


# ------------------------------------------------------------------ reference policies

def gnn_pages(eps, elem, elements):
    """Raw GNN: every episode pages when it opens, with the GNN ranking at that step."""
    return [{"episode_id": e["episode_id"], "t": e["open"], "ranking": elem[e["open"]].tolist(), "kind": "scores"} for e in eps]


def gnn_cl_pages(eps, elem, elements, obs):
    """GNN + change-log rule: as the raw GNN, but no page when a logged drain or planned outage is active (or ended
    within the last hour) when the episode opens. The rule only reads the change log, which is observable."""
    return [p for p, e in zip(gnn_pages(eps, elem, elements), eps) if not change_active(obs, e["open"])]


# ------------------------------------------------------------------ scoring (reads ground truth)

def faults_of(entry, start=START):
    return [f for f in entry["faults"] if f["impact"]["observable"] and f["t_start"] >= start]


def overlaps_change(entry, f):
    return any(m["action"] in CHANGE_ACTIONS and m["t_start"] < f["t_end"] and m["t_end"] > f["t_start"] for m in entry["maintenance"])


def ranking_vector(page, elements):
    if page["kind"] == "scores":
        return np.asarray(page["ranking"], float)
    sc = np.zeros(len(elements))
    seen = []
    for c in page["ranking"]:
        if isinstance(c, str) and c.startswith("link:") and c not in elements and "--" in c:
            c = "link:" + "--".join(reversed(c[5:].split("--", 1)))
        if c in elements and c not in seen:
            seen.append(c)
    for r, c in enumerate(seen[:3]):
        sc[elements.index(c)] = 3 - r
    return sc


def score_scenario(entry, pages, elements, T):
    """Per-scenario counts for one method: pages, false pages, faults detected, delays, root-cause credit."""
    faults = faults_of(entry)
    days = (T - START) / DAY
    out = {"days": days, "pages": len(pages), "false_pages": 0, "faults": len(faults), "detected": 0, "delay_sum": 0.0,
           "top1_sum": 0.0, "top3_sum": 0.0, "change_faults": 0, "change_missed": 0, "per_fault": []}
    hit = set()
    for p in pages:
        owners = [i for i, f in enumerate(faults) if f["t_start"] <= p["t"] < f["t_end"] + DETECT_TAIL]
        if not owners and not any(f["t_start"] <= p["t"] < f["t_end"] + DETECT_TAIL for f in entry["faults"]):
            out["false_pages"] += 1
        hit.update(owners)
    for i, f in enumerate(faults):
        first = min((p for p in pages if f["t_start"] <= p["t"] < f["t_end"] + DETECT_TAIL), key=lambda p: p["t"], default=None)
        rec = {"fault_id": f["id"], "type": f["type"], "root": f["root_cause"], "t_start": f["t_start"], "detected": first is not None,
               "overlaps_change": overlaps_change(entry, f)}
        if first is not None:
            others = {g["root_cause"] for g in entry["faults"] if g is not f and g["t_start"] <= first["t"] < g["t_end"] + LAG}
            sc = ranking_vector(first, elements)
            for r in others:
                if r in elements and r != f["root_cause"]:
                    sc[elements.index(r)] = -np.inf
            c = localisation_credit(sc, elements, f["root_cause"])
            rec.update(delay_steps=first["t"] - f["t_start"], **c)
            out["detected"] += 1
            out["delay_sum"] += first["t"] - f["t_start"]
            out["top1_sum"] += c["top1"]
            out["top3_sum"] += c["top3"]
        if rec["overlaps_change"]:
            out["change_faults"] += 1
            out["change_missed"] += not rec["detected"]
        out["per_fault"].append(rec)
    return out


def window_ranking(elem, t0, t1):
    return window_score(elem[t0:t1], "q90")
