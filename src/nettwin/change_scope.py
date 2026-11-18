"""Phase 13: does a logged change explain what is observed? A KG check for the triage agent.

Phase 6 and Phase 12 found the agent misses too many faults that overlap a logged drain or planned outage: it reads the
change as the explanation. `change_scope` asks the operator inventory KG what each logged change can explain, and lists
the observed symptoms that fall outside it.

What a change can explain (every planned change in this simulator targets one link):
- drain (routing metric raised): the target link loses its traffic; the links on the paths the KG reroutes onto gain it;
  services whose paths change may see a longer path (more latency). Nothing goes down and no traffic is lost.
- planned outage (`down`): as a drain, and the target link is also reported down (or dark) by the poller.
- no-op work: nothing.
A change explains a symptom only while it is active, or within MAINT_TAIL steps of its end (reconvergence).

Symptoms are what an operator sees, as known at the incident's decision step:
- alerts active in the incident window (`alerts_at`, causal);
- links whose utilisation (beyond any network-wide shift) or loss changed against the same steps one day earlier.
Collector gaps, probe timeouts, alerts also seen at the same time one day earlier, and service loss no wider than one
day earlier happen without faults and changes; they are listed as `neutral`, not counted.

Only observable data is used: telemetry, alerts, the change log and the inventory KG. Nothing reads ground truth.
"""
import copy
import warnings

import numpy as np

from nettwin.alerts import alerts_at
from nettwin.evaluation import MAINT_TAIL

CHANGE_ACTIONS = ("drain", "down")
DRAIN_METRIC = 1_000_000          # config.yaml simulator.drain_metric (frozen); a drained link carries no transit
# Telemetry thresholds (development on validation only; decision log, Phase 13).
UTIL_SHIFT = 0.10                 # |util now - util one day earlier|, as a fraction of capacity ...
UTIL_SHIFT_REL = 0.30             # ... and at least this share of the day-earlier level
LOSS_SHIFT_PCT = 0.05             # link loss (discards) change, percent of packets (the link_discards alert level)
NEUTRAL_RULES = ("link_no_data", "link_probe_fail")
SERVICE_LOSS_EXCESS = 2           # services with SLA loss must exceed the day-earlier count by this many to count
MAX_LISTED = 15


def footprint(inventory_kg, change, drain_metric=DRAIN_METRIC):
    """What one logged change can explain, from the inventory KG: the target link, the links whose traffic the change
    moves (losing and gaining), and the services whose routing paths change."""
    target = change["target"]
    kg = copy.deepcopy(inventory_kg)
    services_before = set(kg.services_on(target, t=kg.t))
    n_events = len(kg.events)
    if change["action"] == "drain":
        kg.apply_change(kg.t, metrics={target: drain_metric}, reason="change scope")
    else:
        kg.apply_change(kg.t, links_down=[target], reason="change scope")
    gaining, losing, moved_pairs, cut_pairs = set(), {target}, [], []
    for ev in kg.events[n_events:]:
        if ev["type"] != "reroute":
            continue
        moved_pairs.append(tuple(ev["pair"]))
        if ev.get("unreachable"):
            cut_pairs.append(tuple(ev["pair"]))
        old = {l for p in ev["removed"] for l in kg.links_of_path(p)}
        new = {l for p in ev["added"] for l in kg.links_of_path(p)}
        gaining |= new - old
        losing |= old - new
    services = {s for pair in moved_pairs for s in kg._pair_services.get(pair, ())} | services_before
    a, b = target.split(":", 1)[1].split("--")
    return {"change_id": change["id"], "action": change["action"], "target": target, "target_pops": [f"pop:{a}", f"pop:{b}"],
            "links_losing_traffic": sorted(losing), "links_gaining_traffic": sorted(gaining - {target}),
            "services_rerouted": sorted(services), "pop_pairs_rerouted": len(moved_pairs), "pop_pairs_cut_off": len(cut_pairs),
            "services_cut_off": sorted({s for pair in cut_pairs for s in kg._pair_services.get(pair, ())})}


def _explains(fp, change, sym, floor_rel):
    """Whether one footprint explains one symptom (dict with kind, element, start, end in absolute steps)."""
    if change["action"] not in CHANGE_ACTIONS:
        return False
    if sym["start"] < change["t_start"] - 1 or sym["start"] >= change["t_end"] + MAINT_TAIL:
        return False                                     # starts before the change or after it (plus the tail)
    el, kind = sym["element"], sym["kind"]
    tgt = fp["target"]
    if kind == "link_status_down":
        return el == tgt and change["action"] == "down"
    if kind in ("svc_unreachable", "svc_loss_sla"):
        return el in fp["services_cut_off"]             # a planned outage of a bridge link cuts off what lies beyond it
    if kind == "pop_unreachable":
        return False                                     # planned work is on links; no router stops answering
    if kind == "svc_latency_sla":
        return el in fp["services_rerouted"]
    if kind in ("link_util_high", "util_up"):
        return el in fp["links_gaining_traffic"]
    if kind == "util_down":
        return el in fp["links_losing_traffic"]
    if kind == "link_latency_high":
        return el in fp["links_gaining_traffic"]        # queueing on a link that took the rerouted traffic
    if kind in ("link_discards", "loss_up"):
        return False                                     # the change review keeps every link under 85% load
    return False


def symptoms(inc):
    """Observed symptoms in the incident window [t0, now), as known at now - 1: alerts and telemetry shifts. Returns
    (symptoms, neutral, n_services_with_loss_one_day_earlier)."""
    t0, now, floor = inc.t0, inc.now, inc.floor
    known = alerts_at(inc.alerts, now - 1)
    end = known.t_end.fillna(np.inf)
    sel = known[(known.t_start < now) & (end > t0)]
    out, neutral = [], []
    a0, a1 = t0 - 288, now - 288                       # the same steps one day earlier: background alerts
    bg_sel = known[(known.t_start < a1) & (end > a0) & (known.t_start >= floor)]
    background = set(zip(bg_sel.rule, bg_sel.target))
    bg_loss = int(bg_sel[bg_sel.rule == "svc_loss_sla"].target.nunique())
    for r in sel.itertuples():
        s = {"kind": r.rule, "element": r.target, "start": int(max(r.t_start, floor)),
             "end": None if not np.isfinite(r.t_end) else int(r.t_end), "source": "alert"}
        (neutral if r.rule in NEUTRAL_RULES or (r.rule, r.target) in background else out).append(s)
    obs = inc.obs
    a, b = t0, now
    if b - 288 > floor:
        with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for name, arr, thr, rel in (("util", obs.util, UTIL_SHIFT, UTIL_SHIFT_REL), ("loss", obs.discards, LOSS_SHIFT_PCT, 0.0)):
                cur = np.nanmean(arr[a:b], 0)
                base = np.nanmean(arr[max(floor, a - 288):b - 288], 0)
                if name == "util":                       # a network-wide demand shift (busier or quieter day) is no symptom
                    ok = np.isfinite(cur) & np.isfinite(base) & (base > 0.05)
                    factor = float(np.median(cur[ok] / base[ok])) if ok.sum() >= 3 else 1.0
                    base = base * factor
                d = cur - base
                big = np.isfinite(d) & (np.abs(d) >= np.maximum(thr, rel * np.abs(np.nan_to_num(base))))
                for j in np.flatnonzero(big):
                    if name == "loss" and d[j] < 0:
                        continue
                    seg = arr[a:b, j] - (base[j] if np.isfinite(base[j]) else 0)
                    hit = np.flatnonzero(np.isfinite(seg) & ((seg >= thr) if d[j] > 0 else (seg <= -thr)))
                    start = a + int(hit[0]) if len(hit) else a
                    out.append({"kind": f"{name}_{'up' if d[j] > 0 else 'down'}", "element": inc.links[j], "start": start,
                                "end": None, "source": "telemetry", "change": round(float(d[j]), 3)})
    return out, neutral, bg_loss


def change_scope(inc, inventory_kg=None):
    """The tool: logged drains and planned outages overlapping the incident window (started by now), what each can
    explain according to the KG, and the symptoms none of them explains. Times are relative to the window start."""
    kg = inventory_kg if inventory_kg is not None else inc.kg
    rel = inc.rel
    active = [e for e in inc.obs.events if e.get("type") == "maintenance" and e["action"] in CHANGE_ACTIONS
              and e["t_start"] < inc.now and e["t_end"] + MAINT_TAIL > inc.t0]
    syms, neutral, bg_loss = symptoms(inc)
    fps = [(c, footprint(kg, c)) for c in active]
    explained, unexplained = [], []
    for s in syms:
        by = [c["id"] for c, fp in fps if _explains(fp, c, s, inc.floor)]
        row = {"symptom": s["kind"], "element": s["element"], "start": rel(s["start"]),
               **({"change_vs_day_before": s["change"]} if "change" in s else {})}
        (explained if by else unexplained).append(row | ({"explained_by": by} if by else {}))
    # service SLA loss: probes see some every day; it counts only when more services show it than one day earlier
    loss_now = {r["element"] for r in unexplained if r["symptom"] == "svc_loss_sla"}
    loss_note = {"services_with_unexplained_loss_now": len(loss_now), "services_with_loss_one_day_earlier": bg_loss}
    if len(loss_now) < bg_loss + SERVICE_LOSS_EXCESS:
        neutral += [r for r in unexplained if r["symptom"] == "svc_loss_sla"]
        unexplained = [r for r in unexplained if r["symptom"] != "svc_loss_sla"]
    unexplained.sort(key=lambda r: (r["start"], r["symptom"], r["element"]))
    explained.sort(key=lambda r: (r["start"], r["symptom"], r["element"]))
    fp_out = []
    for c, fp in fps:
        fp_out.append({"change_id": c["id"], "action": c["action"], "target": c["target"], "start": rel(max(c["t_start"], inc.floor)),
                       "end": rel(c["t_end"]), "links_losing_traffic": fp["links_losing_traffic"][:MAX_LISTED],
                       "links_gaining_traffic": fp["links_gaining_traffic"][:MAX_LISTED],
                       "n_links_gaining_traffic": len(fp["links_gaining_traffic"]),
                       "n_services_rerouted": len(fp["services_rerouted"])})
    if not active:
        verdict = "no logged drain or planned outage is active in this window; the change log explains nothing here"
    elif unexplained:
        verdict = (f"{len(unexplained)} symptom(s) fall outside what the logged change(s) can explain: they need another cause")
    else:
        verdict = "every observed symptom is within what the logged change(s) can explain"
    return {"window": [0, rel(inc.now - 1)], "active_changes": fp_out, "n_symptoms": len(syms),
            "n_unexplained": len(unexplained), "unexplained_symptoms": unexplained[:MAX_LISTED * 2],
            "n_explained": len(explained), "explained_symptoms": explained[:MAX_LISTED],
            "neutral": {"n": len(neutral), "service_loss": loss_note,
                        "note": "collector gaps, probe timeouts, alerts also seen at the same time one day earlier, and "
                                "service loss no wider than one day earlier occur without faults or changes; not counted"},
            "verdict": verdict,
            "rules": {"drain": "explains traffic leaving the target link and arriving on the rerouted links, and more latency "
                               "on rerouted services, while active or up to 60 minutes after it ends",
                      "down": "as drain, plus the target link reported down",
                      "never_explained": "any other link down, a router unreachable, discards, service loss or unreachability "
                                         "(except services a planned outage cuts off), "
                                         "traffic shifts on links the change does not reroute, symptoms starting before the "
                                         "change or more than 60 minutes after it ends"}}
